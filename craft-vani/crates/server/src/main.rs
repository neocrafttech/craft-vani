use axum::{
    Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use candle::Device;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tower_http::cors::CorsLayer;

use crate::decoder::Decoder;
use inference::InferenceOutput;
mod decoder;
mod utils;

struct ServerState {
    decoder: Option<Decoder>,
    device: Device,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let device = utils::device(true).unwrap();
    println!("Pre-loading model...");

    // Paths relative to the workspace root if run from there
    let model_dir = Path::new("whisper-tiny");
    let mel_path = Path::new("mel_filters.safetensors");

    let decoder = match Decoder::load_from_dir(model_dir, mel_path, &device) {
        Ok(d) => {
            println!("Model pre-loaded successfully");
            Some(d)
        }
        Err(e) => {
            eprintln!("Failed to pre-load model at {:?}: {}", model_dir, e);
            None
        }
    };

    let state = Arc::new(Mutex::new(ServerState { decoder, device: device.clone() }));

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(tower_http::services::ServeDir::new("."))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<Arc<Mutex<ServerState>>>,
) -> impl IntoResponse {
    ws.max_message_size(512 * 1024 * 1024).on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<Mutex<ServerState>>) {
    println!("New WebSocket connection");
    let (sender, mut receiver) = socket.split();
    
    // Create a channel to send inference tasks to a background task
    let (tx, mut rx) = mpsc::channel::<Vec<f32>>(100);
    let sender = Arc::new(Mutex::new(sender));
    
    // Spawn a background task to handle inference
    let state_clone = state.clone();
    let sender_clone = sender.clone();
    
    tokio::spawn(async move {
        while let Some(samples) = rx.recv().await {
            println!("[Inference] Processing {} samples", samples.len());
            
            let mut state = state_clone.lock().await;
            if let Some(decoder) = &mut state.decoder {
                match decoder.run_raw(&samples) {
                    Ok(segments) => {
                        println!("[Inference] Decoding successful: {} segments", segments.len());
                        decoder.reset_kv_cache();
                        let resp = bincode::serialize::<InferenceOutput>(
                            &InferenceOutput::Decoded(segments),
                        )
                        .unwrap();
                        drop(state);
                        let mut sender = sender_clone.lock().await;
                        let _ = sender.send(Message::Binary(resp.into())).await;
                    }
                    Err(err) => {
                        eprintln!("[Inference] Decoding error: {}", err);
                        let resp = bincode::serialize::<InferenceOutput>(
                            &InferenceOutput::Error(err.to_string()),
                        )
                        .unwrap();
                        drop(state);
                        let mut sender = sender_clone.lock().await;
                        let _ = sender.send(Message::Binary(resp.into())).await;
                    }
                }
            } else {
                eprintln!("[Inference] Decoder not initialized");
                let resp = bincode::serialize::<InferenceOutput>(&InferenceOutput::Error(
                    "Backend model not loaded".to_string(),
                ))
                .unwrap();
                drop(state);
                let mut sender = sender_clone.lock().await;
                let _ = sender.send(Message::Binary(resp.into())).await;
            }
        }
    });

    // Main loop: just receive and queue chunks, don't process them
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Binary(bin) = msg {
            println!("[WebSocket] Received chunk of {} bytes", bin.len());
            
            let samples: Vec<f32> = if bin.len() % 4 == 0 {
                bin.chunks_exact(4)
                    .map(|chunk| {
                        let mut array = [0u8; 4];
                        array.copy_from_slice(chunk);
                        f32::from_le_bytes(array)
                    })
                    .collect()
            } else {
                eprintln!("[WebSocket] ERROR: Received {} bytes which is not divisible by 4", bin.len());
                continue;
            };

            if !samples.is_empty() {
                println!("[WebSocket] Queuing {} samples for inference", samples.len());
                if tx.send(samples).await.is_err() {
                    eprintln!("[WebSocket] Failed to send samples to inference channel");
                    break;
                }
            }
        }
    }
    println!("[WebSocket] Connection closed");
}
