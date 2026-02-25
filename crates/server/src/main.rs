use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use inference::{InferenceOutput, InferenceResponse, SarvamOutput};
use reqwest::multipart;
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{Duration, timeout};
use tower_http::cors::CorsLayer;

use crate::decoder::Decoder;
mod decoder;
mod utils;

const TARGET_SAMPLE_RATE: usize = 16_000;
const MIN_DECODE_SECONDS: usize = 4;
const DEFAULT_OVERLAP_SECONDS: usize = 1;
const RESAMPLE_CHUNK_SIZE: usize = 1024;
const MIN_TEXT_LEN: usize = 3;
const MIN_AVG_LOGPROB: f64 = -1.2;
const MAX_NO_SPEECH_PROB: f64 = 0.5;
const MIN_RMS_ENERGY: f32 = 0.003;
const MAX_RMS_THRESHOLD: f32 = 0.02;
const MIN_VOICED_RATIO: f32 = 0.08;
const VOICED_FRAME_SIZE: usize = 320; // 20ms @ 16kHz
const SARVAM_STT_URL: &str = "https://api.sarvam.ai/speech-to-text";
const SARVAM_MODEL: &str = "saaras:v3";
const SARVAM_TIMEOUT_SECS: u64 = 4;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the whisper model directory
    #[arg(short, long)]
    model: Option<String>,
}

struct ServerState {
    decoder: Option<Decoder>,
    decoder_error: Option<String>,
    sarvam_client: reqwest::Client,
    sarvam_api_key: Option<String>,
}

enum InferenceTask {
    SetConfig {
        sample_rate: Option<usize>,
        whisper_enabled: Option<bool>,
        sarvam_enabled: Option<bool>,
    },
    Samples(Vec<f32>),
}

#[derive(Deserialize)]
struct WsConfigMessage {
    #[serde(rename = "type")]
    message_type: String,
    sample_rate: Option<u32>,
    whisper_enabled: Option<bool>,
    sarvam_enabled: Option<bool>,
}

#[derive(Clone, Copy)]
struct EngineSelection {
    whisper_enabled: bool,
    sarvam_enabled: bool,
}

impl Default for EngineSelection {
    fn default() -> Self {
        Self { whisper_enabled: true, sarvam_enabled: true }
    }
}

struct AudioPipeline {
    in_sample_rate: usize,
    min_decode_samples: usize,
    overlap_samples: usize,
    buffered_pcm: Vec<f32>,
    resampler: FastFixedIn<f32>,
}

#[derive(Default)]
struct TranscriptStabilizer {
    last_emitted: Option<String>,
}

impl TranscriptStabilizer {
    fn update_and_should_emit(&mut self, candidate: &str) -> bool {
        let normalized = normalize_for_stability(candidate);
        if normalized.is_empty() {
            return false;
        }
        if let Some(last) = &self.last_emitted {
            if normalized == *last {
                return false;
            }
            if normalized.starts_with(last) {
                let delta = normalized.len().saturating_sub(last.len());
                if delta < 3 {
                    return false;
                }
            }
        }
        self.last_emitted = Some(normalized);
        true
    }
}

fn normalize_for_stability(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_space = false;
    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            out.push(ch);
            prev_space = false;
        } else if ch.is_whitespace() && !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    out.trim().to_string()
}

fn is_meaningful_text(text: &str) -> bool {
    let t = text.trim();
    t.len() >= MIN_TEXT_LEN && t.chars().any(|c| c.is_alphanumeric())
}

fn is_hallucination_fragment(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    matches!(t.as_str(), ".com" | "com" | ".co" | ".org" | ".net")
        || (t.starts_with('.') && t.len() <= 8)
}

fn is_low_value_text(text: &str) -> bool {
    let t = text.trim();
    if t.len() < MIN_TEXT_LEN {
        return true;
    }
    let alpha_count = t.chars().filter(|c| c.is_alphabetic()).count();
    alpha_count < 2
}

fn keep_segment(segment: &inference::Segment) -> bool {
    let text_ok = is_meaningful_text(&segment.dr.text)
        && !is_hallucination_fragment(&segment.dr.text)
        && !is_low_value_text(&segment.dr.text);
    let logprob_ok =
        segment.dr.avg_logprob.is_finite() && segment.dr.avg_logprob >= MIN_AVG_LOGPROB;
    let no_speech_ok =
        !segment.dr.no_speech_prob.is_finite() || segment.dr.no_speech_prob <= MAX_NO_SPEECH_PROB;
    text_ok && logprob_ok && no_speech_ok
}

fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn voiced_ratio(samples: &[f32], frame_size: usize, frame_rms_threshold: f32) -> f32 {
    if samples.len() < frame_size || frame_size == 0 {
        return 0.0;
    }
    let mut voiced = 0usize;
    let mut total = 0usize;
    for frame in samples.chunks_exact(frame_size) {
        total += 1;
        if rms_energy(frame) >= frame_rms_threshold {
            voiced += 1;
        }
    }
    if total == 0 { 0.0 } else { voiced as f32 / total as f32 }
}

fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let sample = (clamped * i16::MAX as f32) as i16;
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

fn first_non_empty_text(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(v) = value.get(*key).and_then(Value::as_str) {
            let t = v.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn parse_sarvam_text(value: &Value) -> Option<String> {
    if let Some(text) = first_non_empty_text(value, &["transcript", "text"]) {
        return Some(text);
    }

    if let Some(data) = value.get("data") {
        if let Some(text) = first_non_empty_text(data, &["transcript", "text"]) {
            return Some(text);
        }
        if let Some(arr) = data.as_array() {
            for item in arr {
                if let Some(text) = first_non_empty_text(item, &["transcript", "text"]) {
                    return Some(text);
                }
            }
        }
    }
    None
}

async fn transcribe_with_sarvam(
    client: &reqwest::Client, api_key: Option<&str>, pcm: &[f32], sample_rate: usize,
) -> Result<String, String> {
    let key = api_key.ok_or_else(|| "SARVAM_API_KEY not found".to_string())?;
    let wav = encode_wav_bytes(pcm, sample_rate as u32).map_err(|e| e.to_string())?;
    let file_part = multipart::Part::bytes(wav).file_name("audio.wav").mime_str("audio/wav");
    let file_part = file_part.map_err(|e| e.to_string())?;
    let form = multipart::Form::new().part("file", file_part).text("model", SARVAM_MODEL);
    let response = client
        .post(SARVAM_STT_URL)
        .header("api-subscription-key", key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = response.status();
    let body = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("Sarvam HTTP {status}: {body}"));
    }
    let value: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    parse_sarvam_text(&value).ok_or_else(|| "Sarvam response missing transcript text".to_string())
}

fn encode_response(response: InferenceResponse) -> Vec<u8> {
    bincode::serialize::<InferenceResponse>(&response).unwrap()
}

fn empty_response(selection: EngineSelection) -> InferenceResponse {
    let empty_whisper = InferenceOutput::Decoded(vec![]);
    let empty_sarvam = SarvamOutput::Empty;
    match (selection.whisper_enabled, selection.sarvam_enabled) {
        (true, true) => InferenceResponse::Both { whisper: empty_whisper, sarvam: empty_sarvam },
        (true, false) => InferenceResponse::Whisper { whisper: empty_whisper },
        (false, true) => InferenceResponse::Sarvam { sarvam: empty_sarvam },
        (false, false) => InferenceResponse::Whisper { whisper: empty_whisper },
    }
}

impl AudioPipeline {
    fn decode_window_seconds() -> usize {
        let from_env = std::env::var("CRAFT_VANI_MIN_DECODE_SECONDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MIN_DECODE_SECONDS);
        from_env.clamp(1, 15)
    }

    fn new(in_sample_rate: usize) -> anyhow::Result<Self> {
        let sample_rate = in_sample_rate.max(1);
        let resample_ratio = TARGET_SAMPLE_RATE as f64 / sample_rate as f64;
        let decode_window_seconds = Self::decode_window_seconds();
        let min_decode_samples = decode_window_seconds * sample_rate;
        let overlap_seconds = std::env::var("CRAFT_VANI_OVERLAP_SECONDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_OVERLAP_SECONDS)
            .clamp(0, 5);
        let overlap_samples = overlap_seconds * sample_rate;
        let resampler = FastFixedIn::new(
            resample_ratio,
            10.0,
            PolynomialDegree::Septic,
            RESAMPLE_CHUNK_SIZE,
            1,
        )?;
        println!(
            "[Inference] Decode window: {}s (+{}s overlap, {} input samples @ {} Hz)",
            decode_window_seconds, overlap_seconds, min_decode_samples, sample_rate
        );
        Ok(Self {
            in_sample_rate: sample_rate,
            min_decode_samples,
            overlap_samples,
            buffered_pcm: Vec::new(),
            resampler,
        })
    }

    fn set_sample_rate(&mut self, in_sample_rate: usize) -> anyhow::Result<()> {
        let new_pipeline = Self::new(in_sample_rate)?;
        *self = new_pipeline;
        Ok(())
    }

    fn push_and_resample_if_ready(&mut self, samples: &[f32]) -> anyhow::Result<Option<Vec<f32>>> {
        self.buffered_pcm.extend_from_slice(samples);
        if self.buffered_pcm.len() < self.min_decode_samples {
            return Ok(None);
        }

        let full_chunks = self.buffered_pcm.len() / RESAMPLE_CHUNK_SIZE;
        if full_chunks == 0 {
            return Ok(None);
        }
        let processed_len = full_chunks * RESAMPLE_CHUNK_SIZE;
        let mut resampled_pcm = Vec::new();
        for chunk in 0..full_chunks {
            let start = chunk * RESAMPLE_CHUNK_SIZE;
            let end = (chunk + 1) * RESAMPLE_CHUNK_SIZE;
            let input_chunk = &self.buffered_pcm[start..end];
            let pcm = self.resampler.process(&[input_chunk], None)?;
            resampled_pcm.extend_from_slice(&pcm[0]);
        }

        let max_overlap = processed_len.saturating_sub(RESAMPLE_CHUNK_SIZE);
        let overlap = self.overlap_samples.min(max_overlap);
        let keep_from = processed_len.saturating_sub(overlap);
        self.buffered_pcm.copy_within(keep_from.., 0);
        self.buffered_pcm.truncate(self.buffered_pcm.len() - keep_from);

        if resampled_pcm.is_empty() { Ok(None) } else { Ok(Some(resampled_pcm)) }
    }
}

fn choose_model_dir(args_model: Option<String>) -> String {
    if let Some(model) = args_model {
        return model;
    }

    if let Ok(model) = std::env::var("MODEL_NAME")
        && !model.trim().is_empty()
    {
        return model;
    }

    let candidates = ["whisper-small", "whisper-tiny.en", "whisper-base", "whisper-medium"];
    for candidate in candidates {
        if Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }

    "whisper-small".to_string()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let _ = dotenvy::dotenv();

    let device = match utils::device() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to initialize device: {}", e);
            std::process::exit(1);
        }
    };

    println!("Pre-loading model...");

    let args = Args::parse();
    // Paths relative to the workspace root if run from there
    let model_name = choose_model_dir(args.model);
    let model_dir = Path::new(&model_name);
    println!("Using model directory: {}", model_name);

    let (decoder, decoder_error) = match Decoder::load_from_dir(model_dir, &device) {
        Ok(d) => {
            println!("Model pre-loaded successfully");
            (Some(d), None)
        }
        Err(e) => {
            let details = format!(
                "Failed to pre-load model at {}: {}. Provide a valid model via --model <dir> or MODEL_NAME=<dir> and ensure preprocessor_config.json, config.json, tokenizer.json, and model weights are present.",
                model_dir.display(),
                e
            );
            eprintln!("{}", details);
            (None, Some(details))
        }
    };

    let sarvam_api_key = std::env::var("SARVAM_API_KEY").ok();
    if sarvam_api_key.is_none() {
        eprintln!("SARVAM_API_KEY not set; Sarvam transcription will be disabled");
    }
    let sarvam_client = reqwest::Client::builder()
        .build()
        .context("failed to initialize Sarvam HTTP client")
        .unwrap();

    let state =
        Arc::new(Mutex::new(ServerState { decoder, decoder_error, sarvam_client, sarvam_api_key }));

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
    let (tx, mut rx) = mpsc::channel::<InferenceTask>(100);
    let sender = Arc::new(Mutex::new(sender));

    // Spawn a background task to handle inference
    let state_clone = state.clone();
    let sender_clone = sender.clone();

    tokio::spawn(async move {
        let mut pipeline = match AudioPipeline::new(TARGET_SAMPLE_RATE) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[Inference] Failed to initialize audio pipeline: {}", err);
                return;
            }
        };
        let mut noise_floor_rms = MIN_RMS_ENERGY * 0.5;
        let mut stabilizer = TranscriptStabilizer::default();
        let mut selection = EngineSelection::default();

        while let Some(task) = rx.recv().await {
            match task {
                InferenceTask::SetConfig { sample_rate, whisper_enabled, sarvam_enabled } => {
                    if let Some(sample_rate) = sample_rate {
                        if let Err(err) = pipeline.set_sample_rate(sample_rate) {
                            eprintln!("[Inference] Failed to reconfigure sample rate: {}", err);
                        } else {
                            println!("[Inference] Configured input sample rate: {}", sample_rate);
                        }
                    }
                    if let Some(enabled) = whisper_enabled {
                        selection.whisper_enabled = enabled;
                    }
                    if let Some(enabled) = sarvam_enabled {
                        selection.sarvam_enabled = enabled;
                    }
                    println!(
                        "[Inference] Engine selection => whisper: {}, sarvam: {}",
                        selection.whisper_enabled, selection.sarvam_enabled
                    );
                }
                InferenceTask::Samples(samples) => {
                    println!("[Inference] Received {} input samples", samples.len());
                    let resampled = match pipeline.push_and_resample_if_ready(&samples) {
                        Ok(Some(pcm)) => pcm,
                        Ok(None) => {
                            // Keep frontend queue flowing while we accumulate enough audio.
                            let resp = encode_response(empty_response(selection));
                            let mut sender = sender_clone.lock().await;
                            let _ = sender.send(Message::Binary(resp.into())).await;
                            continue;
                        }
                        Err(err) => {
                            eprintln!("[Inference] Resampling error: {}", err);
                            let err = err.to_string();
                            let response =
                                match (selection.whisper_enabled, selection.sarvam_enabled) {
                                    (true, true) => InferenceResponse::Both {
                                        whisper: InferenceOutput::Error(err),
                                        sarvam: SarvamOutput::Empty,
                                    },
                                    (true, false) => InferenceResponse::Whisper {
                                        whisper: InferenceOutput::Error(err),
                                    },
                                    (false, true) => InferenceResponse::Sarvam {
                                        sarvam: SarvamOutput::Error(err),
                                    },
                                    (false, false) => empty_response(selection),
                                };
                            let resp = encode_response(response);
                            let mut sender = sender_clone.lock().await;
                            let _ = sender.send(Message::Binary(resp.into())).await;
                            continue;
                        }
                    };

                    println!(
                        "[Inference] Processing {} samples at {} Hz",
                        resampled.len(),
                        TARGET_SAMPLE_RATE
                    );
                    let rms = rms_energy(&resampled);
                    if rms < noise_floor_rms * 1.5 {
                        noise_floor_rms = 0.95 * noise_floor_rms + 0.05 * rms;
                    }
                    let rms_threshold =
                        (noise_floor_rms * 3.0).clamp(MIN_RMS_ENERGY, MAX_RMS_THRESHOLD);
                    let voiced = voiced_ratio(&resampled, VOICED_FRAME_SIZE, rms_threshold * 0.8);
                    if rms < rms_threshold || voiced < MIN_VOICED_RATIO {
                        println!(
                            "[Inference] Skipping chunk (rms={:.6}, thr={:.6}, voiced={:.2})",
                            rms, rms_threshold, voiced
                        );
                        let resp = encode_response(empty_response(selection));
                        let mut sender = sender_clone.lock().await;
                        let _ = sender.send(Message::Binary(resp.into())).await;
                        continue;
                    }
                    let whisper_output = if selection.whisper_enabled {
                        let mut state = state_clone.lock().await;
                        let output = if let Some(decoder) = &mut state.decoder {
                            match decoder.run_raw(&resampled) {
                                Ok(segments) => {
                                    let total_segments = segments.len();
                                    let filtered_segments: Vec<_> =
                                        segments.into_iter().filter(keep_segment).collect();
                                    let candidate_text = filtered_segments
                                        .iter()
                                        .map(|s| s.dr.text.trim())
                                        .filter(|t| !t.is_empty())
                                        .collect::<Vec<_>>()
                                        .join(" ");
                                    let emit = stabilizer.update_and_should_emit(&candidate_text);
                                    let output_segments =
                                        if emit { filtered_segments } else { vec![] };
                                    println!(
                                        "[Inference] Whisper successful: {} kept / {} total (emit={})",
                                        output_segments.len(),
                                        total_segments,
                                        emit
                                    );
                                    InferenceOutput::Decoded(output_segments)
                                }
                                Err(err) => {
                                    eprintln!("[Inference] Whisper decoding error: {}", err);
                                    InferenceOutput::Error(err.to_string())
                                }
                            }
                        } else {
                            let err = state
                                .decoder_error
                                .clone()
                                .unwrap_or_else(|| "Backend model not loaded".to_string());
                            eprintln!("[Inference] Decoder not initialized: {}", err);
                            InferenceOutput::Error(err)
                        };

                        Some(output)
                    } else {
                        None
                    };

                    let sarvam_output = if selection.sarvam_enabled {
                        let (sarvam_client, sarvam_api_key) = {
                            let state = state_clone.lock().await;
                            (state.sarvam_client.clone(), state.sarvam_api_key.clone())
                        };
                        let sarvam_result = match timeout(
                            Duration::from_secs(SARVAM_TIMEOUT_SECS),
                            transcribe_with_sarvam(
                                &sarvam_client,
                                sarvam_api_key.as_deref(),
                                &resampled,
                                TARGET_SAMPLE_RATE,
                            ),
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_) => Err(format!(
                                "Sarvam request timed out after {}s",
                                SARVAM_TIMEOUT_SECS
                            )),
                        };
                        match sarvam_result {
                            Ok(text) => {
                                println!("[Inference] Sarvam successful");
                                SarvamOutput::Text(text)
                            }
                            Err(err) => {
                                eprintln!("[Inference] Sarvam error: {}", err);
                                SarvamOutput::Error(err)
                            }
                        }
                    } else {
                        SarvamOutput::Empty
                    };

                    let response = match (whisper_output, selection.sarvam_enabled) {
                        (Some(whisper), true) => {
                            InferenceResponse::Both { whisper, sarvam: sarvam_output }
                        }
                        (Some(whisper), false) => InferenceResponse::Whisper { whisper },
                        (None, true) => InferenceResponse::Sarvam { sarvam: sarvam_output },
                        (None, false) => empty_response(selection),
                    };

                    let resp = encode_response(response);
                    let mut sender = sender_clone.lock().await;
                    let _ = sender.send(Message::Binary(resp.into())).await;
                }
            }
        }
    });

    // Main loop: just receive and queue chunks, don't process them
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                if let Ok(config) = serde_json::from_str::<WsConfigMessage>(&text)
                    && config.message_type == "config"
                    && tx
                        .send(InferenceTask::SetConfig {
                            sample_rate: config.sample_rate.map(|sr| sr as usize),
                            whisper_enabled: config.whisper_enabled,
                            sarvam_enabled: config.sarvam_enabled,
                        })
                        .await
                        .is_err()
                {
                    eprintln!("[WebSocket] Failed to send config to inference channel");
                    break;
                }
            }
            Message::Binary(bin) => {
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
                    eprintln!(
                        "[WebSocket] ERROR: Received {} bytes which is not divisible by 4",
                        bin.len()
                    );
                    continue;
                };

                if !samples.is_empty() {
                    println!("[WebSocket] Queuing {} samples for inference", samples.len());
                    if tx.send(InferenceTask::Samples(samples)).await.is_err() {
                        eprintln!("[WebSocket] Failed to send samples to inference channel");
                        break;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    println!("[WebSocket] Connection closed");
}
