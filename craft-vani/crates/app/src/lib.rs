use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AudioContext, AudioContextOptions, BinaryType, MediaStream, ScriptProcessorNode, WebSocket,
};
use yew::{Component, Context, Html, html};

use inference::console_log;
use inference::{InferenceOutput, Segment};

// fetch_url is no longer needed as model loading moved to backend

pub enum Msg {
    UpdateStatus(String),
    WsOut(Result<InferenceOutput, String>),
    ToggleMute,
    RecordingStarted(Result<RecordingState, String>),
    Process,
    WsConnected(WebSocket),
}

pub struct RecordingState {
    pub audio_context: AudioContext,
    pub stream: MediaStream,
    pub processor: ScriptProcessorNode,
    pub _closure: Closure<dyn FnMut(web_sys::AudioProcessingEvent)>,
    pub samples: std::rc::Rc<std::cell::RefCell<Vec<f32>>>,
    pub muted: std::rc::Rc<std::cell::Cell<bool>>,
}

pub struct CurrentDecode {
    pub start_time: Option<f64>,
    pub offset_samples: usize,
}

pub struct App {
    status: String,
    segments: Vec<Segment>,
    current_decode: Option<CurrentDecode>,
    ws: Option<WebSocket>,
    recording: Option<RecordingState>,
    muted: bool,
    _interval: Option<gloo_timers::callback::Interval>,
    decoded_samples: usize,
    chunk_queue: Vec<(Vec<u8>, usize)>, // (bytes, offset_samples)
}

fn performance_now() -> Option<f64> {
    let window = web_sys::window()?;
    let performance = window.performance()?;
    Some(performance.now() / 1000.)
}

async fn start_recording(audio_context: AudioContext, is_muted: bool) -> Msg {
    let result = async {
        let window = web_sys::window().ok_or("window")?;
        let navigator = window.navigator();
        let devices = navigator.media_devices().map_err(|e| format!("{e:?}"))?;

        let constraints = web_sys::MediaStreamConstraints::new();
        constraints.set_audio(&JsValue::from_bool(true));

        let stream_promise = devices
            .get_user_media_with_constraints(&constraints)
            .map_err(|e| format!("{e:?}"))?;
        let stream = JsFuture::from(stream_promise)
            .await
            .map_err(|e| format!("{e:?}"))?;
        let stream = MediaStream::from(stream);

        let tracks = stream.get_audio_tracks();
        for i in 0..tracks.length() {
            let track = web_sys::MediaStreamTrack::from(tracks.get(i));
            track.set_enabled(true);
        }

        // Use the context passed in (created in ToggleRecording to ensure user activation)
        let _ = audio_context.resume();

        let source = audio_context
            .create_media_stream_source(&stream)
            .map_err(|e| format!("{e:?}"))?;

        // 4096 is a buffer size.
        let processor = audio_context
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                4096, 1, 1,
            )
            .map_err(|e| format!("{e:?}"))?;

        let muted = std::rc::Rc::new(std::cell::Cell::new(is_muted));
        let muted_cb = muted.clone();
        let samples = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let samples_cb = samples.clone();
        let closure = Closure::wrap(Box::new(move |e: web_sys::AudioProcessingEvent| {
            if muted_cb.get() {
                return;
            }
            let input_buffer = e.input_buffer().unwrap();
            let data = input_buffer.get_channel_data(0).unwrap();
            samples_cb.borrow_mut().extend_from_slice(&data);
        }) as Box<dyn FnMut(web_sys::AudioProcessingEvent)>);

        processor.set_onaudioprocess(Some(closure.as_ref().unchecked_ref()));
        source
            .connect_with_audio_node(&processor)
            .map_err(|e| format!("{e:?}"))?;
        processor
            .connect_with_audio_node(&audio_context.destination())
            .map_err(|e| format!("{e:?}"))?;

        Ok(RecordingState {
            audio_context,
            stream,
            processor,
            _closure: closure,
            samples,
            muted,
        })
    }
    .await;

    Msg::RecordingStarted(result)
}

fn connect_ws(ctx: &Context<App>) -> Result<WebSocket, JsValue> {
    let ws = WebSocket::new("ws://localhost:3000/ws")?;
    ws.set_binary_type(BinaryType::Arraybuffer);

    let link = ctx.link().clone();
    let onmessage_callback =
        Closure::<dyn FnMut(_)>::wrap(Box::new(move |e: web_sys::MessageEvent| {
            if let Ok(abuffer) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                let vec = js_sys::Uint8Array::new(&abuffer).to_vec();
                if let Ok(output) = bincode::deserialize::<InferenceOutput>(&vec) {
                    link.send_message(Msg::WsOut(Ok(output)));
                }
            }
        }));
    ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
    onmessage_callback.forget();

    let link = ctx.link().clone();
    let onerror_callback =
        Closure::<dyn FnMut(_)>::wrap(Box::new(move |e: web_sys::ErrorEvent| {
            link.send_message(Msg::WsOut(Err(format!("WS Error: {:?}", e.message()))));
        }));
    ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    onerror_callback.forget();

    let link = ctx.link().clone();
    let ws_clone = ws.clone();
    let onopen_callback = Closure::<dyn FnMut()>::wrap(Box::new(move || {
        link.send_message(Msg::WsConnected(ws_clone.clone()));
    }));
    ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
    onopen_callback.forget();

    Ok(ws)
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let status = "connecting to backend...".to_string();

        let _ws = connect_ws(ctx).map_err(|e| console_log!("failed to connect ws: {:?}", e));

        Self {
            status,
            segments: vec![],
            current_decode: None,
            ws: None,
            recording: None,
            muted: true,
            _interval: None,
            decoded_samples: 0,
            chunk_queue: vec![],
        }
    }

    fn rendered(&mut self, _ctx: &Context<Self>, _first_render: bool) {}

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::WsConnected(ws) => {
                self.ws = Some(ws);
                self.status = "connected to backend".to_string();
                true
            }
            Msg::WsOut(output) => {
                let dt = self.current_decode.as_ref().and_then(|current_decode| {
                    current_decode.start_time.and_then(|start_time| {
                        performance_now().map(|stop_time| stop_time - start_time)
                    })
                });
                let current_decode = self.current_decode.take();
                match output {
                    Ok(InferenceOutput::Decoded(new_segments)) => {
                        self.status = match dt {
                            None => "decoding succeeded!".to_string(),
                            Some(dt) => format!("decoding succeeded in {dt:.2}s"),
                        };
                        if let Some(current) = current_decode {
                            let offset_secs = current.offset_samples as f64 / 16000.0;
                            for mut segment in new_segments {
                                segment.start += offset_secs;
                                self.segments.push(segment);
                            }
                        }
                    }
                    Ok(InferenceOutput::Error(err)) => {
                        self.status = format!("backend error: {err}");
                    }
                    Err(err) => {
                        self.status = format!("websocket error: {err}");
                    }
                }
                
                // Try to send the next chunk from the queue
                if self.current_decode.is_none() && !self.chunk_queue.is_empty() {
                    if let Some(ws) = &self.ws {
                        let (bytes, offset_samples) = self.chunk_queue.remove(0);
                        let start_time = performance_now();
                        self.current_decode = Some(CurrentDecode { start_time, offset_samples });
                        console_log!("Sending next queued chunk of {} bytes, {} remaining in queue", bytes.len(), self.chunk_queue.len());
                        let _ = ws.send_with_u8_array(&bytes);
                    }
                }
                
                true
            }
            Msg::UpdateStatus(status) => {
                self.status = status;
                true
            }
            Msg::ToggleMute => {
                self.muted = !self.muted;
                if let Some(recording) = &self.recording {
                    recording.muted.set(self.muted);
                }

                if self.muted {
                    self._interval = None;
                    self.status = "Muted".to_string();
                } else {
                    // Unmuting
                    self.status = "Unmuted, listening...".to_string();
                    if self.recording.is_none() {
                        let options = AudioContextOptions::new();
                        options.set_sample_rate(16000.0);
                        match AudioContext::new_with_context_options(&options) {
                            Ok(audio_context) => {
                                ctx.link().send_future(start_recording(audio_context, self.muted));
                            }
                            Err(err) => {
                                self.status = format!("Failed to create AudioContext: {:?}", err);
                            }
                        }
                    }

                    // Start periodic processing every 2 seconds
                    let link = ctx.link().clone();
                    self._interval = Some(gloo_timers::callback::Interval::new(2000, move || {
                        link.send_message(Msg::Process);
                    }));
                }
                true
            }
            Msg::Process => {
                console_log!(
                    "Msg::Process triggered {} {}",
                    self.muted,
                    self.current_decode.is_none()
                );
                
                // Collect any new samples into the queue
                if !self.muted {
                    if let Some(recording) = &self.recording {
                        let all_samples = recording.samples.borrow();
                        let new_samples_count = all_samples.len();
                        if new_samples_count > self.decoded_samples {
                            let samples = all_samples[self.decoded_samples..].to_vec();
                            let offset_samples = self.decoded_samples;
                            self.decoded_samples = new_samples_count;
                            
                            // Convert samples to bytes and add to queue
                            let mut bytes = Vec::with_capacity(samples.len() * 4);
                            for s in samples {
                                bytes.extend_from_slice(&s.to_le_bytes());
                            }
                            self.chunk_queue.push((bytes, offset_samples));
                            console_log!("Queued chunk of {} bytes, queue size: {}", self.chunk_queue[self.chunk_queue.len() - 1].0.len(), self.chunk_queue.len());
                        }
                    }
                }
                
                // Try to send the next chunk from the queue if backend is ready
                if self.current_decode.is_none() && !self.chunk_queue.is_empty() {
                    if let Some(ws) = &self.ws {
                        let (bytes, offset_samples) = self.chunk_queue.remove(0);
                        let start_time = performance_now();
                        self.current_decode = Some(CurrentDecode { start_time, offset_samples });
                        console_log!("Sending chunk of {} bytes from queue", bytes.len());
                        let _ = ws.send_with_u8_array(&bytes);
                    }
                }
                
                false
            }
            Msg::RecordingStarted(result) => {
                match result {
                    Ok(state) => {
                        self.recording = Some(state);
                        self.status = "recording...".to_string();
                    }
                    Err(err) => {
                        self.status = format!("recording error: {err}");
                    }
                }
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let label = if self.muted { "Unmute" } else { "Mute" };
        html! {
            <div>
                <div>
                  <button class="button" onclick={ctx.link().callback(move |_| Msg::ToggleMute)}> { label }</button>
                </div>
                <h2>
                  {&self.status}
                </h2>
                {
                        html! {
                            <>
                            { if self.current_decode.is_some() {
                                html! { <progress id="progress-bar" aria-label="decoding…"></progress> }
                            } else { html! {} } }
                            <blockquote>
                            <p>
                              {
                                  self.segments.iter().map(|segment| { html! {
                                      <>
                                      {&segment.dr.text}
                                      <br/ >
                                      </>
                                  } }).collect::<Html>()
                              }
                            </p>
                            </blockquote>
                            </>
                        }
                    }
            </div>
        }
    }
}
