use js_sys::Date;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{AudioContext, AudioContextOptions, MediaStream, ScriptProcessorNode};
use yew::{Component, Context, Html, html};
use yew_agent::{Bridge, Bridged};

use crate::console_log;
use crate::worker::{ModelData, Segment, Worker, WorkerInput, WorkerOutput};

async fn fetch_url(url: &str) -> Result<Vec<u8>, JsValue> {
    use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};
    let window = web_sys::window().ok_or("window")?;
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    opts.set_cache(RequestCache::NoCache);
    let request = Request::new_with_str_and_init(url, &opts)?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into()?;
    let data = JsFuture::from(resp.blob()?).await?;
    let blob = web_sys::Blob::from(data);
    let array_buffer = JsFuture::from(blob.array_buffer()).await?;
    let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
    Ok(data)
}

pub enum Msg {
    UpdateStatus(String),
    SetDecoder(ModelData),
    WorkerIn(WorkerInput),
    WorkerOut(Result<WorkerOutput, String>),
    ToggleMute,
    RecordingStarted(Result<RecordingState, String>),
    Process,
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
    loaded: bool,
    segments: Vec<Segment>,
    current_decode: Option<CurrentDecode>,
    worker: Box<dyn Bridge<Worker>>,
    recording: Option<RecordingState>,
    muted: bool,
    _interval: Option<gloo_timers::callback::Interval>,
    decoded_samples: usize,
}

async fn model_data_load() -> Result<ModelData, JsValue> {
    let quantized = false;
    let is_multilingual = false;

    let (tokenizer, mel_filters, weights, config) = if quantized {
        console_log!("loading quantized weights");
        let tokenizer = fetch_url("quantized/tokenizer-tiny-en.json").await?;
        let mel_filters = fetch_url("mel_filters.safetensors").await?;
        let weights = fetch_url("quantized/model-tiny-en-q80.gguf").await?;
        let config = fetch_url("quantized/config-tiny-en.json").await?;
        (tokenizer, mel_filters, weights, config)
    } else {
        console_log!("loading float weights");
        if is_multilingual {
            let mel_filters = fetch_url("mel_filters.safetensors").await?;
            let tokenizer = fetch_url("whisper-tiny/tokenizer.json").await?;
            let weights = fetch_url("whisper-tiny/model.safetensors").await?;
            let config = fetch_url("whisper-tiny/config.json").await?;
            (tokenizer, mel_filters, weights, config)
        } else {
            let mel_filters = fetch_url("mel_filters.safetensors").await?;
            let tokenizer = fetch_url("whisper-tiny.en/tokenizer.json").await?;
            let weights = fetch_url("whisper-tiny.en/model.safetensors").await?;
            let config = fetch_url("whisper-tiny.en/config.json").await?;
            (tokenizer, mel_filters, weights, config)
        }
    };

    let timestamps = true;
    let _task = Some("transcribe".to_string());
    console_log!("{}", weights.len());
    Ok(ModelData {
        tokenizer,
        mel_filters,
        weights,
        config,
        quantized,
        timestamps,
        task: None,
        is_multilingual,
        language: None,
    })
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

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let status = "loading weights".to_string();
        let cb = {
            let link = ctx.link().clone();
            move |e| link.send_message(Self::Message::WorkerOut(e))
        };
        let worker = Worker::bridge(std::rc::Rc::new(cb));
        Self {
            status,
            segments: vec![],
            current_decode: None,
            worker,
            loaded: false,
            recording: None,
            muted: true,
            _interval: None,
            decoded_samples: 0,
        }
    }

    fn rendered(&mut self, ctx: &Context<Self>, first_render: bool) {
        if first_render {
            ctx.link().send_future(async {
                match model_data_load().await {
                    Err(err) => {
                        let status = format!("{err:?}");
                        Msg::UpdateStatus(status)
                    }
                    Ok(model_data) => Msg::SetDecoder(model_data),
                }
            });
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetDecoder(md) => {
                self.status = "weights loaded successfully!".to_string();
                self.loaded = true;
                console_log!("loaded weights");
                self.worker.send(WorkerInput::ModelData(md));
                true
            }
            Msg::WorkerOut(output) => {
                let dt = self.current_decode.as_ref().and_then(|current_decode| {
                    current_decode.start_time.and_then(|start_time| {
                        performance_now().map(|stop_time| stop_time - start_time)
                    })
                });
                let current_decode = self.current_decode.take();
                match output {
                    Ok(WorkerOutput::WeightsLoaded) => self.status = "weights loaded!".to_string(),
                    Ok(WorkerOutput::Decoded(new_segments)) => {
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
                    Err(err) => {
                        self.status = format!("decoding error {err:?}");
                    }
                }
                true
            }
            Msg::WorkerIn(inp) => {
                self.worker.send(inp);
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

                    // Start periodic processing every 5 seconds
                    let link = ctx.link().clone();
                    self._interval = Some(gloo_timers::callback::Interval::new(5000, move || {
                        link.send_message(Msg::Process);
                    }));
                }
                true
            }
            Msg::Process => {
                if !self.muted && self.current_decode.is_none() {
                    if let Some(recording) = &self.recording {
                        let all_samples = recording.samples.borrow();
                        let new_samples_count = all_samples.len();
                        if new_samples_count > self.decoded_samples {
                            let samples = all_samples[self.decoded_samples..].to_vec();
                            let start_time = performance_now();
                            let offset_samples = self.decoded_samples;
                            self.decoded_samples = new_samples_count;
                            self.current_decode =
                                Some(CurrentDecode { start_time, offset_samples });
                            ctx.link().send_message(Msg::WorkerIn(WorkerInput::DecodeTaskRaw {
                                samples,
                            }));
                        }
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
        html! {
            <div>
                <div>
                  { if self.loaded {
                      let label = if self.muted { "Unmute" } else { "Mute" };
                      html!(<button class="button" onclick={ctx.link().callback(move |_| Msg::ToggleMute)}> { label }</button>)
                  } else { html!() } }
                </div>
                <h2>
                  {&self.status}
                </h2>
                {
                    if !self.loaded {
                        html! { <progress id="progress-bar" aria-label="loading weights…"></progress> }
                    } else {
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
                }

                // Display the current date and time the page was rendered
                <p class="footer">
                    { "Rendered: " }
                    { String::from(Date::new_0().to_string()) }
                </p>
            </div>
        }
    }
}
