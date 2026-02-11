#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::safetensors::Load;
use candle::{Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use candle_transformers::models::whisper::{self as m, Config};
use clap::{Parser, ValueEnum};
use rand::{SeedableRng, distr::Distribution};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self, x: &Tensor, xa: &Tensor, flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

pub use inference::inference::{DecodingResult, Segment};

pub struct Decoder {
    pub model: Model,
    pub rng: rand::rngs::StdRng,
    pub task: Option<Task>,
    pub timestamps: bool,
    pub tokenizer: Tokenizer,
    pub suppress_tokens: Tensor,
    pub sot_token: u32,
    pub transcribe_token: u32,
    pub translate_token: u32,
    pub eot_token: u32,
    pub no_speech_token: u32,
    pub no_timestamps_token: u32,
    pub language_token: Option<u32>,
    pub mel_filters: Vec<f32>,
    pub device: Device,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model, tokenizer: Tokenizer, mel_filters: Vec<f32>, seed: u64, device: &Device,
        language_token: Option<u32>, task: Option<Task>, timestamps: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token =
            m::NO_SPEECH_TOKENS.iter().find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
            mel_filters,
            device: device.clone(),
        })
    }

    pub fn load_from_dir(dir: &Path, mel_path: &Path, device: &Device) -> Result<Self> {
        let mel_filters = std::fs::read(mel_path)?;
        let mel_filters = ::safetensors::tensor::SafeTensors::deserialize(&mel_filters)?;
        let mel_filters = mel_filters.tensor("mel_80")?.load(device)?;
        let mel_filters = mel_filters.flatten_all()?.to_vec1::<f32>()?;
        let quantized_dir = Path::new("quantized");
        let quantized_config = quantized_dir.join("config-tiny-en.json");
        let quantized_tokenizer = quantized_dir.join("tokenizer-tiny-en.json");
        let quantized_weights = quantized_dir.join("model-tiny-en-q80.gguf");
        let use_quantized = std::env::var("CRAFT_VANI_USE_QUANTIZED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        let (tokenizer, model) = if use_quantized
            && quantized_config.exists()
            && quantized_tokenizer.exists()
            && quantized_weights.exists()
        {
            let config: Config = serde_json::from_reader(std::fs::File::open(&quantized_config)?)?;
            let tokenizer = Tokenizer::from_file(quantized_tokenizer).map_err(E::msg)?;
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &quantized_weights,
                device,
            )?;
            let model = Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?);
            println!("Using quantized tiny-en model (CRAFT_VANI_USE_QUANTIZED=1)");
            (tokenizer, model)
        } else {
            let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(E::msg)?;
            let config: Config =
                serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
            let weights_path = dir.join("model.safetensors");
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, device)? };
            let model = Model::Normal(m::model::Whisper::load(&vb, config)?);
            println!("Using normal safetensors model");
            (tokenizer, model)
        };

        let decoder = Self::new(
            model,
            tokenizer,
            mel_filters,
            299792458,
            device,
            None,
            Some(Task::Transcribe),
            false, // timestamps
        )?;
        Ok(decoder)
    }

    pub fn run_raw(&mut self, pcm_data: &[f32]) -> Result<Vec<Segment>> {
        let mel = m::audio::pcm_to_mel(self.model.config(), pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let n_mels = self.model.config().num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, n_mels, mel_len / n_mels), &self.device)?;
        let segments = self.run(&mel, None)?;
        Ok(segments)
    }

    pub fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model.decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?.i(0)?.i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distr::weighted::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;
        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            if let Ok(dr) = dr {
                let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                    || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                    return Ok(dr);
                }
            }
        }
        unreachable!()
    }

    pub fn run(&mut self, mel: &Tensor, _times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                continue;
            }
            let segment = Segment { start: time_offset, duration: segment_duration, dr };
            segments.push(segment)
        }
        Ok(segments)
    }

    pub fn reset_kv_cache(&mut self) {
        match &mut self.model {
            Model::Normal(m) => m.reset_kv_cache(),
            Model::Quantized(m) => m.reset_kv_cache(),
        }
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
pub enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    quantized: bool,

    /// Language.
    #[arg(long)]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    task: Option<Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,

    /// The input device to use.
    #[arg(long)]
    device: Option<String>,
}
