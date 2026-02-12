use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum InferenceOutput {
    Decoded(Vec<Segment>),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SarvamOutput {
    Text(String),
    Error(String),
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceResponse {
    Whisper { whisper: InferenceOutput },
    Sarvam { sarvam: SarvamOutput },
    Both { whisper: InferenceOutput, sarvam: SarvamOutput },
}
