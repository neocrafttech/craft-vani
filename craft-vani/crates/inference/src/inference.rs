use serde::{Deserialize, Serialize};

#[macro_export]
macro_rules! inference_log {
    ($($t:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        {
            $crate::worker::log(&format_args!($($t)*).to_string());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!($($t)*);
        }
    };
}

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

#[derive(Serialize, Deserialize, Debug)]
pub enum InferenceOutput {
    Decoded(Vec<Segment>),
    Error(String),
}
