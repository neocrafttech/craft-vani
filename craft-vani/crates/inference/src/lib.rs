pub mod inference;
pub use candle_transformers::models::whisper::{self as m, Config};
pub use inference::{DecodingResult, InferenceOutput, Segment};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format_args!($($t)*).to_string().into());
            #[cfg(not(target_arch = "wasm32"))]
            println!($($t)*);
        }
    }
}
