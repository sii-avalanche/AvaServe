// Re-export everything from openai_protocol so existing `crate::protocols::*` paths continue to work.
pub use openai_protocol::*;

pub mod anthropic;
