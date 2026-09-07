use serde::{Deserialize, Serialize};
use serde_json::Value;

use openai_protocol::common::GenerationRequest;

fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                let text = extract_text_from_block(block);
                (!text.is_empty()).then_some(text)
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn extract_text_from_block(block: &Value) -> String {
    match block {
        Value::String(text) => text.clone(),
        Value::Array(values) => values
            .iter()
            .filter_map(|value| {
                let text = extract_text_from_block(value);
                (!text.is_empty()).then_some(text)
            })
            .collect::<Vec<_>>()
            .join(" "),
        Value::Object(map) => {
            let mut parts = Vec::new();
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                parts.push(text.to_string());
            }
            if let Some(content) = map.get("content") {
                let nested_text = extract_text_from_content(content);
                if !nested_text.is_empty() {
                    parts.push(nested_text);
                }
            }
            parts.join(" ")
        }
        _ => String::new(),
    }
}

pub fn extract_routing_text(messages: &[Value]) -> String {
    messages
        .first()
        .and_then(|msg| msg.get("content"))
        .map(extract_text_from_content)
        .unwrap_or_default()
}

/// Minimal Anthropic Messages API request. Other fields are forwarded unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub rest: serde_json::Map<String, Value>,
}

impl GenerationRequest for AnthropicMessagesRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        extract_routing_text(&self.messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_text_from_anthropic_content_blocks() {
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_result", "content": [{"type": "text", "text": "world"}]}
            ]
        })];
        assert_eq!(extract_routing_text(&messages), "hello world");
    }
}
