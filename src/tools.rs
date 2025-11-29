use crate::{image_utils, server::OpenRouterServer};
use anyhow::Result;
use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    schemars, tool, tool_router,
};
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GenerateImageArgs {
    #[schemars(example = &"一只可爱的小猫穿着宇航服在月球上行走，科幻风格")]
    pub prompt: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EditImageArgs {
    #[schemars(example = &"请将这张图片编辑成一张科幻风格的海报")]
    pub instruction: String,
    #[schemars(example = &"https://example.com/image.jpg")]
    #[schemars(example = &"C:\\Images\\photo.png")]
    #[schemars(example = &"data:image/jpeg;base64,/9j/4AAQ...")]
    pub images: Vec<String>,
}

#[tool_router]
impl OpenRouterServer {
    #[tool(description = "文本生成图像")]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
    ) -> Result<CallToolResult, McpError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let model = self.config.model.clone();
        let content = vec![json!({
            "type": "text",
            "text": args.prompt
        })];
        let request_body = json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": 1000,
            "temperature": 0.7
        });

        match self.client.post(&url).json(&request_body).send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "无法获取错误详情".to_string());
                    return Err(McpError::internal_error(
                        format!("API 请求失败，状态码: {}, 错误: {}", status, error_text),
                        None,
                    ));
                }

                match response.json::<serde_json::Value>().await {
                    Ok(response_data) => {
                        let (content, images_array) = extract_text_and_images(&response_data)?;

                        let current_save_dir = {
                            let save_dir = self.save_directory.read().await;
                            save_dir.clone()
                        };
                        let saved_images = image_utils::save_response_images(
                            &images_array,
                            Some(&current_save_dir),
                            Some("generated_image"),
                            false,
                        );

                        let mut response_text = format!(
                            "**模型:** {}\n**提示词:** {}\n**保存目录:** {}\n**响应:** {}",
                            model, args.prompt, current_save_dir, content
                        );
                        if !images_array.is_empty() {
                            response_text.push_str(&format!(
                                "\n\n**生成的图像:** {} 张图像",
                                images_array.len()
                            ));
                            for (index, img_info) in saved_images.iter().enumerate() {
                                response_text.push_str(&format!(
                                    "\n- 图像 {}: {}...",
                                    index + 1,
                                    &img_info.url[..std::cmp::min(50, img_info.url.len())]
                                ));
                                if let Some(saved_path) = &img_info.saved_path {
                                    response_text
                                        .push_str(&format!("\n  已保存到: {}", saved_path));
                                } else {
                                    response_text
                                        .push_str("\n  ⚠️ 未保存到文件");
                                }
                                if !img_info.debug_info.is_empty() {
                                    response_text
                                        .push_str(&format!("\n  [调试] {}", img_info.debug_info));
                                }
                            }
                        }

                        if let Some(usage) = response_data.get("usage")
                            && let (
                                Some(prompt_tokens),
                                Some(completion_tokens),
                                Some(total_tokens),
                            ) = (
                                usage.get("prompt_tokens").and_then(|t| t.as_u64()),
                                usage.get("completion_tokens").and_then(|t| t.as_u64()),
                                usage.get("total_tokens").and_then(|t| t.as_u64()),
                            )
                        {
                            response_text.push_str(&format!("\n\n**使用统计:**\n- 提示词tokens: {}\n- 完成tokens: {}\n- 总tokens: {}", prompt_tokens, completion_tokens, total_tokens));
                        }

                        Ok(CallToolResult::success(vec![Content::text(response_text)]))
                    }
                    Err(e) => Err(McpError::internal_error(
                        format!("解析响应失败: {}", e),
                        None,
                    )),
                }
            }
            Err(e) => Err(McpError::internal_error(format!("请求失败: {}", e), None)),
        }
    }

    #[tool(
        description = "使用图像模型编辑或分析图像（支持多张图像）。图像可以是：1) URL链接 2) base64编码数据 3) 本地文件路径"
    )]
    async fn edit_image(
        &self,
        Parameters(args): Parameters<EditImageArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.images.is_empty() {
            return Err(McpError::internal_error(
                "❌ 编辑图像时必须传入至少一张图片！\n\n请提供以下格式之一的图片：\n- URL链接 (http:// 或 https://)\n- base64编码数据 (data:image/...)\n- 本地文件路径\n\n示例：\n- URL: https://example.com/image.jpg\n- 本地文件: C:\\Images\\photo.png\n- base64: data:image/jpeg;base64,/9j/4AAQ...",
                None,
            ));
        }

        let url = format!("{}/chat/completions", self.config.base_url);
        let model = self.config.model.clone();
        let mut content = vec![json!({
            "type": "text",
            "text": args.instruction
        })];

        for image_input in &args.images {
            match image_utils::detect_and_process_image_input(image_input) {
                Ok(image_content) => match image_content.content_type.as_str() {
                    "url" => {
                        content.push(json!({
                            "type": "image_url",
                            "image_url": {"url": image_content.data}
                        }));
                    }
                    "base64" => {
                        content.push(json!({
                            "type": "image_url",
                            "image_url": {"url": image_content.data}
                        }));
                    }
                    _ => {
                        content.push(json!({
                            "type": "image_url",
                            "image_url": {"url": image_content.data}
                        }));
                    }
                },
                Err(_) => {
                    let current_save_dir = {
                        let save_dir = self.save_directory.read().await;
                        save_dir.clone()
                    };
                    match image_utils::find_image_in_save_directory(image_input, &current_save_dir)
                    {
                        Ok(image_content) => {
                            content.push(json!({
                                "type": "image_url",
                                "image_url": {"url": image_content.data}
                            }));
                        }
                        Err(_) => {
                            content.push(json!({
                                "type": "image_url",
                                "image_url": {"url": image_input}
                            }));
                        }
                    }
                }
            }
        }

        let request_body = json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": 1000,
            "temperature": 0.7
        });

        match self.client.post(&url).json(&request_body).send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "无法获取错误详情".to_string());
                    return Err(McpError::internal_error(
                        format!("API 请求失败，状态码: {}, 错误: {}", status, error_text),
                        None,
                    ));
                }

                match response.json::<serde_json::Value>().await {
                    Ok(response_data) => {
                        let (content, images_array) = extract_text_and_images(&response_data)?;

                        let current_save_dir = {
                            let save_dir = self.save_directory.read().await;
                            save_dir.clone()
                        };

                        let base_filename = if !args.images.is_empty() {
                            let first_image = &args.images[0];
                            if !first_image.starts_with("http://")
                                && !first_image.starts_with("https://")
                                && !first_image.starts_with("data:image/")
                            {
                                Some(image_utils::extract_filename_without_extension(first_image))
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let saved_images = image_utils::save_response_images(
                            &images_array,
                            Some(&current_save_dir),
                            base_filename.as_deref(),
                            true,
                        );

                        let mut response_text = format!(
                            "**模型:** {}\n**指令:** {}\n**输入图像:** {} 张图像\n**响应:** {}",
                            model,
                            args.instruction,
                            args.images.len(),
                            content
                        );
                        if !images_array.is_empty() {
                            response_text.push_str(&format!(
                                "\n\n**生成的图像:** {} 张图像",
                                images_array.len()
                            ));
                            for (index, img_info) in saved_images.iter().enumerate() {
                                response_text.push_str(&format!(
                                    "\n- 图像 {}: {}...",
                                    index + 1,
                                    &img_info.url[..std::cmp::min(50, img_info.url.len())]
                                ));
                                if let Some(saved_path) = &img_info.saved_path {
                                    response_text
                                        .push_str(&format!("\n  已保存到: {}", saved_path));
                                }
                            }
                        }

                        if let Some(usage) = response_data.get("usage")
                            && let (
                                Some(prompt_tokens),
                                Some(completion_tokens),
                                Some(total_tokens),
                            ) = (
                                usage.get("prompt_tokens").and_then(|t| t.as_u64()),
                                usage.get("completion_tokens").and_then(|t| t.as_u64()),
                                usage.get("total_tokens").and_then(|t| t.as_u64()),
                            )
                        {
                            response_text.push_str(&format!("\n\n**使用统计:**\n- 提示词tokens: {}\n- 完成tokens: {}\n- 总tokens: {}", prompt_tokens, completion_tokens, total_tokens));
                        }

                        Ok(CallToolResult::success(vec![Content::text(response_text)]))
                    }
                    Err(e) => Err(McpError::internal_error(
                        format!("解析响应失败: {}", e),
                        None,
                    )),
                }
            }
            Err(e) => Err(McpError::internal_error(format!("请求失败: {}", e), None)),
        }
    }
}

impl OpenRouterServer {
    pub(crate) fn create_tool_router() -> rmcp::handler::server::router::tool::ToolRouter<Self> {
        Self::tool_router()
    }
}

/// 从 markdown 文本中提取嵌入的 base64 图像，并返回清理后的文本
/// 匹配格式: ![...](data:image/...;base64,...)
/// 返回: (清理后的文本, 提取的图片URLs)
fn extract_images_from_markdown(text: &str) -> (String, Vec<String>) {
    let mut images = Vec::new();
    let mut cleaned_text = text.to_string();

    // 使用循环查找并替换所有的 markdown 图片
    loop {
        if let Some(start_idx) = cleaned_text.find("![") {
            let remaining = &cleaned_text[start_idx..];
            // 找到 ](
            if let Some(paren_idx) = remaining.find("](") {
                let after_paren = &remaining[paren_idx + 2..];
                // 检查是否是 data:image
                if after_paren.starts_with("data:image/") {
                    // 找到匹配的 )
                    if let Some(end_idx) = after_paren.find(')') {
                        let data_url = &after_paren[..end_idx];
                        images.push(data_url.to_string());

                        // 从文本中移除整个 markdown 图片语法
                        let full_match_end = start_idx + paren_idx + 2 + end_idx + 1;
                        cleaned_text.replace_range(start_idx..full_match_end, "");
                        continue;
                    }
                }
            }
            // 如果没匹配到完整的 markdown 图片，跳过这个 ![
            cleaned_text.replace_range(start_idx..start_idx+2, "");
        } else {
            break;
        }
    }

    (cleaned_text.trim().to_string(), images)
}

/// 从 OpenRouter/Gemini 等兼容响应中提取文本和图像
fn extract_text_and_images(response: &Value) -> Result<(String, Vec<Value>), McpError> {
    // 1) 规范错误字段
    if let Some(error) = response.get("error") {
        let error_message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("未知错误");
        return Err(McpError::internal_error(
            format!("API 返回错误: {}", error_message),
            None,
        ));
    }

    // 2) 提取第一条消息（兼容 choices / candidates）
    let message = if let Some(choices) = response.get("choices").and_then(|c| c.as_array()) {
        if choices.is_empty() {
            return Err(McpError::internal_error(
                "API 响应中 'choices' 数组为空".to_string(),
                None,
            ));
        }
        choices[0].get("message").ok_or_else(|| {
            McpError::internal_error("响应格式无效: choices[0].message 缺失".to_string(), None)
        })?
    } else if let Some(candidates) = response.get("candidates").and_then(|c| c.as_array()) {
        // Gemini 风格
        if candidates.is_empty() {
            return Err(McpError::internal_error(
                "API 响应中 'candidates' 数组为空".to_string(),
                None,
            ));
        }
        candidates[0].get("content").ok_or_else(|| {
            McpError::internal_error("响应格式无效: candidates[0].content 缺失".to_string(), None)
        })?
    } else {
        return Err(McpError::internal_error(
            "响应格式无效: 未找到 choices 或 candidates".to_string(),
            None,
        ));
    };

    // 3) 统一提取 content/parts 字段
    let mut texts: Vec<String> = Vec::new();
    let mut images: Vec<Value> = Vec::new();

    let content_field = message
        .get("content")
        .or_else(|| message.get("parts"))
        .unwrap_or(message);

    match content_field {
        Value::String(s) => {
            // 先尝试从文本中提取嵌入的 base64 图像，并获取清理后的文本
            let (cleaned_text, embedded_images) = extract_images_from_markdown(s);
            for img_url in embedded_images {
                images.push(json!({ "image_url": { "url": img_url } }));
            }
            // 只保存清理后的文本（移除了base64图片）
            if !cleaned_text.is_empty() {
                texts.push(cleaned_text);
            }
        }
        Value::Array(parts) => {
            for part in parts {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "text" => {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            // 先尝试从文本中提取嵌入的 base64 图像，并获取清理后的文本
                            let (cleaned_text, embedded_images) = extract_images_from_markdown(t);
                            for img_url in embedded_images {
                                images.push(json!({ "image_url": { "url": img_url } }));
                            }
                            // 只保存清理后的文本（移除了base64图片）
                            if !cleaned_text.is_empty() {
                                texts.push(cleaned_text);
                            }
                        }
                    }
                    "image_url" => {
                        if part.get("image_url").is_some() {
                            images.push(json!({ "image_url": part.get("image_url").cloned().unwrap_or_default() }));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }

    // 4) 兼容 message.images 或 data 数组（如 images 生成接口）
    if let Some(imgs) = message.get("images").and_then(|i| i.as_array()) {
        for img in imgs {
            if let Some(url_obj) = img.get("image_url") {
                images.push(json!({ "image_url": url_obj }));
            }
        }
    }
    if images.is_empty()
        && let Some(data) = response.get("data").and_then(|d| d.as_array())
    {
        for img in data {
            if let Some(b64) = img.get("b64_json").and_then(|b| b.as_str()) {
                let data_url = format!("data:image/png;base64,{}", b64);
                images.push(json!({ "image_url": { "url": data_url } }));
            } else if let Some(url) = img.get("url") {
                images.push(json!({ "image_url": { "url": url } }));
            }
        }
    }

    let merged_text = if texts.is_empty() {
        "无内容".to_string()
    } else {
        texts.join("\n")
    };

    Ok((merged_text, images))
}
