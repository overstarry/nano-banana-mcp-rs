use anyhow::{Result, anyhow};
use std::env;

#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub http_referer: String,
    pub x_title: String,
    pub http_port: u16,
    pub model: String,
    pub sse_keep_alive_secs: Option<u64>,
}

impl OpenRouterConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok(); // 加载 .env 文件，如果存在

        // 首先尝试从命令行参数获取 API key
        let args: Vec<String> = env::args().collect();
        let api_key = Self::get_api_key_from_args(&args)
            .or_else(|| env::var("OPENROUTER_API_KEY").ok())
            .ok_or_else(|| anyhow!("OPENROUTER_API_KEY 环境变量或 --api-key 命令行参数是必需的"))?;

        let base_url = env::var("OPENROUTER_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

        let http_referer =
            env::var("HTTP_REFERER").unwrap_or_else(|_| "http://localhost:3000".to_string());

        let x_title =
            env::var("X_TITLE").unwrap_or_else(|_| "OpenRouter MCP Server (Rust)".to_string());

        let http_port = env::var("MCP_HTTP_PORT")
            .unwrap_or_else(|_| "6621".to_string())
            .parse()
            .unwrap_or(6621);

        // SSE keep-alive 配置（秒），可选
        let sse_keep_alive_secs = env::var("MCP_SSE_KEEP_ALIVE_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());

        // 获取模型配置：优先命令行参数，然后环境变量，最后默认值
        // 默认使用 OpenRouter 的 gemini 图像模型，如果使用第三方 API 服务（如 tu-zi.com），
        // 可能需要设置其他模型名，例如：
        //   - nano-banana (tu-zi.com 的 nano-banana 模型)
        //   - gpt-4o-image (一些服务使用这个名称)
        //   - google/gemini-2.5-flash-image-preview
        //   - google/gemini-3-pro-image-preview
        let model = Self::get_model_from_args(&args)
            .or_else(|| env::var("MCP_MODEL").ok())
            .unwrap_or_else(|| "google/gemini-2.5-flash-preview-06-17".to_string());

        // 不再验证模型名称，允许用户使用任意兼容 OpenAI chat/completions API 的模型
        // 这样可以支持各种第三方 API 转发服务（如 tu-zi.com、one-api 等）

        Ok(Self {
            api_key,
            base_url,
            http_referer,
            x_title,
            http_port,
            model,
            sse_keep_alive_secs,
        })
    }

    /// 从命令行参数中获取 API key
    fn get_api_key_from_args(args: &[String]) -> Option<String> {
        for (i, arg) in args.iter().enumerate() {
            if arg == "--api-key" && i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
            if arg.starts_with("--api-key=") {
                return Some(arg.trim_start_matches("--api-key=").to_string());
            }
        }
        None
    }

    /// 从命令行参数中获取模型
    fn get_model_from_args(args: &[String]) -> Option<String> {
        for (i, arg) in args.iter().enumerate() {
            if arg == "--model" && i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
            if arg.starts_with("--model=") {
                return Some(arg.trim_start_matches("--model=").to_string());
            }
        }
        None
    }

    pub fn get_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();

        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("http-referer"),
            self.http_referer.parse().unwrap(),
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("x-title"),
            self.x_title.parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        headers
    }
}
