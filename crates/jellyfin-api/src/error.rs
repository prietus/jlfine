use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON decode failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("URL parse failed: {0}")]
    Url(#[from] url::ParseError),

    #[error("server returned {status}: {body}")]
    Server { status: u16, body: String },

    #[error("authentication response missing required field: {0}")]
    BadAuthResponse(&'static str),

    #[error("not authenticated — call sign_in() first")]
    NotAuthenticated,

    #[error("invalid header value: {0}")]
    InvalidHeader(#[from] reqwest::header::InvalidHeaderValue),
}

pub type Result<T> = std::result::Result<T, Error>;
