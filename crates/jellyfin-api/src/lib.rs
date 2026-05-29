//! HTTP client for a single Jellyfin server.
//!
//! Stateless and side-effect-free: this crate doesn't touch the keychain,
//! the filesystem, or any global. Persistence of `Identity.device_id`,
//! `access_token`, and `base_url` is the caller's job (see `jelly-storage`).
//!
//! ```ignore
//! use jellyfin_api::{Client, Identity};
//! use url::Url;
//!
//! # async fn run() -> jellyfin_api::Result<()> {
//! let identity = Identity::new("Jelly", "Mac", "stable-uuid", "0.1");
//! let mut client = Client::new(Url::parse("https://jelly.example.com")?, identity)
//!     .with_accept_language("es");
//! let auth = client.sign_in("user", "pw").await?;
//! let me = client.current_user().await?;
//! println!("hi {}", me.name);
//! # Ok(()) }
//! ```

pub mod auth;
pub mod client;
pub mod error;
pub mod playback;
pub mod types;
pub mod urls;

pub use auth::Identity;
pub use client::{Client, ItemsQuery, SortOrder};
pub use error::{Error, Result};
pub use types::{
    AuthenticationResult, BaseItemDto, ItemType, ItemsResponse, MediaSourceInfo, MediaStream,
    NameGuidPair, PersonInfo, PublicSystemInfo, UserDto, UserItemDataDto, seconds_to_ticks,
    ticks_to_seconds,
};
pub use urls::{ImageOptions, ImageType, audio_stream_url, image_url, video_stream_url};
