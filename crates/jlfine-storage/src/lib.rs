//! Persistent storage for the jelly client: server URL, user ID, access
//! token, and a stable device UUID that survives logout.
//!
//! Two backing stores:
//!
//! - **config file** (`<config_dir>/config.json`) — server URL, user ID,
//!   device ID. Plain JSON; not a secret, but the file lives in the OS
//!   user-config directory so other users can't read it.
//! - **OS keyring** — access token. Uses `keyring` (Keychain on macOS,
//!   Secret Service on Linux). The token is referenced by `user_id`
//!   so multiple accounts on the same machine don't clash.
//!
//! The device ID is generated lazily on first read and persists across
//! logouts. Jellyfin tracks devices by this ID, so reusing it keeps the
//! "Sessions" view on the server stable instead of growing a new row
//! every launch.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use url::Url;
use uuid::Uuid;

const KEYRING_SERVICE: &str = "us.priet.jelly";
const CONFIG_FILE: &str = "config.json";

#[derive(Debug, Error)]
pub enum Error {
    #[error("could not locate user config dir")]
    NoConfigDir,

    #[error("config IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("keyring error: {0}")]
    Keyring(#[from] keyring::Error),

    #[error("URL parse error: {0}")]
    Url(#[from] url::ParseError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Storage {
    config_dir: PathBuf,
    keyring_service: String,
}

impl Storage {
    /// Use the OS-conventional config dir for the app.
    pub fn new() -> Result<Self> {
        let dirs = ProjectDirs::from("us", "Priet", "Jelly").ok_or(Error::NoConfigDir)?;
        Ok(Self::with_dirs(
            dirs.config_dir().to_path_buf(),
            KEYRING_SERVICE.into(),
        ))
    }

    /// Inject an arbitrary config dir and keyring service. Tests use this
    /// with `tempfile::TempDir` and a per-test keyring service name to
    /// avoid polluting the real OS keychain.
    pub fn with_dirs(config_dir: PathBuf, keyring_service: String) -> Self {
        Self {
            config_dir,
            keyring_service,
        }
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    /// Get-or-create the persistent device ID. Lazily initialised on first
    /// call so a fresh install picks up a new UUID without an explicit
    /// "first run" step.
    pub fn device_id(&self) -> Result<String> {
        let mut cfg = self.load_config()?;
        if cfg.device_id.is_empty() {
            cfg.device_id = Uuid::new_v4().to_string();
            self.save_config(&cfg)?;
        }
        Ok(cfg.device_id)
    }

    /// Return the persisted session if one is stored AND its token is
    /// still in the keyring. A token missing from the keyring (manually
    /// removed, keychain reset, etc.) is treated as logged out.
    pub fn load_session(&self) -> Result<Option<SavedSession>> {
        let cfg = self.load_config()?;
        let Some(SessionFields {
            server_url,
            user_id,
        }) = cfg.session
        else {
            return Ok(None);
        };
        let server_url = Url::parse(&server_url)?;
        let entry = keyring::Entry::new(&self.keyring_service, &user_id)?;
        match entry.get_password() {
            Ok(token) => Ok(Some(SavedSession {
                server_url,
                user_id,
                token,
            })),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save_session(&self, server_url: &Url, user_id: &str, token: &str) -> Result<()> {
        let entry = keyring::Entry::new(&self.keyring_service, user_id)?;
        entry.set_password(token)?;
        let mut cfg = self.load_config()?;
        cfg.session = Some(SessionFields {
            server_url: server_url.to_string(),
            user_id: user_id.to_string(),
        });
        self.save_config(&cfg)?;
        Ok(())
    }

    /// Preferred audio output device, as an mpv-style id
    /// (`coreaudio/<UID>`, `alsa/hw:1,0`, etc). `None` means the
    /// caller should fall back to the platform default.
    pub fn audio_device(&self) -> Result<Option<String>> {
        let cfg = self.load_config()?;
        Ok(cfg.audio_device.filter(|s| !s.is_empty()))
    }

    /// Persist or clear the preferred audio output device. Pass
    /// `None` to fall back to the platform default.
    pub fn set_audio_device(&self, device: Option<&str>) -> Result<()> {
        let mut cfg = self.load_config()?;
        cfg.audio_device = device.map(str::to_owned);
        self.save_config(&cfg)
    }

    /// Whether music takes exclusive control of the output device.
    /// Defaults to `true` (the v1 bitperfect requirement); the UI
    /// settings screen lets the user flip it off.
    pub fn exclusive_mode(&self) -> Result<bool> {
        let cfg = self.load_config()?;
        Ok(cfg.exclusive_mode.unwrap_or(true))
    }

    pub fn set_exclusive_mode(&self, exclusive: bool) -> Result<()> {
        let mut cfg = self.load_config()?;
        cfg.exclusive_mode = Some(exclusive);
        self.save_config(&cfg)
    }

    /// Drop the session fields from the config and delete the token from
    /// the keyring. Device ID is preserved.
    pub fn clear_session(&self) -> Result<()> {
        let mut cfg = self.load_config()?;
        if let Some(s) = cfg.session.take() {
            // Best-effort delete; if the entry is already gone, that's fine.
            if let Ok(entry) = keyring::Entry::new(&self.keyring_service, &s.user_id) {
                let _ = entry.delete_credential();
            }
        }
        self.save_config(&cfg)?;
        Ok(())
    }

    // ----------------------------------------------------------- internals

    fn config_path(&self) -> PathBuf {
        self.config_dir.join(CONFIG_FILE)
    }

    fn load_config(&self) -> Result<ConfigFile> {
        let path = self.config_path();
        match std::fs::read(&path) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes).unwrap_or_default()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ConfigFile::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Atomic write: serialise to a sibling tmp file, then rename. A crash
    /// mid-write leaves the previous config intact instead of half-written.
    fn save_config(&self, cfg: &ConfigFile) -> Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        let path = self.config_path();
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(cfg)?;
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SavedSession {
    pub server_url: Url,
    pub user_id: String,
    pub token: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    device_id: String,
    #[serde(default)]
    session: Option<SessionFields>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    audio_device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exclusive_mode: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionFields {
    server_url: String,
    user_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_storage() -> (Storage, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Per-test keyring service so concurrent tests don't clash and
        // we don't pollute the developer's real keychain on `cargo test`.
        let service = format!("us.priet.jelly.test.{}", Uuid::new_v4());
        let storage = Storage::with_dirs(dir.path().to_path_buf(), service);
        (storage, dir)
    }

    #[test]
    fn device_id_persists_across_reloads() {
        let (s, _dir) = fresh_storage();
        let id1 = s.device_id().unwrap();
        let id2 = s.device_id().unwrap();
        assert_eq!(id1, id2);
        assert!(!id1.is_empty());
    }

    #[test]
    fn device_id_survives_clear_session() {
        let (s, _dir) = fresh_storage();
        let id_before = s.device_id().unwrap();
        // No session to clear, but the call should still leave device id intact.
        s.clear_session().unwrap();
        let id_after = s.device_id().unwrap();
        assert_eq!(id_before, id_after);
    }

    #[test]
    fn load_session_none_when_empty() {
        let (s, _dir) = fresh_storage();
        assert!(s.load_session().unwrap().is_none());
    }

    // Keyring-touching tests are gated to manual runs. On CI / shared
    // dev boxes the keyring backend may be unavailable or require a
    // graphical session, and we don't want flakes there.
    #[test]
    #[ignore = "touches the OS keychain — run with `cargo test -- --ignored`"]
    fn save_load_clear_session_round_trip() {
        let (s, _dir) = fresh_storage();
        let url = Url::parse("https://jelly.example.com").unwrap();
        s.save_session(&url, "user-42", "tok-xyz").unwrap();

        let loaded = s.load_session().unwrap().expect("session should exist");
        assert_eq!(loaded.server_url, url);
        assert_eq!(loaded.user_id, "user-42");
        assert_eq!(loaded.token, "tok-xyz");

        s.clear_session().unwrap();
        assert!(s.load_session().unwrap().is_none());
    }
}
