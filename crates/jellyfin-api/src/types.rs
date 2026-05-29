use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of `POST /Users/AuthenticateByName`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct AuthenticationResult {
    pub user: Option<UserDto>,
    pub access_token: Option<String>,
    pub server_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct UserDto {
    pub id: String,
    pub name: String,
    pub server_id: Option<String>,
    pub has_password: Option<bool>,
    pub primary_image_tag: Option<String>,
}

/// `GET /System/Info/Public` — used to validate a server URL before login.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PublicSystemInfo {
    pub local_address: Option<String>,
    pub server_name: Option<String>,
    pub version: Option<String>,
    pub id: Option<String>,
    pub startup_wizard_completed: Option<bool>,
}

/// Wrapper around paginated list endpoints (`/Items`, `/Users/{id}/Items`, …).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ItemsResponse {
    pub items: Vec<BaseItemDto>,
    pub total_record_count: Option<i64>,
    pub start_index: Option<i64>,
}

/// Subset of `BaseItemDto` fields that the client uses today. Jellyfin
/// returns ~100 fields depending on what `Fields` we request; everything
/// is `Option` so missing fields decode cleanly.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct BaseItemDto {
    pub id: String,
    pub name: Option<String>,
    #[serde(rename = "Type")]
    pub item_type: Option<ItemType>,
    pub collection_type: Option<String>,
    pub server_id: Option<String>,
    pub parent_id: Option<String>,
    pub series_id: Option<String>,
    pub season_id: Option<String>,
    pub album_id: Option<String>,
    pub album_artist: Option<String>,
    pub album_artists: Option<Vec<NameGuidPair>>,
    pub artists: Option<Vec<String>>,
    pub artist_items: Option<Vec<NameGuidPair>>,
    pub album: Option<String>,
    pub production_year: Option<i32>,
    pub index_number: Option<i32>,
    pub parent_index_number: Option<i32>,
    pub run_time_ticks: Option<i64>,
    pub overview: Option<String>,
    pub taglines: Option<Vec<String>>,
    pub community_rating: Option<f64>,
    pub critic_rating: Option<f64>,
    pub official_rating: Option<String>,
    pub genres: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub studios: Option<Vec<NameGuidPair>>,
    pub people: Option<Vec<PersonInfo>>,
    pub premiere_date: Option<String>,
    pub production_locations: Option<Vec<String>>,
    pub provider_ids: Option<HashMap<String, String>>,
    pub image_tags: Option<HashMap<String, String>>,
    pub backdrop_image_tags: Option<Vec<String>>,
    pub image_blur_hashes: Option<ImageBlurHashes>,
    pub user_data: Option<UserItemDataDto>,
    pub media_sources: Option<Vec<MediaSourceInfo>>,
    pub media_streams: Option<Vec<MediaStream>>,
    pub child_count: Option<i32>,
}

/// One entry from a `BaseItemDto`'s `People` array — cast and crew. The
/// `primary_image_tag`, when present, is the tag for that person's
/// headshot fetched via the `/Items/{id}/Images/Primary` endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PersonInfo {
    pub id: Option<String>,
    pub name: Option<String>,
    /// Character name for actors (e.g. "D'Leh"); empty for crew.
    pub role: Option<String>,
    /// "Actor", "Director", "Writer", "Producer", …
    #[serde(rename = "Type")]
    pub person_type: Option<String>,
    pub primary_image_tag: Option<String>,
}

/// Jellyfin's item-type enum, sent as a string in JSON. Lists below are
/// the ones the client cares about; unknown values decode to `Other(_)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub enum ItemType {
    Movie,
    Series,
    Season,
    Episode,
    MusicAlbum,
    MusicArtist,
    Audio,
    Folder,
    CollectionFolder,
    BoxSet,
    Playlist,
    Video,
    #[serde(untagged)]
    Other(String),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct NameGuidPair {
    pub name: Option<String>,
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ImageBlurHashes {
    #[serde(default)]
    pub primary: HashMap<String, String>,
    #[serde(default)]
    pub backdrop: HashMap<String, String>,
    #[serde(default)]
    pub logo: HashMap<String, String>,
    #[serde(default)]
    pub thumb: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct UserItemDataDto {
    pub playback_position_ticks: Option<i64>,
    pub play_count: Option<i32>,
    pub played: Option<bool>,
    pub is_favorite: Option<bool>,
    pub played_percentage: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct MediaSourceInfo {
    pub id: Option<String>,
    pub path: Option<String>,
    pub container: Option<String>,
    pub size: Option<i64>,
    pub run_time_ticks: Option<i64>,
    pub bitrate: Option<i64>,
    pub media_streams: Option<Vec<MediaStream>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct MediaStream {
    pub codec: Option<String>,
    #[serde(rename = "Type")]
    pub stream_type: Option<String>,
    pub language: Option<String>,
    pub title: Option<String>,
    pub channels: Option<i32>,
    pub sample_rate: Option<i32>,
    pub bit_depth: Option<i32>,
    pub bit_rate: Option<i64>,
    pub height: Option<i32>,
    pub width: Option<i32>,
    pub is_default: Option<bool>,
    pub is_forced: Option<bool>,
    pub video_range: Option<String>,
    pub video_range_type: Option<String>,
}

/// One Jellyfin tick is 100 ns; 10,000,000 ticks per second.
pub fn ticks_to_seconds(ticks: i64) -> f64 {
    ticks as f64 / 10_000_000.0
}

/// Inverse of `ticks_to_seconds`. Saturates at i64 bounds.
pub fn seconds_to_ticks(seconds: f64) -> i64 {
    (seconds * 10_000_000.0).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_type_round_trips() {
        let v: ItemType = serde_json::from_str(r#""MusicAlbum""#).unwrap();
        assert_eq!(v, ItemType::MusicAlbum);
        let v: ItemType = serde_json::from_str(r#""SomethingNew""#).unwrap();
        assert_eq!(v, ItemType::Other("SomethingNew".into()));
    }

    #[test]
    fn ticks_round_trip() {
        let secs = 3.5_f64;
        let t = seconds_to_ticks(secs);
        assert_eq!(t, 35_000_000);
        assert!((ticks_to_seconds(t) - secs).abs() < 1e-9);
    }
}
