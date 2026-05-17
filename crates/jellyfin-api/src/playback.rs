use serde::Serialize;

/// Sent on `POST /Sessions/Playing` when playback starts so the server
/// records the session and resume position is tracked.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PlaybackStartInfo {
    pub item_id: String,
    pub media_source_id: Option<String>,
    pub play_method: PlayMethod,
    pub position_ticks: Option<i64>,
    pub is_paused: bool,
    pub is_muted: bool,
    pub audio_stream_index: Option<i32>,
    pub subtitle_stream_index: Option<i32>,
}

/// Sent every few seconds on `POST /Sessions/Playing/Progress`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PlaybackProgressInfo {
    pub item_id: String,
    pub media_source_id: Option<String>,
    pub play_method: PlayMethod,
    pub position_ticks: i64,
    pub is_paused: bool,
    pub is_muted: bool,
    pub event_name: Option<ProgressEvent>,
}

/// Sent on `POST /Sessions/Playing/Stopped` when playback ends.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PlaybackStopInfo {
    pub item_id: String,
    pub media_source_id: Option<String>,
    pub position_ticks: Option<i64>,
    pub failed: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum PlayMethod {
    DirectPlay,
    DirectStream,
    Transcode,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum ProgressEvent {
    TimeUpdate,
    Pause,
    Unpause,
    VolumeChange,
    Seek,
    AudioTrackChange,
    SubtitleTrackChange,
    QualityChange,
}
