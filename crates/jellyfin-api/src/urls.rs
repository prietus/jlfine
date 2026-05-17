use url::Url;

/// Jellyfin image types served at `/Items/{id}/Images/{type}`.
#[derive(Debug, Clone, Copy)]
pub enum ImageType {
    Primary,
    Backdrop,
    Logo,
    Thumb,
}

impl ImageType {
    fn as_str(self) -> &'static str {
        match self {
            ImageType::Primary => "Primary",
            ImageType::Backdrop => "Backdrop",
            ImageType::Logo => "Logo",
            ImageType::Thumb => "Thumb",
        }
    }
}

/// Build an image URL. `tag` is the metadata-change hash Jellyfin updates
/// when the image file changes; without it, intermediate caches keep
/// serving stale images. `quality` and at least one of `fill_height` /
/// `fill_width` should be set so the server scales server-side.
#[derive(Debug, Clone, Default)]
pub struct ImageOptions {
    pub tag: Option<String>,
    pub fill_height: Option<u32>,
    pub fill_width: Option<u32>,
    pub quality: Option<u8>,
}

pub fn image_url(base: &Url, item_id: &str, image_type: ImageType, opts: &ImageOptions) -> Url {
    let mut url = join_path(base, &["Items", item_id, "Images", image_type.as_str()]);
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("quality", &opts.quality.unwrap_or(90).to_string());
        // Default fill height if neither dimension was supplied — matches
        // the Swift client's "reasonable thumbnail" default.
        if opts.fill_height.is_none() && opts.fill_width.is_none() {
            q.append_pair("fillHeight", "240");
        } else {
            if let Some(h) = opts.fill_height {
                q.append_pair("fillHeight", &h.to_string());
            }
            if let Some(w) = opts.fill_width {
                q.append_pair("fillWidth", &w.to_string());
            }
        }
        if let Some(t) = opts.tag.as_deref() {
            if !t.is_empty() {
                q.append_pair("tag", t);
            }
        }
    }
    url
}

/// Direct-play audio URL with `static=true` so the server serves the
/// original file untouched. `api_key` is the access token — Jellyfin
/// accepts it in the query for stream endpoints, which is the easy
/// path for libmpv (no custom header plumbing needed).
pub fn audio_stream_url(base: &Url, item_id: &str, token: &str) -> Url {
    let mut url = join_path(base, &["Audio", item_id, "stream"]);
    url.query_pairs_mut()
        .append_pair("static", "true")
        .append_pair("api_key", token);
    url
}

/// Direct-play video URL. Same `static=true` rule — required to preserve
/// Dolby Vision RPU and HDR metadata; any transcoding would strip them.
pub fn video_stream_url(base: &Url, item_id: &str, token: &str) -> Url {
    let mut url = join_path(base, &["Videos", item_id, "stream"]);
    url.query_pairs_mut()
        .append_pair("static", "true")
        .append_pair("api_key", token);
    url
}

/// Append segments to the base URL's existing path (Jellyfin may live at
/// `/jellyfin` behind a reverse proxy). `Url::join` doesn't work cleanly
/// for that — it replaces the last segment if the base lacks a trailing
/// slash. So we mutate `path_segments_mut` directly.
fn join_path(base: &Url, segments: &[&str]) -> Url {
    let mut url = base.clone();
    {
        let mut seg = url
            .path_segments_mut()
            .expect("base URL must support path segments");
        // Remove any trailing empty segment so the join doesn't produce
        // `//Items/...` for a base ending in `/`.
        seg.pop_if_empty();
        for s in segments {
            seg.push(s);
        }
    }
    url
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Url {
        Url::parse("https://jelly.example.com").unwrap()
    }

    fn base_with_subpath() -> Url {
        Url::parse("https://example.com/jellyfin").unwrap()
    }

    #[test]
    fn audio_stream_url_static_true() {
        let u = audio_stream_url(&base(), "abc123", "tok");
        assert_eq!(u.path(), "/Audio/abc123/stream");
        let q: Vec<_> = u.query_pairs().collect();
        assert!(q.iter().any(|(k, v)| k == "static" && v == "true"));
        assert!(q.iter().any(|(k, v)| k == "api_key" && v == "tok"));
    }

    #[test]
    fn video_stream_url_preserves_subpath() {
        let u = video_stream_url(&base_with_subpath(), "id", "t");
        assert_eq!(u.path(), "/jellyfin/Videos/id/stream");
    }

    #[test]
    fn image_url_defaults_to_fill_height_240() {
        let u = image_url(&base(), "id", ImageType::Primary, &ImageOptions::default());
        assert_eq!(u.path(), "/Items/id/Images/Primary");
        let q: Vec<_> = u.query_pairs().collect();
        assert!(q.iter().any(|(k, v)| k == "fillHeight" && v == "240"));
        assert!(q.iter().any(|(k, v)| k == "quality" && v == "90"));
    }

    #[test]
    fn image_url_uses_explicit_dimensions() {
        let opts = ImageOptions {
            fill_width: Some(800),
            quality: Some(80),
            tag: Some("hash".into()),
            ..Default::default()
        };
        let u = image_url(&base(), "id", ImageType::Backdrop, &opts);
        let q: Vec<_> = u.query_pairs().collect();
        assert!(q.iter().any(|(k, v)| k == "fillWidth" && v == "800"));
        assert!(q.iter().any(|(k, v)| k == "tag" && v == "hash"));
        assert!(q.iter().any(|(k, v)| k == "quality" && v == "80"));
        assert!(!q.iter().any(|(k, _)| k == "fillHeight"));
    }
}
