use crate::auth::{Identity, authorization_header};
use crate::error::{Error, Result};
use crate::playback::{PlaybackProgressInfo, PlaybackStartInfo, PlaybackStopInfo};
use crate::types::{
    AuthenticationResult, BaseItemDto, ItemType, ItemsResponse, PublicSystemInfo, UserDto,
};
use reqwest::header::{ACCEPT_LANGUAGE, AUTHORIZATION};
use serde::Serialize;
use url::Url;

const AUTHORIZATION_HEADER: &str = "X-Emby-Authorization";

/// Async client for a single Jellyfin server.
///
/// A client is cheap to clone — `reqwest::Client` pools connections
/// internally. Auth state (`token`) is set by `sign_in` or fed in
/// via `with_token` when restoring from persisted storage.
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    base_url: Url,
    identity: Identity,
    accept_language: Option<String>,
    token: Option<String>,
}

impl Client {
    pub fn new(base_url: Url, identity: Identity) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url,
            identity,
            accept_language: None,
            token: None,
        }
    }

    /// Set the `Accept-Language` header sent on every request. Jellyfin
    /// uses it to pick the metadata translation when a library has
    /// metadata in multiple languages.
    pub fn with_accept_language(mut self, lang: impl Into<String>) -> Self {
        self.accept_language = Some(lang.into());
        self
    }

    /// Reuse a previously-obtained access token (from storage).
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    // ---------------------------------------------------------------- auth

    /// `POST /Users/AuthenticateByName`. On success, the access token is
    /// stored on `self` and also returned in the result for the caller
    /// to persist (e.g. via `jelly-storage`).
    pub async fn sign_in(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<AuthenticationResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "PascalCase")]
        struct Body<'a> {
            username: &'a str,
            pw: &'a str,
        }
        let url = self.path(&["Users", "AuthenticateByName"]);
        // add_common_headers attaches the X-Emby-Authorization header; don't
        // duplicate it here. Reverse proxies (nginx in particular) reject
        // requests with multiple Authorization headers as malformed.
        let res = self.http.post(url).json(&Body {
            username,
            pw: password,
        });
        let res = self.add_common_headers(res).send().await?;
        let body: AuthenticationResult = parse_json(res).await?;
        let token = body
            .access_token
            .as_deref()
            .ok_or(Error::BadAuthResponse("AccessToken"))?
            .to_string();
        self.token = Some(token);
        Ok(body)
    }

    // ---------------------------------------------------------------- system

    pub async fn public_system_info(&self) -> Result<PublicSystemInfo> {
        let url = self.path(&["System", "Info", "Public"]);
        let res = self.http.get(url);
        let res = self.add_common_headers(res).send().await?;
        parse_json(res).await
    }

    // ---------------------------------------------------------------- user

    /// `GET /Users/Me` — uses the current token to identify the user.
    pub async fn current_user(&self) -> Result<UserDto> {
        let url = self.path(&["Users", "Me"]);
        let res = self.http.get(url);
        let res = self.add_common_headers(res).send().await?;
        parse_json(res).await
    }

    // ---------------------------------------------------------------- library

    /// `GET /Users/{userId}/Views` — top-level libraries the user has
    /// access to. CollectionType distinguishes Music / Movies / TV / etc.
    pub async fn user_views(&self, user_id: &str) -> Result<ItemsResponse> {
        let url = self.path(&["Users", user_id, "Views"]);
        let res = self.http.get(url);
        let res = self.add_common_headers(res).send().await?;
        parse_json(res).await
    }

    /// `GET /Users/{userId}/Items` — paginated, filterable listing.
    pub async fn items(&self, user_id: &str, query: &ItemsQuery) -> Result<ItemsResponse> {
        let url = self.path(&["Users", user_id, "Items"]);
        let req = self.http.get(url).query(&query.to_pairs());
        let res = self.add_common_headers(req).send().await?;
        parse_json(res).await
    }

    /// `GET /Users/{userId}/Items/{itemId}` — full detail for one item.
    pub async fn item(&self, user_id: &str, item_id: &str) -> Result<BaseItemDto> {
        let url = self.path(&["Users", user_id, "Items", item_id]);
        let res = self.http.get(url);
        let res = self.add_common_headers(res).send().await?;
        parse_json(res).await
    }

    // ---------------------------------------------------------------- playback

    pub async fn report_playback_start(&self, info: &PlaybackStartInfo) -> Result<()> {
        let url = self.path(&["Sessions", "Playing"]);
        let res = self.http.post(url).json(info);
        self.add_common_headers(res)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn report_playback_progress(&self, info: &PlaybackProgressInfo) -> Result<()> {
        let url = self.path(&["Sessions", "Playing", "Progress"]);
        let res = self.http.post(url).json(info);
        self.add_common_headers(res)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn report_playback_stopped(&self, info: &PlaybackStopInfo) -> Result<()> {
        let url = self.path(&["Sessions", "Playing", "Stopped"]);
        let res = self.http.post(url).json(info);
        self.add_common_headers(res)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    // ---------------------------------------------------------------- helpers

    fn path(&self, segments: &[&str]) -> Url {
        let mut url = self.base_url.clone();
        {
            let mut seg = url
                .path_segments_mut()
                .expect("base URL must support path segments");
            seg.pop_if_empty();
            for s in segments {
                seg.push(s);
            }
        }
        url
    }

    fn add_common_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        // Jellyfin accepts the MediaBrowser scheme on either `Authorization`
        // (current name) or `X-Emby-Authorization` (legacy). Sending both
        // keeps us compatible with the long tail of server versions.
        let header = authorization_header(&self.identity, self.token.as_deref());
        let mut req = req
            .header(AUTHORIZATION_HEADER, header.clone())
            .header(AUTHORIZATION, header);
        if let Some(lang) = &self.accept_language {
            req = req.header(ACCEPT_LANGUAGE, lang);
        }
        req
    }
}

/// Parameters for `/Users/{userId}/Items`. All fields are optional;
/// build with struct-update syntax for clarity:
/// `ItemsQuery { parent_id: Some(id), limit: Some(50), ..Default::default() }`.
#[derive(Debug, Default, Clone)]
pub struct ItemsQuery {
    pub parent_id: Option<String>,
    pub include_item_types: Vec<ItemType>,
    pub sort_by: Vec<String>,
    pub sort_order: Option<SortOrder>,
    pub limit: Option<u32>,
    pub start_index: Option<u32>,
    pub recursive: Option<bool>,
    pub fields: Vec<String>,
    pub search_term: Option<String>,
    pub artist_ids: Vec<String>,
    pub album_artist_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum SortOrder {
    Ascending,
    Descending,
}

impl SortOrder {
    fn as_str(self) -> &'static str {
        match self {
            SortOrder::Ascending => "Ascending",
            SortOrder::Descending => "Descending",
        }
    }
}

impl ItemsQuery {
    fn to_pairs(&self) -> Vec<(&'static str, String)> {
        let mut v = Vec::new();
        if let Some(p) = &self.parent_id {
            v.push(("ParentId", p.clone()));
        }
        if !self.include_item_types.is_empty() {
            v.push((
                "IncludeItemTypes",
                self.include_item_types
                    .iter()
                    .map(item_type_as_str)
                    .collect::<Vec<_>>()
                    .join(","),
            ));
        }
        if !self.sort_by.is_empty() {
            v.push(("SortBy", self.sort_by.join(",")));
        }
        if let Some(o) = self.sort_order {
            v.push(("SortOrder", o.as_str().into()));
        }
        if let Some(l) = self.limit {
            v.push(("Limit", l.to_string()));
        }
        if let Some(s) = self.start_index {
            v.push(("StartIndex", s.to_string()));
        }
        if let Some(r) = self.recursive {
            v.push(("Recursive", r.to_string()));
        }
        if !self.fields.is_empty() {
            v.push(("Fields", self.fields.join(",")));
        }
        if let Some(t) = &self.search_term {
            v.push(("SearchTerm", t.clone()));
        }
        if !self.artist_ids.is_empty() {
            v.push(("ArtistIds", self.artist_ids.join(",")));
        }
        if !self.album_artist_ids.is_empty() {
            v.push(("AlbumArtistIds", self.album_artist_ids.join(",")));
        }
        v
    }
}

fn item_type_as_str(t: &ItemType) -> String {
    match t {
        ItemType::Movie => "Movie".into(),
        ItemType::Series => "Series".into(),
        ItemType::Season => "Season".into(),
        ItemType::Episode => "Episode".into(),
        ItemType::MusicAlbum => "MusicAlbum".into(),
        ItemType::MusicArtist => "MusicArtist".into(),
        ItemType::Audio => "Audio".into(),
        ItemType::Folder => "Folder".into(),
        ItemType::CollectionFolder => "CollectionFolder".into(),
        ItemType::BoxSet => "BoxSet".into(),
        ItemType::Playlist => "Playlist".into(),
        ItemType::Video => "Video".into(),
        ItemType::Other(s) => s.clone(),
    }
}

async fn parse_json<T: serde::de::DeserializeOwned>(res: reqwest::Response) -> Result<T> {
    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(Error::Server {
            status: status.as_u16(),
            body,
        });
    }
    let bytes = res.bytes().await?;
    Ok(serde_json::from_slice(&bytes)?)
}
