//! Slint frontend for the jelly client.
//!
//! One main window, two screens (login / library). The UI runs on the
//! main thread under Slint's event loop. A worker thread carries a
//! tokio runtime that handles all networking against jellyfin-api and
//! all persistence against jelly-storage. Commands flow UI → backend
//! via an mpsc channel; updates flow backend → UI via
//! `slint::invoke_from_event_loop`, the only thread-safe way to mutate
//! window state in Slint.
//!
//! Browsing keeps a small nav stack inside `run_authed`. The bottom
//! entry is always the selected library view; pushing happens when
//! the user activates a "container" item (album → tracks, series →
//! seasons → episodes, playlist → items). Going back pops one level
//! and refetches that level's children.

#![allow(clippy::needless_return)]

use anyhow::{Context, Result};
use audio_engine::AudioEngine;
use jelly_storage::Storage;
use jellyfin_api::{
    BaseItemDto, Client, Identity, ImageOptions, ImageType as ApiImageType, ItemType, ItemsQuery,
    SortOrder, audio_stream_url, image_url, ticks_to_seconds, video_stream_url,
};
use slint::{ComponentHandle, Model, ModelRc, SharedPixelBuffer, SharedString, VecModel};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Semaphore;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tracing::{debug, error, info, warn};
use url::Url;

slint::include_modules!();

const ITEM_LIMIT: u32 = 200;
const IMAGE_PARALLELISM: usize = 8;
const IMAGE_FILL_HEIGHT: u32 = 360;

pub fn run() -> Result<()> {
    let storage = Arc::new(Storage::new().context("init storage")?);
    let device_id = storage.device_id().context("get device id")?;

    let window = MainWindow::new()?;
    let weak = window.as_weak();
    let (cmd_tx, cmd_rx) = unbounded_channel::<BackendCmd>();

    {
        let weak = weak.clone();
        let storage = storage.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(backend_loop(weak, storage, device_id, cmd_rx));
        });
    }

    window.on_sign_in_clicked({
        let cmd_tx = cmd_tx.clone();
        move |server, user, pw| {
            let _ = cmd_tx.send(BackendCmd::SignIn {
                server: server.to_string(),
                user: user.to_string(),
                pw: pw.to_string(),
            });
        }
    });
    window.on_sign_out_clicked({
        let cmd_tx = cmd_tx.clone();
        move || {
            let _ = cmd_tx.send(BackendCmd::SignOut);
        }
    });
    window.on_view_selected({
        let cmd_tx = cmd_tx.clone();
        move |view_id, collection_type| {
            let _ = cmd_tx.send(BackendCmd::SelectView {
                view_id: view_id.to_string(),
                collection_type: collection_type.to_string(),
            });
        }
    });
    window.on_activate_item({
        let cmd_tx = cmd_tx.clone();
        move |item_id, item_type| {
            let _ = cmd_tx.send(BackendCmd::ActivateItem {
                item_id: item_id.to_string(),
                item_type: item_type.to_string(),
            });
        }
    });
    window.on_back_clicked({
        let cmd_tx = cmd_tx.clone();
        move || {
            let _ = cmd_tx.send(BackendCmd::GoBack);
        }
    });

    window.run()?;
    Ok(())
}

// ---------------------------------------------------------------- commands

enum BackendCmd {
    SignIn {
        server: String,
        user: String,
        pw: String,
    },
    SignOut,
    SelectView {
        view_id: String,
        collection_type: String,
    },
    ActivateItem {
        item_id: String,
        item_type: String,
    },
    GoBack,
}

// ---------------------------------------------------------------- nav model

#[derive(Clone)]
struct NavEntry {
    /// `parentId` to pass in the items query for this level.
    parent_id: String,
    /// Header title for this level.
    title: String,
    /// Header subtitle for this level (filled in after fetching).
    subtitle: String,
    children: ChildrenType,
}

/// Determines the filter / sort applied when fetching children of a
/// given nav entry. Library roots are filtered by their collection
/// type; deeper levels just enumerate the right item kind.
#[derive(Clone)]
enum ChildrenType {
    LibraryRoot { collection_type: String },
    Tracks,
    Seasons,
    Episodes,
    PlaylistItems,
}

impl ChildrenType {
    fn query(&self, parent_id: &str) -> ItemsQuery {
        let mut q = ItemsQuery {
            parent_id: Some(parent_id.to_string()),
            limit: Some(ITEM_LIMIT),
            recursive: Some(true),
            fields: vec![
                "PrimaryImageAspectRatio".into(),
                "Genres".into(),
                "ChildCount".into(),
            ],
            sort_order: Some(SortOrder::Ascending),
            ..Default::default()
        };
        match self {
            ChildrenType::LibraryRoot { collection_type } => match collection_type.as_str() {
                "music" => {
                    q.include_item_types = vec![ItemType::MusicAlbum];
                    q.sort_by = vec!["SortName".into()];
                }
                "movies" => {
                    q.include_item_types = vec![ItemType::Movie];
                    q.sort_by = vec!["SortName".into(), "ProductionYear".into()];
                }
                "tvshows" => {
                    q.include_item_types = vec![ItemType::Series];
                    q.sort_by = vec!["SortName".into()];
                }
                "playlists" => {
                    q.include_item_types = vec![ItemType::Playlist];
                    q.sort_by = vec!["SortName".into()];
                }
                _ => {
                    q.sort_by = vec!["SortName".into()];
                }
            },
            ChildrenType::Tracks => {
                q.include_item_types = vec![ItemType::Audio];
                q.sort_by = vec!["ParentIndexNumber".into(), "IndexNumber".into()];
                q.recursive = Some(false); // direct children only
            }
            ChildrenType::Seasons => {
                q.include_item_types = vec![ItemType::Season];
                q.sort_by = vec!["IndexNumber".into()];
                q.recursive = Some(false);
            }
            ChildrenType::Episodes => {
                q.include_item_types = vec![ItemType::Episode];
                q.sort_by = vec!["IndexNumber".into()];
                q.recursive = Some(false);
            }
            ChildrenType::PlaylistItems => {
                q.sort_by = vec!["SortName".into()];
                q.recursive = Some(false);
            }
        }
        q
    }

    /// Collection type shown in the Slint window for header context.
    fn collection_label(&self) -> &'static str {
        match self {
            ChildrenType::LibraryRoot { collection_type } => match collection_type.as_str() {
                "music" => "music",
                "movies" => "movies",
                "tvshows" => "tvshows",
                "playlists" => "playlists",
                _ => "",
            },
            ChildrenType::Tracks => "music",
            ChildrenType::Seasons | ChildrenType::Episodes => "tvshows",
            ChildrenType::PlaylistItems => "playlists",
        }
    }

    /// "grid" of cards (default) vs. "tracklist" of rows for music.
    fn display_mode(&self) -> &'static str {
        match self {
            ChildrenType::Tracks => "tracklist",
            _ => "grid",
        }
    }
}

// ---------------------------------------------------------------- backend loop

async fn backend_loop(
    weak: slint::Weak<MainWindow>,
    storage: Arc<Storage>,
    device_id: String,
    mut cmd_rx: UnboundedReceiver<BackendCmd>,
) {
    if let Ok(Some(saved)) = storage.load_session() {
        info!(server = %saved.server_url, user = %saved.user_id, "restoring session");
        let identity = make_identity(&device_id);
        let client = Client::new(saved.server_url.clone(), identity)
            .with_accept_language(preferred_language())
            .with_token(saved.token.clone());
        match client.current_user().await {
            Ok(user) => {
                let first_view = if let Ok(views) = client.user_views(&user.id).await {
                    let first = views.items.first().cloned();
                    push_signed_in(&weak, &user.name, views.items);
                    first
                } else {
                    push_signed_in(&weak, &user.name, vec![]);
                    None
                };
                let state = AuthedState::new(weak.clone(), client, user.id);
                run_authed(state, storage.clone(), first_view, &mut cmd_rx).await;
                return;
            }
            Err(e) => {
                warn!(?e, "stored token rejected, falling back to login");
                let _ = storage.clear_session();
                prefill_server_url(&weak, saved.server_url.as_str());
            }
        }
    }

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCmd::SignIn { server, user, pw } => {
                set_busy(&weak, true);
                set_error(&weak, "");
                let result = attempt_sign_in(&server, &user, &pw, &device_id, &storage).await;
                set_busy(&weak, false);
                match result {
                    Ok((client, user_dto, views)) => {
                        let first = views.first().cloned();
                        push_signed_in(&weak, &user_dto.name, views);
                        let state = AuthedState::new(weak.clone(), client, user_dto.id);
                        run_authed(state, storage.clone(), first, &mut cmd_rx).await;
                        return;
                    }
                    Err(e) => set_error(&weak, &format!("{e:#}")),
                }
            }
            BackendCmd::SignOut
            | BackendCmd::SelectView { .. }
            | BackendCmd::ActivateItem { .. }
            | BackendCmd::GoBack => {}
        }
    }
}

async fn attempt_sign_in(
    server: &str,
    user: &str,
    pw: &str,
    device_id: &str,
    storage: &Storage,
) -> Result<(Client, jellyfin_api::UserDto, Vec<BaseItemDto>)> {
    let base = Url::parse(server).with_context(|| format!("invalid URL: {server}"))?;
    let identity = make_identity(device_id);
    let mut client = Client::new(base.clone(), identity).with_accept_language(preferred_language());
    let auth = client.sign_in(user, pw).await.context("sign_in")?;
    let user_dto = auth.user.context("auth response had no user")?;
    let token = auth.access_token.context("auth response had no token")?;
    storage
        .save_session(&base, &user_dto.id, &token)
        .context("persist session")?;
    let views = client
        .user_views(&user_dto.id)
        .await
        .context("user_views")?;
    Ok((client, user_dto, views.items))
}

// ---------------------------------------------------------------- authed

struct AuthedState {
    weak: slint::Weak<MainWindow>,
    client: Client,
    user_id: String,
    epoch: Arc<AtomicU64>,
    image_http: reqwest::Client,
    image_sem: Arc<Semaphore>,
    audio: AudioEngine,
}

impl AuthedState {
    fn new(weak: slint::Weak<MainWindow>, client: Client, user_id: String) -> Self {
        Self {
            weak,
            client,
            user_id,
            epoch: Arc::new(AtomicU64::new(0)),
            image_http: reqwest::Client::new(),
            image_sem: Arc::new(Semaphore::new(IMAGE_PARALLELISM)),
            audio: AudioEngine::new(),
        }
    }
}

async fn run_authed(
    state: AuthedState,
    storage: Arc<Storage>,
    initial_view: Option<BaseItemDto>,
    cmd_rx: &mut UnboundedReceiver<BackendCmd>,
) {
    let mut nav: Vec<NavEntry> = Vec::new();
    let mut current_items: Vec<BaseItemDto> = Vec::new();

    if let Some(v) = initial_view {
        nav.push(library_root_entry(&v));
        fetch_and_render(&state, &nav, &mut current_items).await;
    }

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCmd::SelectView {
                view_id,
                collection_type,
            } => {
                nav.clear();
                nav.push(NavEntry {
                    parent_id: view_id,
                    title: view_title(&collection_type),
                    subtitle: String::new(),
                    children: ChildrenType::LibraryRoot { collection_type },
                });
                fetch_and_render(&state, &nav, &mut current_items).await;
            }
            BackendCmd::ActivateItem { item_id, item_type } => {
                activate(&state, &mut nav, &mut current_items, &item_id, &item_type).await;
            }
            BackendCmd::GoBack => {
                if nav.len() > 1 {
                    nav.pop();
                    fetch_and_render(&state, &nav, &mut current_items).await;
                }
            }
            BackendCmd::SignOut => {
                info!("signing out");
                if let Err(e) = storage.clear_session() {
                    error!(?e, "clear_session failed");
                }
                push_signed_out(&state.weak);
                return;
            }
            BackendCmd::SignIn { .. } => warn!("sign-in while authed; ignoring"),
        }
    }
}

fn library_root_entry(view: &BaseItemDto) -> NavEntry {
    let ct = view.collection_type.clone().unwrap_or_default();
    NavEntry {
        parent_id: view.id.clone(),
        title: view_title(&ct),
        subtitle: String::new(),
        children: ChildrenType::LibraryRoot {
            collection_type: ct,
        },
    }
}

/// Click handler for an item. Leaves cause playback; containers push
/// a new nav level and refetch.
async fn activate(
    state: &AuthedState,
    nav: &mut Vec<NavEntry>,
    current_items: &mut Vec<BaseItemDto>,
    item_id: &str,
    item_type: &str,
) {
    match item_type {
        "Movie" | "Episode" | "Video" => {
            let Some(token) = state.client.token() else {
                warn!("no token; cannot play");
                return;
            };
            let url = video_stream_url(state.client.base_url(), item_id, token);
            info!(%url, kind = item_type, "play video");
            video_engine::play(url.to_string());
        }
        "Audio" => {
            let Some(token) = state.client.token() else {
                warn!("no token; cannot play");
                return;
            };
            let container = current_items
                .iter()
                .find(|i| i.id == item_id)
                .and_then(|i| i.media_sources.as_ref())
                .and_then(|m| m.first())
                .and_then(|s| s.container.clone());
            let url = audio_stream_url(state.client.base_url(), item_id, token);
            info!(%url, container = ?container, "play audio");
            state.audio.play_track(url.to_string(), container);
        }
        "MusicAlbum" | "Series" | "Season" | "Playlist" => {
            let Some(item) = current_items.iter().find(|i| i.id == item_id).cloned() else {
                warn!(item_id, "activate target not in current_items");
                return;
            };
            let children = match item_type {
                "MusicAlbum" => ChildrenType::Tracks,
                "Series" => ChildrenType::Seasons,
                "Season" => ChildrenType::Episodes,
                "Playlist" => ChildrenType::PlaylistItems,
                _ => unreachable!(),
            };
            let is_album = matches!(children, ChildrenType::Tracks);
            let cover_tag = if is_album {
                item.image_tags
                    .as_ref()
                    .and_then(|t| t.get("Primary"))
                    .cloned()
            } else {
                None
            };
            nav.push(NavEntry {
                parent_id: item.id.clone(),
                title: item.name.clone().unwrap_or_default(),
                subtitle: detail_subtitle(&item, item_type),
                children,
            });
            fetch_and_render(state, nav, current_items).await;
            // Cover load uses the post-fetch epoch so a quick back-out
            // cancels it cleanly via the epoch guard.
            if let Some(tag) = cover_tag {
                spawn_album_cover_fetch(state, item.id, tag);
            }
        }
        other => {
            info!(item = item_id, kind = other, "activation not supported yet");
        }
    }
}

fn detail_subtitle(item: &BaseItemDto, item_type: &str) -> String {
    match item_type {
        "MusicAlbum" => item
            .album_artist
            .clone()
            .or_else(|| item.production_year.map(|y| y.to_string()))
            .unwrap_or_default(),
        "Series" => item
            .production_year
            .map(|y| y.to_string())
            .unwrap_or_default(),
        "Season" => item
            .child_count
            .map(|c| format!("{c} episodes"))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Increment the epoch (cancels image loads from the previous level),
/// push placeholders, fetch this level's items, render them, then kick
/// off concurrent image downloads. Returns after the fetch finishes;
/// images stream in afterwards via background tasks.
async fn fetch_and_render(
    state: &AuthedState,
    nav: &[NavEntry],
    current_items: &mut Vec<BaseItemDto>,
) {
    let top = nav.last().expect("nav non-empty");
    let epoch = state.epoch.fetch_add(1, Ordering::SeqCst) + 1;
    debug!(epoch, title = %top.title, "fetch level");

    set_loading(&state.weak, true);
    set_view_meta(
        &state.weak,
        top.title.clone(),
        if top.subtitle.is_empty() {
            "Loading…".into()
        } else {
            top.subtitle.clone()
        },
        top.children.collection_label().into(),
        top.children.display_mode().into(),
    );
    set_can_go_back(&state.weak, nav.len() > 1);
    set_items(&state.weak, vec![]);
    // Cover only belongs to the album drill-down; clear it everywhere
    // else. The cover load below (when entering an album) repopulates.
    if !matches!(top.children, ChildrenType::Tracks) {
        clear_album_image(&state.weak);
    }

    let resp = match state
        .client
        .items(&state.user_id, &top.children.query(&top.parent_id))
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!(?e, "items fetch failed");
            if state.epoch.load(Ordering::SeqCst) == epoch {
                set_loading(&state.weak, false);
                set_view_meta(
                    &state.weak,
                    "Error".into(),
                    format!("{e:#}"),
                    String::new(),
                    "grid".into(),
                );
            }
            return;
        }
    };

    if state.epoch.load(Ordering::SeqCst) != epoch {
        return;
    }

    let total = resp.total_record_count.unwrap_or(resp.items.len() as i64);
    let count_label = if total as usize > resp.items.len() {
        format!("{} of {} items", resp.items.len(), total)
    } else {
        format!("{} items", resp.items.len())
    };
    let header_subtitle = if top.subtitle.is_empty() {
        count_label
    } else {
        format!("{} · {}", top.subtitle, count_label)
    };

    let items_data: Vec<ItemData> = resp.items.iter().map(build_initial_item_data).collect();

    set_view_meta(
        &state.weak,
        top.title.clone(),
        header_subtitle,
        top.children.collection_label().into(),
        top.children.display_mode().into(),
    );
    set_items(&state.weak, items_data);
    set_loading(&state.weak, false);

    *current_items = resp.items.clone();

    // Image downloads.
    let base = state.client.base_url().clone();
    let to_load: Vec<(usize, String, String)> = resp
        .items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            item.image_tags
                .as_ref()
                .and_then(|t| t.get("Primary"))
                .cloned()
                .map(|tag| (i, item.id.clone(), tag))
        })
        .collect();

    for (i, item_id, tag) in to_load {
        let weak = state.weak.clone();
        let http = state.image_http.clone();
        let sem = state.image_sem.clone();
        let epoch_ref = state.epoch.clone();
        let base = base.clone();
        tokio::spawn(async move {
            let _permit = match sem.acquire().await {
                Ok(p) => p,
                Err(_) => return,
            };
            if epoch_ref.load(Ordering::SeqCst) != epoch {
                return;
            }
            let url = image_url(
                &base,
                &item_id,
                ApiImageType::Primary,
                &ImageOptions {
                    tag: Some(tag),
                    fill_height: Some(IMAGE_FILL_HEIGHT),
                    quality: Some(90),
                    ..Default::default()
                },
            );
            match fetch_image(&http, &url).await {
                Ok((rgba, w, h)) => {
                    if epoch_ref.load(Ordering::SeqCst) == epoch {
                        push_item_image(&weak, epoch_ref.clone(), epoch, i, rgba, w, h);
                    }
                }
                Err(e) => debug!(?e, %url, "image fetch failed"),
            }
        });
    }
}

fn spawn_album_cover_fetch(state: &AuthedState, album_id: String, tag: String) {
    let weak = state.weak.clone();
    let http = state.image_http.clone();
    let sem = state.image_sem.clone();
    let epoch_ref = state.epoch.clone();
    let expected_epoch = epoch_ref.load(Ordering::SeqCst);
    let base = state.client.base_url().clone();
    tokio::spawn(async move {
        let _permit = match sem.acquire().await {
            Ok(p) => p,
            Err(_) => return,
        };
        if epoch_ref.load(Ordering::SeqCst) != expected_epoch {
            return;
        }
        let url = image_url(
            &base,
            &album_id,
            ApiImageType::Primary,
            &ImageOptions {
                tag: Some(tag),
                fill_height: Some(600),
                quality: Some(92),
                ..Default::default()
            },
        );
        match fetch_image(&http, &url).await {
            Ok((rgba, w, h)) => {
                if epoch_ref.load(Ordering::SeqCst) == expected_epoch {
                    set_album_image(&weak, epoch_ref.clone(), expected_epoch, rgba, w, h);
                }
            }
            Err(e) => debug!(?e, %url, "album cover fetch failed"),
        }
    });
}

async fn fetch_image(http: &reqwest::Client, url: &Url) -> Result<(Vec<u8>, u32, u32)> {
    let bytes = http
        .get(url.as_str())
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let decoded = tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, u32, u32)> {
        let img = image::load_from_memory(&bytes)
            .context("decode")?
            .to_rgba8();
        let (w, h) = img.dimensions();
        Ok((img.into_raw(), w, h))
    })
    .await
    .context("join blocking")??;
    Ok(decoded)
}

// ---------------------------------------------------------------- mapping

#[derive(Clone)]
struct ItemData {
    id: String,
    title: String,
    subtitle: String,
    item_type: String,
    track_index: i32,
}

/// Build the UI-side row for a Jellyfin item. Subtitle and visible
/// type both come from the item's own `Type` field, so this works
/// regardless of which nav level we're rendering.
fn build_initial_item_data(item: &BaseItemDto) -> ItemData {
    let item_type = item_type_str(&item.item_type);
    let title = item.name.clone().unwrap_or_default();
    let subtitle = match item_type.as_str() {
        "MusicAlbum" => item
            .album_artist
            .clone()
            .or_else(|| item.artists.as_ref().and_then(|a| a.first().cloned()))
            .or_else(|| item.production_year.map(|y| y.to_string()))
            .unwrap_or_default(),
        "Audio" => item.run_time_ticks.map(format_duration).unwrap_or_default(),
        "Movie" | "Series" => item
            .production_year
            .map(|y| y.to_string())
            .unwrap_or_default(),
        "Season" => item
            .child_count
            .map(|c| format!("{c} episodes"))
            .unwrap_or_default(),
        "Episode" => match (item.parent_index_number, item.index_number) {
            (Some(s), Some(e)) => format!("S{s:02}E{e:02}"),
            (None, Some(e)) => format!("E{e:02}"),
            _ => String::new(),
        },
        "Playlist" => item
            .child_count
            .map(|c| format!("{c} items"))
            .unwrap_or_default(),
        _ => String::new(),
    };
    ItemData {
        id: item.id.clone(),
        title,
        subtitle,
        item_type,
        track_index: item.index_number.unwrap_or(0),
    }
}

fn item_type_str(t: &Option<ItemType>) -> String {
    match t {
        Some(ItemType::Movie) => "Movie".into(),
        Some(ItemType::Series) => "Series".into(),
        Some(ItemType::Season) => "Season".into(),
        Some(ItemType::Episode) => "Episode".into(),
        Some(ItemType::MusicAlbum) => "MusicAlbum".into(),
        Some(ItemType::MusicArtist) => "MusicArtist".into(),
        Some(ItemType::Audio) => "Audio".into(),
        Some(ItemType::Folder) => "Folder".into(),
        Some(ItemType::CollectionFolder) => "CollectionFolder".into(),
        Some(ItemType::BoxSet) => "BoxSet".into(),
        Some(ItemType::Playlist) => "Playlist".into(),
        Some(ItemType::Video) => "Video".into(),
        Some(ItemType::Other(s)) => s.clone(),
        None => String::new(),
    }
}

fn view_title(collection_type: &str) -> String {
    match collection_type {
        "music" => "Music".into(),
        "movies" => "Movies".into(),
        "tvshows" => "TV Shows".into(),
        "playlists" => "Playlists".into(),
        _ => "Library".into(),
    }
}

fn format_duration(ticks: i64) -> String {
    let total = ticks_to_seconds(ticks).round() as i64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

// ---------------------------------------------------------------- helpers

fn make_identity(device_id: &str) -> Identity {
    Identity::new(
        "Jelly",
        "jelly-desktop",
        device_id,
        env!("CARGO_PKG_VERSION"),
    )
}

fn preferred_language() -> String {
    std::env::var("LANG")
        .ok()
        .and_then(|s| s.split('.').next().map(|s| s.replace('_', "-")))
        .unwrap_or_else(|| "en".to_string())
}

// ---------------------------------------------------------------- UI mutation

fn set_busy(weak: &slint::Weak<MainWindow>, busy: bool) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_busy(busy);
        }
    });
}

fn set_error(weak: &slint::Weak<MainWindow>, msg: &str) {
    let weak = weak.clone();
    let msg = SharedString::from(msg);
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_error_message(msg);
        }
    });
}

fn prefill_server_url(weak: &slint::Weak<MainWindow>, url: &str) {
    let weak = weak.clone();
    let url = SharedString::from(url);
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_server_url(url);
        }
    });
}

fn set_loading(weak: &slint::Weak<MainWindow>, loading: bool) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_loading(loading);
        }
    });
}

fn set_view_meta(
    weak: &slint::Weak<MainWindow>,
    title: String,
    subtitle: String,
    collection_type: String,
    view_mode: String,
) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_view_title(SharedString::from(title));
            w.set_view_subtitle(SharedString::from(subtitle));
            w.set_view_collection_type(SharedString::from(collection_type));
            w.set_view_mode(SharedString::from(view_mode));
        }
    });
}

fn set_can_go_back(weak: &slint::Weak<MainWindow>, can: bool) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_can_go_back(can);
        }
    });
}

fn set_items(weak: &slint::Weak<MainWindow>, items: Vec<ItemData>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            let slint_items: Vec<Item> = items
                .into_iter()
                .map(|d| Item {
                    id: SharedString::from(d.id),
                    title: SharedString::from(d.title),
                    subtitle: SharedString::from(d.subtitle),
                    image: slint::Image::default(),
                    has_image: false,
                    item_type: SharedString::from(d.item_type),
                    track_index: d.track_index,
                })
                .collect();
            w.set_items(ModelRc::new(VecModel::from(slint_items)));
        }
    });
}

/// Update a single card's image. Built inside the UI-thread closure
/// because `slint::Image` isn't `Send`; only raw RGBA crosses threads.
fn push_item_image(
    weak: &slint::Weak<MainWindow>,
    epoch_ref: Arc<AtomicU64>,
    expected_epoch: u64,
    index: usize,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if epoch_ref.load(Ordering::SeqCst) != expected_epoch {
            return;
        }
        let Some(w) = weak.upgrade() else {
            return;
        };
        let model = w.get_items();
        let Some(mut item) = model.row_data(index) else {
            return;
        };
        let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
        buffer.make_mut_bytes().copy_from_slice(&rgba);
        item.image = slint::Image::from_rgba8(buffer);
        item.has_image = true;
        model.set_row_data(index, item);
    });
}

fn set_album_image(
    weak: &slint::Weak<MainWindow>,
    epoch_ref: Arc<AtomicU64>,
    expected_epoch: u64,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if epoch_ref.load(Ordering::SeqCst) != expected_epoch {
            return;
        }
        let Some(w) = weak.upgrade() else {
            return;
        };
        let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
        buffer.make_mut_bytes().copy_from_slice(&rgba);
        w.set_album_image(slint::Image::from_rgba8(buffer));
        w.set_album_has_image(true);
    });
}

fn clear_album_image(weak: &slint::Weak<MainWindow>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_album_image(slint::Image::default());
            w.set_album_has_image(false);
        }
    });
}

fn push_signed_in(weak: &slint::Weak<MainWindow>, user_name: &str, views: Vec<BaseItemDto>) {
    let weak = weak.clone();
    let user_name = SharedString::from(user_name);
    let view_items: Vec<ViewItem> = views
        .into_iter()
        .map(|v| ViewItem {
            id: SharedString::from(v.id),
            name: SharedString::from(v.name.unwrap_or_default()),
            collection_type: SharedString::from(v.collection_type.unwrap_or_default()),
        })
        .collect();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_user_name(user_name);
            let model = VecModel::from(view_items);
            w.set_views(ModelRc::new(model));
            w.set_selected_view(if w.get_views().row_count() > 0 { 0 } else { -1 });
            w.set_error_message(SharedString::default());
            w.set_password(SharedString::default());
            w.set_screen(Screen::Library);
        }
    });
}

fn push_signed_out(weak: &slint::Weak<MainWindow>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_password(SharedString::default());
            w.set_user_name(SharedString::default());
            w.set_views(ModelRc::new(VecModel::<ViewItem>::default()));
            w.set_items(ModelRc::new(VecModel::<Item>::default()));
            w.set_view_title(SharedString::default());
            w.set_view_subtitle(SharedString::default());
            w.set_can_go_back(false);
            w.set_screen(Screen::Login);
        }
    });
}
