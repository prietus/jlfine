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
    MediaStream, PersonInfo, SortOrder, audio_stream_url, image_url, ticks_to_seconds,
    video_stream_url,
};
use slint::{ComponentHandle, Model, ModelRc, SharedPixelBuffer, SharedString, VecModel};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Semaphore;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tracing::{debug, error, info, warn};
use url::Url;
use video_engine::AudioDevice;

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
    window.on_open_settings({
        let cmd_tx = cmd_tx.clone();
        let weak = weak.clone();
        move || {
            // Switch screen synchronously so the panel appears
            // immediately even while the device enumeration is in
            // flight; the dropdown shows "Loading…" until results
            // arrive.
            if let Some(w) = weak.upgrade() {
                w.set_screen(Screen::Settings);
            }
            let _ = cmd_tx.send(BackendCmd::RefreshDevices);
        }
    });
    window.on_select_audio_device({
        let cmd_tx = cmd_tx.clone();
        move |id| {
            let _ = cmd_tx.send(BackendCmd::SelectAudioDevice { id: id.to_string() });
        }
    });
    window.on_set_exclusive_mode({
        let cmd_tx = cmd_tx.clone();
        move |v| {
            let _ = cmd_tx.send(BackendCmd::SetExclusiveMode { value: v });
        }
    });
    window.on_refresh_devices({
        let cmd_tx = cmd_tx.clone();
        move || {
            let _ = cmd_tx.send(BackendCmd::RefreshDevices);
        }
    });
    // Spawn the user's default browser asynchronously. Failures get
    // logged — we never want a missing handler to surface as a panic
    // inside the Slint callback.
    window.on_open_url(|u| {
        let url = u.to_string();
        std::thread::spawn(move || {
            if let Err(e) = open_external_url(&url) {
                warn!(%url, ?e, "open url failed");
            }
        });
    });
    window.on_download_album({
        let cmd_tx = cmd_tx.clone();
        move || {
            let _ = cmd_tx.send(BackendCmd::DownloadAlbum);
        }
    });
    window.on_play_detail({
        let cmd_tx = cmd_tx.clone();
        move || {
            let _ = cmd_tx.send(BackendCmd::PlayDetail);
        }
    });
    window.on_download_detail({
        let cmd_tx = cmd_tx.clone();
        move || {
            let _ = cmd_tx.send(BackendCmd::DownloadDetail);
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
    RefreshDevices,
    SelectAudioDevice {
        id: String,
    },
    SetExclusiveMode {
        value: bool,
    },
    DownloadAlbum,
    /// Reproducir on the movie detail page.
    PlayDetail,
    /// Descargar on the movie detail page.
    DownloadDetail,
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
    /// Populated only when this nav level is an album's tracklist —
    /// the artist for path construction at download time.
    album_artist: Option<String>,
}

/// Determines the filter / sort applied when fetching children of a
/// given nav entry. Library roots are filtered by their collection
/// type; deeper levels just enumerate the right item kind.
#[derive(Clone)]
enum ChildrenType {
    LibraryRoot {
        collection_type: String,
    },
    Tracks,
    Episodes,
    PlaylistItems,
    /// A movie's metadata detail page — a leaf with no children list.
    MovieDetail,
    /// A series' metadata detail page; its children (seasons) render
    /// below the metadata.
    SeriesDetail,
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
                "Overview".into(),
                // MediaSources isn't returned by default; we need it
                // for download (correct file extension via .path) and
                // it lets us hint symphonia properly during playback.
                "MediaSources".into(),
                "Path".into(),
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
            ChildrenType::SeriesDetail => {
                q.include_item_types = vec![ItemType::Season];
                q.sort_by = vec!["IndexNumber".into()];
                q.recursive = Some(false);
            }
            // No children list to fetch; activate() short-circuits the
            // fetch and just populates the detail page.
            ChildrenType::MovieDetail => {}
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
            ChildrenType::Episodes | ChildrenType::SeriesDetail => "tvshows",
            ChildrenType::MovieDetail => "movies",
            ChildrenType::PlaylistItems => "playlists",
        }
    }

    /// "grid" of cards (default), "tracklist" of rows for music, or
    /// "detail" for the movie/series metadata page.
    fn display_mode(&self) -> &'static str {
        match self {
            ChildrenType::Tracks => "tracklist",
            ChildrenType::MovieDetail | ChildrenType::SeriesDetail => "detail",
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
                let state = AuthedState::new(weak.clone(), client, user.id, storage.clone());
                push_initial_audio_state(&state);
                spawn_device_refresh(&state, /* auto_pick = */ true);
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
                        let state =
                            AuthedState::new(weak.clone(), client, user_dto.id, storage.clone());
                        push_initial_audio_state(&state);
                        spawn_device_refresh(&state, /* auto_pick = */ true);
                        run_authed(state, storage.clone(), first, &mut cmd_rx).await;
                        return;
                    }
                    Err(e) => set_error(&weak, &format!("{e:#}")),
                }
            }
            BackendCmd::SignOut
            | BackendCmd::SelectView { .. }
            | BackendCmd::ActivateItem { .. }
            | BackendCmd::GoBack
            | BackendCmd::RefreshDevices
            | BackendCmd::SelectAudioDevice { .. }
            | BackendCmd::SetExclusiveMode { .. }
            | BackendCmd::DownloadAlbum
            | BackendCmd::PlayDetail
            | BackendCmd::DownloadDetail => {}
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
    storage: Arc<Storage>,
    /// Last device list fetched from mpv. Used to resolve a selected
    /// id back to a human label and to drive the auto-pick of the
    /// first bitperfect device on a fresh install.
    audio_devices: Arc<Mutex<Vec<AudioDevice>>>,
    /// Album currently displayed in the tracklist (used by the
    /// download task to decide whether to push progress updates to the
    /// UI — if the user navigated away, the task keeps writing files
    /// but stops touching the window).
    current_album_id: Arc<Mutex<Option<String>>>,
    /// Cancel flag for the in-flight download. Replaced on each new
    /// click; the previous Arc gets dropped after we flip it true.
    download_cancel: Arc<Mutex<Option<Arc<std::sync::atomic::AtomicBool>>>>,
}

impl AuthedState {
    fn new(
        weak: slint::Weak<MainWindow>,
        client: Client,
        user_id: String,
        storage: Arc<Storage>,
    ) -> Self {
        Self {
            weak,
            client,
            user_id,
            epoch: Arc::new(AtomicU64::new(0)),
            image_http: reqwest::Client::new(),
            image_sem: Arc::new(Semaphore::new(IMAGE_PARALLELISM)),
            audio: AudioEngine::new(),
            storage,
            audio_devices: Arc::new(Mutex::new(Vec::new())),
            current_album_id: Arc::new(Mutex::new(None)),
            download_cancel: Arc::new(Mutex::new(None)),
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
                    album_artist: None,
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
            BackendCmd::RefreshDevices => {
                spawn_device_refresh(&state, /* auto_pick = */ false);
            }
            BackendCmd::SelectAudioDevice { id } => {
                let id_opt = if id.is_empty() {
                    None
                } else {
                    Some(id.as_str())
                };
                if let Err(e) = storage.set_audio_device(id_opt) {
                    error!(?e, "persist audio device failed");
                }
                let label = resolve_device_label(&state, &id);
                set_selected_audio_device(&state.weak, &id, &label);
            }
            BackendCmd::SetExclusiveMode { value } => {
                if let Err(e) = storage.set_exclusive_mode(value) {
                    error!(?e, "persist exclusive mode failed");
                }
                set_exclusive_mode(&state.weak, value);
            }
            BackendCmd::DownloadAlbum => {
                start_album_download(&state, nav.last(), &current_items);
            }
            BackendCmd::PlayDetail => match nav.last() {
                Some(top) if matches!(top.children, ChildrenType::MovieDetail) => {
                    play_video(&state, &top.parent_id);
                }
                _ => warn!("play-detail with no playable detail on top of nav"),
            },
            BackendCmd::DownloadDetail => {
                info!("video download from detail page not implemented yet");
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
        album_artist: None,
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
        // Episodes still play on click; movies/series open a detail
        // page first (Reproducir plays from there).
        "Episode" | "Video" => {
            play_video(state, item_id);
        }
        "Movie" => {
            open_detail(
                state,
                nav,
                current_items,
                item_id,
                /* is_series = */ false,
            )
            .await;
        }
        "Series" => {
            open_detail(
                state,
                nav,
                current_items,
                item_id,
                /* is_series = */ true,
            )
            .await;
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
            let audio_device = state.storage.audio_device().ok().flatten();
            let exclusive = state.storage.exclusive_mode().unwrap_or(true);
            info!(%url, container = ?container, ?audio_device, exclusive, "play audio");
            state
                .audio
                .play_track(url.to_string(), container, audio_device, exclusive);
        }
        "MusicAlbum" | "Season" | "Playlist" => {
            let Some(item) = current_items.iter().find(|i| i.id == item_id).cloned() else {
                warn!(item_id, "activate target not in current_items");
                return;
            };
            let children = match item_type {
                "MusicAlbum" => ChildrenType::Tracks,
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
                album_artist: if is_album {
                    Some(item.album_artist.clone().unwrap_or_default())
                } else {
                    None
                },
            });
            if is_album {
                *state.current_album_id.lock().unwrap() = Some(item.id.clone());
                reset_album_download_ui(&state.weak);
            } else {
                *state.current_album_id.lock().unwrap() = None;
            }
            fetch_and_render(state, nav, current_items).await;
            // Cover load uses the post-fetch epoch so a quick back-out
            // cancels it cleanly via the epoch guard.
            if let Some(tag) = cover_tag {
                spawn_album_cover_fetch(state, item.id.clone(), tag);
            }
            if is_album {
                // Jellyfin's listing endpoint sometimes ships a thin
                // version of each item (Overview/ProductionYear can be
                // dropped even when they're in `fields`). Refetch the
                // single item to get the full record — small request,
                // and it's the standard pattern for album detail.
                let full = match state.client.item(&state.user_id, &item.id).await {
                    Ok(f) => f,
                    Err(e) => {
                        warn!(?e, album = %item.id, "item refetch failed; using listing copy");
                        item.clone()
                    }
                };
                debug!(
                    year = ?full.production_year,
                    has_overview = full.overview.is_some(),
                    n_genres = full.genres.as_ref().map(|g| g.len()).unwrap_or(0),
                    "album detail"
                );
                let (meta_line, genres) = album_header_strings(&full, current_items);
                let artist = full.album_artist.clone().unwrap_or_default();
                let overview = full.overview.clone().unwrap_or_default();
                let wiki = extract_wikipedia_url(&overview);
                set_album_meta(&state.weak, artist, meta_line, genres, overview, wiki);
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

/// Direct-play a movie/episode by id (Reproducir, or an episode click).
fn play_video(state: &AuthedState, item_id: &str) {
    let Some(token) = state.client.token() else {
        warn!("no token; cannot play");
        return;
    };
    let url = video_stream_url(state.client.base_url(), item_id, token);
    let audio_device = state.storage.audio_device().ok().flatten();
    info!(%url, ?audio_device, "play video");
    video_engine::play(url.to_string(), audio_device);
}

/// Push a movie/series detail page onto the nav stack. For a series we
/// also fetch its seasons (rendered under the metadata); a movie has no
/// children. The thin listing copy is replaced by a full single-item
/// refetch for the rich metadata, then images stream in.
async fn open_detail(
    state: &AuthedState,
    nav: &mut Vec<NavEntry>,
    current_items: &mut Vec<BaseItemDto>,
    item_id: &str,
    is_series: bool,
) {
    let Some(item) = current_items.iter().find(|i| i.id == item_id).cloned() else {
        warn!(item_id, "detail target not in current_items");
        return;
    };
    nav.push(NavEntry {
        parent_id: item.id.clone(),
        title: item.name.clone().unwrap_or_default(),
        subtitle: String::new(),
        children: if is_series {
            ChildrenType::SeriesDetail
        } else {
            ChildrenType::MovieDetail
        },
        album_artist: None,
    });
    *state.current_album_id.lock().unwrap() = None;
    // Flips view-mode to "detail"; fetches seasons for a series, no-op
    // children for a movie.
    fetch_and_render(state, nav, current_items).await;
    let full = match state.client.item(&state.user_id, &item.id).await {
        Ok(f) => f,
        Err(e) => {
            warn!(?e, id = %item.id, "detail refetch failed; using listing copy");
            item.clone()
        }
    };
    apply_detail(state, &full, is_series);
}

/// Build every detail-page string/model from a full item record, push
/// them to the UI, then spawn the backdrop + cast-photo loads.
fn apply_detail(state: &AuthedState, item: &BaseItemDto, is_series: bool) {
    let actors: Vec<PersonInfo> = item
        .people
        .as_ref()
        .map(|ps| {
            ps.iter()
                .filter(|p| p.person_type.as_deref() == Some("Actor"))
                .take(24)
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let cast: Vec<(String, String)> = actors
        .iter()
        .map(|p| {
            (
                p.name.clone().unwrap_or_default(),
                p.role.clone().unwrap_or_default(),
            )
        })
        .collect();

    let (imdb_url, tmdb_url) = provider_urls(item, is_series);
    let premiere = item
        .premiere_date
        .as_deref()
        .and_then(format_premiere_date)
        .map(|d| format!("Estreno: {d}"))
        .unwrap_or_default();
    let country = item
        .production_locations
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| format!("País: {}", v.join(", ")))
        .unwrap_or_default();
    let studios = item
        .studios
        .as_ref()
        .map(|s| s.iter().filter_map(|x| x.name.clone()).collect::<Vec<_>>())
        .filter(|v| !v.is_empty())
        .map(|v| format!("Estudios: {}", v.join(", ")))
        .unwrap_or_default();

    set_detail_meta(
        &state.weak,
        DetailMeta {
            meta: detail_meta_line(item),
            tagline: item
                .taglines
                .as_ref()
                .and_then(|t| t.first())
                .cloned()
                .unwrap_or_default(),
            overview: item.overview.clone().unwrap_or_default(),
            director: people_line(item, "Director", "Director"),
            writers: people_line(item, "Writer", "Guionistas"),
            genres: item.genres.clone().unwrap_or_default(),
            cast,
            premiere,
            country,
            studios,
            tags: item.tags.clone().unwrap_or_default(),
            media_info: build_media_info(item),
            imdb_url,
            tmdb_url,
            can_play: !is_series,
            children_label: if is_series {
                "Temporadas".to_string()
            } else {
                String::new()
            },
        },
    );

    if let Some(tag) = item
        .backdrop_image_tags
        .as_ref()
        .and_then(|v| v.first())
        .cloned()
    {
        spawn_backdrop_fetch(state, item.id.clone(), tag);
    }
    for (i, p) in actors.iter().enumerate() {
        if let (Some(id), Some(tag)) = (p.id.clone(), p.primary_image_tag.clone()) {
            spawn_cast_fetch(state, i, id, tag);
        }
    }
}

/// "2008 · 1h 42m · PG-13 · ★ 5.1 · 🍅 9%" — only the parts present.
fn detail_meta_line(item: &BaseItemDto) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(y) = item.production_year {
        parts.push(y.to_string());
    }
    if let Some(t) = item.run_time_ticks.filter(|t| *t > 0) {
        parts.push(format_runtime(t));
    }
    if let Some(r) = item.official_rating.clone().filter(|s| !s.is_empty()) {
        parts.push(r);
    }
    if let Some(c) = item.community_rating {
        parts.push(format!("★ {c:.1}"));
    }
    if let Some(cr) = item.critic_rating {
        parts.push(format!("🍅 {}%", cr.round() as i64));
    }
    parts.join("  ·  ")
}

/// Runtime ticks → "1h 42m" / "42m".
fn format_runtime(ticks: i64) -> String {
    let total = ticks_to_seconds(ticks).round() as i64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    if h > 0 {
        format!("{h}h {m:02}m")
    } else {
        format!("{m}m")
    }
}

/// "Director: A, B" / "Guionistas: A, B" from the people list, filtered
/// by Jellyfin person type. Empty when nobody matches.
fn people_line(item: &BaseItemDto, person_type: &str, label: &str) -> String {
    let names: Vec<String> = item
        .people
        .as_ref()
        .map(|ps| {
            ps.iter()
                .filter(|p| p.person_type.as_deref() == Some(person_type))
                .filter_map(|p| p.name.clone())
                .collect()
        })
        .unwrap_or_default();
    if names.is_empty() {
        String::new()
    } else {
        format!("{label}: {}", names.join(", "))
    }
}

/// External-link URLs from `ProviderIds`. Keys are matched case-
/// insensitively ("Imdb"/"Tmdb"). Empty string when absent.
fn provider_urls(item: &BaseItemDto, is_series: bool) -> (String, String) {
    let Some(ids) = item.provider_ids.as_ref() else {
        return (String::new(), String::new());
    };
    let get = |key: &str| {
        ids.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.clone())
            .filter(|v| !v.is_empty())
    };
    let imdb = get("Imdb")
        .map(|id| format!("https://www.imdb.com/title/{id}/"))
        .unwrap_or_default();
    let tmdb = get("Tmdb")
        .map(|id| {
            let kind = if is_series { "tv" } else { "movie" };
            format!("https://www.themoviedb.org/{kind}/{id}")
        })
        .unwrap_or_default();
    (imdb, tmdb)
}

/// ISO date ("2008-03-04T…") → "4 de marzo de 2008". None if unparseable.
fn format_premiere_date(iso: &str) -> Option<String> {
    let date = iso.split('T').next()?;
    let mut it = date.split('-');
    let y: i32 = it.next()?.parse().ok()?;
    let mo: usize = it.next()?.parse().ok()?;
    let d: u32 = it.next()?.parse().ok()?;
    const MESES: [&str; 12] = [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ];
    let mes = MESES.get(mo.checked_sub(1)?)?;
    Some(format!("{d} de {mes} de {y}"))
}

/// "Información del medio" chips: resolution, video codec, HDR range,
/// each audio track (codec/channels/lang), container, bitrate.
fn build_media_info(item: &BaseItemDto) -> Vec<String> {
    let source = item.media_sources.as_ref().and_then(|m| m.first());
    let streams: &[MediaStream] = item
        .media_streams
        .as_deref()
        .or_else(|| source.and_then(|s| s.media_streams.as_deref()))
        .unwrap_or(&[]);

    let mut out: Vec<String> = Vec::new();
    if let Some(v) = streams
        .iter()
        .find(|s| s.stream_type.as_deref() == Some("Video"))
    {
        if let Some(h) = v.height {
            out.push(resolution_label(h));
        }
        if let Some(c) = v.codec.as_deref() {
            out.push(video_codec_label(c));
        }
        if let Some(r) = v
            .video_range
            .as_deref()
            .filter(|r| !r.is_empty() && !r.eq_ignore_ascii_case("SDR"))
        {
            out.push(r.to_uppercase());
        }
    }
    for a in streams
        .iter()
        .filter(|s| s.stream_type.as_deref() == Some("Audio"))
    {
        out.push(audio_label(a));
    }
    if let Some(c) = source
        .and_then(|s| s.container.as_deref())
        .filter(|c| !c.is_empty())
    {
        out.push(c.to_uppercase());
    }
    if let Some(b) = source.and_then(|s| s.bitrate).filter(|b| *b > 0) {
        out.push(format!("{:.1} Mbps", b as f64 / 1_000_000.0));
    }
    out.retain(|s| !s.is_empty());
    out
}

fn resolution_label(h: i32) -> String {
    match h {
        x if x >= 2000 => "4K".into(),
        x if x >= 1400 => "1440p".into(),
        x if x >= 1000 => "1080p".into(),
        x if x >= 700 => "720p".into(),
        x if x >= 540 => "576p".into(),
        x if x > 0 => format!("{x}p"),
        _ => String::new(),
    }
}

fn video_codec_label(c: &str) -> String {
    match c.to_ascii_lowercase().as_str() {
        "hevc" | "h265" => "HEVC".into(),
        "h264" | "avc" => "H.264".into(),
        "av1" => "AV1".into(),
        "vp9" => "VP9".into(),
        "mpeg2video" => "MPEG-2".into(),
        other => other.to_uppercase(),
    }
}

fn audio_label(a: &MediaStream) -> String {
    let codec = a.codec.as_deref().unwrap_or("").to_uppercase();
    let ch = a.channels.map(channel_label).unwrap_or_default();
    let lang = a
        .language
        .as_deref()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_uppercase())
        .unwrap_or_default();
    [codec, ch, lang]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn channel_label(ch: i32) -> String {
    match ch {
        1 => "Mono".into(),
        2 => "2.0".into(),
        6 => "5.1".into(),
        8 => "7.1".into(),
        n => format!("{n}ch"),
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
        *state.current_album_id.lock().unwrap() = None;
    }
    // Detail props are stale the moment we leave a detail page; wiping
    // them here avoids the previous movie's backdrop/overview flashing
    // before apply_detail repopulates (after the single-item refetch).
    clear_detail(&state.weak);
    // A movie has no children to list — skip the items query entirely
    // and let apply_detail fill the page.
    if matches!(top.children, ChildrenType::MovieDetail) {
        *current_items = Vec::new();
        set_view_meta(
            &state.weak,
            top.title.clone(),
            String::new(),
            top.children.collection_label().into(),
            "detail".into(),
        );
        set_loading(&state.weak, false);
        return;
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
    // The series detail page shows its own meta row; don't tack a
    // "N items" count onto the nav subtitle there.
    let header_subtitle = if matches!(top.children, ChildrenType::SeriesDetail) {
        top.subtitle.clone()
    } else if top.subtitle.is_empty() {
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

// ---------------------------------------------------------------- audio settings

/// Push whatever the storage already knows into the settings panel.
/// Selected-id can point at a device we don't have in the cached
/// list yet (we haven't called mpv) — that's fine, the dropdown
/// shows the id-as-label until [`spawn_device_refresh`] resolves a
/// real description.
fn push_initial_audio_state(state: &AuthedState) {
    let id = state
        .storage
        .audio_device()
        .ok()
        .flatten()
        .unwrap_or_default();
    let exclusive = state.storage.exclusive_mode().unwrap_or(true);
    let label = if id.is_empty() {
        "System default".to_string()
    } else {
        id.clone()
    };
    set_selected_audio_device(&state.weak, &id, &label);
    set_exclusive_mode(&state.weak, exclusive);
    set_devices_loading(&state.weak, true);
}

/// Enumerate audio outputs via mpv on a blocking thread, push the
/// result to the UI, and (when `auto_pick` is true and storage has
/// no preference yet) write the first bitperfect device back to
/// storage. The first-launch auto-pick is the only thing that
/// changes selection without explicit user input.
fn spawn_device_refresh(state: &AuthedState, auto_pick: bool) {
    let weak = state.weak.clone();
    let cache = state.audio_devices.clone();
    let storage = state.storage.clone();
    set_devices_loading(&weak, true);
    std::thread::spawn(move || {
        let devices = match video_engine::list_audio_devices() {
            Ok(d) => d,
            Err(e) => {
                error!(?e, "list_audio_devices failed");
                set_devices_loading(&weak, false);
                return;
            }
        };
        {
            let mut guard = cache.lock().unwrap();
            *guard = devices.clone();
        }

        // Auto-pick on first launch only.
        let mut selected_id = storage.audio_device().ok().flatten().unwrap_or_default();
        if auto_pick
            && selected_id.is_empty()
            && let Some(first) = devices.iter().find(|d| d.bitperfect)
        {
            if let Err(e) = storage.set_audio_device(Some(&first.id)) {
                warn!(?e, "auto-pick persist failed");
            } else {
                info!(id = %first.id, desc = %first.description, "auto-picked bitperfect device");
                selected_id = first.id.clone();
            }
        }

        let label = if selected_id.is_empty() {
            "System default".to_string()
        } else {
            devices
                .iter()
                .find(|d| d.id == selected_id)
                .map(|d| d.description.clone())
                .unwrap_or_else(|| selected_id.clone())
        };

        set_audio_devices(&weak, devices);
        set_selected_audio_device(&weak, &selected_id, &label);
        set_devices_loading(&weak, false);
    });
}

fn resolve_device_label(state: &AuthedState, id: &str) -> String {
    if id.is_empty() {
        return "System default".to_string();
    }
    let guard = state.audio_devices.lock().unwrap();
    guard
        .iter()
        .find(|d| d.id == id)
        .map(|d| d.description.clone())
        .unwrap_or_else(|| id.to_string())
}

// ---------------------------------------------------------------- mapping

#[derive(Clone)]
struct ItemData {
    id: String,
    title: String,
    subtitle: String,
    item_type: String,
    track_index: i32,
    /// Per-track technical badge for audio rows (e.g. "DSD64",
    /// "FLAC 24/96"); empty for everything else.
    tech: String,
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
    let tech = if item_type == "Audio" {
        track_tech(item)
    } else {
        String::new()
    };
    ItemData {
        id: item.id.clone(),
        title,
        subtitle,
        item_type,
        track_index: item.index_number.unwrap_or(0),
        tech,
    }
}

/// Compact technical badge for an audio track, sourced from its audio
/// MediaStream: DSD streams show their rate as "DSD64/128/…", PCM/
/// lossless show "<CODEC> <bitDepth>/<kHz>" (e.g. "FLAC 24/96"). Empty
/// when no usable stream metadata is present.
fn track_tech(item: &BaseItemDto) -> String {
    let streams = item.media_streams.as_ref().or_else(|| {
        item.media_sources
            .as_ref()
            .and_then(|m| m.first())
            .and_then(|s| s.media_streams.as_ref())
    });
    let Some(audio) = streams.and_then(|s| {
        s.iter()
            .find(|st| st.stream_type.as_deref() == Some("Audio"))
    }) else {
        return String::new();
    };

    let codec = audio.codec.as_deref().unwrap_or("");
    let sample_rate = audio.sample_rate.unwrap_or(0);
    let codec_upper = codec.to_ascii_uppercase();

    // DSD: the "sample rate" is the 1-bit rate; express it as the
    // familiar DSDxx multiple of the 44.1 kHz CD base (2.8224 MHz = 64).
    if codec_upper.contains("DSD") || codec_upper.contains("DST") {
        if sample_rate > 0 {
            let multiple = (sample_rate as f64 / 44_100.0).round() as i64;
            return format!("DSD{multiple}");
        }
        return "DSD".to_string();
    }

    let mut parts: Vec<String> = Vec::new();
    if !codec.is_empty() {
        parts.push(codec_upper);
    }
    if let Some(depth) = audio.bit_depth
        && sample_rate > 0
    {
        let khz = sample_rate as f64 / 1000.0;
        let khz_str = if khz.fract() == 0.0 {
            format!("{khz:.0}")
        } else {
            format!("{khz:.1}")
        };
        parts.push(format!("{depth}/{khz_str}"));
    } else if sample_rate > 0 {
        let khz = sample_rate as f64 / 1000.0;
        let khz_str = if khz.fract() == 0.0 {
            format!("{khz:.0}")
        } else {
            format!("{khz:.1}")
        };
        parts.push(format!("{khz_str} kHz"));
    }
    parts.join(" ")
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
                    tech: SharedString::from(d.tech),
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

// ---------------------------------------------------------------- detail page

/// Everything the detail page needs as ready-to-display strings. Built
/// on the backend thread by [`apply_detail`]; pushed to Slint by
/// [`set_detail_meta`]. Cast images are filled in afterwards by index.
struct DetailMeta {
    meta: String,
    tagline: String,
    overview: String,
    director: String,
    writers: String,
    genres: Vec<String>,
    cast: Vec<(String, String)>,
    premiere: String,
    country: String,
    studios: String,
    tags: Vec<String>,
    media_info: Vec<String>,
    imdb_url: String,
    tmdb_url: String,
    can_play: bool,
    children_label: String,
}

fn string_model(v: Vec<String>) -> ModelRc<SharedString> {
    let s: Vec<SharedString> = v.into_iter().map(SharedString::from).collect();
    ModelRc::new(VecModel::from(s))
}

fn set_detail_meta(weak: &slint::Weak<MainWindow>, d: DetailMeta) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        let Some(w) = weak.upgrade() else {
            return;
        };
        w.set_detail_meta(SharedString::from(d.meta));
        w.set_detail_tagline(SharedString::from(d.tagline));
        w.set_detail_overview(SharedString::from(d.overview));
        w.set_detail_director(SharedString::from(d.director));
        w.set_detail_writers(SharedString::from(d.writers));
        w.set_detail_genres(string_model(d.genres));
        let cast: Vec<CastMember> = d
            .cast
            .into_iter()
            .map(|(name, character)| CastMember {
                name: SharedString::from(name),
                character: SharedString::from(character),
                image: slint::Image::default(),
                has_image: false,
            })
            .collect();
        w.set_detail_cast(ModelRc::new(VecModel::from(cast)));
        w.set_detail_premiere(SharedString::from(d.premiere));
        w.set_detail_country(SharedString::from(d.country));
        w.set_detail_studios(SharedString::from(d.studios));
        w.set_detail_tags(string_model(d.tags));
        w.set_detail_media_info(string_model(d.media_info));
        w.set_detail_imdb_url(SharedString::from(d.imdb_url));
        w.set_detail_tmdb_url(SharedString::from(d.tmdb_url));
        w.set_detail_can_play(d.can_play);
        w.set_detail_children_label(SharedString::from(d.children_label));
    });
}

fn clear_detail(weak: &slint::Weak<MainWindow>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        let Some(w) = weak.upgrade() else {
            return;
        };
        w.set_detail_has_backdrop(false);
        w.set_detail_backdrop(slint::Image::default());
        w.set_detail_meta(SharedString::new());
        w.set_detail_tagline(SharedString::new());
        w.set_detail_overview(SharedString::new());
        w.set_detail_director(SharedString::new());
        w.set_detail_writers(SharedString::new());
        w.set_detail_genres(ModelRc::new(VecModel::from(Vec::<SharedString>::new())));
        w.set_detail_cast(ModelRc::new(VecModel::from(Vec::<CastMember>::new())));
        w.set_detail_premiere(SharedString::new());
        w.set_detail_country(SharedString::new());
        w.set_detail_studios(SharedString::new());
        w.set_detail_tags(ModelRc::new(VecModel::from(Vec::<SharedString>::new())));
        w.set_detail_media_info(ModelRc::new(VecModel::from(Vec::<SharedString>::new())));
        w.set_detail_imdb_url(SharedString::new());
        w.set_detail_tmdb_url(SharedString::new());
        w.set_detail_can_play(false);
        w.set_detail_children_label(SharedString::new());
    });
}

fn set_detail_backdrop(
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
        w.set_detail_backdrop(slint::Image::from_rgba8(buffer));
        w.set_detail_has_backdrop(true);
    });
}

fn push_cast_image(
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
        let model = w.get_detail_cast();
        let Some(mut member) = model.row_data(index) else {
            return;
        };
        let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
        buffer.make_mut_bytes().copy_from_slice(&rgba);
        member.image = slint::Image::from_rgba8(buffer);
        member.has_image = true;
        model.set_row_data(index, member);
    });
}

fn spawn_backdrop_fetch(state: &AuthedState, item_id: String, tag: String) {
    let weak = state.weak.clone();
    let http = state.image_http.clone();
    let sem = state.image_sem.clone();
    let epoch_ref = state.epoch.clone();
    let expected = epoch_ref.load(Ordering::SeqCst);
    let base = state.client.base_url().clone();
    tokio::spawn(async move {
        let _permit = match sem.acquire().await {
            Ok(p) => p,
            Err(_) => return,
        };
        if epoch_ref.load(Ordering::SeqCst) != expected {
            return;
        }
        let url = image_url(
            &base,
            &item_id,
            ApiImageType::Backdrop,
            &ImageOptions {
                tag: Some(tag),
                fill_width: Some(1280),
                quality: Some(90),
                ..Default::default()
            },
        );
        match fetch_image(&http, &url).await {
            Ok((rgba, w, h)) => {
                if epoch_ref.load(Ordering::SeqCst) == expected {
                    set_detail_backdrop(&weak, epoch_ref.clone(), expected, rgba, w, h);
                }
            }
            Err(e) => debug!(?e, %url, "backdrop fetch failed"),
        }
    });
}

fn spawn_cast_fetch(state: &AuthedState, index: usize, person_id: String, tag: String) {
    let weak = state.weak.clone();
    let http = state.image_http.clone();
    let sem = state.image_sem.clone();
    let epoch_ref = state.epoch.clone();
    let expected = epoch_ref.load(Ordering::SeqCst);
    let base = state.client.base_url().clone();
    tokio::spawn(async move {
        let _permit = match sem.acquire().await {
            Ok(p) => p,
            Err(_) => return,
        };
        if epoch_ref.load(Ordering::SeqCst) != expected {
            return;
        }
        let url = image_url(
            &base,
            &person_id,
            ApiImageType::Primary,
            &ImageOptions {
                tag: Some(tag),
                fill_height: Some(200),
                quality: Some(90),
                ..Default::default()
            },
        );
        match fetch_image(&http, &url).await {
            Ok((rgba, w, h)) => {
                if epoch_ref.load(Ordering::SeqCst) == expected {
                    push_cast_image(&weak, epoch_ref.clone(), expected, index, rgba, w, h);
                }
            }
            Err(e) => debug!(?e, %url, "cast image fetch failed"),
        }
    });
}

fn set_audio_devices(weak: &slint::Weak<MainWindow>, devices: Vec<AudioDevice>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            let slint_items: Vec<AudioDeviceItem> = devices
                .into_iter()
                .map(|d| AudioDeviceItem {
                    id: SharedString::from(d.id),
                    description: SharedString::from(d.description),
                    bitperfect: d.bitperfect,
                })
                .collect();
            w.set_audio_devices(ModelRc::new(VecModel::from(slint_items)));
        }
    });
}

fn set_selected_audio_device(weak: &slint::Weak<MainWindow>, id: &str, label: &str) {
    let weak = weak.clone();
    let id = SharedString::from(id);
    let label = SharedString::from(label);
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_selected_audio_device(id);
            w.set_selected_audio_device_label(label);
        }
    });
}

fn set_exclusive_mode(weak: &slint::Weak<MainWindow>, value: bool) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_exclusive_mode(value);
        }
    });
}

fn set_devices_loading(weak: &slint::Weak<MainWindow>, loading: bool) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_devices_loading(loading);
        }
    });
}

fn clear_album_image(weak: &slint::Weak<MainWindow>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_album_image(slint::Image::default());
            w.set_album_has_image(false);
            w.set_album_artist(SharedString::new());
            w.set_album_meta(SharedString::new());
            w.set_album_genres(ModelRc::new(VecModel::from(Vec::<SharedString>::new())));
            w.set_album_overview(SharedString::new());
            w.set_album_wikipedia_url(SharedString::new());
        }
    });
}

fn set_album_meta(
    weak: &slint::Weak<MainWindow>,
    artist: String,
    meta: String,
    genres: Vec<String>,
    overview: String,
    wikipedia_url: Option<String>,
) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_album_artist(SharedString::from(artist));
            w.set_album_meta(SharedString::from(meta));
            let chips: Vec<SharedString> = genres.into_iter().map(SharedString::from).collect();
            w.set_album_genres(ModelRc::new(VecModel::from(chips)));
            w.set_album_overview(SharedString::from(overview));
            w.set_album_wikipedia_url(SharedString::from(wikipedia_url.unwrap_or_default()));
        }
    });
}

/// Build the meta line ("2023 · 12 pistas · 45:32") and the genre list
/// from the album item + its already-fetched tracks. Genres come off the
/// album itself; track count + total duration are summed from `tracks`.
fn album_header_strings(album: &BaseItemDto, tracks: &[BaseItemDto]) -> (String, Vec<String>) {
    let mut parts: Vec<String> = Vec::new();
    if let Some(y) = album.production_year {
        parts.push(y.to_string());
    }
    let n = tracks.len();
    if n > 0 {
        parts.push(format!("{n} pista{}", if n == 1 { "" } else { "s" }));
    }
    let total_ticks: i64 = tracks
        .iter()
        .filter_map(|t| t.run_time_ticks)
        .filter(|t| *t > 0)
        .sum();
    if total_ticks > 0 {
        parts.push(format_ticks_long(total_ticks));
    }
    let meta = parts.join(" · ");
    let genres = album.genres.clone().unwrap_or_default();
    (meta, genres)
}

/// Scan `overview` for the first Wikipedia article URL. Jellyfin's
/// metadata providers often dump links into the overview text (esp.
/// TheAudioDB), so we don't need fancy parsing — just grab the URL
/// substring up to the first whitespace / closing bracket.
fn extract_wikipedia_url(overview: &str) -> Option<String> {
    for marker in ["https://", "http://"] {
        let mut search = overview;
        while let Some(start) = search.find(marker) {
            let candidate = &search[start..];
            let end = candidate
                .find(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '"' | ')' | ']'))
                .unwrap_or(candidate.len());
            let url = &candidate[..end];
            if url.contains("wikipedia.org/wiki/") {
                return Some(url.trim_end_matches(['.', ',', ';', ':']).to_string());
            }
            search = &candidate[marker.len()..];
        }
    }
    None
}

/// Hand off to the OS's default URL opener. Avoids pulling in the
/// `open` crate for a 6-line wrapper. macOS uses `open`, Linux uses
/// `xdg-open`; both background by default so we don't block.
fn open_external_url(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    let prog = "open";
    #[cfg(target_os = "linux")]
    let prog = "xdg-open";
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let prog = "";
    if prog.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "no URL opener on this platform",
        ));
    }
    std::process::Command::new(prog)
        .arg(url)
        .spawn()
        .map(|_| ())
}

/// Long form for album totals: H:MM:SS over an hour, M:SS otherwise.
/// Different from the per-track formatter which is always M:SS.
fn format_ticks_long(ticks: i64) -> String {
    // Jellyfin ticks: 10,000,000 per second.
    let total_secs = (ticks / 10_000_000).max(0) as u64;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

// ---------------------------------------------------------------- album download

/// Kick off (or replace) the download for the album currently on screen.
///
/// Called on the backend thread holding `current_items`. We snapshot
/// the tracks into a thread-safe struct, spawn a tokio task that runs
/// the actual fetch loop, and stash the cancel flag on `state` so a
/// subsequent click on a different album can cancel this one cleanly.
fn start_album_download(
    state: &AuthedState,
    top: Option<&NavEntry>,
    current_items: &[BaseItemDto],
) {
    let Some(top) = top else {
        warn!("download requested with empty nav");
        return;
    };
    if !matches!(top.children, ChildrenType::Tracks) {
        warn!("download requested outside an album view");
        return;
    }
    let album_id = top.parent_id.clone();
    let album_name = top.title.clone();
    let artist = top
        .album_artist
        .clone()
        .unwrap_or_else(|| "Unknown Artist".to_string());

    // Cancel any in-flight download from a previous album. New token
    // for this run; we hold one Arc, the task holds the other.
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut slot = state.download_cancel.lock().unwrap();
        if let Some(prev) = slot.take() {
            prev.store(true, Ordering::SeqCst);
        }
        *slot = Some(cancel.clone());
    }

    let tracks: Vec<DownloadTrack> = current_items
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let source = t.media_sources.as_ref().and_then(|m| m.first());
            DownloadTrack {
                id: t.id.clone(),
                order: i + 1,
                track_index: t.index_number,
                title: t.name.clone().unwrap_or_else(|| format!("Track {}", i + 1)),
                extension: pick_extension(
                    source.and_then(|s| s.path.as_deref()),
                    source.and_then(|s| s.container.as_deref()),
                ),
            }
        })
        .collect();
    if tracks.is_empty() {
        warn!(album = %album_id, "no tracks to download");
        return;
    }

    let Some(token) = state.client.token().map(|s| s.to_string()) else {
        warn!("no token; cannot download");
        return;
    };
    let base = state.client.base_url().clone();
    let weak = state.weak.clone();
    let current_album_id = state.current_album_id.clone();

    // Lock the button right away. The task will keep updating the
    // label as each track completes.
    push_download_status(
        &weak,
        &current_album_id,
        &album_id,
        format!("Descargando 0/{}…", tracks.len()),
        /* enabled = */ false,
        /* active = */ true,
    );

    tokio::spawn(async move {
        let result = run_album_download(
            base,
            token,
            artist.clone(),
            album_name.clone(),
            tracks.clone(),
            cancel.clone(),
            &weak,
            &current_album_id,
            &album_id,
        )
        .await;
        match result {
            Ok(()) => {
                push_download_status(
                    &weak,
                    &current_album_id,
                    &album_id,
                    "✓ Descargado".to_string(),
                    false,
                    false,
                );
            }
            Err(e) if cancel.load(Ordering::SeqCst) => {
                debug!(?e, "download cancelled");
            }
            Err(e) => {
                error!(?e, album = %album_id, "album download failed");
                push_download_status(
                    &weak,
                    &current_album_id,
                    &album_id,
                    "Error — reintentar".to_string(),
                    true,
                    false,
                );
            }
        }
    });
}

#[derive(Clone)]
struct DownloadTrack {
    id: String,
    /// 1-based position in the album's track listing.
    order: usize,
    track_index: Option<i32>,
    title: String,
    /// File extension already resolved — "flac", "m4a", "wav", "dsf",
    /// "mp3". Preferred from `MediaSourceInfo.path` (most accurate),
    /// falling back to `container` and then a sane default.
    extension: String,
}

#[allow(clippy::too_many_arguments)]
async fn run_album_download(
    base: Url,
    token: String,
    artist: String,
    album_name: String,
    tracks: Vec<DownloadTrack>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    weak: &slint::Weak<MainWindow>,
    current_album_id: &Arc<Mutex<Option<String>>>,
    album_id: &str,
) -> Result<()> {
    let dir = album_download_dir(&artist, &album_name)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("create dir {}", dir.display()))?;
    info!(dir = %dir.display(), tracks = tracks.len(), "starting album download");

    let total = tracks.len();
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("build http client")?;

    for (i, t) in tracks.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("cancelled"));
        }
        let filename = format!(
            "{:02} - {}.{}",
            t.track_index.map(|n| n as usize).unwrap_or(t.order),
            sanitize_path_component(&t.title),
            t.extension
        );
        let dest = dir.join(filename);
        let url = audio_stream_url(&base, &t.id, &token);

        debug!(track = %t.title, %url, dest = %dest.display(), "downloading");
        let resp = http.get(url).send().await?.error_for_status()?;
        let bytes = resp.bytes().await?;
        tokio::task::spawn_blocking({
            let dest = dest.clone();
            let bytes = bytes.clone();
            move || std::fs::write(&dest, &bytes)
        })
        .await
        .context("write task join")??;

        push_download_status(
            weak,
            current_album_id,
            album_id,
            format!("Descargando {}/{}…", i + 1, total),
            false,
            true,
        );
    }
    Ok(())
}

/// `~/Music/Jelly/<artist>/<album>/`. The `directories` crate already
/// hands us `audio_dir` cross-platform (`$XDG_MUSIC_DIR` or `~/Music`
/// on Linux, `~/Music` on macOS). Falls back to `$HOME/Music` if the
/// crate can't resolve user dirs.
fn album_download_dir(artist: &str, album: &str) -> Result<std::path::PathBuf> {
    let music = directories::UserDirs::new()
        .and_then(|d| d.audio_dir().map(|p| p.to_path_buf()))
        .or_else(dirs_fallback)
        .ok_or_else(|| anyhow::anyhow!("no user music dir"))?;
    Ok(music
        .join("Jelly")
        .join(sanitize_path_component(artist))
        .join(sanitize_path_component(album)))
}

fn dirs_fallback() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Music"))
}

/// Strip path separators, control chars, and other characters that
/// confuse Finder/Nautilus or break shells. Keeps the name readable;
/// not a security boundary. Quotes are *removed* rather than mapped
/// to `_` so that `"Heroes"` ends up as `Heroes` instead of `_Heroes_`.
fn sanitize_path_component(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .filter_map(|c| match c {
            '"' => None,
            '/' | '\\' | ':' | '*' | '?' | '<' | '>' | '|' => Some('_'),
            c if (c as u32) < 0x20 => Some('_'),
            c => Some(c),
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c: char| c == '.' || c.is_whitespace());
    if trimmed.is_empty() {
        "Untitled".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Resolve a sane file extension for a track download.
///
/// Preference order:
/// 1. The extension of the server-side path (`MediaSourceInfo.path`).
///    This is the most reliable — it's literally the file on disk.
/// 2. `MediaSourceInfo.container`, normalised (split on commas, take
///    the first entry, lowercase, map common Jellyfin spellings to
///    the conventional extension).
/// 3. `flac` as a last-resort default — every lossless source the
///    user listens to is one of FLAC/ALAC/WAV/DSD, and `.flac` is the
///    most likely to be correct *and* not destructive (Finder shows
///    it as audio either way, and we never overwrite a different
///    file silently because the title is in the filename).
fn pick_extension(path: Option<&str>, container: Option<&str>) -> String {
    if let Some(p) = path
        && let Some(ext) = std::path::Path::new(p).extension()
        && let Some(s) = ext.to_str()
    {
        let lower = s.to_ascii_lowercase();
        if !lower.is_empty() {
            return lower;
        }
    }
    if let Some(c) = container {
        let raw = c
            .split(',')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let normalised = match raw.as_str() {
            "" => None,
            // Jellyfin sometimes returns "mp4" for ALAC-in-m4a.
            "mp4" => Some("m4a"),
            "ogg" | "vorbis" => Some("ogg"),
            "wave" => Some("wav"),
            other => Some(other),
        };
        if let Some(n) = normalised {
            return n.to_string();
        }
    }
    "flac".to_string()
}

/// Only touch the UI when the user is still looking at this album.
/// If they navigated away, the task keeps writing files but stops
/// poking the window (the next album view will reset state anyway).
fn push_download_status(
    weak: &slint::Weak<MainWindow>,
    current_album_id: &Arc<Mutex<Option<String>>>,
    album_id: &str,
    label: String,
    enabled: bool,
    active: bool,
) {
    let same = current_album_id
        .lock()
        .unwrap()
        .as_deref()
        .is_some_and(|cur| cur == album_id);
    if !same {
        return;
    }
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_album_download_label(SharedString::from(label));
            w.set_album_download_enabled(enabled);
            w.set_album_download_active(active);
        }
    });
}

fn reset_album_download_ui(weak: &slint::Weak<MainWindow>) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_album_download_label(SharedString::from("⬇  Descargar álbum"));
            w.set_album_download_enabled(true);
            w.set_album_download_active(false);
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
