//! Slint frontend for the jelly client.
//!
//! One main window, two screens (login / library). The UI runs on the
//! main thread under Slint's event loop. A worker thread carries a
//! tokio runtime that handles all networking against jellyfin-api and
//! all persistence against jelly-storage. Commands flow UI → backend
//! via an mpsc channel; updates flow backend → UI via
//! `slint::invoke_from_event_loop`, the only thread-safe way to mutate
//! window state in Slint.

#![allow(clippy::needless_return)]

use anyhow::{Context, Result};
use jelly_storage::Storage;
use jellyfin_api::{
    BaseItemDto, Client, Identity, ImageOptions, ImageType as ApiImageType, ItemType, ItemsQuery,
    SortOrder, image_url,
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
const IMAGE_FILL_HEIGHT: u32 = 360; // 2x the card height for retina

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

    window.run()?;
    Ok(())
}

// ---------------------------------------------------------------- backend

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
}

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
            BackendCmd::SignOut | BackendCmd::SelectView { .. } => {}
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
        }
    }
}

async fn run_authed(
    state: AuthedState,
    storage: Arc<Storage>,
    initial_view: Option<jellyfin_api::BaseItemDto>,
    cmd_rx: &mut UnboundedReceiver<BackendCmd>,
) {
    if let Some(v) = initial_view {
        select_view(&state, &v.id, v.collection_type.as_deref().unwrap_or(""));
    }

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCmd::SelectView {
                view_id,
                collection_type,
            } => {
                select_view(&state, &view_id, &collection_type);
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

/// Increment the epoch, fetch items, then fire-and-forget the image
/// downloads. Returns immediately; each image push checks the current
/// epoch and skips itself if the user has navigated away.
fn select_view(state: &AuthedState, view_id: &str, collection_type: &str) {
    let epoch = state.epoch.fetch_add(1, Ordering::SeqCst) + 1;
    debug!(view = view_id, ct = collection_type, epoch, "select view");

    let weak = state.weak.clone();
    let client = state.client.clone();
    let user_id = state.user_id.clone();
    let view_id = view_id.to_string();
    let collection_type = collection_type.to_string();
    let epoch_for_task = state.epoch.clone();
    let image_http = state.image_http.clone();
    let image_sem = state.image_sem.clone();

    set_loading(&weak, true);
    set_view_title(&weak, view_title(&collection_type, ""), "Loading…".into());
    set_items(&weak, vec![]);

    tokio::spawn(async move {
        let query = items_query_for(&view_id, &collection_type);
        let resp = match client.items(&user_id, &query).await {
            Ok(r) => r,
            Err(e) => {
                error!(?e, "items fetch failed");
                if epoch_for_task.load(Ordering::SeqCst) == epoch {
                    set_loading(&weak, false);
                    set_view_title(&weak, "Error".into(), format!("{e:#}"));
                }
                return;
            }
        };

        if epoch_for_task.load(Ordering::SeqCst) != epoch {
            return;
        }

        let total = resp.total_record_count.unwrap_or(resp.items.len() as i64);
        let count_label = if total as usize > resp.items.len() {
            format!("{} of {} items", resp.items.len(), total)
        } else {
            format!("{} items", resp.items.len())
        };

        let items_data = resp
            .items
            .iter()
            .map(|i| build_initial_item_data(i, &collection_type))
            .collect::<Vec<_>>();

        set_view_title(&weak, view_title(&collection_type, ""), count_label);
        set_items(&weak, items_data);
        set_loading(&weak, false);

        // Snapshot of (index, item_id, image_tag) for items that have
        // a Primary image. The image URL is built locally; tag goes in
        // the query so server caches don't serve stale art.
        let base = client.base_url().clone();
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
            let weak = weak.clone();
            let http = image_http.clone();
            let sem = image_sem.clone();
            let epoch_ref = epoch_for_task.clone();
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
    });
}

/// Decode an image off the UI thread. Returns raw RGBA bytes plus
/// dimensions so the caller can ship them across the Slint event-loop
/// boundary (`slint::Image` is not `Send`, so we build it on the UI
/// thread instead).
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

fn items_query_for(view_id: &str, collection_type: &str) -> ItemsQuery {
    let (types, sort) = match collection_type {
        "music" => (vec![ItemType::MusicAlbum], vec!["SortName".into()]),
        "movies" => (
            vec![ItemType::Movie],
            vec!["SortName".into(), "ProductionYear".into()],
        ),
        "tvshows" => (vec![ItemType::Series], vec!["SortName".into()]),
        "playlists" => (vec![ItemType::Playlist], vec!["SortName".into()]),
        _ => (vec![], vec!["SortName".into()]),
    };
    ItemsQuery {
        parent_id: Some(view_id.to_string()),
        include_item_types: types,
        sort_by: sort,
        sort_order: Some(SortOrder::Ascending),
        limit: Some(ITEM_LIMIT),
        recursive: Some(true),
        fields: vec!["PrimaryImageAspectRatio".into(), "Genres".into()],
        ..Default::default()
    }
}

/// Plain-Rust mirror of the Slint `Item` struct used for crossing the
/// event-loop boundary. The Slint type contains a non-`Send`
/// `slint::Image`, so we build it inside the UI-thread closure.
#[derive(Clone)]
struct ItemData {
    id: String,
    title: String,
    subtitle: String,
}

fn build_initial_item_data(item: &BaseItemDto, collection_type: &str) -> ItemData {
    let title = item.name.clone().unwrap_or_default();
    let subtitle = match collection_type {
        "music" => item
            .album_artist
            .clone()
            .or_else(|| item.artists.as_ref().and_then(|a| a.first().cloned()))
            .or_else(|| item.production_year.map(|y| y.to_string()))
            .unwrap_or_default(),
        "movies" | "tvshows" => item
            .production_year
            .map(|y| y.to_string())
            .unwrap_or_default(),
        _ => String::new(),
    };
    ItemData {
        id: item.id.clone(),
        title,
        subtitle,
    }
}

fn view_title(collection_type: &str, fallback: &str) -> String {
    match collection_type {
        "music" => "Music".into(),
        "movies" => "Movies".into(),
        "tvshows" => "TV Shows".into(),
        "playlists" => "Playlists".into(),
        _ => {
            if fallback.is_empty() {
                "Library".into()
            } else {
                fallback.into()
            }
        }
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

fn set_view_title(weak: &slint::Weak<MainWindow>, title: String, subtitle: String) {
    let weak = weak.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_view_title(SharedString::from(title));
            w.set_view_subtitle(SharedString::from(subtitle));
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
                })
                .collect();
            w.set_items(ModelRc::new(VecModel::from(slint_items)));
        }
    });
}

/// Mutate a single item's image in place. We build the `slint::Image`
/// inside the UI-thread closure because Slint's image types aren't
/// `Send`; only the raw bytes cross the thread boundary. Epoch is
/// re-checked on the UI thread so two `set_items` calls in quick
/// succession can't race with image arrivals from a previous view.
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

fn push_signed_in(
    weak: &slint::Weak<MainWindow>,
    user_name: &str,
    views: Vec<jellyfin_api::BaseItemDto>,
) {
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
            w.set_screen(Screen::Login);
        }
    });
}
