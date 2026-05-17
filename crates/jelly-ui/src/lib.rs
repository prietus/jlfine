//! Slint frontend for the jelly client.
//!
//! One main window, two screens (login / library). The UI runs on the
//! main thread under Slint's event loop. A worker thread carries a
//! tokio runtime that handles all networking against jellyfin-api and
//! all persistence against jelly-storage. Commands flow UI → backend
//! via an mpsc channel; updates flow backend → UI via
//! `slint::invoke_from_event_loop`, which is the only thread-safe way
//! to mutate window state in Slint.

#![allow(clippy::needless_return)]

use anyhow::{Context, Result};
use jelly_storage::Storage;
use jellyfin_api::{Client, Identity};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tracing::{error, info, warn};
use url::Url;

slint::include_modules!();

/// Entry point. Boots the UI, the tokio runtime, and the storage layer
/// in the right order, then runs the Slint event loop until the window
/// is closed.
pub fn run() -> Result<()> {
    let storage = Arc::new(Storage::new().context("init storage")?);
    let device_id = storage.device_id().context("get device id")?;

    let window = MainWindow::new()?;
    let weak = window.as_weak();

    let (cmd_tx, cmd_rx) = unbounded_channel::<BackendCmd>();

    // Backend thread owns the tokio runtime and the long-lived Client.
    // Any state the UI needs lives in Slint-managed properties; the
    // backend only ever talks to the window through invoke_from_event_loop.
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

    // UI → backend callbacks.
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
    // view-selected is purely visual for now; no backend work needed.

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
}

async fn backend_loop(
    weak: slint::Weak<MainWindow>,
    storage: Arc<Storage>,
    device_id: String,
    mut cmd_rx: tokio::sync::mpsc::UnboundedReceiver<BackendCmd>,
) {
    // Try to restore a saved session up front. If the token is still
    // valid we jump straight into the library screen; otherwise the
    // user sees the login screen with the previous server URL prefilled.
    if let Ok(Some(saved)) = storage.load_session() {
        info!(server = %saved.server_url, user = %saved.user_id, "restoring session");
        let identity = make_identity(&device_id);
        let client = Client::new(saved.server_url.clone(), identity)
            .with_accept_language(preferred_language())
            .with_token(saved.token.clone());
        match client.current_user().await {
            Ok(user) => {
                if let Ok(views) = client.user_views(&user.id).await {
                    push_signed_in(&weak, &user.name, views.items);
                } else {
                    push_signed_in(&weak, &user.name, vec![]);
                }
                // store the client so subsequent commands can use it
                run_authed(weak.clone(), storage.clone(), client, &mut cmd_rx).await;
                return;
            }
            Err(e) => {
                warn!(?e, "stored token rejected, falling back to login");
                let _ = storage.clear_session();
                prefill_server_url(&weak, saved.server_url.as_str());
            }
        }
    }

    // Main login loop. Stays here until a successful sign-in upgrades
    // us into the authenticated state.
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCmd::SignIn { server, user, pw } => {
                set_busy(&weak, true);
                set_error(&weak, "");
                let result = attempt_sign_in(&server, &user, &pw, &device_id, &storage).await;
                set_busy(&weak, false);
                match result {
                    Ok((client, user_name, views)) => {
                        push_signed_in(&weak, &user_name, views);
                        run_authed(weak.clone(), storage.clone(), client, &mut cmd_rx).await;
                        return;
                    }
                    Err(e) => {
                        set_error(&weak, &format!("{e:#}"));
                    }
                }
            }
            BackendCmd::SignOut => {
                // already signed out — ignore
            }
        }
    }
}

async fn attempt_sign_in(
    server: &str,
    user: &str,
    pw: &str,
    device_id: &str,
    storage: &Storage,
) -> Result<(Client, String, Vec<jellyfin_api::BaseItemDto>)> {
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
    Ok((client, user_dto.name, views.items))
}

/// After sign-in succeeds, stay in this loop handling logout (and
/// future authenticated commands) without dropping the Client.
async fn run_authed(
    weak: slint::Weak<MainWindow>,
    storage: Arc<Storage>,
    _client: Client,
    cmd_rx: &mut tokio::sync::mpsc::UnboundedReceiver<BackendCmd>,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BackendCmd::SignOut => {
                info!("signing out");
                if let Err(e) = storage.clear_session() {
                    error!(?e, "clear_session failed");
                }
                push_signed_out(&weak);
                return;
            }
            BackendCmd::SignIn { .. } => {
                warn!("sign-in attempted while already authed; ignoring");
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
// Every UI mutation must hop through invoke_from_event_loop so it
// happens on the Slint main thread, regardless of which tokio worker
// fired the result.

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
            w.set_screen(Screen::Login);
        }
    });
}

// Keep the unused import quiet on platforms where it isn't pulled in.
#[allow(dead_code)]
fn _silence_unused_warning(_tx: UnboundedSender<BackendCmd>) {}
