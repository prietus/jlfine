//! Video playback engine backed by libmpv.
//!
//! Each `play` call spawns an mpv instance that owns its own native
//! window (cocoa-cb on macOS, X11/Wayland on Linux) and renders via
//! `gpu-next` for HDR10 / Dolby Vision support through libplacebo +
//! libdovi. The host application stays free of any rendering surface
//! concerns — see project memory `video-hdr-dv-requirements` for the
//! design rationale.
//!
//! NSApplication on macOS is not initialised here: the host (Slint
//! via winit) already provides one. The libmpv `play_video_poc` PoC
//! had to bootstrap NSApp itself because it ran from a bare CLI; the
//! desktop app does not need that wrapper.

use libmpv2::{Mpv, events::Event};
use serde::Deserialize;
#[cfg(target_os = "macos")]
use std::sync::Arc;
use std::thread;
use tracing::{debug, error, info, warn};

#[cfg(target_os = "macos")]
mod mac;
#[cfg(target_os = "macos")]
mod mac_gl;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mpv: {0}")]
    Mpv(String),
}

/// One entry from mpv's `audio-device-list`. The `id` is the opaque
/// string mpv expects back in its `audio-device` property; the
/// `description` is the human-readable name. `bitperfect` flags
/// outputs that bypass system mixers (CoreAudio with a specific UID
/// on macOS, ALSA `hw:` direct on Linux); everything else routes
/// through PulseAudio/Pipewire/JACK/dmix and may resample.
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub id: String,
    pub description: String,
    pub bitperfect: bool,
}

/// Enumerate audio outputs visible to mpv. Spins up a short-lived
/// mpv instance, queries `audio-device-list` (returned as JSON when
/// read as a string), and drops the instance immediately. Cheap
/// enough to call lazily from the UI when the user opens the
/// settings screen.
pub fn list_audio_devices() -> Result<Vec<AudioDevice>, Error> {
    let mpv = Mpv::new().map_err(|e| Error::Mpv(format!("init: {e}")))?;
    let json: String = mpv
        .get_property("audio-device-list")
        .map_err(|e| Error::Mpv(format!("get audio-device-list: {e}")))?;

    #[derive(Deserialize)]
    struct Raw {
        name: String,
        description: String,
    }
    let raw: Vec<Raw> = serde_json::from_str(&json)
        .map_err(|e| Error::Mpv(format!("parse audio-device-list: {e}")))?;

    let mut out: Vec<AudioDevice> = raw
        .into_iter()
        .map(|d| AudioDevice {
            bitperfect: is_bitperfect(&d.name),
            id: d.name,
            description: d.description,
        })
        .collect();
    // Bitperfect-capable devices first so first-launch defaults land
    // on something useful; preserves mpv's order within each group.
    out.sort_by_key(|d| !d.bitperfect);
    Ok(out)
}

/// A device id is bitperfect when it identifies a specific hardware
/// interface: `coreaudio/<UID>` (macOS, a concrete device, not the
/// generic `coreaudio/` follow-system-default) or `alsa/hw:CARD,DEV`
/// (Linux raw hardware). Everything else — Pulse, Pipewire, JACK,
/// `alsa/default`, `alsa/plughw:*` — passes through a mixer.
fn is_bitperfect(id: &str) -> bool {
    if let Some(uid) = id.strip_prefix("coreaudio/") {
        !uid.is_empty()
    } else {
        id.starts_with("alsa/hw:")
    }
}

/// Fire-and-forget playback. Returns immediately after dispatching
/// the work; the mpv window stays open until EOF or until the caller
/// process exits. `audio_device` is the mpv-style id from
/// [`list_audio_devices`]; `None` lets mpv pick its default.
///
/// Callable from any thread. On macOS the host (this crate) owns the
/// `NSWindow` and mpv attaches to its contentView via `wid` so the
/// window stays inside the host's responder chain — the NSWindow
/// creation is hopped over to the AppKit main thread internally.
pub fn play(url: impl Into<String>, audio_device: Option<String>) {
    let url = url.into();

    #[cfg(target_os = "macos")]
    {
        // jelly-ui's backend loop runs inside a tokio multi_thread
        // runtime, so this call lands on a worker thread. The
        // NSWindow needs the main thread (AppKit); mpv itself
        // expects to live off the main thread (its mac VO dispatches
        // back to main internally via GCD). So: window on main,
        // mpv on a worker, key monitor hopped back to main once
        // mpv exists.
        mac::run_on_main(move || {
            let window = mac::VideoWindow::new("jelly · video");
            let view_ptr = window.view_ptr_for_mpv();
            let slot_id = mac::register_window(window);

            thread::spawn(move || {
                let mpv = match build_mpv(audio_device, Some(view_ptr)) {
                    Ok(m) => Arc::new(m),
                    Err(e) => {
                        error!(?e, "mpv init failed");
                        mac::run_on_main(move || mac::unregister_window(slot_id));
                        return;
                    }
                };
                // Hop back to main to install the NSEvent monitor —
                // mpv handle is Send+Sync so cloning the Arc across
                // threads is fine.
                let mpv_for_keys = mpv.clone();
                mac::run_on_main(move || {
                    mac::install_key_handler(slot_id, mpv_for_keys);
                });

                if let Err(e) = mpv.command("loadfile", &[&url]) {
                    error!(?e, "loadfile failed");
                }
                pump_mpv_events(&mpv);
                drop(mpv);
                mac::run_on_main(move || mac::unregister_window(slot_id));
            });
        });
        return;
    }

    #[cfg(not(target_os = "macos"))]
    {
        thread::spawn(move || {
            let mpv = match build_mpv(audio_device, None) {
                Ok(m) => m,
                Err(e) => {
                    error!(?e, "mpv init failed");
                    return;
                }
            };
            install_default_keybindings(&mpv);
            if let Err(e) = mpv.command("loadfile", &[&url]) {
                error!(?e, "loadfile failed");
                return;
            }
            pump_mpv_events(&mpv);
        });
    }
}

/// Register the standard mpv keys at runtime. Belt and braces around
/// the `input-default-bindings` / `input-builtin-bindings` toggle —
/// on some libmpv builds neither flag actually loads the bundled
/// input.conf, so the user sees `[input] No key binding found for
/// key 'f'.` and nothing happens. Calling `keybind` explicitly
/// guarantees the bindings exist whatever the build's behaviour.
fn install_default_keybindings(mpv: &Mpv) {
    // (key, command) pairs. Same mapping as the macOS NSEvent
    // monitor — pause/seek/mute/fullscreen/stats/audio/sub cycle/
    // quit. Mouse: left-click pauses, double-click toggles
    // fullscreen, scroll seeks (which mpv handles via WHEEL_UP/DOWN).
    let bindings: &[(&str, &str)] = &[
        ("SPACE", "cycle pause"),
        ("LEFT", "seek -5"),
        ("RIGHT", "seek 5"),
        ("UP", "seek 60"),
        ("DOWN", "seek -60"),
        ("f", "cycle fullscreen"),
        ("m", "cycle mute"),
        ("a", "cycle audio"),
        ("s", "cycle sub"),
        ("i", "script-binding stats/display-stats"),
        ("I", "script-binding stats/display-stats-toggle"),
        ("q", "quit"),
        ("ESC", "quit"),
        ("MBTN_LEFT", "cycle pause"),
        ("MBTN_LEFT_DBL", "cycle fullscreen"),
        ("WHEEL_UP", "seek 5"),
        ("WHEEL_DOWN", "seek -5"),
    ];
    for (key, cmd) in bindings {
        if let Err(e) = mpv.command("keybind", &[key, cmd]) {
            warn!(?e, key, cmd, "keybind failed");
        }
    }
}

/// Construct an `Mpv` handle with the project's standard settings —
/// HDR-aware `gpu-next`, hardware decoding, on-screen controller,
/// stats overlay. On macOS the caller passes the host NSView's
/// pointer via `mac_view_ptr` so mpv embeds rendering there instead
/// of opening its own window.
fn build_mpv(
    audio_device: Option<String>,
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] mac_view_ptr: Option<isize>,
) -> Result<Mpv, libmpv2::Error> {
    Mpv::with_initializer(|init| {
        let verbose = std::env::var("MPV_VERBOSE").is_ok();
        init.set_property("terminal", "yes")?;
        init.set_property(
            "msg-level",
            if verbose {
                "cplayer=info,vo=v,vd=v,placebo=v"
            } else {
                "all=warn"
            },
        )?;
        init.set_property("vo", "gpu-next")?;
        init.set_property("hwdec", "auto-safe")?;
        init.set_property("target-colorspace-hint", "yes")?;
        init.set_property("keep-open", "yes")?;
        // `force-window=yes` makes mpv create its OWN window even
        // when we hand it one via `wid` — a phantom NSWindow sits
        // alongside ours. On macOS with wid set, we don't need it:
        // the host window already exists. On Linux/Wayland mpv owns
        // the window so we keep force-window on as a safety net.
        if mac_view_ptr.is_none() {
            init.set_property("force-window", "yes")?;
        }
        // OSC + OSD on so the user gets a control bar on mouse move
        // and overlay messages on seek/volume. mpv's own key bindings
        // stay enabled too: on Linux they reach mpv directly through
        // its window, and even on macOS the OSC's mouse-driven
        // controls still use them internally.
        //
        // mpv 0.36 renamed `input-default-bindings` to
        // `input-builtin-bindings`; old mpv only knows the old name,
        // new mpv only the new one. Try both and ignore the error
        // from whichever isn't recognized so f/space/arrows/etc.
        // actually do something.
        let _ = init.set_property("input-builtin-bindings", "yes");
        let _ = init.set_property("input-default-bindings", "yes");
        init.set_property("input-vo-keyboard", "yes")?;
        init.set_property("input-media-keys", "yes")?;
        init.set_property("osc", "yes")?;
        init.set_property("osd-bar", "yes")?;
        init.set_property("load-stats-overlay", "yes")?;
        init.set_property("cursor-autohide", "1000")?;
        // Linux/Wayland: ask mpv for a bordered window. mpv defaults
        // to yes but compositors that don't speak server-side
        // decorations (GNOME, sway) need mpv to draw its own via
        // libdecor — that's only available if mpv was built with
        // it. If your titlebar is still missing after this, the mpv
        // build lacks libdecor support.
        init.set_property("border", "yes")?;

        #[cfg(target_os = "macos")]
        if let Some(ptr) = mac_view_ptr {
            init.set_property("wid", ptr.to_string())?;
        }

        if let Some(dev) = audio_device.as_deref() {
            init.set_property("audio-device", dev)?;
        }
        Ok(())
    })
}

/// Drain mpv events until shutdown or end-file. Runs on a worker
/// thread; nothing touched here is main-thread-only.
fn pump_mpv_events(mpv: &Mpv) {
    info!("mpv event loop started");
    loop {
        match mpv.wait_event(60.0) {
            Some(Ok(Event::EndFile(reason))) => {
                debug!(?reason, "end-file");
                break;
            }
            Some(Ok(Event::Shutdown)) => {
                debug!("shutdown");
                break;
            }
            Some(Ok(e)) => debug!(?e, "mpv event"),
            Some(Err(e)) => warn!(?e, "mpv event error"),
            None => {}
        }
    }
    info!("mpv event loop exited");
}
