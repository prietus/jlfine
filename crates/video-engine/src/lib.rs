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
use std::thread;
use tracing::{debug, error, info, warn};

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

/// Fire-and-forget playback. Returns immediately after spawning the
/// worker thread; the mpv window stays open until the user closes it
/// or EOF is reached. `audio_device` is the mpv-style id from
/// [`list_audio_devices`]; `None` lets mpv pick its default.
pub fn play(url: impl Into<String>, audio_device: Option<String>) {
    let url = url.into();
    thread::spawn(move || run(url, audio_device));
}

fn run(url: String, audio_device: Option<String>) {
    info!(%url, ?audio_device, "opening mpv window");

    let mpv = match Mpv::with_initializer(|init| {
        // terminal=yes ships mpv's own logs through our stderr. msg-level
        // stays at warn by default so we don't drown the host's logs;
        // bump via MPV_VERBOSE=1 when debugging.
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
        init.set_property("force-window", "yes")?;
        init.set_property("keep-open", "yes")?;
        if let Some(dev) = audio_device.as_deref() {
            init.set_property("audio-device", dev)?;
        }
        Ok(())
    }) {
        Ok(m) => m,
        Err(e) => {
            error!(?e, "mpv init failed");
            return;
        }
    };

    if let Err(e) = mpv.command("loadfile", &[&url]) {
        error!(?e, "loadfile failed");
        return;
    }

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

    info!("mpv window closed");
}
