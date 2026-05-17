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
use std::thread;
use tracing::{debug, error, info, warn};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mpv: {0}")]
    Mpv(String),
}

/// Fire-and-forget playback. Returns immediately after spawning the
/// worker thread; the mpv window stays open until the user closes it
/// or EOF is reached.
pub fn play(url: impl Into<String>) {
    let url = url.into();
    thread::spawn(move || run(url));
}

fn run(url: String) {
    info!(%url, "opening mpv window");

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
