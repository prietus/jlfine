use anyhow::{Context, Result};
use libmpv2::{Mpv, events::Event};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: poc-libmpv <video-file>")?;

    #[cfg(target_os = "macos")]
    mac::init_nsapp();

    let mpv = Mpv::with_initializer(|init| {
        init.set_property("terminal", "yes")?;
        init.set_property("msg-level", "cplayer=info,vo=v,vd=v,placebo=v")?;
        init.set_property("vo", "gpu-next")?;
        init.set_property("hwdec", "auto-safe")?;
        init.set_property("target-colorspace-hint", "yes")?;
        init.set_property("force-window", "yes")?;
        init.set_property("keep-open", "yes")?;
        Ok(())
    })
    .map_err(|e| anyhow::anyhow!("mpv init failed: {e:?}"))?;

    mpv.command("loadfile", &[&path])
        .map_err(|e| anyhow::anyhow!("loadfile failed: {e:?}"))?;

    let mpv = std::sync::Arc::new(mpv);

    #[cfg(target_os = "macos")]
    {
        let worker_mpv = mpv.clone();
        std::thread::spawn(move || event_loop(&worker_mpv));
        mac::run_nsapp(); // diverges
    }

    #[cfg(not(target_os = "macos"))]
    {
        event_loop(&mpv);
        Ok(())
    }
}

fn event_loop(mpv: &Mpv) {
    loop {
        match mpv.wait_event(600.0) {
            Some(Ok(Event::EndFile(reason))) => {
                eprintln!("[end] reason: {reason:?}");
                break;
            }
            Some(Ok(Event::Shutdown)) => {
                eprintln!("[shutdown]");
                break;
            }
            Some(Ok(Event::FileLoaded)) => eprintln!("[file-loaded]"),
            Some(Ok(Event::VideoReconfig)) => {
                eprintln!("[video-reconfig]");
                dump_video_params(mpv);
            }
            Some(Ok(Event::PlaybackRestart)) => {
                eprintln!("[playback-restart]");
                dump_video_params(mpv);
            }
            Some(Ok(other)) => eprintln!("[event] {other:?}"),
            Some(Err(e)) => eprintln!("[event-error] {e:?}"),
            None => {}
        }
    }
}

fn dump_video_params(mpv: &Mpv) {
    let keys = [
        "video-params/pixelformat",
        "video-params/colormatrix",
        "video-params/primaries",
        "video-params/gamma",
        "video-params/sig-peak",
        "video-params/colorlevels",
        "video-codec",
        "container-fps",
        "width",
        "height",
        "video-dec-params/dolby-vision-profile",
        "video-dec-params/dolby-vision-level",
        "hwdec-current",
    ];
    eprintln!("--- video params ---");
    for k in keys {
        match mpv.get_property::<String>(k) {
            Ok(v) => eprintln!("  {k:40} = {v}"),
            Err(e) => eprintln!("  {k:40} = <err: {e:?}>"),
        }
    }
    eprintln!("--------------------");
}

#[cfg(target_os = "macos")]
mod mac {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    pub fn init_nsapp() {
        let mtm = MainThreadMarker::new().expect("must call from main thread");
        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
        app.activate();
    }

    pub fn run_nsapp() -> ! {
        let mtm = MainThreadMarker::new().expect("must call from main thread");
        let app = NSApplication::sharedApplication(mtm);
        app.run();
        std::process::exit(0);
    }
}
