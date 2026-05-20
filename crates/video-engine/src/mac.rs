//! macOS host window for libmpv playback.
//!
//! Owns the `NSWindow` and content `NSView` that mpv renders into via
//! the `--wid` option. The previous design let mpv create its own
//! window, but when libmpv is embedded under another app's
//! `NSApplication` (Slint/winit), that window never enters the
//! responder chain and the keyboard never reaches mpv. By owning the
//! window we make it a normal first-class window of the host app —
//! it becomes key on click, accepts events, and we forward those
//! events into mpv via `mpv_command`.
//!
//! This module only knows about AppKit. The actual mpv lifecycle and
//! key→command mapping live in `lib.rs`.
//!
//! All public APIs must be invoked from the main thread.

use block2::RcBlock;
use libmpv2::Mpv;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_app_kit::{
    NSBackingStoreType, NSEvent, NSEventMask, NSEventModifierFlags, NSView, NSWindow,
    NSWindowStyleMask, NSWindowWillCloseNotification,
};
use objc2_foundation::{
    MainThreadMarker, NSNotification, NSNotificationCenter, NSPoint, NSRect, NSSize, NSString,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

/// Owns the NSWindow + content NSView. Dropping it removes the
/// NSEvent monitor + close observer (if installed) and closes the
/// window.
pub struct VideoWindow {
    window: Retained<NSWindow>,
    view: Retained<NSView>,
    /// Opaque token returned by `addLocalMonitorForEventsMatchingMask`.
    /// Held so we can pass it back to `removeMonitor` on drop.
    monitor: Option<Retained<AnyObject>>,
    /// Token returned by `NSNotificationCenter::addObserverForName:`
    /// for the window close notification. Held so we can hand it back
    /// to `removeObserver:` on drop.
    close_observer: Option<Retained<objc2_foundation::NSObject>>,
}

impl VideoWindow {
    /// Create a titled, resizable NSWindow with a layer-backed
    /// content view ready to host mpv's renderer. Must be called on
    /// the main thread.
    pub fn new(title: &str) -> Self {
        // SAFETY: callers must invoke this from the AppKit main
        // thread. video-engine's `play` runs inside the Slint event
        // loop which is pinned to the main thread.
        let mtm = unsafe { MainThreadMarker::new_unchecked() };

        // Default frame: 1280x720. mpv will resize the view to the
        // video's aspect ratio after the first frame, and the user
        // can drag the window edges.
        let frame = NSRect::new(NSPoint::new(120.0, 120.0), NSSize::new(1280.0, 720.0));
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Resizable
            | NSWindowStyleMask::Miniaturizable;

        let window: Retained<NSWindow> = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc::<NSWindow>(),
                frame,
                style,
                NSBackingStoreType::NSBackingStoreBuffered,
                false,
            )
        };

        window.setTitle(&NSString::from_str(title));
        // Don't auto-release: we hold the Retained<NSWindow> and
        // need close() to be idempotent under Drop semantics.
        unsafe { window.setReleasedWhenClosed(false) };

        // Layer-backed content view so mpv can attach a CAMetalLayer
        // (gpu-next uses Metal under the hood on macOS). Without
        // this the view falls back to software compositing and HDR
        // breaks.
        let view: Retained<NSView> = window
            .contentView()
            .expect("freshly-created NSWindow has a contentView");
        view.setWantsLayer(true);
        window.makeKeyAndOrderFront(None);

        Self {
            window,
            view,
            monitor: None,
            close_observer: None,
        }
    }

    /// Pointer to pass to mpv as the `wid` option. mpv interprets
    /// this as an `intptr_t` cast of an `NSView*` and attaches its
    /// renderer to that view.
    pub fn view_ptr_for_mpv(&self) -> isize {
        Retained::as_ptr(&self.view) as isize
    }

    /// Install an `NSEvent` local monitor for `keyDown` and forward
    /// the matching keys to mpv as commands — same mapping as a
    /// vanilla mpv CLI session (space=pause, arrows=seek, f=fs,
    /// i=stats, m=mute, q=quit, a/s cycle audio/subs). Other key
    /// events pass through untouched so Slint's library window keeps
    /// its shortcuts.
    ///
    /// The block holds a `Weak<Mpv>` rather than a strong `Arc<Mpv>`
    /// so the worker thread's strong reference is guaranteed to be
    /// the last one. That way `Mpv::drop` (which calls the
    /// main-thread-blocking `mpv_terminate_destroy`) runs on the
    /// worker thread when it exits, not later on the main thread
    /// when the monitor is removed — that ordering would deadlock.
    pub fn install_key_handler(&mut self, mpv: Arc<Mpv>) {
        let mpv_weak: Weak<Mpv> = Arc::downgrade(&mpv);
        // Filter monitor events to our window by NSInteger window
        // number; `NSEvent::window()` would also work but requires
        // re-fetching a `Retained<NSWindow>` per event.
        let our_window_num = unsafe { self.window.windowNumber() };
        // Clone the Retained — same Objective-C object, second
        // strong ref so the block can outlive a hypothetical
        // assignment to self.window. The block runs on the main
        // thread, so holding a !Send Retained inside is fine.
        let window_for_fs = self.window.clone();

        let block: RcBlock<dyn Fn(NonNull<NSEvent>) -> *mut NSEvent> =
            RcBlock::new(move |event_ptr: NonNull<NSEvent>| -> *mut NSEvent {
                // SAFETY: AppKit hands us a live NSEvent inside the
                // dispatch of `keyDown`; we don't keep it past the
                // closure body.
                let event = unsafe { event_ptr.as_ref() };
                let win_num = unsafe { event.windowNumber() };
                if win_num != our_window_num {
                    return event_ptr.as_ptr();
                }
                let key = unsafe { event.keyCode() };
                let mods = unsafe { event.modifierFlags() };
                let shift = mods.contains(NSEventModifierFlags::NSEventModifierFlagShift);

                // F: native macOS fullscreen instead of mpv's cycle
                // fullscreen. Animation, dedicated Space, Mission
                // Control integration — all stock AppKit behavior.
                if key == 3 {
                    window_for_fs.toggleFullScreen(None);
                    return std::ptr::null_mut();
                }

                let Some(mpv) = mpv_weak.upgrade() else {
                    // mpv already gone (worker exited); just consume.
                    return std::ptr::null_mut();
                };
                if let Some((cmd, args)) = translate_key(key, shift)
                    && mpv
                        .command(cmd, &args)
                        .map_err(|e| warn!(?e, cmd, "mpv command failed"))
                        .is_ok()
                {
                    // Consume so AppKit doesn't beep at us.
                    return std::ptr::null_mut();
                }
                event_ptr.as_ptr()
            });

        let token = unsafe {
            NSEvent::addLocalMonitorForEventsMatchingMask_handler(
                NSEventMask::KeyDown,
                &block,
            )
        };
        self.monitor = token;

        // Wire the close button: when AppKit posts
        // NSWindowWillCloseNotification for our window, tell mpv to
        // quit. The worker thread will see the Shutdown event, exit,
        // and the run_on_main cleanup will drop this VideoWindow.
        // close() at that point is a no-op since the window is
        // already closing.
        let mpv_weak_for_close: Weak<Mpv> = Arc::downgrade(&mpv);
        let close_block: RcBlock<dyn Fn(NonNull<NSNotification>)> =
            RcBlock::new(move |_note: NonNull<NSNotification>| {
                if let Some(mpv) = mpv_weak_for_close.upgrade() {
                    let _ = mpv
                        .command("quit", &[])
                        .map_err(|e| warn!(?e, "mpv quit on window close failed"));
                }
            });
        let center = unsafe { NSNotificationCenter::defaultCenter() };
        let observer = unsafe {
            center.addObserverForName_object_queue_usingBlock(
                Some(NSWindowWillCloseNotification),
                Some(self.window.as_ref()),
                None,
                &close_block,
            )
        };
        self.close_observer = Some(observer);
    }
}

/// Map a macOS virtual key code (kVK_* — physical key position) to
/// the mpv command + args we want to fire. Same physical mapping as
/// `~/jelly`'s PlayerKeyHandler.swift.
fn translate_key(key_code: u16, shift: bool) -> Option<(&'static str, Vec<&'static str>)> {
    match key_code {
        49 => Some(("cycle", vec!["pause"])),        // space
        123 => Some(("seek", vec!["-5"])),            // left
        124 => Some(("seek", vec!["5"])),             // right
        125 => Some(("seek", vec!["-60"])),           // down
        126 => Some(("seek", vec!["60"])),            // up
        53 => Some(("quit", vec![])),                 // escape
        // F is handled separately by the block — it calls AppKit's
        // toggleFullScreen on our NSWindow instead of mpv's cycle.
        46 => Some(("cycle", vec!["mute"])),          // m
        0 => Some(("cycle", vec!["audio"])),          // a
        1 => Some(("cycle", vec!["sub"])),            // s
        // i alone toggles a single stats page; Shift+I cycles pages.
        34 if shift => Some(("script-binding", vec!["stats/display-stats-toggle"])),
        34 => Some(("script-binding", vec!["stats/display-stats"])),
        12 => Some(("quit", vec![])),                 // q
        _ => None,
    }
}

impl Drop for VideoWindow {
    fn drop(&mut self) {
        if let Some(observer) = self.close_observer.take() {
            unsafe {
                NSNotificationCenter::defaultCenter().removeObserver(&observer);
            }
        }
        if let Some(monitor) = self.monitor.take() {
            unsafe { NSEvent::removeMonitor(&monitor) };
        }
        // close() must run on the main thread; the host (Slint event
        // loop) is the only thread that owns this struct. If the
        // user already clicked the X button, the close-observer path
        // got here first and this is a harmless no-op.
        self.window.close();
    }
}

// ---------------------------------------------------------------- dispatch helper

#[link(name = "System", kind = "dylib")]
unsafe extern "C" {
    static _dispatch_main_q: c_void;
    fn dispatch_async_f(
        queue: *const c_void,
        context: *mut c_void,
        work: extern "C" fn(*mut c_void),
    );
}

/// Post a closure to the main thread's run loop. Used by the mpv
/// worker thread to drop the `VideoWindow` after mpv emits
/// `Shutdown` — `Retained<NSWindow>::drop` must run on the main
/// thread, so we can't drop it from the worker directly.
pub fn run_on_main<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    extern "C" fn trampoline<F: FnOnce() + Send + 'static>(ctx: *mut c_void) {
        // Reconstruct the Box and run the closure exactly once.
        let f = unsafe { Box::from_raw(ctx as *mut F) };
        f();
    }
    let boxed: Box<F> = Box::new(f);
    let ctx = Box::into_raw(boxed) as *mut c_void;
    unsafe {
        dispatch_async_f(
            &raw const _dispatch_main_q as *const c_void,
            ctx,
            trampoline::<F>,
        );
    }
}

// ---------------------------------------------------------------- window registry
//
// `VideoWindow` holds `Retained<NSWindow>` which is `!Send`, so it
// can't live in a `Mutex` static. Instead we keep one main-thread
// `RefCell<HashMap>` keyed by an increasing id. `play()` registers
// the window from the main thread; the mpv worker, on exit,
// dispatches `unregister_window` back to the main thread so the
// `Drop` of `VideoWindow` (and thus `NSWindow.close()`) runs there.

thread_local! {
    static WINDOWS: RefCell<HashMap<u64, VideoWindow>> = RefCell::new(HashMap::new());
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Stash a fresh `VideoWindow` on the main thread and return its id.
/// Caller must be on the main thread.
pub fn register_window(w: VideoWindow) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    WINDOWS.with(|cell| {
        cell.borrow_mut().insert(id, w);
    });
    id
}

/// Drop the window with the given id. No-op if already gone (e.g.
/// the user closed it via the X button and we tore down earlier).
/// Caller must be on the main thread.
pub fn unregister_window(id: u64) {
    WINDOWS.with(|cell| {
        cell.borrow_mut().remove(&id);
    });
}

/// Install the NSEvent key monitor on the window with the given id.
/// No-op if the window is already gone. Caller must be on the main
/// thread.
pub fn install_key_handler(id: u64, mpv: Arc<Mpv>) {
    WINDOWS.with(|cell| {
        if let Some(win) = cell.borrow_mut().get_mut(&id) {
            win.install_key_handler(mpv);
        }
    });
}
