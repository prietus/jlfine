//! macOS keyboard bridge for libmpv playback.
//!
//! mpv on macOS opens and owns its own video window. The `--wid`
//! embedding path is unreliable with the Metal-backed `gpu-next` VO —
//! mpv ignores the host view and spawns its own window anyway — so we
//! stop fighting it: mpv keeps its window and we only solve keyboard.
//!
//! libmpv runs embedded under the host's `NSApplication` (Slint via
//! winit). mpv's window never reliably enters that responder chain,
//! so mpv's own key handling stays dead. We bridge keys with an
//! app-wide `NSEvent` local monitor: it fires for every key event the
//! app dispatches regardless of which window is key, we translate the
//! ones aimed at mpv's window into mpv commands, and we leave the
//! jelly-ui (Slint) windows untouched.
//!
//! "Which window is mpv's" is decided by exclusion: [`snapshot_windows`]
//! records the window numbers that exist *before* mpv starts (the
//! jelly-ui windows); any other key window is mpv's.
//!
//! This module only knows about AppKit. The mpv lifecycle lives in
//! `lib.rs`.
//!
//! `run_on_main` / `run_on_main_sync` are callable from any thread;
//! every other public API must run on the main thread.

use block2::RcBlock;
use libmpv2::Mpv;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_app_kit::{NSApplication, NSEvent, NSEventMask, NSEventModifierFlags};
use objc2_foundation::MainThreadMarker;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Weak;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

/// Snapshot the window numbers of every window the app currently owns
/// — these are jelly-ui's Slint windows, captured just before mpv
/// opens its own. The key monitor uses this set to tell jelly-ui
/// windows (leave alone) from mpv's window (forward keys). Must run
/// on the main thread.
pub fn snapshot_windows() -> Vec<isize> {
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    let app = NSApplication::sharedApplication(mtm);
    let windows = app.windows();
    let count = windows.count();
    let mut nums = Vec::with_capacity(count);
    for i in 0..count {
        let w = unsafe { windows.objectAtIndex(i) };
        nums.push(unsafe { w.windowNumber() });
    }
    nums
}

/// Install an app-wide `NSEvent` local monitor for `keyDown` and
/// forward keys aimed at mpv's window into mpv as commands — same
/// mapping as a vanilla mpv CLI session (space=pause, arrows=seek,
/// f=fullscreen, i=stats, m=mute, q=quit, a/s cycle audio/subs). Key
/// events whose window is one of `pre_windows` (jelly-ui's) pass
/// through untouched so Slint keeps its shortcuts.
///
/// The block holds a `Weak<Mpv>` rather than a strong `Arc<Mpv>` so
/// the worker thread's strong reference is guaranteed to be the last
/// one. `Mpv::drop` calls the main-thread-blocking
/// `mpv_terminate_destroy`; keeping it off the monitor means it runs
/// on the worker when playback ends, not on the main thread when the
/// monitor is removed — that ordering would deadlock.
///
/// Must run on the main thread.
pub fn install_key_monitor(id: u64, pre_windows: Vec<isize>, mpv: Weak<Mpv>) {
    let block: RcBlock<dyn Fn(NonNull<NSEvent>) -> *mut NSEvent> =
        RcBlock::new(move |event_ptr: NonNull<NSEvent>| -> *mut NSEvent {
            // SAFETY: AppKit hands us a live NSEvent for the duration
            // of this dispatch; we never keep it past the closure.
            let event = unsafe { event_ptr.as_ref() };
            let win_num = unsafe { event.windowNumber() };
            // A pre-existing (jelly-ui) window has focus — don't touch
            // its keys.
            if pre_windows.contains(&win_num) {
                return event_ptr.as_ptr();
            }
            let Some(mpv) = mpv.upgrade() else {
                // mpv already gone (worker exited, monitor not yet
                // removed). Leave the event alone.
                return event_ptr.as_ptr();
            };
            let key = unsafe { event.keyCode() };
            let mods = unsafe { event.modifierFlags() };
            let shift = mods.contains(NSEventModifierFlags::NSEventModifierFlagShift);
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
        NSEvent::addLocalMonitorForEventsMatchingMask_handler(NSEventMask::KeyDown, &block)
    };
    MONITORS.with(|cell| {
        cell.borrow_mut().insert(id, token);
    });
}

/// Remove the key monitor with the given id. No-op if already gone.
/// Must run on the main thread.
pub fn remove_key_monitor(id: u64) {
    let token = MONITORS.with(|cell| cell.borrow_mut().remove(&id));
    if let Some(Some(monitor)) = token {
        unsafe { NSEvent::removeMonitor(&monitor) };
    }
}

/// Map a macOS virtual key code (kVK_* — physical key position) to
/// the mpv command + args we want to fire. Same physical mapping as
/// `~/jelly`'s PlayerKeyHandler.swift. mpv owns the window, so `f`
/// goes through mpv's own `cycle fullscreen` (native macOS fullscreen
/// of its window).
fn translate_key(key_code: u16, shift: bool) -> Option<(&'static str, Vec<&'static str>)> {
    match key_code {
        49 => Some(("cycle", vec!["pause"])),     // space
        123 => Some(("seek", vec!["-5"])),        // left
        124 => Some(("seek", vec!["5"])),         // right
        125 => Some(("seek", vec!["-60"])),       // down
        126 => Some(("seek", vec!["60"])),        // up
        53 => Some(("quit", vec![])),             // escape
        3 => Some(("cycle", vec!["fullscreen"])), // f
        46 => Some(("cycle", vec!["mute"])),      // m
        0 => Some(("cycle", vec!["audio"])),      // a
        1 => Some(("cycle", vec!["sub"])),        // s
        // i alone toggles a single stats page; Shift+I cycles pages.
        34 if shift => Some(("script-binding", vec!["stats/display-stats-toggle"])),
        34 => Some(("script-binding", vec!["stats/display-stats"])),
        12 => Some(("quit", vec![])), // q
        _ => None,
    }
}

// ---------------------------------------------------------------- dispatch helpers

#[link(name = "System", kind = "dylib")]
unsafe extern "C" {
    static _dispatch_main_q: c_void;
    fn dispatch_async_f(
        queue: *const c_void,
        context: *mut c_void,
        work: extern "C" fn(*mut c_void),
    );
    fn dispatch_sync_f(
        queue: *const c_void,
        context: *mut c_void,
        work: extern "C" fn(*mut c_void),
    );
}

/// Post a closure to the main thread's run loop, fire-and-forget.
/// Used by the mpv worker to install/remove the key monitor (AppKit
/// APIs are main-thread-only).
pub fn run_on_main<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    extern "C" fn trampoline<F: FnOnce() + Send + 'static>(ctx: *mut c_void) {
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

/// Run a closure on the main thread and block until it returns its
/// value. Used to snapshot the jelly-ui windows before mpv opens its
/// own. Safe to call from the worker because the main thread (Slint
/// event loop) services the main dispatch queue and is not waiting on
/// the worker at this point.
pub fn run_on_main_sync<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    struct Payload<F, T> {
        f: Option<F>,
        result: Option<T>,
    }
    extern "C" fn trampoline<F: FnOnce() -> T, T>(ctx: *mut c_void) {
        let payload = unsafe { &mut *(ctx as *mut Payload<F, T>) };
        let f = payload.f.take().expect("payload closure taken once");
        payload.result = Some(f());
    }
    let mut payload = Payload {
        f: Some(f),
        result: None,
    };
    let ctx = &mut payload as *mut Payload<F, T> as *mut c_void;
    unsafe {
        dispatch_sync_f(
            &raw const _dispatch_main_q as *const c_void,
            ctx,
            trampoline::<F, T>,
        );
    }
    payload
        .result
        .take()
        .expect("main-thread closure produced no result")
}

// ---------------------------------------------------------------- monitor registry
//
// The monitor token is a `Retained<AnyObject>` (`!Send`), so it can't
// live in a `Mutex` static. We keep it in a main-thread-only
// `RefCell<HashMap>` keyed by an increasing id. The worker generates
// the id via `next_monitor_id`, dispatches `install_key_monitor` to
// main, and on exit dispatches `remove_key_monitor` back to main.

thread_local! {
    static MONITORS: RefCell<HashMap<u64, Option<Retained<AnyObject>>>> =
        RefCell::new(HashMap::new());
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Allocate a fresh monitor id. Callable from any thread.
pub fn next_monitor_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
