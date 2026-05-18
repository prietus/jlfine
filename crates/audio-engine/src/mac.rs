//! macOS CoreAudio HAL backend (bitperfect, streaming).
//!
//! Acquires HogMode, switches the device's nominal sample rate to
//! match the source, and pumps PCM from a SPSC ring buffer to the DAC
//! via an `AudioDeviceIOProc`. The decoder thread (in `super`) feeds
//! the ring buffer at decode speed; the IOProc consumes it in
//! real-time and emits silence on underrun.

use core_foundation::{
    base::TCFType,
    string::{CFString, CFStringRef},
};
use coreaudio_sys::*;
use rtrb::Consumer;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::{Duration, Instant};
use std::{mem, process, ptr, thread};

/// HAL-level errors. The `i32` payload is the raw `OSStatus`; it's
/// consumed via `Debug` when bubbling up through
/// `Error::Backend(format!(...))`, which dead-code analysis can't see.
#[derive(Debug)]
#[allow(dead_code)]
pub enum HalError {
    NoDefaultDevice(i32),
    EnumerateDevices(i32),
    HogBusy(i32),
    SampleRate(i32),
    CreateIoProc(i32),
    StartDevice(i32),
}

pub fn play_stream(
    consumer: Consumer<f32>,
    sample_rate: f64,
    channels: u32,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: &Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
) -> Result<(), HalError> {
    let device_id = resolve_device(audio_device)?;
    let device_name = get_device_name(device_id).unwrap_or_else(|| "?".into());
    tracing::info!(device_id, device_name = %device_name, exclusive, "selected output device");

    // Wait for ~200 ms of audio to accumulate (or EOF if the track is
    // shorter than the prefill) before we touch the DAC. This + the
    // 500 ms pre-roll below keeps the first packets from arriving as
    // underrun-silence.
    {
        let target = ((sample_rate as usize) * (channels as usize)) / 5;
        let start = Instant::now();
        loop {
            if cancel.load(Ordering::Acquire) {
                return Ok(());
            }
            if eof.load(Ordering::Acquire) {
                break;
            }
            if consumer.slots() >= target {
                break;
            }
            if start.elapsed() > Duration::from_secs(5) {
                tracing::warn!(slots = consumer.slots(), "prefill timeout");
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        tracing::info!(slots = consumer.slots(), "prefill done");
    }

    let mut session = Session {
        device_id,
        proc_id: None,
        took_hog: false,
        running: false,
        state_ptr: ptr::null_mut(),
    };

    // Exclusive (bitperfect) path: take HogMode and force the
    // device's nominal sample rate to match the source. Shared path:
    // leave both alone — the audio stays interruptible by other apps
    // and the system mixer will resample. Caller knows what they
    // asked for.
    if exclusive {
        session.took_hog = acquire_hog_mode(device_id)?;
        tracing::info!(took_hog = session.took_hog, pid = process::id(), "hog mode");

        let prev_rate = get_nominal_sample_rate(device_id)?;
        set_nominal_sample_rate(device_id, sample_rate)?;
        let new_rate = get_nominal_sample_rate(device_id)?;
        tracing::info!(
            prev_rate,
            target = sample_rate,
            now = new_rate,
            "device rate"
        );
    } else {
        tracing::info!("shared mode: skipping HogMode and rate switch");
    }

    let state = Box::new(PlayerState {
        consumer,
        eof: eof.clone(),
        channels,
        frames_played: AtomicI64::new(0),
        finished: AtomicBool::new(false),
        pre_roll_active: AtomicBool::new(true),
    });
    let state_ptr = Box::into_raw(state);
    session.state_ptr = state_ptr;

    // Pre-roll: zeros for ~500 ms while the DAC PLL locks onto the
    // new physical sample rate. Otherwise the listener loses the
    // first second of the track.
    {
        let s = state_ptr as usize;
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(500));
            unsafe {
                (*(s as *mut PlayerState))
                    .pre_roll_active
                    .store(false, Ordering::Release);
            }
        });
    }

    let proc_id = create_io_proc(device_id, state_ptr as *mut c_void)?;
    session.proc_id = proc_id;
    start_device(device_id, proc_id)?;
    session.running = true;
    tracing::info!("io proc started");

    // Wait for natural finish OR external cancel. 100 ms is fine —
    // cancellation responsiveness isn't tight, and we keep CPU low.
    while !unsafe { &*state_ptr }.finished.load(Ordering::Acquire)
        && !cancel.load(Ordering::Acquire)
    {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!(
        cancelled = cancel.load(Ordering::Acquire),
        "playback finished"
    );
    // Session::drop tears down IOProc, releases HogMode, and frees
    // the player state (which drops the Consumer) — order matters.
    Ok(())
}

// ----------------------------------------------------------- shared state

struct PlayerState {
    consumer: Consumer<f32>,
    eof: Arc<AtomicBool>,
    channels: u32,
    frames_played: AtomicI64,
    finished: AtomicBool,
    pre_roll_active: AtomicBool,
}
unsafe impl Send for PlayerState {}
unsafe impl Sync for PlayerState {}

struct Session {
    device_id: AudioObjectID,
    /// `AudioDeviceIOProcID` is already `Option<fn>` from bindgen; no
    /// extra wrapper needed.
    proc_id: AudioDeviceIOProcID,
    took_hog: bool,
    running: bool,
    state_ptr: *mut PlayerState,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            if self.proc_id.is_some() {
                if self.running {
                    AudioDeviceStop(self.device_id, self.proc_id);
                }
                AudioDeviceDestroyIOProcID(self.device_id, self.proc_id);
            }
            if self.took_hog {
                let _ = release_hog_mode(self.device_id);
            }
            if !self.state_ptr.is_null() {
                drop(Box::from_raw(self.state_ptr));
            }
        }
    }
}

// ----------------------------------------------------- CoreAudio props

fn hog_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyHogMode,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}
fn rate_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}
fn default_output_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}
fn devices_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}
fn name_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}
fn uid_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

// ------------------------------------------------------ discovery

/// Map the persisted `audio_device` id (mpv-style) to a HAL device.
/// `coreaudio/<UID>` resolves via `kAudioDevicePropertyDeviceUID`;
/// anything else (None, `coreaudio/` with no UID, a non-CoreAudio id
/// that drifted in from another platform) falls back to the system
/// default output.
fn resolve_device(audio_device: Option<&str>) -> Result<AudioObjectID, HalError> {
    let uid = audio_device
        .and_then(|s| s.strip_prefix("coreaudio/"))
        .filter(|s| !s.is_empty());
    if let Some(uid) = uid {
        for id in list_devices()? {
            if let Some(have) = get_device_uid(id)
                && have == uid
            {
                return Ok(id);
            }
        }
        tracing::warn!(uid, "preferred device UID not found, using default output");
    }
    default_output_device()
}

fn list_devices() -> Result<Vec<AudioObjectID>, HalError> {
    let addr = devices_property();
    let mut size: u32 = 0;
    let s = unsafe {
        AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, ptr::null(), &mut size)
    };
    if s != 0 {
        return Err(HalError::EnumerateDevices(s));
    }
    let count = size as usize / mem::size_of::<AudioObjectID>();
    let mut ids = vec![0u32; count];
    let s = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            0,
            ptr::null(),
            &mut size,
            ids.as_mut_ptr() as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::EnumerateDevices(s));
    }
    Ok(ids)
}

fn default_output_device() -> Result<AudioObjectID, HalError> {
    let addr = default_output_property();
    let mut id: AudioObjectID = 0;
    let mut size = mem::size_of::<AudioObjectID>() as u32;
    let s = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut id as *mut _ as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::NoDefaultDevice(s));
    }
    Ok(id)
}

fn get_device_name(id: AudioObjectID) -> Option<String> {
    get_cfstring_property(id, name_property())
}

fn get_device_uid(id: AudioObjectID) -> Option<String> {
    get_cfstring_property(id, uid_property())
}

fn get_cfstring_property(id: AudioObjectID, addr: AudioObjectPropertyAddress) -> Option<String> {
    let mut cf: CFStringRef = ptr::null();
    let mut size = mem::size_of::<CFStringRef>() as u32;
    let s = unsafe {
        AudioObjectGetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut cf as *mut _ as *mut c_void,
        )
    };
    if s != 0 || cf.is_null() {
        return None;
    }
    Some(unsafe { CFString::wrap_under_create_rule(cf) }.to_string())
}

// --------------------------------------------------- hog + rate

fn acquire_hog_mode(id: AudioObjectID) -> Result<bool, HalError> {
    let addr = hog_property();
    let me = process::id() as i32;

    let mut current: i32 = -1;
    let mut size = mem::size_of::<i32>() as u32;
    unsafe {
        AudioObjectGetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut current as *mut _ as *mut c_void,
        );
    }
    if current == me {
        return Ok(true);
    }
    if current != -1 {
        return Err(HalError::HogBusy(current));
    }

    let mut new_hog = me;
    let s = unsafe {
        AudioObjectSetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            mem::size_of::<i32>() as u32,
            &mut new_hog as *mut _ as *mut c_void,
        )
    };
    if s != 0 {
        tracing::warn!(
            status = s,
            "hog mode set failed; continuing without exclusive lock"
        );
        return Ok(false);
    }
    Ok(true)
}

fn release_hog_mode(id: AudioObjectID) -> Result<(), HalError> {
    let addr = hog_property();
    let mut release: i32 = -1;
    unsafe {
        AudioObjectSetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            mem::size_of::<i32>() as u32,
            &mut release as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

fn get_nominal_sample_rate(id: AudioObjectID) -> Result<f64, HalError> {
    let addr = rate_property();
    let mut rate: f64 = 0.0;
    let mut size = mem::size_of::<f64>() as u32;
    let s = unsafe {
        AudioObjectGetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut rate as *mut _ as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::SampleRate(s));
    }
    Ok(rate)
}

fn set_nominal_sample_rate(id: AudioObjectID, rate: f64) -> Result<(), HalError> {
    let addr = rate_property();
    let mut r = rate;
    let s = unsafe {
        AudioObjectSetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            mem::size_of::<f64>() as u32,
            &mut r as *mut _ as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::SampleRate(s));
    }
    thread::sleep(Duration::from_millis(150));
    Ok(())
}

// ----------------------------------------------------------- IOProc

fn create_io_proc(id: AudioObjectID, ctx: *mut c_void) -> Result<AudioDeviceIOProcID, HalError> {
    let mut proc_id: AudioDeviceIOProcID = None;
    let s = unsafe { AudioDeviceCreateIOProcID(id, Some(io_proc), ctx, &mut proc_id) };
    if s != 0 {
        return Err(HalError::CreateIoProc(s));
    }
    Ok(proc_id)
}

fn start_device(id: AudioObjectID, proc_id: AudioDeviceIOProcID) -> Result<(), HalError> {
    let s = unsafe { AudioDeviceStart(id, proc_id) };
    if s != 0 {
        return Err(HalError::StartDevice(s));
    }
    Ok(())
}

/// Real-time audio callback. No allocations, no locks — just an
/// SPSC ring-buffer pop and a memcpy.
unsafe extern "C" fn io_proc(
    _in_device: AudioObjectID,
    _in_now: *const AudioTimeStamp,
    _in_input_data: *const AudioBufferList,
    _in_input_time: *const AudioTimeStamp,
    out_output_data: *mut AudioBufferList,
    _in_output_time: *const AudioTimeStamp,
    in_client_data: *mut c_void,
) -> OSStatus {
    let state = unsafe { &mut *(in_client_data as *mut PlayerState) };
    let abl = unsafe { &mut *out_output_data };

    if abl.mNumberBuffers == 0 {
        return 0;
    }
    let buffer = unsafe { &mut *(abl.mBuffers.as_mut_ptr()) };
    let data = buffer.mData as *mut f32;
    let bytes = buffer.mDataByteSize as usize;
    if data.is_null() || bytes == 0 {
        return 0;
    }

    let n_samples = bytes / mem::size_of::<f32>();
    let out = unsafe { std::slice::from_raw_parts_mut(data, n_samples) };

    if state.pre_roll_active.load(Ordering::Acquire) {
        for s in out.iter_mut() {
            *s = 0.0;
        }
        return 0;
    }

    let avail = state.consumer.slots();
    let take = avail.min(n_samples);
    if take > 0 {
        // `take <= avail`, so `read_chunk` is guaranteed to succeed.
        if let Ok(chunk) = state.consumer.read_chunk(take) {
            let (s1, s2) = chunk.as_slices();
            out[..s1.len()].copy_from_slice(s1);
            out[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
            chunk.commit_all();
        }
    }
    if take < n_samples {
        for s in out[take..].iter_mut() {
            *s = 0.0;
        }
    }

    state
        .frames_played
        .fetch_add((take / state.channels as usize) as i64, Ordering::Relaxed);

    // EOF from decoder + ring drained = natural end of track.
    if state.eof.load(Ordering::Acquire) && state.consumer.slots() == 0 {
        state.finished.store(true, Ordering::Release);
    }

    0
}
