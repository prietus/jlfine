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
    EnumerateStreams(i32),
    NoOutputStream,
    EnumeratePhysicalFormats(i32),
    PhysicalFormatUnsupported,
    SetPhysicalFormat(i32),
    GetPhysicalFormat(i32),
    DopRequiresExclusive,
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

// ============================================================== DoP path
//
// DoP (DSD-over-PCM) playback. Differences vs the PCM path:
//
// 1. HogMode is mandatory: changing a stream's physical format is
//    illegal without it.
// 2. We switch the output stream's *physical* format to 24-bit
//    signed integer, big-endian, aligned-high in a 32-bit container,
//    at the DoP PCM rate (176.4 / 352.8 kHz for DSD64 / DSD128).
// 3. The IOProc re-stamps the DoP marker on every sample from its
//    own counter rather than trusting whatever marker the decoder
//    produced. That way an underrun (filled with DoP silence bytes)
//    doesn't break the alternating 0x05/0xFA pattern the DAC needs
//    to stay in DSD mode.

use coreaudio_sys::AudioStreamRangedDescription;
use std::sync::atomic::AtomicU8;

const DOP_MARKER_A: u8 = 0x05;
const DOP_MARKER_B: u8 = 0xFA;

/// DSD "silence" cell: 0x69 followed by 0x96 keeps the DSD pipe at
/// near-DC and is what most DAC vendors recommend for muted DoP.
const DOP_SILENCE_HI: u8 = 0x69;
const DOP_SILENCE_LO: u8 = 0x96;

struct DopPlayerState {
    consumer: Consumer<u32>,
    eof: Arc<AtomicBool>,
    channels: u32,
    finished: AtomicBool,
    /// Marker the next sample must carry. Lives in the consumer so
    /// underruns can keep the schedule going.
    next_marker: AtomicU8,
}
unsafe impl Send for DopPlayerState {}
unsafe impl Sync for DopPlayerState {}

struct DopSession {
    device_id: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    took_hog: bool,
    running: bool,
    state_ptr: *mut DopPlayerState,
    /// (stream id, format) captured before our override. Best-effort
    /// restore in Drop so the user doesn't reboot to get their DAC
    /// back into Float32.
    saved_format: Option<(AudioStreamID, AudioStreamBasicDescription)>,
    /// Set by the worker when the next queued play is also DSD —
    /// restoring the physical format only to overwrite it again
    /// makes the DAC's display flash through the previous rate.
    skip_restore: Arc<AtomicBool>,
}

impl Drop for DopSession {
    fn drop(&mut self) {
        unsafe {
            if self.proc_id.is_some() {
                if self.running {
                    AudioDeviceStop(self.device_id, self.proc_id);
                }
                AudioDeviceDestroyIOProcID(self.device_id, self.proc_id);
            }
            if let Some((stream_id, prev)) = self.saved_format.take() {
                if self.skip_restore.load(Ordering::SeqCst) {
                    tracing::info!(
                        stream_id,
                        "skipping physical-format restore (next play is also DSD)"
                    );
                } else {
                    let _ = set_physical_format(stream_id, &prev);
                }
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

#[allow(clippy::too_many_arguments)]
pub fn play_stream_dop(
    consumer: Consumer<u32>,
    pcm_rate: f64,
    channels: u32,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: &Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
    skip_restore: Arc<AtomicBool>,
) -> Result<(), HalError> {
    if !exclusive {
        // No HogMode = no physical-format change. Refuse rather than
        // silently fall back to PCM and feed DSD bits to the mixer.
        return Err(HalError::DopRequiresExclusive);
    }

    let device_id = resolve_device(audio_device)?;
    let device_name = get_device_name(device_id).unwrap_or_else(|| "?".into());
    tracing::info!(
        device_id,
        device_name = %device_name,
        pcm_rate,
        channels,
        "selected output device for DoP"
    );

    let mut session = DopSession {
        device_id,
        proc_id: None,
        took_hog: false,
        running: false,
        state_ptr: ptr::null_mut(),
        saved_format: None,
        skip_restore,
    };

    session.took_hog = acquire_hog_mode(device_id)?;
    tracing::info!(took_hog = session.took_hog, pid = process::id(), "hog mode");

    // Pick the first output stream of the device. Multi-stream
    // devices are rare for music DACs; if we ever care, we'll add
    // selection here.
    let stream_id = first_output_stream(device_id)?;
    let target = dop_format(pcm_rate, channels);

    log_available_physical_formats(stream_id);

    // Bail out early with a clean error rather than poke the device
    // and hope.
    if !physical_format_supported(stream_id, &target)? {
        tracing::warn!(
            stream_id,
            sample_rate = target.mSampleRate,
            bits = target.mBitsPerChannel,
            "device does not advertise 24-bit packed integer at the DoP rate"
        );
        return Err(HalError::PhysicalFormatUnsupported);
    }

    let prev = get_physical_format(stream_id)?;
    if asbd_matches_target(&prev, &target) {
        // Already in the right format (e.g. previous track was also
        // DSD64). Skip the set entirely so the DAC doesn't show any
        // intermediate rate flicker between tracks.
        tracing::info!(stream_id, "physical format already DoP — skipping set");
    } else {
        set_physical_format(stream_id, &target)?;
        session.saved_format = Some((stream_id, prev));
        let actual = get_physical_format(stream_id).ok();
        tracing::info!(
            stream_id,
            target_rate = target.mSampleRate,
            target_bits = target.mBitsPerChannel,
            target_flags = target.mFormatFlags,
            target_bytes_per_frame = target.mBytesPerFrame,
            actual_rate = actual.map(|f| f.mSampleRate),
            actual_bits = actual.map(|f| f.mBitsPerChannel),
            actual_flags = actual.map(|f| f.mFormatFlags),
            actual_bytes_per_frame = actual.map(|f| f.mBytesPerFrame),
            "physical format set for DoP"
        );
    }

    // No `set_nominal_sample_rate` here: setting the physical format
    // already establishes the wire rate, and a second call would
    // make the DAC briefly re-lock (visible as "176.4 kHz PCM"
    // showing on the display before DSD64 takes over).

    // Short PLL lock window. Long enough that the DAC has finished
    // re-locking by the time the first DoP packet arrives; short
    // enough that the user doesn't notice the gap between hitting
    // play and audio starting.
    thread::sleep(Duration::from_millis(200));

    let state = Box::new(DopPlayerState {
        consumer,
        eof: eof.clone(),
        channels,
        finished: AtomicBool::new(false),
        next_marker: AtomicU8::new(DOP_MARKER_A),
    });
    let state_ptr = Box::into_raw(state);
    session.state_ptr = state_ptr;

    let proc_id = create_io_proc_with_callback(device_id, state_ptr as *mut c_void, dop_io_proc)?;
    session.proc_id = proc_id;
    start_device(device_id, proc_id)?;
    session.running = true;
    tracing::info!("DoP io proc started");

    while !unsafe { &*state_ptr }.finished.load(Ordering::Acquire)
        && !cancel.load(Ordering::Acquire)
    {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!(
        cancelled = cancel.load(Ordering::Acquire),
        "DoP playback finished"
    );
    Ok(())
}

fn dop_format(pcm_rate: f64, channels: u32) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        mSampleRate: pcm_rate,
        mFormatID: kAudioFormatLinearPCM,
        // 24-bit AlignedHigh in a 32-bit container with NonMixable.
        // Chosen from the DAC's advertised physical formats — the
        // XD-05 BAL (and most USB-Audio Class DACs on macOS) doesn't
        // expose 24-bit Packed (3-byte) at all; it only offers the
        // 24-in-32-AlignedHigh variant, in Mixable and NonMixable
        // flavours. Under HogMode we want NonMixable so the system
        // never inserts conversion stages between us and the DAC.
        // On the wire the 4-byte sample is [pad, dsd_lo, dsd_hi,
        // marker] (LE memory order), which a DAC reading 24-bit
        // AlignedHigh reconstructs as (marker<<16)|(dsd_hi<<8)|dsd_lo
        // — marker lands on bits 23..16 where ESS Sabre / AKM DoP
        // detectors look.
        mFormatFlags: kAudioFormatFlagIsSignedInteger
            | kAudioFormatFlagIsAlignedHigh
            | kAudioFormatFlagIsNonMixable,
        mBytesPerPacket: 4 * channels,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4 * channels,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 24,
        mReserved: 0,
    }
}

fn create_io_proc_with_callback(
    id: AudioObjectID,
    ctx: *mut c_void,
    cb: unsafe extern "C" fn(
        AudioObjectID,
        *const AudioTimeStamp,
        *const AudioBufferList,
        *const AudioTimeStamp,
        *mut AudioBufferList,
        *const AudioTimeStamp,
        *mut c_void,
    ) -> OSStatus,
) -> Result<AudioDeviceIOProcID, HalError> {
    let mut proc_id: AudioDeviceIOProcID = None;
    let s = unsafe { AudioDeviceCreateIOProcID(id, Some(cb), ctx, &mut proc_id) };
    if s != 0 {
        return Err(HalError::CreateIoProc(s));
    }
    Ok(proc_id)
}

/// DoP IOProc.
///
/// Layout in `out`: 24-bit packed per channel, 3 bytes per sample,
/// no padding. Memory order matches USB-Audio Class wire order:
/// `[dsd_lo, dsd_hi, marker]` per channel, repeated across frames.
///
/// The marker comes from the consumer's own counter; whatever the
/// producer encoded in bits 23..16 of each ring-buffer u32 is
/// ignored. Underruns emit DoP silence at the same marker tempo.
unsafe extern "C" fn dop_io_proc(
    _in_device: AudioObjectID,
    _in_now: *const AudioTimeStamp,
    _in_input_data: *const AudioBufferList,
    _in_input_time: *const AudioTimeStamp,
    out_output_data: *mut AudioBufferList,
    _in_output_time: *const AudioTimeStamp,
    in_client_data: *mut c_void,
) -> OSStatus {
    let state = unsafe { &mut *(in_client_data as *mut DopPlayerState) };
    let abl = unsafe { &mut *out_output_data };
    if abl.mNumberBuffers == 0 {
        return 0;
    }
    let buffer = unsafe { &mut *(abl.mBuffers.as_mut_ptr()) };
    let data = buffer.mData as *mut u8;
    let bytes = buffer.mDataByteSize as usize;
    if data.is_null() || bytes == 0 {
        return 0;
    }

    let channels = state.channels as usize;
    let bytes_per_frame = 4 * channels;
    let frames = bytes / bytes_per_frame;
    let out = unsafe { std::slice::from_raw_parts_mut(data, frames * bytes_per_frame) };

    let avail = state.consumer.slots();
    let take_samples = avail.min(frames * channels);
    let take_frames = take_samples / channels;

    let mut marker = state.next_marker.load(Ordering::Relaxed);

    // Real samples: pull from ring, strip whatever marker the producer put
    // in bits 23..16, restamp with `marker`, emit as 24-bit packed LE.
    if take_frames > 0 {
        if let Ok(chunk) = state.consumer.read_chunk(take_frames * channels) {
            let (s1, s2) = chunk.as_slices();
            let mut frame_idx = 0usize;
            for slice in [s1, s2] {
                for samples in slice.chunks_exact(channels) {
                    let frame_off = frame_idx * bytes_per_frame;
                    write_dop_frame(&mut out[frame_off..frame_off + bytes_per_frame], samples, marker);
                    marker = flip_marker(marker);
                    frame_idx += 1;
                }
            }
            chunk.commit_all();
        }
    }

    // Tail: any frames the ring couldn't fill go out as DoP silence
    // with the schedule's continuing marker, so the DAC never sees
    // the alternation break.
    let silence_lo_hi_toggle_start = take_frames % 2 == 1;
    for i in take_frames..frames {
        let frame_off = i * bytes_per_frame;
        let toggled = (i - take_frames) % 2 == 1;
        let (hi, lo) = if toggled ^ silence_lo_hi_toggle_start {
            (DOP_SILENCE_LO, DOP_SILENCE_HI)
        } else {
            (DOP_SILENCE_HI, DOP_SILENCE_LO)
        };
        write_dop_silence_frame(&mut out[frame_off..frame_off + bytes_per_frame], channels, marker, hi, lo);
        marker = flip_marker(marker);
    }

    state.next_marker.store(marker, Ordering::Relaxed);

    if state.eof.load(Ordering::Acquire) && state.consumer.slots() == 0 {
        state.finished.store(true, Ordering::Release);
    }
    0
}

#[inline]
fn flip_marker(m: u8) -> u8 {
    match m {
        DOP_MARKER_A => DOP_MARKER_B,
        _ => DOP_MARKER_A,
    }
}

#[inline]
fn write_dop_frame(out: &mut [u8], samples: &[u32], marker: u8) {
    // `samples` is one ring-buffer u32 per channel. Bits 15..0 are
    // the 16 DSD bits; bits 23..16 (the producer's marker) get
    // overwritten with the consumer's schedule. Output is 24-bit
    // AlignedHigh in a 32-bit slot, native (LE) endian: 4 bytes
    // per channel = [pad, dsd_lo, dsd_hi, marker]. The DAC reads
    // the high 3 bytes as the 24-bit sample, putting `marker` on
    // bits 23..16 where its DoP detector looks.
    for (ch, s) in samples.iter().enumerate() {
        let dsd_lo = (s & 0xFF) as u8;
        let dsd_hi = ((s >> 8) & 0xFF) as u8;
        let base = ch * 4;
        out[base] = 0;
        out[base + 1] = dsd_lo;
        out[base + 2] = dsd_hi;
        out[base + 3] = marker;
    }
}

#[inline]
fn write_dop_silence_frame(out: &mut [u8], channels: usize, marker: u8, hi: u8, lo: u8) {
    for ch in 0..channels {
        let base = ch * 4;
        out[base] = 0;
        out[base + 1] = lo;
        out[base + 2] = hi;
        out[base + 3] = marker;
    }
}

// ---------------------------------------------------- stream + format helpers

fn streams_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyStreams,
        mScope: kAudioObjectPropertyScopeOutput,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn physical_format_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioStreamPropertyPhysicalFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn available_physical_formats_property() -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: kAudioStreamPropertyAvailablePhysicalFormats,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn first_output_stream(device_id: AudioObjectID) -> Result<AudioStreamID, HalError> {
    let addr = streams_property();
    let mut size: u32 = 0;
    let s = unsafe {
        AudioObjectGetPropertyDataSize(device_id, &addr, 0, ptr::null(), &mut size)
    };
    if s != 0 {
        return Err(HalError::EnumerateStreams(s));
    }
    let count = size as usize / mem::size_of::<AudioStreamID>();
    if count == 0 {
        return Err(HalError::NoOutputStream);
    }
    let mut ids = vec![0u32; count];
    let s = unsafe {
        AudioObjectGetPropertyData(
            device_id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            ids.as_mut_ptr() as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::EnumerateStreams(s));
    }
    Ok(ids[0])
}

/// Emit one log line per advertised physical format on a stream.
/// Pure debug aid for narrowing down "device refused our format"
/// scenarios — easy to compare against `target` in the next log line.
fn log_available_physical_formats(stream_id: AudioStreamID) {
    let addr = available_physical_formats_property();
    let mut size: u32 = 0;
    let s = unsafe {
        AudioObjectGetPropertyDataSize(stream_id, &addr, 0, ptr::null(), &mut size)
    };
    if s != 0 || size == 0 {
        tracing::debug!(stream_id, status = s, "no available physical formats");
        return;
    }
    let count = size as usize / mem::size_of::<AudioStreamRangedDescription>();
    let mut buf = vec![AudioStreamRangedDescription::default(); count];
    let s = unsafe {
        AudioObjectGetPropertyData(
            stream_id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            buf.as_mut_ptr() as *mut c_void,
        )
    };
    if s != 0 {
        tracing::debug!(stream_id, status = s, "failed to read available formats");
        return;
    }
    for d in &buf {
        tracing::debug!(
            stream_id,
            min_rate = d.mSampleRateRange.mMinimum,
            max_rate = d.mSampleRateRange.mMaximum,
            rate = d.mFormat.mSampleRate,
            bits = d.mFormat.mBitsPerChannel,
            flags = d.mFormat.mFormatFlags,
            bytes_per_frame = d.mFormat.mBytesPerFrame,
            channels = d.mFormat.mChannelsPerFrame,
            "available physical format"
        );
    }
}

fn physical_format_supported(
    stream_id: AudioStreamID,
    target: &AudioStreamBasicDescription,
) -> Result<bool, HalError> {
    let addr = available_physical_formats_property();
    let mut size: u32 = 0;
    let s = unsafe {
        AudioObjectGetPropertyDataSize(stream_id, &addr, 0, ptr::null(), &mut size)
    };
    if s != 0 {
        return Err(HalError::EnumeratePhysicalFormats(s));
    }
    let count = size as usize / mem::size_of::<AudioStreamRangedDescription>();
    if count == 0 {
        return Ok(false);
    }
    let mut buf = vec![AudioStreamRangedDescription::default(); count];
    let s = unsafe {
        AudioObjectGetPropertyData(
            stream_id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            buf.as_mut_ptr() as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::EnumeratePhysicalFormats(s));
    }
    Ok(buf.iter().any(|d| asbd_compatible(&d.mFormat, target, &d.mSampleRateRange)))
}

/// Strict match for "already at the DoP target" — every field that
/// affects the wire layout must agree. Used to skip a redundant
/// physical-format set between consecutive DSD tracks.
fn asbd_matches_target(
    current: &AudioStreamBasicDescription,
    target: &AudioStreamBasicDescription,
) -> bool {
    current.mFormatID == target.mFormatID
        && current.mFormatFlags == target.mFormatFlags
        && current.mBitsPerChannel == target.mBitsPerChannel
        && current.mBytesPerFrame == target.mBytesPerFrame
        && current.mChannelsPerFrame == target.mChannelsPerFrame
        && (current.mSampleRate - target.mSampleRate).abs() < 0.5
}

fn asbd_compatible(
    have: &AudioStreamBasicDescription,
    want: &AudioStreamBasicDescription,
    range: &coreaudio_sys::AudioValueRange,
) -> bool {
    have.mFormatID == want.mFormatID
        && have.mBitsPerChannel == want.mBitsPerChannel
        && have.mChannelsPerFrame == want.mChannelsPerFrame
        // Require signed integer; tolerate float-flag missing,
        // AlignedHigh-vs-Packed differences, and endian differences.
        // We re-set the format to exactly what we want — this match
        // just gates "is the device fundamentally able to do 24-bit
        // signed integer at this rate?"
        && (have.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0
        && (have.mFormatFlags & kAudioFormatFlagIsFloat) == 0
        && want.mSampleRate >= range.mMinimum
        && want.mSampleRate <= range.mMaximum
}

fn get_physical_format(stream_id: AudioStreamID) -> Result<AudioStreamBasicDescription, HalError> {
    let addr = physical_format_property();
    let mut fmt = AudioStreamBasicDescription::default();
    let mut size = mem::size_of::<AudioStreamBasicDescription>() as u32;
    let s = unsafe {
        AudioObjectGetPropertyData(
            stream_id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            &mut fmt as *mut _ as *mut c_void,
        )
    };
    if s != 0 {
        return Err(HalError::GetPhysicalFormat(s));
    }
    Ok(fmt)
}

fn set_physical_format(
    stream_id: AudioStreamID,
    fmt: &AudioStreamBasicDescription,
) -> Result<(), HalError> {
    let addr = physical_format_property();
    let s = unsafe {
        AudioObjectSetPropertyData(
            stream_id,
            &addr,
            0,
            ptr::null(),
            mem::size_of::<AudioStreamBasicDescription>() as u32,
            fmt as *const _ as *const c_void,
        )
    };
    if s != 0 {
        return Err(HalError::SetPhysicalFormat(s));
    }
    thread::sleep(Duration::from_millis(50));
    Ok(())
}
