//! PoC: bitperfect playback on macOS via the CoreAudio HAL.
//!
//! Port of MacHALPlayer.swift narrowed to the minimum needed to validate
//! the audio path: pick a device, acquire hog mode, switch the device's
//! nominal sample rate to match the file, and pump PCM frames straight
//! into the IOProc. ExtAudioFile decodes the source (FLAC/ALAC/WAV/AIFF).
//!
//! No pause/seek/drain. Plays a file from start to EOF, then cleans up
//! (including releasing hog mode) and exits.

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    mac::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("poc-bitperfect-mac runs only on macOS");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
mod mac {
    use anyhow::{Context, Result, anyhow, bail};
    use core_foundation::{
        base::TCFType,
        string::{CFString, CFStringRef},
        url::CFURL,
    };
    use coreaudio_sys::*;
    use std::{
        ffi::c_void,
        mem, process, ptr,
        sync::atomic::{AtomicBool, AtomicI64, Ordering},
        thread,
        time::Duration,
    };

    const DEFAULT_DEVICE_SUBSTRING: &str = "xDuoo";

    pub fn run() -> Result<()> {
        let mut args = std::env::args().skip(1);
        let file_arg = args
            .next()
            .context("usage: poc-bitperfect-mac <file> [--device <substring>]")?;
        let mut device_substr: Option<String> = None;
        while let Some(a) = args.next() {
            match a.as_str() {
                "--device" => device_substr = args.next(),
                other => bail!("unknown arg: {other}"),
            }
        }
        let device_substr = device_substr.unwrap_or_else(|| DEFAULT_DEVICE_SUBSTRING.to_string());

        let device_id = select_device(&device_substr)?;
        let name = get_device_name(device_id).unwrap_or_else(|| "?".into());
        println!("[device] id={device_id} name={name:?}");

        let url =
            CFURL::from_path(&file_arg, false).ok_or_else(|| anyhow!("bad path: {file_arg}"))?;
        let (ext, fmt, total_frames) = open_ext_audio_file(&url)?;
        println!(
            "[file]   sr={} ch={} bits={} frames={} duration={:.2}s",
            fmt.mSampleRate,
            fmt.mChannelsPerFrame,
            fmt.mBitsPerChannel,
            total_frames,
            total_frames as f64 / fmt.mSampleRate,
        );

        let channels = fmt.mChannelsPerFrame.max(1);
        let sample_rate = fmt.mSampleRate;
        set_client_format(ext, sample_rate, channels)?;

        // Session owns lifecycle; Drop releases everything even on panic / early return.
        let mut session = Session {
            device_id,
            ext,
            proc_id: None, // AudioDeviceIOProcID = Option<fn>
            took_hog: false,
            running: false,
            state_ptr: ptr::null_mut(),
        };

        session.took_hog = acquire_hog_mode(device_id)?;
        println!("[hog]    took={} pid={}", session.took_hog, process::id());

        let prev_rate = get_nominal_sample_rate(device_id)?;
        set_nominal_sample_rate(device_id, sample_rate)?;
        let new_rate = get_nominal_sample_rate(device_id)?;
        println!("[rate]   prev={prev_rate} target={sample_rate} now={new_rate}");

        let state = Box::new(PlayerState {
            ext,
            channels,
            frames_played: AtomicI64::new(0),
            finished: AtomicBool::new(false),
            pre_roll_active: AtomicBool::new(true),
        });
        let state_ptr = Box::into_raw(state);
        session.state_ptr = state_ptr;

        // Pre-roll: zeros for ~500ms so the DAC's PLL locks on the new rate
        // before the first real sample reaches it. Without this the listener
        // loses the first second of the song to PLL settling artefacts.
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
        println!("[io]     proc started");

        while !unsafe { &*state_ptr }.finished.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(100));
            let played = unsafe { &*state_ptr }.frames_played.load(Ordering::Relaxed);
            let pos = played as f64 / sample_rate;
            let dur = total_frames as f64 / sample_rate;
            eprint!("\r[play]   {pos:7.2}s / {dur:7.2}s");
        }
        eprintln!();
        println!("[done]");
        // Session::drop releases hog, stops, destroys proc, disposes ext file.
        Ok(())
    }

    // ----------------------------------------------------------- shared state

    #[repr(C)]
    struct PlayerState {
        ext: ExtAudioFileRef,
        channels: u32,
        frames_played: AtomicI64,
        finished: AtomicBool,
        pre_roll_active: AtomicBool,
    }
    unsafe impl Send for PlayerState {}
    unsafe impl Sync for PlayerState {}

    struct Session {
        device_id: AudioObjectID,
        ext: ExtAudioFileRef,
        // AudioDeviceIOProcID is already `Option<fn>` from bindgen; no extra wrap.
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
                if !self.ext.is_null() {
                    ExtAudioFileDispose(self.ext);
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

    // ----------------------------------------------------- property addresses

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

    // ------------------------------------------------------ device discovery

    fn select_device(substr: &str) -> Result<AudioObjectID> {
        for id in list_devices()? {
            if let Some(n) = get_device_name(id) {
                if n.to_lowercase().contains(&substr.to_lowercase()) {
                    return Ok(id);
                }
            }
        }
        eprintln!("[device] '{substr}' not found, falling back to default output");
        default_output_device()
    }

    fn list_devices() -> Result<Vec<AudioObjectID>> {
        let addr = devices_property();
        let mut size: u32 = 0;
        let s = unsafe {
            AudioObjectGetPropertyDataSize(
                kAudioObjectSystemObject,
                &addr,
                0,
                ptr::null(),
                &mut size,
            )
        };
        if s != 0 {
            bail!("AudioObjectGetPropertyDataSize devices: {s}");
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
            bail!("AudioObjectGetPropertyData devices: {s}");
        }
        Ok(ids)
    }

    fn default_output_device() -> Result<AudioObjectID> {
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
            bail!("get default output: {s}");
        }
        Ok(id)
    }

    fn get_device_name(id: AudioObjectID) -> Option<String> {
        let addr = name_property();
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

    // -------------------------------------------------- ExtAudioFile helpers

    fn open_ext_audio_file(
        url: &CFURL,
    ) -> Result<(ExtAudioFileRef, AudioStreamBasicDescription, i64)> {
        let mut ext: ExtAudioFileRef = ptr::null_mut();
        // core-foundation and coreaudio-sys both bind __CFURL but as distinct
        // Rust types. The underlying CFURLRef pointer is the same C type, so
        // cast across the bindings.
        let url_ref = url.as_concrete_TypeRef() as *const _ as CFURLRef;
        let s = unsafe { ExtAudioFileOpenURL(url_ref, &mut ext) };
        if s != 0 || ext.is_null() {
            bail!("ExtAudioFileOpenURL: {s}");
        }

        let mut fmt: AudioStreamBasicDescription = unsafe { mem::zeroed() };
        let mut size = mem::size_of::<AudioStreamBasicDescription>() as u32;
        let s = unsafe {
            ExtAudioFileGetProperty(
                ext,
                kExtAudioFileProperty_FileDataFormat,
                &mut size,
                &mut fmt as *mut _ as *mut c_void,
            )
        };
        if s != 0 {
            unsafe { ExtAudioFileDispose(ext) };
            bail!("FileDataFormat: {s}");
        }

        let mut frames: i64 = 0;
        let mut fsize = mem::size_of::<i64>() as u32;
        unsafe {
            ExtAudioFileGetProperty(
                ext,
                kExtAudioFileProperty_FileLengthFrames,
                &mut fsize,
                &mut frames as *mut _ as *mut c_void,
            );
        }
        Ok((ext, fmt, frames))
    }

    fn set_client_format(ext: ExtAudioFileRef, sr: f64, ch: u32) -> Result<()> {
        let fmt = AudioStreamBasicDescription {
            mSampleRate: sr,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: ch * 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: ch * 4,
            mChannelsPerFrame: ch,
            mBitsPerChannel: 32,
            mReserved: 0,
        };
        let s = unsafe {
            ExtAudioFileSetProperty(
                ext,
                kExtAudioFileProperty_ClientDataFormat,
                mem::size_of::<AudioStreamBasicDescription>() as u32,
                &fmt as *const _ as *const c_void,
            )
        };
        if s != 0 {
            bail!("ClientDataFormat: {s}");
        }
        Ok(())
    }

    // -------------------------------------------------- hog + rate

    fn acquire_hog_mode(id: AudioObjectID) -> Result<bool> {
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
            bail!("device busy: hog held by pid {current}");
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
            eprintln!("[hog]    set failed: {s}; continuing without");
            return Ok(false);
        }
        Ok(true)
    }

    fn release_hog_mode(id: AudioObjectID) -> Result<()> {
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

    fn get_nominal_sample_rate(id: AudioObjectID) -> Result<f64> {
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
            bail!("get nominal rate: {s}");
        }
        Ok(rate)
    }

    fn set_nominal_sample_rate(id: AudioObjectID, rate: f64) -> Result<()> {
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
            bail!("set nominal rate: {s}");
        }
        thread::sleep(Duration::from_millis(150));
        Ok(())
    }

    // ----------------------------------------------------------- IOProc

    fn create_io_proc(id: AudioObjectID, ctx: *mut c_void) -> Result<AudioDeviceIOProcID> {
        let mut proc_id: AudioDeviceIOProcID = None;
        let s = unsafe { AudioDeviceCreateIOProcID(id, Some(io_proc), ctx, &mut proc_id) };
        if s != 0 {
            bail!("create IOProc: {s}");
        }
        Ok(proc_id)
    }

    fn start_device(id: AudioObjectID, proc_id: AudioDeviceIOProcID) -> Result<()> {
        let s = unsafe { AudioDeviceStart(id, proc_id) };
        if s != 0 {
            bail!("AudioDeviceStart: {s}");
        }
        Ok(())
    }

    /// Real-time audio callback. No allocations, no locks.
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
        let data = buffer.mData;
        let bytes = buffer.mDataByteSize;
        if data.is_null() || bytes == 0 {
            return 0;
        }

        if state.pre_roll_active.load(Ordering::Acquire) {
            unsafe { ptr::write_bytes(data as *mut u8, 0, bytes as usize) };
            return 0;
        }

        let bytes_per_frame = 4 * state.channels;
        let requested = bytes / bytes_per_frame;

        let mut render = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer {
                mNumberChannels: state.channels,
                mDataByteSize: bytes,
                mData: data,
            }],
        };

        let mut frames_read: u32 = requested;
        let s = unsafe { ExtAudioFileRead(state.ext, &mut frames_read, &mut render) };

        if s == 0 && frames_read > 0 {
            state
                .frames_played
                .fetch_add(frames_read as i64, Ordering::Relaxed);
            if frames_read < requested {
                let consumed = (frames_read * bytes_per_frame) as usize;
                let tail = bytes as usize - consumed;
                unsafe {
                    ptr::write_bytes((data as *mut u8).add(consumed), 0, tail);
                }
                state.finished.store(true, Ordering::Release);
            }
            return 0;
        }

        unsafe { ptr::write_bytes(data as *mut u8, 0, bytes as usize) };
        state.finished.store(true, Ordering::Release);
        0
    }
}
