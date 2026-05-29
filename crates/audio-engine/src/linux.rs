//! Linux ALSA backend.
//!
//! Opens an ALSA PCM directly on the user-picked device (parsed
//! from `alsa/hw:CARD,DEV` → `hw:CARD,DEV`) and pumps interleaved
//! f32 from a SPSC ring buffer to the kernel via blocking
//! `snd_pcm_writei`. `hw:` devices in ALSA already provide
//! exclusive access at the kernel level (`EBUSY` if anyone else
//! holds them) and refuse format conversions — that is the
//! bitperfect guarantee on Linux.
//!
//! Exclusive toggle semantics (mirrors macOS HogMode):
//! - `exclusive=true` opens the raw `hw:` PCM. The user-picked id
//!   is smart-upgraded: `alsa/<plugin>:CARD=X[,DEV=Y]` (e.g.
//!   `front:`/`plughw:`/`sysdefault:`/`dsnoop:`) is rewritten to
//!   `hw:CARD=X,DEV=Y` so we skip dmix/plug → real bitperfect on
//!   the picked card. Ids without a `CARD=` param (`pipewire`,
//!   `pulse`, bare `default`) can never be bitperfect and we
//!   refuse them so the user picks an actual hardware device.
//! - `exclusive=false` opens the device as-is (Pipewire/Pulse
//!   passthrough is fine), with the system mixer doing whatever
//!   conversions it wants.

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug)]
#[allow(dead_code)]
pub enum AlsaError {
    /// `exclusive=true` but the user picked an id we can't upgrade
    /// to a raw `hw:` (no `CARD=` param — `pipewire`, `pulse`,
    /// bare `default`).
    NotBitperfect(String),
    Open(alsa::Error),
    HwParams(alsa::Error),
    Write(alsa::Error),
    UnsupportedFormat,
    DopRequiresExclusive,
}

pub fn play_stream(
    mut consumer: Consumer<f32>,
    sample_rate: f64,
    channels: u32,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: &Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
) -> Result<(), AlsaError> {
    let device_name = resolve_device(audio_device, exclusive)?;
    tracing::info!(device = %device_name, exclusive, "opening ALSA PCM");

    let pcm = PCM::new(&device_name, Direction::Playback, false).map_err(AlsaError::Open)?;

    // Prefer FloatLE — symphonia hands us f32 and writei accepts it
    // natively, no integer conversion. Most consumer DACs accept it;
    // hw: devices that don't (some pro cards locked at S32_LE) need
    // the fallback. Try FloatLE → S32LE → fail.
    let format = pick_format(&pcm, sample_rate, channels).map_err(|e| {
        tracing::error!(?e, "no acceptable hw format negotiated");
        AlsaError::UnsupportedFormat
    })?;
    tracing::info!(format = ?format, "negotiated format");

    // ~85 ms buffer at 48k: 4 periods × 1024 frames. Small enough to
    // stay responsive on stop/seek, large enough to absorb decoder
    // jitter without underrunning a hw: PCM.
    let buffer_frames: u32 = 4096;
    let period_frames: u32 = 1024;
    {
        let hwp = HwParams::any(&pcm).map_err(AlsaError::HwParams)?;
        hwp.set_channels(channels).map_err(AlsaError::HwParams)?;
        hwp.set_rate(sample_rate as u32, ValueOr::Nearest)
            .map_err(AlsaError::HwParams)?;
        hwp.set_format(format).map_err(AlsaError::HwParams)?;
        hwp.set_access(Access::RWInterleaved)
            .map_err(AlsaError::HwParams)?;
        hwp.set_buffer_size_near(buffer_frames as i64)
            .map_err(AlsaError::HwParams)?;
        hwp.set_period_size_near(period_frames as i64, ValueOr::Nearest)
            .map_err(AlsaError::HwParams)?;
        pcm.hw_params(&hwp).map_err(AlsaError::HwParams)?;
    }

    // Prefill ~200 ms before opening the floodgates — same rationale
    // as macOS: tiny safety margin so the very first writei sees a
    // healthy chunk and the DAC's PLL has time to lock onto the
    // chosen sample rate before audio actually starts coming out.
    {
        let target = (sample_rate as usize) * (channels as usize) / 5;
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

    pcm.prepare().map_err(AlsaError::Open)?;

    // Working buffer the size of one period. We pull from rtrb in
    // chunks of period_frames (channels * period_frames samples)
    // and feed snd_pcm_writei. writei blocks until there's room,
    // so the throttling falls out for free — no manual sleep.
    let chunk_samples = (period_frames as usize) * (channels as usize);
    let mut f32_buf: Vec<f32> = vec![0.0; chunk_samples];
    let mut i32_buf: Vec<i32> = if matches!(format, Format::S32LE) {
        vec![0; chunk_samples]
    } else {
        Vec::new()
    };

    loop {
        if cancel.load(Ordering::Acquire) {
            break;
        }

        let avail = consumer.slots();
        if avail == 0 {
            if eof.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        let take = avail.min(chunk_samples);

        // Drain the chunk out of the ring buffer.
        let chunk = match consumer.read_chunk(take) {
            Ok(c) => c,
            Err(_) => {
                thread::sleep(Duration::from_millis(1));
                continue;
            }
        };
        let (s1, s2) = chunk.as_slices();
        f32_buf[..s1.len()].copy_from_slice(s1);
        f32_buf[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
        chunk.commit_all();

        let frames_to_write = take / channels as usize;
        let write_result = match format {
            Format::FloatLE => {
                let io = pcm.io_f32().map_err(AlsaError::Write)?;
                io.writei(&f32_buf[..take])
            }
            Format::S32LE => {
                // Saturating f32 → i32. Symphonia normalises to
                // [-1.0, 1.0] but rounding can drift; clamp before
                // scaling so we never wrap past `i32::MIN/MAX`.
                for (dst, src) in i32_buf[..take].iter_mut().zip(&f32_buf[..take]) {
                    let v = src.clamp(-1.0, 1.0) * 2147483647.0;
                    *dst = v as i32;
                }
                let io = pcm.io_i32().map_err(AlsaError::Write)?;
                io.writei(&i32_buf[..take])
            }
            _ => return Err(AlsaError::UnsupportedFormat),
        };

        match write_result {
            Ok(written) if written == frames_to_write => {}
            Ok(written) => {
                tracing::trace!(written, frames_to_write, "short write");
            }
            Err(e) => {
                // Hand it to ALSA's own recovery first: it knows
                // about EPIPE (underrun) vs ESTRPIPE (suspended)
                // and will `snd_pcm_prepare` for us where
                // appropriate. Only escalate if recovery itself
                // fails.
                tracing::warn!(?e, "writei error, attempting recover");
                pcm.try_recover(e, true).map_err(AlsaError::Write)?;
            }
        }
    }

    // Best-effort drain. If the user hit stop we don't care; if EOF
    // brought us here, this empties the kernel ring so the last few
    // ms actually reach the DAC.
    let _ = pcm.drain();
    Ok(())
}

fn resolve_device(audio_device: Option<&str>, exclusive: bool) -> Result<String, AlsaError> {
    let raw = audio_device.unwrap_or("");
    let alsa_inner = raw.strip_prefix("alsa/");

    if exclusive {
        // `hw:` already → ideal, use as-is.
        // `<plugin>:CARD=X[,DEV=Y]` (front/plughw/sysdefault/dsnoop/…)
        // → upgrade to `hw:CARD=X,DEV=Y` so we open the raw hardware
        // and skip dmix/plug → real bitperfect on the user's DAC.
        // Anything else (Pipewire, Pulse, bare `default`, non-alsa id)
        // can never be bitperfect → refuse so the user picks a real
        // hardware id.
        let Some(rest) = alsa_inner else {
            return Err(AlsaError::NotBitperfect(raw.to_string()));
        };
        if rest.starts_with("hw:") {
            return Ok(rest.to_string());
        }
        if let Some(hw) = upgrade_to_hw(rest) {
            tracing::info!(from = %raw, to = %hw, "upgraded ALSA device id to raw hw: for bitperfect");
            return Ok(hw);
        }
        Err(AlsaError::NotBitperfect(raw.to_string()))
    } else {
        match alsa_inner {
            Some(rest) if !rest.is_empty() => Ok(rest.to_string()),
            _ => Ok("default".to_string()),
        }
    }
}

/// Rewrite an ALSA device id with `CARD=`/`DEV=` hints to the raw
/// `hw:` form. `front:CARD=x20,DEV=0` → `hw:CARD=x20,DEV=0` —
/// bypasses dmix/plug and lets the kernel give us the raw PCM. DEV
/// defaults to `0` when omitted (e.g. `sysdefault:CARD=X`). Returns
/// `None` for ids without a `CARD=` param (Pipewire, Pulse, bare
/// `default`), which can never be bitperfect.
fn upgrade_to_hw(rest: &str) -> Option<String> {
    let (_plugin, params) = rest.split_once(':')?;
    let card = params.split(',').find_map(|p| p.strip_prefix("CARD="))?;
    let dev = params
        .split(',')
        .find_map(|p| p.strip_prefix("DEV="))
        .unwrap_or("0");
    Some(format!("hw:CARD={card},DEV={dev}"))
}

fn pick_format(pcm: &PCM, sample_rate: f64, channels: u32) -> Result<Format, alsa::Error> {
    for fmt in [Format::FloatLE, Format::S32LE] {
        let hwp = HwParams::any(pcm)?;
        if hwp.set_channels(channels).is_err() {
            continue;
        }
        if hwp.set_rate(sample_rate as u32, ValueOr::Nearest).is_err() {
            continue;
        }
        if hwp.set_access(Access::RWInterleaved).is_err() {
            continue;
        }
        if hwp.set_format(fmt).is_ok() {
            return Ok(fmt);
        }
    }
    // Fall back through a generic params probe to surface a real
    // error from ALSA rather than our own enum variant.
    let hwp = HwParams::any(pcm)?;
    hwp.set_channels(channels)?;
    hwp.set_rate(sample_rate as u32, ValueOr::Nearest)?;
    hwp.set_access(Access::RWInterleaved)?;
    hwp.set_format(Format::FloatLE)?;
    Ok(Format::FloatLE)
}

// ============================================================== DSD paths
//
// Two ALSA configurations, picked by the caller based on what the
// hardware advertises:
//
// - **DSD native** (`DSDU32BE` at `dsd_rate/32`). The DAC eats DSD
//   bits directly; nothing alters the bitstream. Maximum fidelity,
//   restricted hardware support.
// - **DoP** (`S32LE` at `dsd_rate/16`). 24-bit PCM with marker
//   bytes; every DAC that accepts 24/176.4 works. Bit-identical to
//   the source DSD bits — the DAC strips the marker and routes.
//
// Both share the same ring buffer element type (`u32`) but the
// interpretation differs: native is "32 DSD bits MSB-first packed
// LE-in-memory"; DoP is "24-bit AlignedHigh PCM sample packed
// LE-in-memory with marker in bits 23..16".

/// Probe whether the chosen ALSA device accepts native DSD playback
/// at the given native frame rate. Opens, asks hw_params for
/// `DSDU32BE` + rate + channels, then closes. Cheap and non-
/// destructive — no playback occurs.
pub fn supports_dsd_native(
    audio_device: Option<&str>,
    exclusive: bool,
    alsa_rate: u32,
    channels: u32,
) -> bool {
    let Ok(device_name) = resolve_device(audio_device, exclusive) else {
        return false;
    };
    let Ok(pcm) = PCM::new(&device_name, Direction::Playback, false) else {
        return false;
    };
    let Ok(hwp) = HwParams::any(&pcm) else {
        return false;
    };
    if hwp.set_channels(channels).is_err() {
        return false;
    }
    if hwp.set_rate(alsa_rate, ValueOr::Nearest).is_err() {
        return false;
    }
    if hwp.set_access(Access::RWInterleaved).is_err() {
        return false;
    }
    hwp.set_format(Format::DSDU32BE).is_ok()
}

/// Selects the per-sample transform applied in the DSD write loop.
/// Native and DoP share the open/hwparams/prefill setup but differ
/// in the typed IO and the bit alignment of the u32 they're handed
/// (see `play_stream_dsd_generic`).
enum DsdMode {
    /// `Format::DSDU32BE`. Write `u32` straight from the ring —
    /// alsa-rs's `io_i32` rejects DSDU32BE in its type/format check,
    /// so we bind `io_u32`.
    Native,
    /// `Format::S32_LE` carrying DoP. The `DopPacker` emits the
    /// 24-bit value AlignedLow (`0x00_MM_DD_DD`); ALSA's 24-in-32
    /// convention is AlignedHigh — the marker has to be in the top
    /// byte where the DAC actually looks — so we shift each sample
    /// left by 8 (`0xMM_DD_DD_00`) before writing.
    Dop,
}

pub fn play_stream_dop(
    consumer: Consumer<u32>,
    pcm_rate: u32,
    channels: u32,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: &Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
) -> Result<(), AlsaError> {
    // DoP at the system mixer would be a lie — the mixer would treat
    // the DoP bytes as garbage PCM. Refuse outright.
    if !exclusive {
        return Err(AlsaError::DopRequiresExclusive);
    }
    play_stream_dsd_generic(
        consumer,
        Format::S32LE,
        pcm_rate,
        channels,
        audio_device,
        exclusive,
        cancel,
        eof,
        DsdMode::Dop,
    )
}

pub fn play_stream_dsd_native(
    consumer: Consumer<u32>,
    alsa_rate: u32,
    channels: u32,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: &Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
) -> Result<(), AlsaError> {
    if !exclusive {
        return Err(AlsaError::DopRequiresExclusive);
    }
    play_stream_dsd_generic(
        consumer,
        Format::DSDU32BE,
        alsa_rate,
        channels,
        audio_device,
        exclusive,
        cancel,
        eof,
        DsdMode::Native,
    )
}

#[allow(clippy::too_many_arguments)]
fn play_stream_dsd_generic(
    mut consumer: Consumer<u32>,
    format: Format,
    rate: u32,
    channels: u32,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: &Arc<AtomicBool>,
    eof: Arc<AtomicBool>,
    mode: DsdMode,
) -> Result<(), AlsaError> {
    let device_name = resolve_device(audio_device, exclusive)?;
    tracing::info!(
        device = %device_name,
        format = ?format,
        rate,
        channels,
        "opening ALSA PCM for DSD"
    );
    let pcm = PCM::new(&device_name, Direction::Playback, false).map_err(AlsaError::Open)?;

    // The buffer geometry matches the PCM path. ~85 ms at the chosen
    // rate is enough headroom for jittery HTTP without dragging
    // start-of-track latency higher than the user would notice.
    let buffer_frames: u32 = 4096;
    let period_frames: u32 = 1024;
    {
        let hwp = HwParams::any(&pcm).map_err(AlsaError::HwParams)?;
        hwp.set_channels(channels).map_err(AlsaError::HwParams)?;
        hwp.set_rate(rate, ValueOr::Nearest)
            .map_err(AlsaError::HwParams)?;
        hwp.set_format(format)
            .map_err(|_| AlsaError::UnsupportedFormat)?;
        hwp.set_access(Access::RWInterleaved)
            .map_err(AlsaError::HwParams)?;
        hwp.set_buffer_size_near(buffer_frames as i64)
            .map_err(AlsaError::HwParams)?;
        hwp.set_period_size_near(period_frames as i64, ValueOr::Nearest)
            .map_err(AlsaError::HwParams)?;
        pcm.hw_params(&hwp).map_err(AlsaError::HwParams)?;
    }

    // Prefill ~200 ms so writei never runs the kernel ring dry on
    // its very first call. Match the PCM path so behaviour is
    // predictable.
    {
        let target = (rate as usize) * (channels as usize) / 5;
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
                tracing::warn!(slots = consumer.slots(), "DSD prefill timeout");
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        tracing::info!(slots = consumer.slots(), "DSD prefill done");
    }

    pcm.prepare().map_err(AlsaError::Open)?;

    let chunk_samples = (period_frames as usize) * (channels as usize);
    let mut u32_buf: Vec<u32> = vec![0; chunk_samples];

    // Two write loops sharing the same ring-buffer drain. They
    // differ in the typed IO (DSDU32BE needs io_u32; alsa-rs rejects
    // io_i32 against DSDU32BE with "Operation not supported") and,
    // for DoP, the AlignedLow → AlignedHigh shift so the marker lands
    // in the top byte the DAC actually inspects.
    match mode {
        DsdMode::Native => {
            let io = pcm.io_u32().map_err(AlsaError::Write)?;
            loop {
                if cancel.load(Ordering::Acquire) {
                    break;
                }
                let avail = consumer.slots();
                if avail == 0 {
                    if eof.load(Ordering::Acquire) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                let take = avail.min(chunk_samples);
                let chunk = match consumer.read_chunk(take) {
                    Ok(c) => c,
                    Err(_) => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                };
                let (s1, s2) = chunk.as_slices();
                u32_buf[..s1.len()].copy_from_slice(s1);
                u32_buf[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
                chunk.commit_all();

                if let Err(e) = io.writei(&u32_buf[..take]) {
                    tracing::warn!(?e, "DSD writei error, attempting recover");
                    pcm.try_recover(e, true).map_err(AlsaError::Write)?;
                }
            }
        }
        DsdMode::Dop => {
            let io = pcm.io_i32().map_err(AlsaError::Write)?;
            let mut i32_buf: Vec<i32> = vec![0; chunk_samples];
            loop {
                if cancel.load(Ordering::Acquire) {
                    break;
                }
                let avail = consumer.slots();
                if avail == 0 {
                    if eof.load(Ordering::Acquire) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                let take = avail.min(chunk_samples);
                let chunk = match consumer.read_chunk(take) {
                    Ok(c) => c,
                    Err(_) => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                };
                let (s1, s2) = chunk.as_slices();
                u32_buf[..s1.len()].copy_from_slice(s1);
                u32_buf[s1.len()..s1.len() + s2.len()].copy_from_slice(s2);
                chunk.commit_all();

                // AlignedLow → AlignedHigh: marker moves from bits
                // 23..16 to bits 31..24 so the DAC's DoP detector
                // sees it in the MSB of the 32-bit S32_LE sample.
                for (dst, src) in i32_buf[..take].iter_mut().zip(&u32_buf[..take]) {
                    *dst = (src << 8) as i32;
                }

                if let Err(e) = io.writei(&i32_buf[..take]) {
                    tracing::warn!(?e, "DSD writei error, attempting recover");
                    pcm.try_recover(e, true).map_err(AlsaError::Write)?;
                }
            }
        }
    }

    let _ = pcm.drain();
    Ok(())
}
