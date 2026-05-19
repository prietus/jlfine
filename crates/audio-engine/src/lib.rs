//! Bitperfect audio playback engine.
//!
//! Streams audio from a URL (typically Jellyfin `static=true` lossless),
//! decodes with `symphonia`, and pumps PCM through a SPSC ring buffer
//! into the platform's bitperfect output (CoreAudio HAL HogMode on
//! macOS; ALSA `hw:` planned for Linux). The decoder runs on its own
//! thread at decode speed; the IOProc consumes the buffer in real time.
//! See the `project-jelly-rs` memory for the requirements behind this
//! design.
//!
//! v1 limitations: no gapless, no seek/pause, no DSD.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("download failed: {0}")]
    Download(#[from] reqwest::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("audio backend: {0}")]
    Backend(String),
    #[error("audio playback is unsupported on this platform")]
    Unsupported,
}

pub type Result<T> = std::result::Result<T, Error>;

enum Cmd {
    Play {
        url: String,
        container: Option<String>,
        audio_device: Option<String>,
        exclusive: bool,
    },
    Stop,
}

/// Cheap to clone — the actual worker thread lives behind the channel.
#[derive(Clone)]
pub struct AudioEngine {
    tx: mpsc::Sender<Cmd>,
}

impl AudioEngine {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Cmd>();
        thread::Builder::new()
            .name("audio-engine".into())
            .spawn(move || worker(rx))
            .expect("spawn audio-engine thread");
        Self { tx }
    }

    /// Replace the current playback. Returns immediately; the worker
    /// thread cancels any in-flight track, waits for HogMode release,
    /// then opens the new HTTP stream and starts decode + playback.
    ///
    /// `container` is a hint for symphonia's format probe (e.g.
    /// `Some("flac")`, `Some("m4a")`).
    ///
    /// `audio_device` is the mpv-style id (`coreaudio/<UID>` on
    /// macOS, `alsa/hw:CARD,DEV` on Linux). `None` falls back to the
    /// platform default output.
    ///
    /// `exclusive=true` is the bitperfect path: HogMode + sample-rate
    /// switching on macOS. `false` opens the device without taking
    /// exclusive control or touching the nominal sample rate — the
    /// system mixer may resample, so this is *not* bitperfect.
    pub fn play_track(
        &self,
        url: impl Into<String>,
        container: Option<String>,
        audio_device: Option<String>,
        exclusive: bool,
    ) {
        let _ = self.tx.send(Cmd::Play {
            url: url.into(),
            container,
            audio_device,
            exclusive,
        });
    }

    pub fn stop(&self) {
        let _ = self.tx.send(Cmd::Stop);
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn worker(rx: mpsc::Receiver<Cmd>) {
    let mut current_cancel: Option<Arc<AtomicBool>> = None;
    let mut current_handle: Option<JoinHandle<()>> = None;

    while let Ok(cmd) = rx.recv() {
        // Stop the previous playback FIRST so HogMode is released
        // before the new acquire attempt. Joining here is essential —
        // otherwise the new thread might race the old one for the
        // device.
        if let Some(c) = current_cancel.take() {
            c.store(true, Ordering::SeqCst);
        }
        if let Some(h) = current_handle.take() {
            let _ = h.join();
        }

        match cmd {
            Cmd::Stop => continue,
            Cmd::Play {
                url,
                container,
                audio_device,
                exclusive,
            } => {
                let cancel = Arc::new(AtomicBool::new(false));
                let cancel_clone = cancel.clone();
                let handle = thread::Builder::new()
                    .name("audio-playback".into())
                    .spawn(move || {
                        if let Err(e) = play_blocking(
                            &url,
                            container.as_deref(),
                            audio_device.as_deref(),
                            exclusive,
                            cancel_clone,
                        ) {
                            tracing::error!(?e, "audio playback failed");
                        }
                    })
                    .expect("spawn playback thread");
                current_cancel = Some(cancel);
                current_handle = Some(handle);
            }
        }
    }
}

fn play_blocking(
    url: &str,
    container: Option<&str>,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let ext = container.unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        // DSD goes through a separate decode path: symphonia doesn't
        // decode DSD, and the backend has to negotiate either DoP
        // (24-bit PCM with marker bytes) or a native DSD format on
        // Linux. PCM/lossless containers all flow through symphonia.
        "dsf" | "dff" => play_dsd_blocking(url, &ext, audio_device, exclusive, cancel),
        _ => play_pcm_blocking(url, container, audio_device, exclusive, cancel),
    }
}

#[cfg(target_os = "macos")]
fn play_pcm_blocking(
    url: &str,
    container: Option<&str>,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    use symphonia::core::codecs::{Decoder, DecoderOptions};
    use symphonia::core::formats::{FormatOptions, FormatReader};
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    tracing::info!(%url, ?container, "opening stream");
    let resp = reqwest::blocking::get(url)?.error_for_status()?;
    let stream = HttpStream::new(resp);
    let mss = MediaSourceStream::new(Box::new(stream), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = container {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::Decode(e.to_string()))?;
    let format: Box<dyn FormatReader> = probed.format;

    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .ok_or_else(|| Error::Decode("no default track".into()))?;
        (track.id, track.codec_params.clone())
    };
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| Error::Decode("missing sample rate".into()))?;
    let channels = codec_params
        .channels
        .ok_or_else(|| Error::Decode("missing channels".into()))?
        .count() as u32;
    tracing::info!(sample_rate, channels, "stream format probed");

    let decoder: Box<dyn Decoder> = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| Error::Decode(e.to_string()))?;

    // 2 seconds of interleaved f32 at the file's sample rate.
    let capacity = (sample_rate as usize) * (channels as usize) * 2;
    let (producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);

    let eof = Arc::new(AtomicBool::new(false));
    let eof_dec = eof.clone();
    let cancel_dec = cancel.clone();
    let decoder_handle = thread::Builder::new()
        .name("audio-decoder".into())
        .spawn(move || {
            let mut producer = producer;
            decode_loop(format, decoder, track_id, &mut producer, &cancel_dec);
            eof_dec.store(true, Ordering::Release);
            tracing::info!("decoder finished");
        })
        .expect("spawn decoder thread");

    let result = mac::play_stream(
        consumer,
        sample_rate as f64,
        channels,
        audio_device,
        exclusive,
        &cancel,
        eof.clone(),
    );

    // If HAL bails out first, the decoder may still be blocked in a
    // socket read — flip cancel so the next read break causes it to
    // exit, then reap the thread.
    cancel.store(true, Ordering::SeqCst);
    let _ = decoder_handle.join();
    result.map_err(|e| Error::Backend(format!("{e:?}")))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn play_pcm_blocking(
    url: &str,
    container: Option<&str>,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    use symphonia::core::codecs::{Decoder, DecoderOptions};
    use symphonia::core::formats::{FormatOptions, FormatReader};
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    tracing::info!(%url, ?container, "opening stream");
    let resp = reqwest::blocking::get(url)?.error_for_status()?;
    let stream = HttpStream::new(resp);
    let mss = MediaSourceStream::new(Box::new(stream), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = container {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::Decode(e.to_string()))?;
    let format: Box<dyn FormatReader> = probed.format;

    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .ok_or_else(|| Error::Decode("no default track".into()))?;
        (track.id, track.codec_params.clone())
    };
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| Error::Decode("missing sample rate".into()))?;
    let channels = codec_params
        .channels
        .ok_or_else(|| Error::Decode("missing channels".into()))?
        .count() as u32;
    tracing::info!(sample_rate, channels, "stream format probed");

    let decoder: Box<dyn Decoder> = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| Error::Decode(e.to_string()))?;

    let capacity = (sample_rate as usize) * (channels as usize) * 2;
    let (producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);

    let eof = Arc::new(AtomicBool::new(false));
    let eof_dec = eof.clone();
    let cancel_dec = cancel.clone();
    let decoder_handle = thread::Builder::new()
        .name("audio-decoder".into())
        .spawn(move || {
            let mut producer = producer;
            decode_loop(format, decoder, track_id, &mut producer, &cancel_dec);
            eof_dec.store(true, Ordering::Release);
            tracing::info!("decoder finished");
        })
        .expect("spawn decoder thread");

    let result = linux::play_stream(
        consumer,
        sample_rate as f64,
        channels,
        audio_device,
        exclusive,
        &cancel,
        eof.clone(),
    );

    cancel.store(true, Ordering::SeqCst);
    let _ = decoder_handle.join();
    result.map_err(|e| Error::Backend(format!("{e:?}")))?;
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn play_pcm_blocking(
    _url: &str,
    _container: Option<&str>,
    _audio_device: Option<&str>,
    _exclusive: bool,
    _cancel: Arc<AtomicBool>,
) -> Result<()> {
    Err(Error::Unsupported)
}

#[cfg(target_os = "macos")]
fn play_dsd_blocking(
    url: &str,
    ext: &str,
    audio_device: Option<&str>,
    exclusive: bool,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    if ext != "dsf" {
        return Err(Error::Backend(format!("unsupported DSD container: {ext}")));
    }

    tracing::info!(%url, "opening DSF stream");
    let resp = reqwest::blocking::get(url)?.error_for_status()?;
    let stream = HttpStream::new(resp);
    let reader =
        dsf::DsfReader::new(stream).map_err(|e| Error::Decode(format!("dsf header: {e}")))?;

    let dsd_rate = reader.header().sampling_frequency;
    let channels = reader.header().channels;
    let pcm_rate = dop::DopPacker::pcm_sample_rate(dsd_rate);
    tracing::info!(dsd_rate, pcm_rate, channels, "DSF -> DoP");

    // 2 seconds of u32 DoP samples at the PCM rate.
    let capacity = (pcm_rate as usize) * (channels as usize) * 2;
    let (producer, consumer) = rtrb::RingBuffer::<u32>::new(capacity);

    let eof = Arc::new(AtomicBool::new(false));
    let eof_dec = eof.clone();
    let cancel_dec = cancel.clone();
    let decoder_handle = thread::Builder::new()
        .name("dsd-packer".into())
        .spawn(move || {
            let mut producer = producer;
            pack_dsf_to_dop(reader, channels as usize, &mut producer, &cancel_dec);
            eof_dec.store(true, Ordering::Release);
            tracing::info!("DSD packer finished");
        })
        .expect("spawn DSD packer thread");

    let result = mac::play_stream_dop(
        consumer,
        pcm_rate as f64,
        channels,
        audio_device,
        exclusive,
        &cancel,
        eof.clone(),
    );
    cancel.store(true, Ordering::SeqCst);
    let _ = decoder_handle.join();
    result.map_err(|e| Error::Backend(format!("{e:?}")))?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn play_dsd_blocking(
    _url: &str,
    _ext: &str,
    _audio_device: Option<&str>,
    _exclusive: bool,
    _cancel: Arc<AtomicBool>,
) -> Result<()> {
    Err(Error::Backend(
        "DSD playback not yet implemented on this platform".into(),
    ))
}

/// Pull DSD bytes from the DSF reader, hand pairs of bytes per
/// channel to the DoP packer, push the resulting 24-bit-in-32-bit
/// PCM samples into the ring buffer. Two DSD bytes per channel per
/// PCM frame: that's where the DSD-rate / 16 PCM rate falls out.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn pack_dsf_to_dop<R: std::io::Read>(
    mut reader: dsf::DsfReader<R>,
    channels: usize,
    producer: &mut rtrb::Producer<u32>,
    cancel: &AtomicBool,
) {
    let mut packer = dop::DopPacker::new(channels);
    let mut dsd_pair = vec![0u8; channels * 2]; // [ch0_hi, ch0_lo, ch1_hi, ch1_lo, ...]
    let mut byte_per_ch = vec![0u8; channels];
    let mut out = vec![0u32; channels];

    loop {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        // Read 2 DSD bytes per channel: the older byte first, then
        // the newer. EOF before completing the pair = end of stream.
        match reader.next_byte_per_channel(&mut byte_per_ch) {
            Ok(true) => {
                for ch in 0..channels {
                    dsd_pair[ch * 2] = byte_per_ch[ch];
                }
            }
            Ok(false) => return,
            Err(e) => {
                tracing::warn!(?e, "DSF read error, ending");
                return;
            }
        }
        match reader.next_byte_per_channel(&mut byte_per_ch) {
            Ok(true) => {
                for ch in 0..channels {
                    dsd_pair[ch * 2 + 1] = byte_per_ch[ch];
                }
            }
            Ok(false) => return,
            Err(e) => {
                tracing::warn!(?e, "DSF read error, ending");
                return;
            }
        }
        packer.pack_frame(&dsd_pair, &mut out);
        push_u32_all(producer, &out, cancel);
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn push_u32_all(producer: &mut rtrb::Producer<u32>, mut samples: &[u32], cancel: &AtomicBool) {
    use std::time::Duration;
    while !samples.is_empty() {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        let avail = producer.slots();
        if avail == 0 {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        let n = avail.min(samples.len());
        match producer.write_chunk_uninit(n) {
            Ok(mut chunk) => {
                let (s1, s2) = chunk.as_mut_slices();
                let (a, b) = samples[..n].split_at(s1.len());
                for (slot, val) in s1.iter_mut().zip(a) {
                    slot.write(*val);
                }
                for (slot, val) in s2.iter_mut().zip(b) {
                    slot.write(*val);
                }
                unsafe { chunk.commit_all() };
                samples = &samples[n..];
            }
            Err(_) => thread::sleep(Duration::from_millis(2)),
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn decode_loop(
    mut format: Box<dyn symphonia::core::formats::FormatReader>,
    mut decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    producer: &mut rtrb::Producer<f32>,
    cancel: &AtomicBool,
) {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::errors::Error as SymError;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return,
            Err(SymError::ResetRequired) => {
                tracing::warn!("decoder reset required, ending");
                return;
            }
            Err(e) => {
                tracing::warn!(?e, "next_packet error, ending");
                return;
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                if sample_buf.is_none() {
                    let spec = *audio_buf.spec();
                    let cap = audio_buf.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(cap, spec));
                }
                let buf = sample_buf.as_mut().unwrap();
                buf.copy_interleaved_ref(audio_buf);
                push_all(producer, buf.samples(), cancel);
            }
            Err(SymError::DecodeError(msg)) => {
                tracing::warn!(msg, "decode error, skipping packet");
                continue;
            }
            Err(e) => {
                tracing::warn!(?e, "decoder fatal, ending");
                return;
            }
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn push_all(producer: &mut rtrb::Producer<f32>, mut samples: &[f32], cancel: &AtomicBool) {
    use std::time::Duration;
    while !samples.is_empty() {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        let avail = producer.slots();
        if avail == 0 {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        let n = avail.min(samples.len());
        match producer.write_chunk_uninit(n) {
            Ok(mut chunk) => {
                let (s1, s2) = chunk.as_mut_slices();
                let (a, b) = samples[..n].split_at(s1.len());
                for (slot, val) in s1.iter_mut().zip(a) {
                    slot.write(*val);
                }
                for (slot, val) in s2.iter_mut().zip(b) {
                    slot.write(*val);
                }
                unsafe { chunk.commit_all() };
                samples = &samples[n..];
            }
            Err(_) => thread::sleep(Duration::from_millis(2)),
        }
    }
}

/// Adapter so `reqwest::blocking::Response` can back symphonia's
/// `MediaSourceStream`. We can't seek a one-shot HTTP body, so
/// `is_seekable` is false and `seek` errors. FLAC / WAV probe fine
/// without seek; ALAC-in-mp4 needs a moov atom at the head of the
/// file (Jellyfin static files generally do).
struct HttpStream {
    inner: std::io::BufReader<reqwest::blocking::Response>,
}

impl HttpStream {
    fn new(resp: reqwest::blocking::Response) -> Self {
        Self {
            inner: std::io::BufReader::with_capacity(64 * 1024, resp),
        }
    }
}

impl std::io::Read for HttpStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl std::io::Seek for HttpStream {
    fn seek(&mut self, _pos: std::io::SeekFrom) -> std::io::Result<u64> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "http stream is not seekable",
        ))
    }
}

impl symphonia::core::io::MediaSource for HttpStream {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

#[cfg(target_os = "macos")]
mod mac;

#[cfg(target_os = "linux")]
mod linux;

mod dop;
mod dsf;
