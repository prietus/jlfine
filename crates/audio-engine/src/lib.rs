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
    /// `container` is a hint for symphonia's format probe (e.g.
    /// `Some("flac")`, `Some("m4a")`).
    pub fn play_track(&self, url: impl Into<String>, container: Option<String>) {
        let _ = self.tx.send(Cmd::Play {
            url: url.into(),
            container,
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
            Cmd::Play { url, container } => {
                let cancel = Arc::new(AtomicBool::new(false));
                let cancel_clone = cancel.clone();
                let handle = thread::Builder::new()
                    .name("audio-playback".into())
                    .spawn(move || {
                        if let Err(e) = play_blocking(&url, container.as_deref(), cancel_clone) {
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

#[cfg(target_os = "macos")]
fn play_blocking(url: &str, container: Option<&str>, cancel: Arc<AtomicBool>) -> Result<()> {
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

    let result = mac::play_stream(consumer, sample_rate as f64, channels, &cancel, eof.clone());

    // If HAL bails out first, the decoder may still be blocked in a
    // socket read — flip cancel so the next read break causes it to
    // exit, then reap the thread.
    cancel.store(true, Ordering::SeqCst);
    let _ = decoder_handle.join();
    result.map_err(|e| Error::Backend(format!("{e:?}")))?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn play_blocking(_url: &str, _container: Option<&str>, _cancel: Arc<AtomicBool>) -> Result<()> {
    Err(Error::Unsupported)
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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
