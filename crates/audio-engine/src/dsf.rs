//! DSF (Sony Direct Stream File) reader for non-seekable streams.
//!
//! DSF lays audio out as fixed-size blocks per channel: for stereo at
//! the typical block size of 4096, the file holds 4096 bytes of L,
//! then 4096 bytes of R, then 4096 bytes of L, … We rebuild the
//! per-channel byte stream into MSB-first-in-time DSD bytes so the
//! caller (the DoP packer) can pull one byte per channel at a time
//! and pack 16 DSD bits per channel into each PCM sample.
//!
//! "Bit-per-sample" in the DSF spec controls intra-byte bit order:
//! `1` means LSB is the oldest sample, `8` means MSB is the oldest.
//! Almost every DSF file in the wild is LSB-first; we normalise by
//! reversing each byte's bits at read time so downstream code only
//! has to deal with MSB-first.
//!
//! Last block padding: the file's declared `sample_count` (samples
//! per channel, in bits) usually does not align to `block_size * 8`,
//! so the final block is zero-padded. We truncate to the declared
//! length so the DAC never gets fed silence-coded-as-DSD past EOF.

use std::io::{self, Read};

const HEADER_ID: &[u8; 4] = b"DSD ";
const FMT_ID: &[u8; 4] = b"fmt ";
const DATA_ID: &[u8; 4] = b"data";

const FMT_VERSION: u32 = 1;
const FMT_FORMAT_ID_DSD_RAW: u32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("not a DSF stream (bad magic {0:?})")]
    BadMagic([u8; 4]),
    #[error("expected chunk {expected:?}, got {got:?}")]
    UnexpectedChunk {
        expected: &'static [u8; 4],
        got: [u8; 4],
    },
    #[error("unsupported DSF: {0}")]
    Unsupported(&'static str),
    #[error("invalid DSF field: {0}")]
    Invalid(&'static str),
}

/// Parsed `fmt ` chunk.
#[derive(Debug, Clone, Copy)]
pub struct DsfHeader {
    pub channels: u32,
    /// DSD sample rate, in DSD bits per second per channel
    /// (e.g. 2_822_400 for DSD64, 5_644_800 for DSD128).
    pub sampling_frequency: u32,
    /// Total samples per channel (in DSD bits), used to truncate the
    /// padding from the last block.
    pub sample_count: u64,
    /// Per-channel block size in bytes (almost always 4096).
    pub block_size: u32,
}

impl DsfHeader {
    /// Total DSD bytes per channel after truncation.
    pub fn bytes_per_channel(&self) -> u64 {
        self.sample_count.div_ceil(8)
    }
}

/// Streaming DSF reader.
///
/// Each `next_byte_per_channel` call yields one DSD byte per channel
/// in channel order, with bits in MSB-first time order. Returns
/// `Ok(false)` at end of stream.
pub struct DsfReader<R: Read> {
    inner: R,
    header: DsfHeader,
    /// One scratch block per channel. Refilled together because the
    /// file groups blocks by channel: ch0 block, ch1 block, ch0, ch1…
    channel_blocks: Vec<Vec<u8>>,
    /// Byte index inside the current block, shared across channels
    /// since both buffers are the same length.
    pos_in_block: usize,
    /// Bytes already yielded (per channel). When this hits
    /// `bytes_per_channel()` we stop, even if the padded block still
    /// has bytes left.
    bytes_yielded_per_channel: u64,
    /// Captured at construction so we don't fold bytes_per_channel
    /// into the hot loop.
    cap_per_channel: u64,
    /// True if the file's bit-per-sample was 1 (LSB-first within each
    /// byte). We reverse on read so the public API is always MSB-first.
    lsb_first: bool,
}

impl<R: Read> DsfReader<R> {
    pub fn new(mut inner: R) -> Result<Self, Error> {
        let (_dsd_size, _total_file_size, _meta_offset) = read_dsd_chunk(&mut inner)?;
        let (header, lsb_first) = read_fmt_chunk(&mut inner)?;
        read_data_chunk_header(&mut inner)?;

        let cap_per_channel = header.bytes_per_channel();
        let block_bytes = header.block_size as usize;
        let channel_blocks = (0..header.channels as usize)
            .map(|_| vec![0u8; block_bytes])
            .collect();

        let mut me = Self {
            inner,
            header,
            channel_blocks,
            pos_in_block: block_bytes, // force refill on first read
            bytes_yielded_per_channel: 0,
            cap_per_channel,
            lsb_first,
        };
        me.refill()?;
        Ok(me)
    }

    pub fn header(&self) -> &DsfHeader {
        &self.header
    }

    /// Read one DSD byte per channel into `out` (size = channels).
    /// Returns `Ok(false)` at EOF.
    pub fn next_byte_per_channel(&mut self, out: &mut [u8]) -> Result<bool, Error> {
        if out.len() != self.header.channels as usize {
            return Err(Error::Invalid("output slice size != channel count"));
        }
        if self.bytes_yielded_per_channel >= self.cap_per_channel {
            return Ok(false);
        }
        if self.pos_in_block >= self.header.block_size as usize {
            self.refill()?;
            // refill can short-read at EOF; if so, we're done.
            if self.bytes_yielded_per_channel >= self.cap_per_channel {
                return Ok(false);
            }
        }
        for (ch, slot) in out.iter_mut().enumerate() {
            let raw = self.channel_blocks[ch][self.pos_in_block];
            *slot = if self.lsb_first {
                raw.reverse_bits()
            } else {
                raw
            };
        }
        self.pos_in_block += 1;
        self.bytes_yielded_per_channel += 1;
        Ok(true)
    }

    fn refill(&mut self) -> Result<(), Error> {
        // Each block is `block_size` bytes per channel, stored in
        // channel order. A short read here = EOF (last block may be
        // partial if the writer didn't pad; we accept that).
        for ch in 0..self.header.channels as usize {
            let block = &mut self.channel_blocks[ch];
            let mut filled = 0;
            while filled < block.len() {
                match self.inner.read(&mut block[filled..]) {
                    Ok(0) => {
                        // Truncate the block to what we got and let
                        // bytes_yielded_per_channel cap the output.
                        block.truncate(filled);
                        break;
                    }
                    Ok(n) => filled += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::Io(e)),
                }
            }
        }
        self.pos_in_block = 0;
        Ok(())
    }
}

// ----------------------------------------------------------- chunk readers

fn read_dsd_chunk<R: Read>(r: &mut R) -> Result<(u64, u64, u64), Error> {
    let mut id = [0u8; 4];
    r.read_exact(&mut id)?;
    if &id != HEADER_ID {
        return Err(Error::BadMagic(id));
    }
    let size = read_u64_le(r)?;
    let total = read_u64_le(r)?;
    let meta = read_u64_le(r)?;
    if size != 28 {
        return Err(Error::Invalid("DSD chunk size != 28"));
    }
    Ok((size, total, meta))
}

fn read_fmt_chunk<R: Read>(r: &mut R) -> Result<(DsfHeader, bool), Error> {
    let mut id = [0u8; 4];
    r.read_exact(&mut id)?;
    if &id != FMT_ID {
        return Err(Error::UnexpectedChunk {
            expected: FMT_ID,
            got: id,
        });
    }
    let size = read_u64_le(r)?;
    if size != 52 {
        return Err(Error::Invalid("fmt chunk size != 52"));
    }
    let version = read_u32_le(r)?;
    if version != FMT_VERSION {
        return Err(Error::Unsupported("fmt version != 1"));
    }
    let format_id = read_u32_le(r)?;
    if format_id != FMT_FORMAT_ID_DSD_RAW {
        return Err(Error::Unsupported("format id != 0 (DSD raw)"));
    }
    let _channel_type = read_u32_le(r)?;
    let channels = read_u32_le(r)?;
    if !(1..=6).contains(&channels) {
        return Err(Error::Invalid("channel count out of range"));
    }
    let sampling_frequency = read_u32_le(r)?;
    if !matches!(
        sampling_frequency,
        2_822_400 | 5_644_800 | 11_289_600 | 22_579_200
    ) {
        return Err(Error::Unsupported("unexpected DSD sampling frequency"));
    }
    let bits_per_sample = read_u32_le(r)?;
    let lsb_first = match bits_per_sample {
        1 => true,
        8 => false,
        _ => return Err(Error::Invalid("bits per sample must be 1 or 8")),
    };
    let sample_count = read_u64_le(r)?;
    let block_size = read_u32_le(r)?;
    if block_size == 0 || block_size > 1 << 16 {
        return Err(Error::Invalid("implausible block size"));
    }
    let _reserved = read_u32_le(r)?;

    Ok((
        DsfHeader {
            channels,
            sampling_frequency,
            sample_count,
            block_size,
        },
        lsb_first,
    ))
}

fn read_data_chunk_header<R: Read>(r: &mut R) -> Result<u64, Error> {
    let mut id = [0u8; 4];
    r.read_exact(&mut id)?;
    if &id != DATA_ID {
        return Err(Error::UnexpectedChunk {
            expected: DATA_ID,
            got: id,
        });
    }
    let size = read_u64_le(r)?;
    Ok(size)
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid DSF buffer with 2 channels, DSD64, the
    /// given per-channel sample count (in DSD bits), block size, and
    /// payload. Payload layout matches the file: `block_size` bytes
    /// of ch0, then ch1, repeating.
    fn build_dsf(
        channels: u32,
        sampling_frequency: u32,
        sample_count: u64,
        block_size: u32,
        bits_per_sample: u32,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // DSD chunk
        buf.extend_from_slice(HEADER_ID);
        buf.extend_from_slice(&28u64.to_le_bytes());
        let total = 28u64 + 52 + 12 + payload.len() as u64;
        buf.extend_from_slice(&total.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // no metadata
        // fmt chunk
        buf.extend_from_slice(FMT_ID);
        buf.extend_from_slice(&52u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u32.to_le_bytes()); // format id
        buf.extend_from_slice(&2u32.to_le_bytes()); // channel type stereo
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sampling_frequency.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(&sample_count.to_le_bytes());
        buf.extend_from_slice(&block_size.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        // data chunk
        buf.extend_from_slice(DATA_ID);
        let data_size = 12u64 + payload.len() as u64;
        buf.extend_from_slice(&data_size.to_le_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn rejects_non_dsf_magic() {
        let buf = b"RIFF....".to_vec();
        match DsfReader::new(&buf[..]) {
            Err(Error::BadMagic(_)) => {}
            other => panic!("expected BadMagic, got {:?}", other.err()),
        }
    }

    #[test]
    fn parses_header_fields() {
        // 2 channels, DSD64, 8 samples per channel (=1 byte), block
        // size 4 just to keep the payload small.
        let payload = vec![0xAA, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00];
        let buf = build_dsf(2, 2_822_400, 8, 4, 8, &payload);
        let r = DsfReader::new(&buf[..]).unwrap();
        let h = r.header();
        assert_eq!(h.channels, 2);
        assert_eq!(h.sampling_frequency, 2_822_400);
        assert_eq!(h.sample_count, 8);
        assert_eq!(h.block_size, 4);
        assert_eq!(h.bytes_per_channel(), 1);
    }

    #[test]
    fn yields_bytes_in_channel_order_msb_first_when_bit_per_sample_is_8() {
        // block_size=2, 16 samples/ch = 2 bytes/ch; one refill of
        // both channels, then EOF.
        let payload = vec![
            // ch0 block: A1, A2
            0xA1, 0xA2, // ch1 block: B1, B2
            0xB1, 0xB2,
        ];
        let buf = build_dsf(2, 2_822_400, 16, 2, 8, &payload);
        let mut r = DsfReader::new(&buf[..]).unwrap();
        let mut out = [0u8; 2];

        assert!(r.next_byte_per_channel(&mut out).unwrap());
        assert_eq!(out, [0xA1, 0xB1]);
        assert!(r.next_byte_per_channel(&mut out).unwrap());
        assert_eq!(out, [0xA2, 0xB2]);
        assert!(!r.next_byte_per_channel(&mut out).unwrap());
    }

    #[test]
    fn reverses_bits_when_bit_per_sample_is_1() {
        // 0b1000_0001 reversed = 0b1000_0001 (palindrome) — useless
        // for the test. Use 0b1010_0000 → 0b0000_0101.
        let payload = vec![
            // ch0 block: one byte, padded to block_size=2
            0b1010_0000,
            0x00,
            // ch1 block
            0b0000_1111,
            0x00,
        ];
        let buf = build_dsf(2, 2_822_400, 8, 2, 1, &payload);
        let mut r = DsfReader::new(&buf[..]).unwrap();
        let mut out = [0u8; 2];

        assert!(r.next_byte_per_channel(&mut out).unwrap());
        assert_eq!(out, [0b0000_0101, 0b1111_0000]);
        // sample_count = 8 means 1 byte per channel; the second byte
        // of each block is padding and must be skipped.
        assert!(!r.next_byte_per_channel(&mut out).unwrap());
    }

    #[test]
    fn refills_multiple_blocks() {
        // block_size=1, 24 samples/ch = 3 bytes/ch → 3 refills.
        let payload = vec![
            0xA1, 0xB1, // block 0
            0xA2, 0xB2, // block 1
            0xA3, 0xB3, // block 2
        ];
        let buf = build_dsf(2, 2_822_400, 24, 1, 8, &payload);
        let mut r = DsfReader::new(&buf[..]).unwrap();
        let mut out = [0u8; 2];
        let mut got = Vec::new();
        while r.next_byte_per_channel(&mut out).unwrap() {
            got.extend_from_slice(&out);
        }
        assert_eq!(got, vec![0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]);
    }

    #[test]
    fn truncates_padded_last_block() {
        // 12 samples/ch = 2 bytes/ch (12 bits rounded up to 16 = 2
        // bytes), block_size=4 → one block per channel, but only the
        // first 2 bytes of each are real audio.
        let payload = vec![
            0xA1, 0xA2, 0x00, 0x00, // ch0 (last 2 bytes padding)
            0xB1, 0xB2, 0x00, 0x00, // ch1
        ];
        let buf = build_dsf(2, 2_822_400, 12, 4, 8, &payload);
        let mut r = DsfReader::new(&buf[..]).unwrap();
        let mut out = [0u8; 2];
        let mut count = 0;
        while r.next_byte_per_channel(&mut out).unwrap() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn rejects_unsupported_sampling_frequency() {
        let payload = vec![0u8; 4];
        let buf = build_dsf(2, 1_234_567, 8, 2, 8, &payload);
        match DsfReader::new(&buf[..]) {
            Err(Error::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {:?}", other.err()),
        }
    }
}
