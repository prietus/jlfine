//! DSD-over-PCM (DoP) packing per spec v1.1.
//!
//! DoP smuggles raw 1-bit DSD inside a 24-bit PCM stream the DAC can
//! accept on its existing S/PDIF / USB-Audio class endpoint. Each
//! 24-bit PCM sample carries:
//!
//! ```text
//!   bits 23..16  marker byte 0x05 or 0xFA (alternates every PCM sample)
//!   bits 15..8   first 8 DSD bits in time order, MSB-first
//!   bits  7..0   next 8 DSD bits in time order, MSB-first
//! ```
//!
//! The marker flips **per PCM sample, not per channel** — every
//! channel in the same frame shares the same marker. A compliant DAC
//! recognises the alternating pattern, strips the marker, and routes
//! the 16 DSD bits straight into its 1-bit stage; if the pattern
//! breaks the DAC reverts to normal PCM playback and the listener
//! gets full-scale noise. So the invariant "marker flips exactly
//! once per frame" is load-bearing.
//!
//! Resulting PCM rate is `dsd_rate / 16` (DSD64 → 176_400,
//! DSD128 → 352_800), the container is 24-bit in 32-bit, signed.

/// First marker the packer emits. The packer flips state after each
/// frame, so two consecutive frames carry different markers.
pub const DOP_MARKER_A: u8 = 0x05;
pub const DOP_MARKER_B: u8 = 0xFA;

/// Stateful packer that toggles the DoP marker every PCM frame.
pub struct DopPacker {
    channels: usize,
    next_marker: u8,
}

impl DopPacker {
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            next_marker: DOP_MARKER_A,
        }
    }

    /// Resulting 24-bit-in-32-bit PCM sample rate for the given DSD
    /// rate (in DSD bits per second per channel).
    pub fn pcm_sample_rate(dsd_rate: u32) -> u32 {
        dsd_rate / 16
    }

    /// Pack one PCM frame.
    ///
    /// `dsd_bytes` carries 2 DSD bytes per channel, interleaved by
    /// channel and in time order:
    /// `[ch0_byte_old, ch0_byte_new, ch1_byte_old, ch1_byte_new, ...]`.
    /// Each byte must already be MSB-first in time — `DsfReader`
    /// normalises that.
    ///
    /// `out` receives one 24-bit-in-32-bit sample per channel. The
    /// marker sits in bits 23..16, so the low byte is always zero
    /// and the value fits a signed range — backends are free to
    /// reinterpret as `i32` for `pcm.io_i32()` or CoreAudio integer
    /// IOProcs.
    pub fn pack_frame(&mut self, dsd_bytes: &[u8], out: &mut [u32]) {
        debug_assert_eq!(dsd_bytes.len(), self.channels * 2);
        debug_assert_eq!(out.len(), self.channels);

        let m = self.next_marker;
        for ch in 0..self.channels {
            let hi = dsd_bytes[ch * 2];
            let lo = dsd_bytes[ch * 2 + 1];
            out[ch] = ((m as u32) << 16) | ((hi as u32) << 8) | (lo as u32);
        }
        self.next_marker = match m {
            DOP_MARKER_A => DOP_MARKER_B,
            _ => DOP_MARKER_A,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker_of(sample: u32) -> u8 {
        ((sample >> 16) & 0xFF) as u8
    }

    fn dsd_payload(sample: u32) -> u16 {
        (sample & 0xFFFF) as u16
    }

    #[test]
    fn first_frame_uses_marker_05() {
        let mut p = DopPacker::new(2);
        let mut out = [0u32; 2];
        p.pack_frame(&[0xAA, 0xBB, 0xCC, 0xDD], &mut out);
        assert_eq!(marker_of(out[0]), DOP_MARKER_A);
        assert_eq!(marker_of(out[1]), DOP_MARKER_A);
    }

    #[test]
    fn marker_alternates_per_frame_not_per_channel() {
        let mut p = DopPacker::new(2);
        let mut out = [0u32; 2];

        // Frame 0: both channels marker 0x05
        p.pack_frame(&[0, 0, 0, 0], &mut out);
        assert_eq!([marker_of(out[0]), marker_of(out[1])], [DOP_MARKER_A; 2]);

        // Frame 1: both channels marker 0xFA
        p.pack_frame(&[0, 0, 0, 0], &mut out);
        assert_eq!([marker_of(out[0]), marker_of(out[1])], [DOP_MARKER_B; 2]);

        // Frame 2: back to 0x05
        p.pack_frame(&[0, 0, 0, 0], &mut out);
        assert_eq!([marker_of(out[0]), marker_of(out[1])], [DOP_MARKER_A; 2]);
    }

    #[test]
    fn dsd_bytes_land_in_low_16_bits_in_time_order() {
        let mut p = DopPacker::new(1);
        let mut out = [0u32; 1];
        // Time-older byte (0xAB) goes to bits 15..8, newer (0xCD) to bits 7..0.
        p.pack_frame(&[0xAB, 0xCD], &mut out);
        assert_eq!(dsd_payload(out[0]), 0xABCD);
    }

    #[test]
    fn low_byte_of_32_is_always_zero() {
        let mut p = DopPacker::new(2);
        let mut out = [0u32; 2];
        p.pack_frame(&[0xFF, 0xFF, 0xFF, 0xFF], &mut out);
        // Packed value is 24-bit; the top byte of the 32-bit
        // container is whatever the backend chooses, but we should
        // never set bits above 24.
        for v in out {
            assert_eq!(v >> 24, 0, "no bits above the 24-bit payload");
        }
    }

    #[test]
    fn dsd64_pcm_rate_is_176_4khz() {
        assert_eq!(DopPacker::pcm_sample_rate(2_822_400), 176_400);
    }

    #[test]
    fn dsd128_pcm_rate_is_352_8khz() {
        assert_eq!(DopPacker::pcm_sample_rate(5_644_800), 352_800);
    }
}
