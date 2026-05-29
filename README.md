# jlfine

A native desktop client for [Jellyfin](https://jellyfin.org), written in Rust,
with a focus on **bit-perfect audio** and **HDR / Dolby Vision video**.

> **Status:** early beta. macOS and Linux.

## Features

- Browse your Jellyfin libraries: music, movies, TV shows, playlists.
- **Movie & series detail pages** — backdrop, synopsis, cast, ratings,
  genres, technical media info, and IMDb / TMDB links.
- **Bit-perfect audio** — exclusive device mode, sample-rate matching, and
  system-mixer bypass, with native DSD / DoP support.
- **HDR10 / HLG / Dolby Vision video** through libmpv (`gpu-next` renderer +
  libplacebo).
- Per-track technical badges in albums (codec, sample rate, bit depth —
  `DSD64`, `FLAC 24/96`, …).
- Album download to local disk.

## Platforms

- **macOS** (Apple Silicon / Intel)
- **Linux** (Wayland by default; X11 works too)

Windows is not supported yet.

## Building from source

### Prerequisites

- **Rust 1.85+** (edition 2024) — install via [rustup](https://rustup.rs).
- **libmpv** — the client links against it for video playback:
  - macOS: `brew install mpv`
  - Arch: `sudo pacman -S mpv`
  - Debian / Ubuntu: `sudo apt install libmpv-dev`
- On **Linux**, you also need ALSA, D-Bus, fontconfig and xkbcommon headers.
  On Debian / Ubuntu:

  ```sh
  sudo apt install pkg-config libmpv-dev libasound2-dev libdbus-1-dev \
    libfontconfig-dev libxkbcommon-dev
  ```

### Build & run

```sh
cargo run --release -p jlfine
```

On first launch, sign in with your Jellyfin server URL, username and password.
Audio output and exclusive (bit-perfect) mode are configured in **Settings**.

## Project layout

This is a Cargo workspace:

| Crate                 | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| `apps/jlfine`         | Desktop application entry point                           |
| `crates/jelly-ui`     | Slint UI (login, library, detail pages, settings)         |
| `crates/jellyfin-api` | Jellyfin HTTP API client                                  |
| `crates/video-engine` | libmpv-backed video playback                              |
| `crates/audio-engine` | Bit-perfect audio playback (CoreAudio on macOS, ALSA on Linux) |
| `crates/jelly-storage`| Session and settings persistence                          |

## License

[MIT](LICENSE) © 2026 Carlos Prieto Ortiz
