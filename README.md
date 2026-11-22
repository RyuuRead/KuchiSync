# KuchiSync

Modular Blender add-on and extension for **A-I-U-E-O** Shape Key lip-sync animation from speech audio or audio paired with a text transcript.

## Compatibility

- **Blender 4.4.3** (including Goo Engine)
- **Blender 4.5 LTS**
- **Blender 5.2 LTS** (5.0+ Slotted Actions & VSE `strips` API)

## Features

- **Pure Shape Key Lip-Sync (`A / I / U / E / O`):** Auto-detects VRChat (`vrc.v_*`), MMD/Japanese kana (`あ / い / う / え / お`), and alphabet Shape Key names, with partial-rig fallback (`1-4` mapped vowels) so rigs without all five vowels remain usable.
- **Per-Vowel Strength (`0.0 - 1.5`):** Independent multiplier slider beside each `A, I, U, E, O` slot, automatically expanding `key_block.slider_max` when combined peaks exceed `1.0`.
- **Two Detection Engines:**
  - **Pure Audio (Formant):** Pure-NumPy Levinson-Durbin LPC formant tracking with voice-type pitch-floor bandpass (`High / Child`, `Female / Standard`, `Male / Deep`), sibilant rejection, temporal smoothing, and syllable hold consolidation.
  - **Audio + Transcript:** Indonesian and English Grapheme-to-Phoneme (G2P) rules aligned to speech islands using monotonic DTW, preserving repeated-vowel syllable articulation via acoustic valley dips.
- **Sparse, Editable Keyframes:** Generates Attack-Sustain-Release keyframes with configurable `Max Mouth Open`, `Min Phoneme Hold`, `Transition Speed`, `Co-Articulation Overlap`, and `Bezier (Auto Clamped)` / `Linear` / `Constant` interpolation.
- **VSE Audio Sync:** Automatically inserts the dialogue clip into an open Video Sequencer channel and sets `scene.sync_mode = 'AUDIO_SYNC'`.
- **Zero Mandatory External Dependencies:** Runs out-of-the-box on Blender's bundled NumPy (`wave`, `aud`, `ffmpeg` fallback), with an optional background **Install SciPy Automatically** button in the panel.

## Installation

1. Download or build the extension package `dist/kuchisync.zip`:
   ```bash
   python3 scripts/package_addon.py
   ```
2. In Blender, open **Edit > Preferences > Get Extensions** (or **Add-ons**), click the top-right menu arrow, choose **Install from Disk...**, and select `dist/kuchisync.zip`.
3. Enable **KuchiSync** in the extensions list.

## Usage

1. Select your character mesh object with Shape Keys in the 3D Viewport.
2. Open the **N-Panel** (`N`) and switch to the **KuchiSync** tab (`View3D > UI > KuchiSync`).
3. Click **Auto-Detect Shape Keys** (or map `A, I, U, E, O` manually) and tune per-vowel strength multipliers if needed.
4. Choose your dialogue audio file (`.wav`, `.mp3`, `.ogg`, `.flac`) and `Language` (`Indonesian` or `English`).
5. Pick **Pure Audio (Formant)** or **Audio + Transcript**, adjust **Animation & Curve Refinement**, and click **Generate Auto Lip-Sync**.

## Changelog

### b1.0 (from a0.2)

- **Zero-Overhead Startup:** Moved `audio_dsp`, `keyframe_gen`, `transcript_align`, `subprocess`, and `scipy` imports out of add-on registration into `Operator.execute()` and lazy runtime helpers.
- **Lazy Dependency & G2P Initialization:** Deferred SciPy/ffmpeg probing into `audio_dsp.ensure_runtime_ready()` and wrapped `transcript_align` / `keyframe_gen` regexes and ARPAbet lexicons in `@functools.lru_cache(maxsize=1)`.
- **Cached Panel Redraws:** `safe_icon()` caches UILayout icon names in a `frozenset` on first draw, and `Panel.draw()` reads cached backend status without disk I/O, `find_spec`, or `subprocess` calls.
- **Measured Boot Time (`import` + `register`):** Reduced from `128.255 ms` to `1.500 ms` on Blender 4.5.16 LTS (`1.659 ms` on Goo Engine 4.4.3, `2.224 ms` on Blender 5.2.3 LTS).

## License

Licensed under **GPL-3.0-or-later**.
