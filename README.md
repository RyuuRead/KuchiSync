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

## Development & Tests

Run the unit test suite:

```bash
python3 -W error -m unittest discover -s tests -p "test_*.py" -v
```

Build the extension ZIP:

```bash
python3 scripts/package_addon.py
```

Run headless verification inside Blender:

```bash
blender -b --factory-startup --command extension validate dist/kuchisync.zip
blender -b --factory-startup --python-exit-code 1 --python tests/blender_smoke_test.py
```

## License

Licensed under **GPL-3.0-or-later**.
