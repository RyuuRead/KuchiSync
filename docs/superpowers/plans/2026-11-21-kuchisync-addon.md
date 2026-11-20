# KuchiSync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the modular `KuchiSync` (`kuchisync`) Blender add-on delivering pure-NumPy/optional-SciPy A-I-U-E-O Shape Key lip-sync (Mode A Formant F1/F2 and Mode B Transcript DTW alignment), sparse editable keyframes with Per-Vowel Strength (`0.0-1.5`), and full compatibility across Goo Engine 4.4.3, Blender 4.5 LTS, and Blender 5.2 LTS.

**Architecture:** Decouple pure signal processing (`audio_dsp.py`), G2P + DTW syllable alignment (`transcript_align.py`), and Shape Key scoring / sparse ASR + Co-Articulation scheduling (`keyframe_gen.py`) from Blender's `bpy` / `aud` APIs so core logic is testable via `unittest` outside Blender and integrates cleanly via thin Blender 4.4-5.2+ adapters and `ui_panel.py`. Packaging pairs `blender_manifest.toml` with `bl_info` in `__init__.py` for zero-warning installation as a modern Blender extension.

**Tech Stack:** Python 3.11+, NumPy, stdlib `wave`, Blender `aud` + `ffmpeg` fallback, optional SciPy (`scipy.signal`, `scipy.io.wavfile`), `bpy` / `bpy_extras.anim_utils` / `ImportHelper`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-21-kuchisync-design.md` and `DESIGN.md`

## Global Constraints

- Target Blender versions: Goo Engine 4.4.3 (Blender 4.4), Blender 4.5 LTS, and Blender 5.2 LTS (5.0+).
- Package folder name: `kuchisync`; add-on name: `KuchiSync`; every Python file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- `blender_manifest.toml` must declare `schema_version = "1.0.0"`, `id = "kuchisync"`, `blender_version_min = "4.2.0"`, `license = ["SPDX:GPL-3.0-or-later"]`, and `[permissions]` for `files` and `network`; `__init__.py` retains `bl_info` (`"blender": (4, 4, 0)`).
- Mode A and Mode B must both run out-of-the-box using pure NumPy without requiring `scipy` installed.
- Never bake per-frame keyframes; generate sparse Attack-Sustain-Release keyframes with optional Co-Articulation overlap (`0.0-0.5`).
- Option 1 peak scaling: allow combined `energy_scale * max_mouth_open * vowel_strength` up to `1.5` and raise `key_block.slider_max` when `peak > key_block.slider_max`.
- User decisions locked: `1A` (`Clear Existing LipSync Keys First` clears only mapped A/I/U/E/O keys within `[clip_start, clip_end]`; Panel 5 `Clear Lip-Sync Keyframes` clears all mapped A/I/U/E/O keys across timeline), `2A` (minimum 1 mapped vowel required; unmapped vowels remap to closest mapped vowel in normalized F1/F2 space), `3A` (`Silence Threshold (dB)` shared in Panel 3 across Mode A and Mode B; Mode B uses speaker-normalized formants).
- Adding VSE audio must automatically set `scene.sync_mode = 'AUDIO_SYNC'`.
- `antislop` + `antislop-code` + `antislop-ui` + `antislop-human`: zero em dashes (`—`), zero buzzwords (`AI Powered`, `Seamless`, `Revolutionary`), zero emoji, no decorative banner comments or vague TODOs, and every icon/status indicator pairs icon shape with explicit text/state cues.

## Review Focus

- **Consecutive same-vowel syllables in Mode B (e.g. "saya makan", "ini"):** must preserve distinct syllable articulations via an acoustic valley dip rather than merging into a single frozen mouth hold -> pinned by tests in Task 3 and Task 4.
- **Unvoiced sibilants / fricatives (`s`, `f`, `sh`) in Mode A:** high-ZCR low/mid-band energy frames must classify as `SILENCE` instead of false `I`/`E` keyframes -> pinned by test in Task 2.
- **Blender 5.0+ Slotted Actions & VSE `strips` vs `sequences`:** F-Curve reading/handle mutation/clearing and VSE sound insertion + `scene.sync_mode = 'AUDIO_SYNC'` must work without `AttributeError` on 4.4, 4.5 LTS, and 5.2 LTS -> pinned by mock tests in Task 4 and headless Blender smoke tests in Task 6.
- **Partial Shape Key mappings + Option 1 `> 1.0` + unrelated Shape Key preservation:** missing vowels remap to closest F1/F2 mapped vowel, `slider_max` expands up to `1.5`, and clearing LipSync keys never deletes unrelated `Blink`/`Brow` keyframes -> pinned by tests in Task 4 and Task 6.
- **Blender UI states & read-only Python SciPy install:** Panel 5 handles empty/invalid object, missing audio, empty transcript, and offline/read-only `pip` (`--target` user `scripts/modules`) without crashing `draw()` or using invalid icons -> pinned by tests in Task 5 and Task 6.

---

### Task 1: Extension Manifest, Package Scaffolding & Entry Point

**Files:**
- Create: `kuchisync/blender_manifest.toml`
- Create: `kuchisync/__init__.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `kuchisync/blender_manifest.toml`
  - `kuchisync.bl_info: dict`
  - `kuchisync.register() -> None`
  - `kuchisync.unregister() -> None`

- [ ] **Step 1: Write the failing test**

```python
import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestManifestAndInit(unittest.TestCase):
    def test_manifest_fields_and_license(self):
        manifest_path = ROOT / "kuchisync" / "blender_manifest.toml"
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["id"], "kuchisync")
        self.assertEqual(data["type"], "add-on")
        self.assertEqual(data["blender_version_min"], "4.2.0")
        self.assertEqual(data["license"], ["SPDX:GPL-3.0-or-later"])
        self.assertIn("files", data["permissions"])
        self.assertIn("network", data["permissions"])

    def test_init_imports_cleanly_outside_blender_and_has_spdx(self):
        init_path = ROOT / "kuchisync" / "__init__.py"
        text = init_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# SPDX-License-Identifier: GPL-3.0-or-later"))
        import kuchisync
        self.assertEqual(kuchisync.bl_info["name"], "KuchiSync")
        self.assertEqual(kuchisync.bl_info["blender"], (4, 4, 0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_manifest.py -v`
Expected: FAIL with `FileNotFoundError` for `kuchisync/blender_manifest.toml`

- [ ] **Step 3: Implement `kuchisync/blender_manifest.toml` and `kuchisync/__init__.py`**

Write `blender_manifest.toml` with exact manifest metadata (`id = "kuchisync"`, `blender_version_min = "4.2.0"`, `SPDX:GPL-3.0-or-later`, `tags = ["Animation", "Rigging"]`, `[permissions]` `files` and `network` without trailing periods per Blender validator rules). Write `__init__.py` starting with `# SPDX-License-Identifier: GPL-3.0-or-later`, defining `bl_info`, and guarding `ui_panel` registration cleanly when `bpy` is present so `import kuchisync` also succeeds in standalone Python tests.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kuchisync/blender_manifest.toml kuchisync/__init__.py tests/test_manifest.py
git commit -m "feat: scaffold KuchiSync extension manifest and package entry"
```

---

### Task 2: Audio Decoding, Resampling, VAD & LPC Formant Tracker (`audio_dsp.py`)

**Files:**
- Create: `kuchisync/audio_dsp.py`
- Test: `tests/test_audio_dsp.py`

**Interfaces:**
- Consumes: standard library (`wave`, `math`, `shutil`, `subprocess`, `importlib`), `numpy`, optional `aud`, optional `scipy`
- Produces:
  - `class AudioLoadError(RuntimeError)`
  - `AudioClip(samples: np.ndarray, sample_rate: int, duration_sec: float)`
  - `FormantFrame(time_sec: float, f1: float | None, f2: float | None, rms_db: float, zcr: float, is_voiced: bool, vowel: str | None, scores: dict[str, float])`
  - `has_scipy() -> bool`
  - `refresh_scipy() -> bool`
  - `load_audio(filepath: str, target_sr: int = 16000) -> AudioClip`
  - `resample_mono(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray`
  - `bandpass_vocal(samples: np.ndarray, sr: int = 16000, low_hz: float = 300.0, high_hz: float = 3000.0) -> np.ndarray`
  - `analyze_audio_frames(samples: np.ndarray, sr: int = 16000, silence_threshold_db: float = -38.0, voice_type: str = "FEMALE_STANDARD", language: str = "INDONESIAN") -> list[FormantFrame]`
  - `compute_speaker_normalized_scores(frames: list[FormantFrame]) -> list[dict[str, float]]`

- [ ] **Step 1: Write the failing test**

```python
import pathlib
import struct
import sys
import tempfile
import types
import unittest
import wave
import numpy as np
from kuchisync import audio_dsp


def _synth_vowel_wav(path: pathlib.Path, formants_hz: tuple[float, float], sr: int = 44100,
                     dur: float = 0.25, sampwidth: int = 2, channels: int = 2) -> None:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    f1, f2 = formants_hz
    sig = 0.45 * np.sin(2 * np.pi * f1 * t) + 0.30 * np.sin(2 * np.pi * f2 * t)
    sig = np.clip(sig, -0.95, 0.95)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        if sampwidth == 2:
            pcm = (sig * 32767.0).astype(np.int16)
            if channels == 2:
                pcm = np.column_stack([pcm, pcm]).ravel()
            wf.writeframes(pcm.tobytes())
        elif sampwidth == 3:
            ints = (sig * 8388607.0).astype(np.int32)
            raw = bytearray()
            for v in ints:
                b = struct.pack("<i", int(v))[:3]
                raw.extend(b * channels)
            wf.writeframes(bytes(raw))


class TestAudioDsp(unittest.TestCase):
    def test_load_24bit_stereo_wav_and_classify_vowels(self):
        expected = {
            "A": (820.0, 1280.0),
            "I": (330.0, 2300.0),
            "U": (360.0, 830.0),
            "E": (540.0, 1950.0),
            "O": (540.0, 920.0),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for vowel, pair in expected.items():
                wav_path = pathlib.Path(tmp) / f"{vowel}.wav"
                _synth_vowel_wav(wav_path, pair, sr=44100, sampwidth=3, channels=2)
                clip = audio_dsp.load_audio(str(wav_path), target_sr=16000)
                self.assertEqual(clip.sample_rate, 16000)
                frames = audio_dsp.analyze_audio_frames(clip.samples, 16000, -38.0, "FEMALE_STANDARD", "INDONESIAN")
                voiced_vowels = [f.vowel for f in frames if f.is_voiced and f.vowel]
                self.assertTrue(voiced_vowels, f"No voiced frames for {vowel}")
                majority = max(set(voiced_vowels), key=voiced_vowels.count)
                self.assertEqual(majority, vowel)

    def test_silence_and_sibilant_suppression(self):
        sr = 16000
        silence = np.zeros(int(sr * 0.2), dtype=np.float32)
        rng = np.random.default_rng(7)
        sibilant = (0.015 * np.where(np.arange(int(sr * 0.2)) % 2 == 0, 1.0, -1.0)
                    + 0.005 * rng.standard_normal(int(sr * 0.2))).astype(np.float32)
        frames_s = audio_dsp.analyze_audio_frames(silence, sr, -38.0)
        frames_sib = audio_dsp.analyze_audio_frames(sibilant, sr, -38.0)
        self.assertTrue(all(not f.is_voiced for f in frames_s))
        self.assertTrue(all(not f.is_voiced for f in frames_sib))

    def test_aud_contract_specs_and_2d_data(self):
        fake_aud = types.ModuleType("aud")
        class FakeSound:
            specs = (48000.0, 2)
            def rechannel(self, _c): return self
            def resample(self, _r, high_quality=True): return self
            def cache(self): return self
            def data(self): return np.ones((1600, 1), dtype=np.float32) * 0.1
        fake_aud.Sound = types.SimpleNamespace(file=lambda _p: FakeSound())
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "fake.ogg"
            p.write_bytes(b"OggSdummy")
            old = sys.modules.get("aud")
            sys.modules["aud"] = fake_aud
            try:
                clip = audio_dsp.load_audio(str(p), target_sr=16000)
                self.assertEqual(clip.sample_rate, 16000)
                self.assertEqual(clip.samples.ndim, 1)
                self.assertEqual(len(clip.samples), 1600)
            finally:
                if old is None:
                    sys.modules.pop("aud", None)
                else:
                    sys.modules["aud"] = old

    def test_missing_file_raises_audio_load_error(self):
        with self.assertRaises(audio_dsp.AudioLoadError):
            audio_dsp.load_audio("/nonexistent/path/audio.wav")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_audio_dsp.py -v`
Expected: FAIL with `ImportError: cannot import name 'audio_dsp' from 'kuchisync'`

- [ ] **Step 3: Implement `kuchisync/audio_dsp.py`**

Implement `# SPDX-License-Identifier: GPL-3.0-or-later`, SciPy optional detection/reload (`has_scipy`, `refresh_scipy`), multi-format audio loading (`wave` 8/16/24/32-bit PCM -> `aud` using `.specs` + `.rechannel(1).resample(target_sr, True).cache().data()` -> `ffmpeg` raw `f32le` pipe -> `scipy.io.wavfile`), pure-NumPy FFT anti-alias resampling (`resample_mono`) + FFT bandpass (`bandpass_vocal`) with automatic `scipy.signal` upgrade when present, 25ms window / 10ms hop pre-emphasis `0.97` + Hamming + VAD (RMS dB + ZCR/band-ratio sibilant guard), order-16 autocorrelation Levinson-Durbin LPC root formant extraction (`200-3500 Hz`, `bw <= 600 Hz`), `HIGH_CHILD`/`FEMALE_STANDARD`/`MALE_DEEP` calibration, Indonesian/English centroid scoring, and `compute_speaker_normalized_scores()` for Mode B.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_audio_dsp.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kuchisync/audio_dsp.py tests/test_audio_dsp.py
git commit -m "feat: add pure-NumPy LPC formant audio analysis and multi-format loader"
```

---

### Task 3: Indonesian/English G2P & Speech-Island DTW Alignment (`transcript_align.py`)

**Files:**
- Create: `kuchisync/transcript_align.py`
- Test: `tests/test_transcript_align.py`

**Interfaces:**
- Consumes: `kuchisync.audio_dsp.bandpass_vocal`, `kuchisync.audio_dsp.analyze_audio_frames`, `kuchisync.audio_dsp.compute_speaker_normalized_scores`
- Produces:
  - `class TranscriptError(ValueError)`
  - `PhonemeTarget(vowel: str, weight: float, phrase_index: int, same_as_prev_in_word: bool)`
  - `AlignedSyllable(vowel: str, start_frame: float, end_frame: float, peak_rms_db: float, valley_dip_before: bool)`
  - `extract_vowels(text: str, language: str = "INDONESIAN") -> list[list[PhonemeTarget]]`
  - `detect_speech_islands(samples: np.ndarray, sr: int, silence_threshold_db: float, hop_ms: float = 10.0) -> list[tuple[int, int]]`
  - `align_transcript(samples: np.ndarray, sr: int, transcript: str, language: str, silence_threshold_db: float, fps: float, start_frame: int) -> list[AlignedSyllable]`

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np
from kuchisync import transcript_align


class TestTranscriptAlign(unittest.TestCase):
    def test_indonesian_g2p_diphthongs_and_stress_weights(self):
        phrases = transcript_align.extract_vowels("saya makan pisau amboi!", "INDONESIAN")
        self.assertEqual(len(phrases), 1)
        vowels = [t.vowel for t in phrases[0]]
        self.assertEqual(vowels, ["A", "A", "A", "A", "I", "A", "U", "A", "O", "I"])
        self.assertTrue(phrases[0][1].same_as_prev_in_word)
        self.assertGreater(phrases[0][0].weight, 1.0)

    def test_english_g2p_silent_e_open_monosyllables_and_lexicon(self):
        phrases = transcript_align.extract_vowels("make time hope, be you.", "ENGLISH")
        self.assertEqual(len(phrases), 2)
        self.assertEqual([t.vowel for t in phrases[0]], ["E", "A", "I", "O"])
        self.assertEqual([t.vowel for t in phrases[1]], ["I", "U"])

    def test_empty_transcript_raises_transcript_error(self):
        with self.assertRaises(transcript_align.TranscriptError):
            transcript_align.extract_vowels("... !!!", "INDONESIAN")

    def test_align_transcript_places_phrases_across_speech_islands(self):
        sr = 16000
        t1 = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
        p1 = 0.4 * np.sin(2 * np.pi * 800 * t1) + 0.25 * np.sin(2 * np.pi * 1250 * t1)
        gap = np.zeros(int(sr * 0.35))
        t2 = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
        p2 = 0.4 * np.sin(2 * np.pi * 330 * t2) + 0.25 * np.sin(2 * np.pi * 2250 * t2)
        audio = np.concatenate([p1, gap, p2]).astype(np.float32)

        aligned = transcript_align.align_transcript(
            audio, sr, "a, i", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned], ["A", "I"])
        self.assertLess(aligned[0].end_frame, 12.0)
        self.assertGreater(aligned[1].start_frame, 16.0)

    def test_consecutive_same_vowel_marks_valley_dip(self):
        sr = 16000
        def tone(dur, amp):
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            return amp * (0.4 * np.sin(2 * np.pi * 820 * t) + 0.25 * np.sin(2 * np.pi * 1280 * t))
        audio = np.concatenate([tone(0.18, 1.0), tone(0.06, 0.15), tone(0.18, 1.0)]).astype(np.float32)
        aligned = transcript_align.align_transcript(
            audio, sr, "saya", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned], ["A", "A"])
        self.assertTrue(aligned[1].valley_dip_before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_transcript_align.py -v`
Expected: FAIL with `ImportError: cannot import name 'transcript_align' from 'kuchisync'`

- [ ] **Step 3: Implement `kuchisync/transcript_align.py`**

Implement `# SPDX-License-Identifier: GPL-3.0-or-later`, Indonesian deterministic syllable splitter with diphthong expansion (`ai->a,i`, `au->a,u`, `oi->o,i`, `ei->e,i`) and penultimate/phrase-final weights, English common-word ARPAbet exception dictionary + rule-based silent-`e` & digraph G2P, `bandpass_vocal` (`300-3000 Hz`) envelope + speech island segmentation (`<= 120ms` micro-pause bridging), phrase-to-island matching, syllable peak prominence + speaker-normalized formant constrained DTW alignment, and `valley_dip_before` detection for consecutive identical-vowel syllables.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_transcript_align.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kuchisync/transcript_align.py tests/test_transcript_align.py
git commit -m "feat: add Indonesian/English G2P and DTW syllable alignment"
```

---

### Task 4: Shape Key Auto-Detect, Sparse Keyframe Scheduler & Blender 4.4-5.2+ Writer (`keyframe_gen.py`)

**Files:**
- Create: `kuchisync/keyframe_gen.py`
- Test: `tests/test_keyframe_gen.py`

**Interfaces:**
- Consumes: `kuchisync.audio_dsp.FormantFrame`, `kuchisync.transcript_align.AlignedSyllable`, optional `bpy` / `bpy_extras.anim_utils`
- Produces:
  - `VOWELS = ("A", "I", "U", "E", "O")`
  - `KeyframeEvent(frame: int, value: float)`
  - `detect_shapekeys(key_names: Sequence[str]) -> dict[str, str]`
  - `resolve_vowel_fallback(vowel: str, mapped_vowels: Collection[str]) -> str`
  - `build_mode_a_segments(frames: Sequence[FormantFrame], fps: float, start_frame: int, min_phoneme_frames: int) -> list[AlignedSyllable]`
  - `schedule_keyframes(syllables: Sequence[AlignedSyllable], mapped_shapes: dict[str, str], vowel_strengths: dict[str, float], start_frame: int, max_mouth_open: float, min_phoneme_frames: int, transition_frames: int, co_articulation: float, preserve_syllables: bool) -> dict[str, list[KeyframeEvent]]`
  - `iter_shape_key_fcurves(shape_keys) -> list`
  - `clear_lipsync_keyframes(obj, target_shape_names: Collection[str], frame_range: tuple[int, int] | None = None) -> int`
  - `apply_keyframes_to_object(obj, scheduled: dict[str, list[KeyframeEvent]], interpolation_mode: str = "BEZIER") -> int`
  - `add_audio_to_vse(scene, filepath: str, frame_start: int) -> None`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from kuchisync import keyframe_gen
from kuchisync.transcript_align import AlignedSyllable


class _FakeKP:
    def __init__(self, frame, value):
        self.co = [float(frame), float(value)]
        self.interpolation = "BEZIER"
        self.handle_left_type = "AUTO"
        self.handle_right_type = "AUTO"


class _FakeKPs(list):
    def remove(self, kp, fast=False):
        super().remove(kp)


class _FakeFCurve:
    def __init__(self, data_path, points=None):
        self.data_path = data_path
        self.keyframe_points = _FakeKPs(points or [])
        self.updated = False
    def update(self):
        self.updated = True


class _FakeKeyBlock:
    def __init__(self, name, fcurve_store):
        self.name = name
        self.value = 0.0
        self.slider_max = 1.0
        self._store = fcurve_store
    def keyframe_insert(self, data_path="value", frame=1):
        dp = f'key_blocks["{self.name}"].{data_path}'
        fc = self._store.setdefault(dp, _FakeFCurve(dp))
        fc.keyframe_points.append(_FakeKP(frame, self.value))


class TestKeyframeGen(unittest.TestCase):
    def test_auto_detect_alphabet_kana_and_no_false_positives(self):
        names = ["Basis", "Angry", "Happy", "Blink", "vrc.v_aa", "い", "mouth_u", "viseme_eh", "お"]
        detected = keyframe_gen.detect_shapekeys(names)
        self.assertEqual(detected, {"A": "vrc.v_aa", "I": "い", "U": "mouth_u", "E": "viseme_eh", "O": "お"})

    def test_partial_mapping_fallback(self):
        self.assertEqual(keyframe_gen.resolve_vowel_fallback("E", ["A", "I", "U", "O"]), "I")

    def test_sparse_asr_sustain_coarticulation_and_strength_above_one(self):
        syllables = [
            AlignedSyllable("A", 5.0, 14.0, -16.0, False),
            AlignedSyllable("I", 15.0, 20.0, -16.0, False),
        ]
        mapped = {"A": "mouth_a", "I": "mouth_i", "U": "", "E": "", "O": ""}
        strengths = {"A": 1.1, "I": 0.8, "U": 1.0, "E": 1.0, "O": 1.0}
        scheduled = keyframe_gen.schedule_keyframes(
            syllables, mapped, strengths, start_frame=1, max_mouth_open=1.0,
            min_phoneme_frames=3, transition_frames=2, co_articulation=0.25, preserve_syllables=True
        )
        a_keys = {k.frame: k.value for k in scheduled["mouth_a"]}
        self.assertEqual(len(a_keys), 4)
        self.assertAlmostEqual(a_keys[6], 1.1, places=2)
        self.assertAlmostEqual(a_keys[13], 1.1 * 0.9, places=2)
        i_frames = [k.frame for k in scheduled["mouth_i"]]
        self.assertLess(min(i_frames), max(a_keys.keys()))

    def test_mode_b_same_vowel_keeps_valley_dip(self):
        syllables = [
            AlignedSyllable("A", 2.0, 6.0, -16.0, False),
            AlignedSyllable("A", 7.0, 11.0, -16.0, True),
        ]
        scheduled = keyframe_gen.schedule_keyframes(
            syllables, {"A": "mouth_a"}, {"A": 1.0}, start_frame=1, max_mouth_open=1.0,
            min_phoneme_frames=3, transition_frames=2, co_articulation=0.25, preserve_syllables=True
        )
        vals = [k.value for k in scheduled["mouth_a"]]
        self.assertTrue(any(0.1 < v < 0.6 for v in vals))

    def test_slotted_action_writer_scoped_clear_and_vse_sync_mode(self):
        fcurves_map = {'key_blocks["Blink"].value': _FakeFCurve('key_blocks["Blink"].value', [_FakeKP(5, 1.0)]),
                       'key_blocks["mouth_a"].value': _FakeFCurve('key_blocks["mouth_a"].value', [_FakeKP(5, 0.8), _FakeKP(50, 0.9)])}
        class FakeBag:
            @property
            def fcurves(self): return list(fcurves_map.values())
        class FakeStrip:
            def channelbag(self, _slot): return FakeBag()
        class FakeLayer:
            strips = [FakeStrip()]
        class FakeAction:
            slots = ["slot0"]
            layers = [FakeLayer()]
        class FakeAnim:
            action = FakeAction()
            action_slot = "slot0"
        kb_a = _FakeKeyBlock("mouth_a", fcurves_map)
        kb_blink = _FakeKeyBlock("Blink", fcurves_map)
        class FakeSK:
            animation_data = FakeAnim()
            key_blocks = {"mouth_a": kb_a, "Blink": kb_blink}
        class FakeObj:
            data = type("D", (), {"shape_keys": FakeSK()})()

        removed = keyframe_gen.clear_lipsync_keyframes(FakeObj(), ["mouth_a"], frame_range=(40, 60))
        self.assertEqual(removed, 1)
        self.assertEqual(len(fcurves_map['key_blocks["Blink"].value'].keyframe_points), 1)
        self.assertEqual([kp.co[0] for kp in fcurves_map['key_blocks["mouth_a"].value'].keyframe_points], [5.0])

        written = keyframe_gen.apply_keyframes_to_object(
            FakeObj(), {"mouth_a": [keyframe_gen.KeyframeEvent(42, 1.15)]}, "BEZIER"
        )
        self.assertEqual(written, 1)
        self.assertGreaterEqual(kb_a.slider_max, 1.15)
        self.assertEqual(fcurves_map['key_blocks["mouth_a"].value'].keyframe_points[-1].handle_left_type, "AUTO_CLAMPED")

        created = []
        class FakeStrips(list):
            def new_sound(self, name, filepath, channel, frame_start):
                s = type("S", (), {"name": name, "sound": type("SND", (), {"filepath": filepath})(), "channel": channel, "frame_start": frame_start, "type": "SOUND"})()
                created.append(s)
                self.append(s)
                return s
        class FakeScene:
            sync_mode = "NONE"
            sequence_editor = type("SE", (), {"strips": FakeStrips()})()
            def sequence_editor_create(self): pass
        scene = FakeScene()
        keyframe_gen.add_audio_to_vse(scene, "/tmp/test.wav", 10)
        self.assertEqual(scene.sync_mode, "AUDIO_SYNC")
        self.assertEqual(len(created), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_keyframe_gen.py -v`
Expected: FAIL with `ImportError: cannot import name 'keyframe_gen' from 'kuchisync'`

- [ ] **Step 3: Implement `kuchisync/keyframe_gen.py`**

Implement `# SPDX-License-Identifier: GPL-3.0-or-later`, exact spec auto-detection scoring (`100` exact including VRChat & Japanese kana, `70` token boundary, `40` safe regex without false positives, greedy unique assignment), normalized `(F1/130, F2/260)` vowel fallback (`resolve_vowel_fallback`), Mode A segment aggregation + spike filtering (`build_mode_a_segments`), sparse ASR + hold + Co-Articulation overlap + Mode B partial valley dip + Option 1 strength scaling (`schedule_keyframes`), Blender 4.4-5.2+ `iter_shape_key_fcurves` + `clear_lipsync_keyframes` + `apply_keyframes_to_object` (`slider_max` auto-expansion, `AUTO_CLAMPED` / `LINEAR` / `CONSTANT`), and `add_audio_to_vse` setting `scene.sync_mode = 'AUDIO_SYNC'`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_keyframe_gen.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kuchisync/keyframe_gen.py tests/test_keyframe_gen.py
git commit -m "feat: add shape key detection, sparse co-articulated scheduler, and 4.4-5.2+ writer"
```

---

### Task 5: Blender N-Panel UI, Operators, SciPy Installer & Packaging (`ui_panel.py`, `__init__.py`)

**Files:**
- Create: `kuchisync/ui_panel.py`
- Modify: `kuchisync/__init__.py`
- Test: `tests/test_ui_logic.py`

**Interfaces:**
- Consumes: `kuchisync.audio_dsp`, `kuchisync.transcript_align`, `kuchisync.keyframe_gen`
- Produces:
  - `safe_icon(icon_name: str, fallback: str = "NONE") -> str`
  - `validate_generation_context(obj, props) -> tuple[bool, str]`
  - `KuchiSyncSettings(bpy.types.PropertyGroup)` with `shape_a..o`, `strength_a..o` (`0.0-1.5`), `audio_filepath`, `language`, `start_frame`, `add_to_vse`, `clear_existing_keys`, `detection_mode`, `voice_type`, `silence_threshold_db`, `transcript_text`, `transcript_datablock`, `max_mouth_open`, `min_phoneme_frames`, `transition_frames`, `co_articulation`, `interpolation_mode`
  - Operators: `lipsync.auto_detect_shapekeys`, `kuchisync.browse_audio`, `kuchisync.set_start_frame`, `kuchisync.load_transcript`, `kuchisync.generate_lipsync`, `kuchisync.clear_lipsync`, `kuchisync.install_scipy`
  - 5 N-Panels in `VIEW_3D > UI > LipSync`: `KUCHISYNC_PT_rig_shapekeys`, `KUCHISYNC_PT_audio_timing`, `KUCHISYNC_PT_detection_engine`, `KUCHISYNC_PT_animation_curve`, `KUCHISYNC_PT_execution`
  - Packaging helper command producing `dist/kuchisync.zip`

- [ ] **Step 1: Write the failing test**

```python
import pathlib
import tempfile
import unittest
from kuchisync import ui_panel

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestUiLogicAndHygiene(unittest.TestCase):
    def test_validate_generation_context_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = pathlib.Path(tmp) / "line.wav"
            audio.write_bytes(b"RIFF0000WAVE")

            def make_props(**kw):
                base = dict(
                    shape_a="mouth_a", shape_i="", shape_u="", shape_e="", shape_o="",
                    audio_filepath=str(audio), detection_mode="PURE_AUDIO",
                    transcript_text="", transcript_datablock=None,
                )
                base.update(kw)
                return type("Props", (), base)()

            mesh_obj = type("Obj", (), {
                "type": "MESH",
                "data": type("Data", (), {"shape_keys": type("SK", (), {"key_blocks": {"mouth_a": object()}})()})(),
            })()

            ok, _ = ui_panel.validate_generation_context(None, make_props())
            self.assertFalse(ok)
            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(shape_a=""))
            self.assertFalse(ok)
            self.assertIn("Shape Key", msg)
            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(audio_filepath=""))
            self.assertFalse(ok)
            self.assertIn("audio", msg.lower())
            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(detection_mode="TRANSCRIPT"))
            self.assertFalse(ok)
            self.assertIn("transcript", msg.lower())
            ok, _ = ui_panel.validate_generation_context(mesh_obj, make_props())
            self.assertTrue(ok)

    def test_antislop_code_and_copy_hygiene(self):
        banned_buzz = ("AI Powered", "Next Generation", "Revolutionary", "Seamless", "Cutting Edge")
        for py in (ROOT / "kuchisync").glob("*.py"):
            src = py.read_text(encoding="utf-8")
            self.assertTrue(src.startswith("# SPDX-License-Identifier: GPL-3.0-or-later"), str(py))
            self.assertNotIn("\u2014", src, f"Em dash found in {py}")
            for word in banned_buzz:
                self.assertNotIn(word.lower(), src.lower(), f"Buzzword {word} found in {py}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_ui_logic.py -v`
Expected: FAIL with `ImportError: cannot import name 'ui_panel' from 'kuchisync'`

- [ ] **Step 3: Implement `kuchisync/ui_panel.py` and wire registration in `kuchisync/__init__.py`**

Implement `ui_panel.py` with `# SPDX-License-Identifier: GPL-3.0-or-later`, clean pure-Python `validate_generation_context` & `get_effective_transcript` helpers at top (importable outside Blender), and when `bpy` is available: `safe_icon()`, `KuchiSyncSettings` (`PropertyGroup` attached to `Scene.kuchisync`), all 7 operators (including `lipsync.auto_detect_shapekeys`, `kuchisync.install_scipy` checking `bpy.app.online_access` + `--target` user `scripts/modules` + `0.5s` `bpy.app.timers` poll + `site.addsitedir()` + `audio_dsp.refresh_scipy()`), and all 5 `VIEW_3D > UI > LipSync` panels adhering strictly to `DESIGN.md` (1-row-per-vowel mapping with Per-Vowel Strength `0.0-1.5`, shared `Silence Threshold (dB)` across Mode A/B, empty/running/error states, non-color-only icon+text status). Update `kuchisync/__init__.py` to register/unregister `ui_panel` cleanly inside Blender, including cancelling any active SciPy install timer/process on `unregister()`.

- [ ] **Step 4: Run all unit tests and build `dist/kuchisync.zip`**

Run: `python3 -m unittest discover -s tests -p "test_*.py" -v`
Expected: PASS across all test suites

- [ ] **Step 5: Commit**

```bash
git add kuchisync/ui_panel.py kuchisync/__init__.py tests/test_ui_logic.py
git commit -m "feat: add N-panel UI, operators, SciPy installer, and extension registration"
```

---

### Task 6: Cross-Version Blender Integration Verification (4.4.3 Goo Engine, 4.5 LTS, 5.2 LTS) & Delivery Gate

**Files:**
- Create: `tests/blender_smoke_test.py`

**Interfaces:**
- Consumes: built `dist/kuchisync.zip` / `kuchisync/` package installed and run headlessly inside Goo Engine 4.4.3, Blender 4.5 LTS, and Blender 5.2 LTS

- [ ] **Step 1: Write the headless Blender verification script `tests/blender_smoke_test.py`**

Create a headless `bpy` script that:
1. Installs/loads `kuchisync` cleanly without errors or legacy warnings.
2. Creates a test Mesh object with shape keys `Basis`, `vrc.v_aa`, `い`, `mouth_u`, `viseme_eh`, `お`, `Blink`, `Angry`.
3. Runs `bpy.ops.lipsync.auto_detect_shapekeys()`, asserting all 5 vowels map uniquely and `Angry`/`Blink` are ignored.
4. Generates a synthetic multi-vowel `.wav` (plus `.ogg`/`.mp3` via `ffmpeg` if available) and runs BOTH Mode A (`PURE_AUDIO`) and Mode B (`TRANSCRIPT` with `"saya makan, halo!"`) with `strength_a = 1.1`, `strength_o = 0.7`, `add_to_vse = True`, `clear_existing_keys = True`, testing `BEZIER` (`AUTO_CLAMPED`), `LINEAR`, and `CONSTANT` interpolation modes.
5. Verifies `slider_max >= 1.1` on `vrc.v_aa`, `scene.sync_mode == 'AUDIO_SYNC'`, VSE strip exists, keyframe counts are sparse (fewer keys than total audio frames), `Blink` keyframes placed beforehand remain intact after scoped clear, and `kuchisync.clear_lipsync` clears all mapped lip-sync keys.
6. Verifies every icon in `ui_panel` resolves cleanly against the active Blender build's `UILayout.bl_rna.properties['icon'].enum_items`.

- [ ] **Step 2: Run extension validation, package zip, and execute `tests/blender_smoke_test.py` across all 3 Blender versions**

Run:
1. Build `dist/kuchisync.zip`
2. Locate Blender binaries inside `/home/ilham/.local/share/Steam/steamapps/common/` (4.4.3 Goo Engine & 4.5 LTS) and `~/.local/share/Blender 5.2 LTS/` (requiring sandbox bypass approval for those external paths)
3. Run `--command extension validate` and `-b --factory-startup --python tests/blender_smoke_test.py` on each of the 3 Blender builds
Expected: EXIT 0 and all assertions PASS on Goo Engine 4.4.3, Blender 4.5 LTS, and Blender 5.2 LTS (if any Blender build exposes a real bug, fix it in source, re-run tests, and commit).

- [ ] **Step 3: Run `antislop` Delivery Gate & `antislop-code` / `antislop-ui` / `antislop-human` checklists**

Verify all unit tests pass after final edits, zero em dashes, zero buzzwords, every interactive element verified, and compile the PASS/FAIL Delivery Gate report.

- [ ] **Step 4: Commit verification script & any compatibility fixes**

```bash
git add tests/blender_smoke_test.py
git commit -m "test: add cross-version headless Blender verification suite"
```

---

## Self-Review Results

- **Spec coverage:** Section 1 (4.4-5.2+, Slotted Actions, VSE + `AUDIO_SYNC`) -> Tasks 1, 4, 6; Section 2 (auto-detect alphabet/kana + partial mapping 2A + Per-Vowel Strength 0.0-1.5) -> Tasks 4, 5; Section 3 (WAV/MP3/OGG/FLAC + LPC F1/F2 + ZCR/VAD + voice presets + Mode B G2P/DTW + shared `Silence Threshold` 3A) -> Tasks 2, 3; Section 4 (sparse ASR, co-articulation, Option 1 `> 1.0`, scoped clear 1A, BEZIER/LINEAR/CONSTANT) -> Task 4; Section 5 (pure-NumPy + non-blocking SciPy installer) -> Tasks 2, 5; Section 6 & `DESIGN.md` (5 N-Panels + states) -> Task 5; Section 7 (`blender_manifest.toml`, `bl_info`, SPDX GPL-3.0-or-later) -> Tasks 1, 5, 6.
- **Type & signature consistency:** Verified `AudioClip`, `FormantFrame`, `PhonemeTarget`, `AlignedSyllable`, `KeyframeEvent`, `lipsync.auto_detect_shapekeys`, and `KuchiSyncSettings` fields match across all tasks.
- **No placeholders / no transcripts:** Steps pin interfaces, values, and tests unambiguously without dumping full implementation code.
