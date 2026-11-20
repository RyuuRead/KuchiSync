# KuchiSync Design Specification

**Date:** 2026-11-21
**Status:** Ready for Approval
**Target Environment:** Goo Engine 4.4.3 (Blender 4.4), Blender 4.5 LTS, Blender 5.2 LTS (5.0+)
**License:** GPL-3.0-or-later

## 1. Goals and Scope

KuchiSync is a modular Blender add-on (`kuchisync`) that generates editable, sparse **A-I-U-E-O** Shape Key lip-sync keyframes for 3D characters from speech audio alone or audio paired with a text transcript (Indonesian and English).

Key user priorities:
1. Works out-of-the-box in stock Blender / Goo Engine Python without requiring manual dependency setup (`numpy` pure-Python DSP fallback; optional `scipy` upgrade).
2. Clean N-Panel workflow (`View3D > UI > LipSync`) inspired by Rhubarb Lip Sync, cgtinker facial tools, Faceit, and LipSync Pro, including **Per-Vowel Strength / Multiplier (`0.0 - 1.5`)** right next to each `A, I, U, E, O` Shape Key slot.
3. Supports peak values above `1.0` (up to `1.5`) when `Max Mouth Open` and `Per-Vowel Strength` combine, automatically raising the target Shape Key's `slider_max` when needed so values like `1.1` evaluate in Blender.
4. Produces sparse, non-baked Attack-Sustain-Release keyframes with optional Co-Articulation cross-fades that stay clean in Graph Editor and Dope Sheet.

Non-goals:
- Armature bone pose baking (Shape Keys only).
- Real-time microphone streaming or external CLI binaries like Rhubarb executable.

## 2. Architecture Overview (Approach A)

```text
kuchisync.zip
└── kuchisync/
    ├── blender_manifest.toml   # Blender 4.2+ extension metadata & permissions
    ├── __init__.py             # bl_info + class/property registration + timer cleanup
    ├── audio_dsp.py            # Audio decoding, VAD, Levinson-Durbin LPC F1/F2, vowel classifier
    ├── transcript_align.py     # Indonesian/English G2P + speech island & DTW/peak alignment
    ├── keyframe_gen.py         # Shape key auto-detect, sparse ASR/co-articulation, 4.4-5.2+ writer & VSE
    └── ui_panel.py             # Scene PropertyGroup, Operators, 5 N-Panels
tests/                          # Pure-Python unit & integration tests (outside zip package)
```

Pure DSP (`audio_dsp.py`), G2P/alignment (`transcript_align.py`), Shape Key name scoring, vowel fallback remapping, and sparse keyframe scheduling (`keyframe_gen.py`) are decoupled from `bpy` so every mathematical and timing invariant can be tested with `unittest` outside Blender as well as verified directly in Goo Engine 4.4.3, Blender 4.5 LTS, and Blender 5.2 LTS.

## 3. Packaging and Blender 4.4-5.2+ Compatibility

### 3.1 Dual Extension Manifest + `bl_info`
- `kuchisync/blender_manifest.toml` uses `schema_version = "1.0.0"`, `id = "kuchisync"`, `type = "add-on"`, `blender_version_min = "4.2.0"`, `license = ["SPDX:GPL-3.0-or-later"]`, `tags = ["Animation", "Rigging"]`, and `[permissions]` declaring `files` (reading audio/transcript files) and `network` (optional user-triggered SciPy pip install).
- `kuchisync/__init__.py` also keeps a standard `bl_info` dictionary (`"blender": (4, 4, 0)`) for compatibility with legacy loading paths while modern Blender versions prioritize `blender_manifest.toml`.
- Build/package verification tests both `blender --command extension validate` and `blender --command extension build` to confirm clean installability across 4.4.3, 4.5 LTS, and 5.2 LTS.

### 3.2 Layered / Slotted Actions & Scoped Keyframe Clearing (Blender 4.4, 4.5 LTS, 5.2 LTS)
Blender 4.4 introduced Slotted Actions (`layers -> strips -> channelbags`) and Blender 5.0 removed `action.fcurves`.
- Keyframes are inserted via `key_block.keyframe_insert(data_path="value", frame=frame)` after ensuring `shape_keys.animation_data` exists.
- Reading and modifying F-Curves (`interpolation`, `handle_left_type`, `handle_right_type`, clearing keys) uses a unified adapter `iter_shape_key_fcurves(shape_keys)`:
  1. Ensure `anim_data.action_slot` is bound if `action.slots` exists and `action_slot` is None.
  2. Try `bpy_extras.anim_utils.action_get_channelbag_for_slot(action, anim_data.action_slot)` when `anim_utils` exposes it.
  3. Fallback: iterate `action.layers -> layer.strips -> strip.channelbag(slot)` or `strip.channelbags`.
  4. Legacy fallback: `action.fcurves` when `hasattr(action, "fcurves")`.
- **Scoped clearing rule (User Decision 1A):**
  - `Clear Existing LipSync Keys First` (checkbox during `Generate Auto Lip-Sync`): clears keyframes ONLY on the mapped `A/I/U/E/O` Shape Keys within the generated clip range `[max(0, start_frame - transition_frames), end_frame + transition_frames]`, protecting earlier/later dialogue takes on the same character.
  - `Clear Lip-Sync Keyframes` (secondary button in Panel 5): clears ALL keyframes on the mapped `A/I/U/E/O` Shape Keys across the entire timeline.
  - Neither operation ever deletes keyframes on unrelated facial Shape Keys (blinks, brows, emotions).

### 3.3 Video Sequence Editor (VSE) Compatibility (Blender 4.4-5.2+)
Blender 5.0 renamed `SequenceEditor.sequences` to `SequenceEditor.strips`.
- `add_audio_to_vse(scene, filepath, frame_start)`:
  1. Call `scene.sequence_editor_create()` if `scene.sequence_editor` is None.
  2. Resolve container via `getattr(se, "strips", None)` else `getattr(se, "sequences", None)`.
  3. Add the sound cleanly via `new_sound(name=..., filepath=..., channel=..., frame_start=...)` on an available audio channel (avoiding duplicate identical strip at the same start frame when regenerating).

## 4. Shape Key Auto-Detection, Partial Mappings & Per-Vowel Strength

### 4.1 Auto-Detection (`detect_shapekeys`)
`detect_shapekeys(key_names: Sequence[str]) -> dict[str, str]` ignores `"Basis"` and scores every remaining Shape Key per vowel (`A`, `I`, `U`, `E`, `O`), case-insensitive:
1. **Exact / Standard (score 100):** exact spec lists including VRChat (`vrc.v_aa`, `vrc.v_ih`, `vrc.v_ou`, `vrc.v_e`, `vrc.v_oh`), generic (`a`, `aa`, `ah`, `mouth_a`, `viseme_a`, `mth_a`, etc.), and Japanese Hiragana/Katakana (`あ`, `ア`, `ぁ`, `ァ`, etc.).
2. **Boundary / Token Match (score 70):** splitting on `_`, `.`, `-`, whitespace, or digit boundaries where tokens match vowel identifiers (e.g. `face_mouth_a`, `lip.oh.01`).
3. **Safe Substring Fallback (score 40):** bounded regex `(?<![a-z])(aa?|ah)(?![a-z])` or kana characters (`あ`/`ア`, etc.) so words like `angry` or `happy` never false-match `a`.
4. **Conflict Resolution (Builder Call):** greedy assignment ordered by `(score, exact_length_bonus)` ensuring distinct Shape Keys across `A/I/U/E/O` whenever distinct candidates exist.

### 4.2 Partial Mapping Support & Vowel Fallback (User Decision 2A)
- Generation requires **at least 1 mapped vowel** (if 0 vowels are mapped, Panel 5 blocks execution with an explicit message).
- When 1 to 4 vowels are mapped, any unmapped vowel `v` is automatically mapped via `resolve_vowel_fallback(v, mapped_vowels)` to the closest mapped vowel in normalized `(F1/130, F2/260)` formant space before scheduling keyframes (for example, missing `E` maps to the nearest mapped `I` or `A`). Thus the character's mouth keeps articulating naturally instead of closing mid-word.

### 4.3 Per-Vowel Strength (`0.0 - 1.5`)
Each vowel (`A`, `I`, `U`, `E`, `O`) stores:
- `shape_<v>`: `StringProperty` (bound in UI via `prop_search` against `obj.data.shape_keys.key_blocks`).
- `strength_<v>`: `FloatProperty(min=0.0, max=1.5, default=1.0)` shown directly beside the mapping row.

## 5. Audio Decoding & Pure-Audio Formant Engine (`audio_dsp.py`)

### 5.1 Decoding & Resampling Chain
Target internal representation: mono `float32` normalized to `[-1.0, 1.0]` relative to bit-depth full scale (preserving true RMS dB) at `16,000 Hz`.
1. **WAV (`wave` stdlib):** supports 8-bit unsigned, 16-bit signed, 24-bit signed (manual 3-byte sign extension), and 32-bit signed PCM.
2. **Blender `aud` (`aud.Sound`):** primary decoder inside Blender for `.mp3`, `.ogg`, `.flac`, and IEEE float WAV without external binaries. Read sample rate from `sound.specs[0]` and samples from `sound.rechannel(1).resample(target_sr, True).cache().data()` (note: `sound.data()` returns a 2D `ndarray`, NOT a `(data, sr)` tuple; fallback to manual mono mixdown + resampler if `.rechannel`/`.resample` fails).
3. **FFmpeg subprocess fallback:** `ffmpeg -v error -i <path> -ac 1 -ar 16000 -f f32le pipe:1` when `aud` is unavailable (standalone tests or custom builds).
4. **SciPy WAV fallback:** `scipy.io.wavfile` if installed and earlier paths do not handle a WAV variant.
5. **Resampling:** `scipy.signal.resample_poly` when SciPy is available; pure-NumPy FFT anti-alias low-pass + `np.interp` otherwise.

### 5.2 Formant F1/F2 Tracking (Pure NumPy)
- Frame window = `25 ms` (`400` samples), hop = `10 ms` (`160` samples).
- **VAD:** Compute RMS dB on un-emphasized window and Zero-Crossing Rate (`zcr`).
  - Frames with `db < silence_threshold_db` become `SILENCE`.
  - Sibilant/fricative guard: frames with `zcr > 0.35` and `db < silence_threshold_db + 12.0` (or high-band `3500-8000 Hz` energy exceeding `200-3000 Hz` vocal energy by `> 1.5x`) are suppressed as `SILENCE`.
- **LPC Order `p = 16`:**
  - Pre-emphasis `y[n] = x[n] - 0.97 * x[n-1]`, multiplied by `np.hamming(N)`.
  - Autocorrelation `r[0..16]` with small diagonal floor (`r[0] *= 1.0001`) + Levinson-Durbin recursion.
  - Complex roots via `np.roots(a)`; keep roots with `imag > 0`, convert phase to Hz `f = arctan2(Im, Re) * (fs / (2 * pi))`, compute bandwidth `bw = -(fs / (2 * pi)) * ln(|r|)`.
  - Keep poles in `200 Hz .. 3500 Hz` with `bw <= 600 Hz`; sort ascending to extract `F1` and `F2`.
- **Voice Type Calibration & Vowel Classifier:**
  - Voice presets scale expected formant space: `HIGH_CHILD` (`1.16, 1.18`), `FEMALE_STANDARD` (`1.00, 1.00`), `MALE_DEEP` (`0.85, 0.87`).
  - Centroids `(F1, F2)` with normalized Gaussian/weighted distance (`sigma_f1 = 130 Hz`, `sigma_f2 = 260 Hz`):
    - **A:** `(825, 1275)` (English slightly centered at `(780, 1320)`)
    - **I:** `(325, 2300)`
    - **U:** `(365, 825)`
    - **E:** `(550, 1950)`
    - **O:** `(550, 925)`
  - Also computes speaker-normalized relative F1/F2 log-scores across voiced frames of the clip so Mode B (`Audio + Transcript`) does not require `Voice Type` (`User Decision 3A`).

## 6. Transcript Alignment Engine (`transcript_align.py`)

### 6.1 Grapheme-to-Phoneme (G2P)
Split text into punctuation-delimited phrases (`.`, `,`, `?`, `!`, `;`, `:`, newline).
- **Indonesian (`INDONESIAN`):** deterministic syllable segmenter (`V`, `KV`, `VK`, `KVK`, digraphs `ng`, `ny`, `kh`, `sy`), handling diphthongs (`ai -> a, i`, `au -> a, u`, `oi -> o, i`, `ei -> e, i`) and assigning duration weights (`1.35` penultimate syllable, `1.25` phrase-final syllable, `0.90` other syllables).
- **English (`ENGLISH`):** built-in common-word exception lexicon (irregular high-frequency words mapped via ARPAbet categories `AA/AE/AH -> A`, `IY/IH -> I`, `UW/UH -> U`, `EH/EY -> E`, `AO/OW -> O`, diphthongs `AY -> A,I`, `AW -> A,U`, `OY -> O,I`) plus rule-based fallback for digraphs (`ee/ea/oo/ou/ow/ai/ay/oi/oy/au/aw/oa`) and silent trailing `'e'` (`make`, `time`, `hope`, excluding open monosyllables like `be`, `he`, `she`, `me`, `we`, `the`).

### 6.2 Acoustic Speech Islands & DTW / Peak-Aligned Syllable Mapping
1. Filter audio to `300 - 3000 Hz` (SciPy `butter`/`sosfiltfilt` when available; pure-NumPy zero-phase FFT bandpass otherwise).
2. Smooth vocal energy envelope (`10 ms` frames, `40 ms` moving window) and detect speech islands above shared `silence_threshold_db` (User Decision 3A), bridging micro-pauses `<= 120 ms` (consonant closures inside words).
3. Match phrase groups to speech islands (merging/splitting proportionally if island count differs from punctuation count).
4. Within each speech island + target vowel sequence:
   - Detect local syllable energy peaks (nuclei) and combine normalized peak prominence with speaker-normalized formant likelihoods from `audio_dsp.py`.
   - Run constrained monotonic Dynamic Time Warping (DTW) weighted by Indonesian/English stress weights, outputting discrete `AlignedSyllable(vowel, start_frame, end_frame, peak_rms_db)` instances rather than flattening back into raw frame labels.

## 7. Sparse Keyframe Pipeline & Co-Articulation (`keyframe_gen.py`)

### 7.1 Mode A vs Mode B Segment Preparation
- **Mode A (`PURE_AUDIO`):** aggregates frame-level vowel predictions into segments, merges adjacent identical vowel segments separated by `<= 1` frame, and drops micro-spikes shorter than `min_phoneme_frames` (default `3`).
- **Mode B (`TRANSCRIPT`, Builder Call):** preserves each aligned G2P syllable instance intact (protecting fast diphthongs and transcript syllables as short as `2` frames from `min_phoneme_frames` pruning). When two consecutive syllables share the same vowel (e.g. Indonesian `"saya makan"` or `"ini"`) with gap `<= transition_frames`, insert a partial articulation valley keyframe (`0.35 * min(peak_prev, peak_next)`) at the acoustic consonant energy minimum between them instead of freezing the mouth wide open across the entire word.
- Apply `resolve_vowel_fallback(vowel, mapped_vowels)` (User Decision 2A) when 1-4 vowels are mapped.

### 7.2 Dynamic Scaling (Option 1), Attack-Sustain-Release & Co-Articulation
1. **Dynamic Scaling (Option 1):**
   - Map local segment RMS energy relative to voiced utterance RMS into `[0.65, 1.0]`.
   - `peak = clamp(energy_scale * max_mouth_open * vowel_strength[vowel], 0.0, 1.5)`.
   - Skip writing non-zero keys when `vowel_strength[vowel] == 0.0`.
   - When `peak > key_block.slider_max`, expand `key_block.slider_max` up to `min(1.5, max(key_block.slider_max, peak))` before inserting keyframes.
2. **Attack - Sustain - Release + Co-Articulation:**
   - Start zero at `max(start_frame, seg_start - transition_frames)` if not already active on that vowel.
   - Attack peak at `seg_start + 1` (or `seg_start` if `seg_end - seg_start < 2`).
   - If `(seg_end - seg_start) > 5`, add hold keyframe `peak * 0.9` at `seg_end - 1`.
   - Release zero at `seg_end + transition_frames`.
   - **Co-Articulation Overlap (`0.0 - 0.5`):** when transitioning directly from vowel `V1` to distinct vowel `V2` (gap `<= transition_frames`), overlap `V1`'s release and `V2`'s attack by `overlap = max(1, round(transition_frames * co_articulation * 2))` frames (when `co_articulation > 0.0`) so `V2` begins rising while `V1` eases down without snapping to `0.0` mid-word.
3. **Interpolation & Handles:**
   - `BEZIER`: `BEZIER` with `AUTO_CLAMPED` left/right handles.
   - `LINEAR`: linear segments.
   - `CONSTANT`: stepped animation for 2D/NPR workflows.

## 8. SciPy Dependency Management

- Startup and panel draw check `has_scipy()` lazily (zero blocking work during `draw()`).
- If missing, Panel 5 renders an info box explaining that pure-NumPy mode is active and SciPy is optional, with **`Install SciPy Automatically`** (`kuchisync.install_scipy`).
- Pressing it checks `getattr(bpy.app, "online_access", True)` and runs `subprocess.Popen([sys.executable, "-m", "pip", "install", "scipy", "--target", target_dir])` targeting `bpy.utils.user_resource("SCRIPTS", path="modules", create=True)` (so read-only Steam/system Blender folders do not block installation), monitored on a `0.5s` `bpy.app.timers` callback.
- On exit code `0`, adds `target_dir` via `site.addsitedir()`, calls `importlib.invalidate_caches()` + `audio_dsp.refresh_scipy()`, and triggers 3D Viewport redraw.

## 9. UI Design and Blender N-Panel Structure (`ui_panel.py`)

Follows `DESIGN.md` (`View3D > UI > LipSync`), using `safe_icon()` for Blender 4.4-5.2 compatibility:
1. `KUCHISYNC_PT_rig_shapekeys`: Active object validation + `Auto-Detect Shape Keys` + 5 aligned rows (`[Status Icon] [A/I/U/E/O] [Shape Key prop_search] [Strength 0.0-1.5]`) + partial-mapping notice when 1-4 vowels are mapped.
2. `KUCHISYNC_PT_audio_timing`: Audio filepath + filter browser (`.wav;.mp3;.ogg;.flac`), `Language` (`Indonesian` / `English`), `Start Frame` + sync-to-playhead button, `Add Audio to VSE`, `Clear Existing LipSync Keys First`.
3. `KUCHISYNC_PT_detection_engine`: Mode selector (`Pure Audio (Formant)` vs `Audio + Transcript`), shared `Silence Threshold (dB)` (`-60` to `-10`, default `-38`), plus `Voice Type` (in Pure Audio) or `Transcript` string input + optional `bpy.types.Text` datablock + `Load .TXT File` + preview line/vowel count (in Audio + Transcript).
4. `KUCHISYNC_PT_animation_curve`: `Max Mouth Open`, `Min Phoneme Hold`, `Transition Speed`, `Co-Articulation Overlap`, `Keyframe Interpolation`.
5. `KUCHISYNC_PT_execution`: Readiness / error summary, primary `Generate Auto Lip-Sync` (`scale_y = 1.5`), secondary `Clear Lip-Sync Keyframes`, conditional SciPy install box.

## 10. Builder Calls Recorded

1. **Greedy uniqueness in auto-detection:** Prefer assigning distinct Shape Keys across `A/I/U/E/O` by descending match score rather than letting two vowels claim the same Shape Key unless no alternative exists.
2. **Mode B syllable preservation & same-vowel valley dip:** Keep aligned Mode B G2P syllables as distinct instances instead of collapsing consecutive same-vowel syllables (`"saya makan"`, `"ini"`) into one frozen hold, inserting a partial consonant dip (`0.35 * min_peak`) at the acoustic energy valley between them.
3. **Audaspace (`aud`) API contract:** Read sample rate from `sound.specs[0]` and samples from `sound.data()` (never unpack a tuple), chaining `.rechannel(1).resample(16000, True).cache()` safely with fallback to `ffmpeg` and `wave`.
4. **Read-only Blender SciPy install target:** Install SciPy to `bpy.utils.user_resource("SCRIPTS", path="modules", create=True)` via `--target` and activate with `site.addsitedir()` so Steam/system Blender installs do not fail on read-only `site-packages`.
5. **Multi-line transcript support in Blender UI:** Pair `StringProperty` inline text with an optional `bpy.types.Text` datablock pointer and `Load .TXT File` operator because Blender's `UILayout.prop()` renders string properties as single-line widgets.
6. **Safe icon lookup (`safe_icon`):** Guard built-in Blender UI icons against `UILayout.bl_rna.properties['icon'].enum_items` so 4.4 / 4.5 / 5.2 icon differences cannot break panel drawing.
7. **Slider range expansion for Option 1:** When `peak > 1.0`, raise `key_block.slider_max` to cover `peak` (up to `1.5`) so Blender evaluates and displays the slider cleanly without silent truncation.
8. **Silent-frame sibilant guard:** Combine high `ZCR > 0.35` with spectral high/mid band ratio check so unvoiced fricatives (`s`, `f`, `sh`) stay closed rather than triggering false `I`/`E` peaks.
