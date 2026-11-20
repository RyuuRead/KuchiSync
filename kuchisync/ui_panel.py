# SPDX-License-Identifier: GPL-3.0-or-later

import importlib
import importlib.util
import math
import os
import site
import subprocess
import sys
import tempfile

from . import audio_dsp, keyframe_gen, transcript_align

MAX_LIVE_TRANSCRIPT_PREVIEW_CHARS = 10000


def _has_module(module_name: str) -> bool:
    """Check module availability safely even if bare mocks without __spec__ exist."""
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    return importlib.util.find_spec(module_name) is not None


if _has_module("bpy"):
    import bpy

    try:
        from bpy_extras.io_utils import ImportHelper
    except Exception:
        class ImportHelper:
            """Minimal ImportHelper fallback when bpy_extras is absent in tests."""

            filepath: str = ""
else:
    bpy = None
    ImportHelper = object


def _get_bpy():
    """Return active bpy module, supporting runtime test mocks in sys.modules."""
    if bpy is not None:
        return bpy
    return sys.modules.get("bpy")


def _icon_exists(enum_items, icon_name: str) -> bool:
    """Return True when icon_name exists in Blender's UILayout icon enum items."""
    if not icon_name or enum_items is None:
        return False
    try:
        if icon_name in enum_items:
            return True
    except TypeError:
        pass
    try:
        return any(getattr(item, "identifier", None) == icon_name for item in enum_items)
    except TypeError:
        return False


def _get_icon_enum_items(bpy_mod):
    """Locate Blender UILayout icon enum_items via bl_rna.functions['prop'|'label']."""
    ui_layout = getattr(getattr(bpy_mod, "types", None), "UILayout", None)
    bl_rna = getattr(ui_layout, "bl_rna", None)
    functions = getattr(bl_rna, "functions", None)
    if functions is None:
        return None

    for fn_name in ("prop", "label"):
        try:
            if fn_name not in functions:
                continue
            fn = functions[fn_name]
        except Exception:
            continue
        params = getattr(fn, "parameters", None)
        if params is None:
            continue
        try:
            if "icon" not in params:
                continue
            icon_param = params["icon"]
        except Exception:
            continue
        enum_items = getattr(icon_param, "enum_items", None)
        if enum_items is not None:
            return enum_items
    return None


def safe_icon(icon_name: str, fallback: str = "NONE") -> str:
    """Resolve a Blender UI icon against UILayout RNA enum items with safe fallback."""
    bpy_mod = _get_bpy()
    if bpy_mod is None:
        return fallback
    try:
        enum_items = _get_icon_enum_items(bpy_mod)
        if enum_items is None:
            return fallback
        if _icon_exists(enum_items, icon_name):
            return icon_name
        if _icon_exists(enum_items, fallback):
            return fallback
        return "NONE"
    except Exception:
        return fallback


def _read_text_datablock(text_block) -> str:
    """Extract string contents from a Blender Text datablock or compatible mock."""
    if text_block is None:
        return ""
    as_string = getattr(text_block, "as_string", None)
    if callable(as_string):
        return str(as_string() or "")
    lines = getattr(text_block, "lines", None)
    if lines is not None:
        return "\n".join(str(getattr(line, "body", "")) for line in lines)
    return ""


def get_effective_transcript(props) -> str:
    """Return trimmed transcript text from Text datablock (if non-empty) or inline field."""
    if props is None:
        return ""
    datablock = getattr(props, "transcript_datablock", None)
    if datablock is not None:
        db_text = _read_text_datablock(datablock).strip()
        if db_text:
            return db_text
    return str(getattr(props, "transcript_text", "") or "").strip()


def resolve_audio_path(filepath: str) -> str:
    """Resolve Blender // relative paths when bpy is available and normalize OS path."""
    raw = str(filepath or "").strip()
    if not raw:
        return ""
    bpy_mod = _get_bpy()
    if (
        bpy_mod is not None
        and hasattr(bpy_mod, "path")
        and callable(getattr(bpy_mod.path, "abspath", None))
    ):
        raw = str(bpy_mod.path.abspath(raw))
    if raw.startswith("//"):
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(raw)))


def _key_block_exists(key_blocks, shape_name: str) -> bool:
    """Check whether shape_name exists on a Blender key_blocks collection or mock."""
    if not shape_name or key_blocks is None:
        return False
    try:
        if shape_name in key_blocks:
            return True
    except TypeError:
        pass
    try:
        return any(getattr(kb, "name", None) == shape_name for kb in key_blocks)
    except TypeError:
        return False


def _key_block_names(key_blocks) -> list[str]:
    """Collect Shape Key names cleanly across Blender collections and dict mocks."""
    if key_blocks is None:
        return []
    if isinstance(key_blocks, dict):
        return [str(name) for name in key_blocks.keys()]
    return [str(getattr(kb, "name", kb)) for kb in key_blocks]


def _validate_target_object(obj) -> tuple[bool, str]:
    """Validate that obj is a Mesh with at least one Shape Key block."""
    if obj is None:
        return False, "No active object. Select a Mesh with Shape Keys."
    if getattr(obj, "type", None) != "MESH":
        obj_name = getattr(obj, "name", "Active object")
        return False, f"Object '{obj_name}' is {getattr(obj, 'type', 'unknown')}, not a Mesh."
    shape_keys = getattr(getattr(obj, "data", None), "shape_keys", None)
    key_blocks = getattr(shape_keys, "key_blocks", None) if shape_keys is not None else None
    if shape_keys is None or not key_blocks:
        obj_name = getattr(obj, "name", "Active Mesh")
        return False, f"Mesh '{obj_name}' has no Shape Keys."
    return True, ""


def _has_valid_mesh_with_shapekeys(obj) -> bool:
    """Return True if obj is a Mesh with at least one Shape Key block."""
    ok, _ = _validate_target_object(obj)
    return ok


def _scene_fps(scene) -> float:
    """Return effective timeline frame rate (fps / fps_base) with safe 24.0 fallback."""
    render = getattr(scene, "render", None)
    raw_fps = float(getattr(render, "fps", 24) or 24)
    fps_base = float(getattr(render, "fps_base", 1.0) or 1.0)
    fps = raw_fps / fps_base if fps_base > 0.0 else raw_fps
    return fps if fps > 0.0 else 24.0


def _collect_rig_mapping(props, key_blocks) -> tuple[dict[str, str], dict[str, float]]:
    """Collect valid mapped vowel Shape Keys and per-vowel strength multipliers."""
    mapped_shapes: dict[str, str] = {}
    vowel_strengths: dict[str, float] = {}
    for vowel in keyframe_gen.VOWELS:
        v_low = vowel.lower()
        shape_name = str(getattr(props, f"shape_{v_low}", "") or "").strip()
        if shape_name and _key_block_exists(key_blocks, shape_name):
            mapped_shapes[vowel] = shape_name
        vowel_strengths[vowel] = float(getattr(props, f"strength_{v_low}", 1.0))
    return mapped_shapes, vowel_strengths


def validate_generation_context(obj, props) -> tuple[bool, str]:
    """Validate active rig object, Shape Key mapping, audio file, and transcript mode."""
    if props is None:
        return False, "KuchiSync settings are unavailable."

    target_ok, target_msg = _validate_target_object(obj)
    if not target_ok:
        return False, target_msg

    key_blocks = obj.data.shape_keys.key_blocks

    mapped_valid = 0
    missing_names: list[str] = []
    for vowel in keyframe_gen.VOWELS:
        shape_name = str(getattr(props, f"shape_{vowel.lower()}", "") or "").strip()
        if not shape_name:
            continue
        if _key_block_exists(key_blocks, shape_name):
            mapped_valid += 1
        else:
            missing_names.append(f"{vowel}: {shape_name}")

    if mapped_valid == 0:
        if missing_names:
            return (
                False,
                f"Mapped Shape Key not found on active Mesh: {', '.join(missing_names)}.",
            )
        return False, "Map at least one vowel (A, I, U, E, O) to a Shape Key."

    raw_audio = str(getattr(props, "audio_filepath", "") or "").strip()
    if not raw_audio:
        return False, "Select an audio file."
    resolved_audio = resolve_audio_path(raw_audio)
    if not resolved_audio or not os.path.isfile(resolved_audio):
        return False, f"Audio file not found: {raw_audio}"

    mode = str(getattr(props, "detection_mode", "PURE_AUDIO") or "PURE_AUDIO").strip().upper()
    if mode == "TRANSCRIPT" and not get_effective_transcript(props):
        return False, "Enter a transcript or select a Text datablock."

    return True, "Ready to generate lip-sync keyframes."


_scipy_install_state: dict[str, object] = {
    "running": False,
    "process": None,
    "log_file": None,
    "target_dir": "",
    "status_message": "",
    "error_message": "",
}


def _close_install_log() -> None:
    """Close temporary pip output log handle if open."""
    log_file = _scipy_install_state.get("log_file")
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
        _scipy_install_state["log_file"] = None


def _read_install_log_tail() -> str:
    """Read a concise error tail from the pip temporary log file."""
    log_file = _scipy_install_state.get("log_file")
    if log_file is None:
        return ""
    try:
        log_file.flush()
        log_file.seek(0)
        raw = log_file.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        return lines[-1][:180]
    except Exception:
        return ""


def _redraw_view3d() -> None:
    """Request a UI redraw on open 3D Viewport areas."""
    bpy_mod = _get_bpy()
    if bpy_mod is None:
        return
    wm_list = []
    ctx_wm = getattr(getattr(bpy_mod, "context", None), "window_manager", None)
    if ctx_wm is not None:
        wm_list.append(ctx_wm)
    else:
        wm_list.extend(getattr(getattr(bpy_mod, "data", None), "window_managers", ()))

    for wm in wm_list:
        for window in getattr(wm, "windows", ()):
            screen = getattr(window, "screen", None)
            if screen is None:
                continue
            for area in getattr(screen, "areas", ()):
                if getattr(area, "type", None) == "VIEW_3D":
                    area.tag_redraw()


def _poll_scipy_install():
    """Poll background pip SciPy installation every 0.5s without blocking Blender UI."""
    proc = _scipy_install_state.get("process")
    if proc is None:
        _scipy_install_state["running"] = False
        _close_install_log()
        return None

    rc = proc.poll()
    if rc is None:
        return 0.5

    target_dir = str(_scipy_install_state.get("target_dir") or "")
    tail = _read_install_log_tail()
    _close_install_log()
    _scipy_install_state["process"] = None
    _scipy_install_state["running"] = False

    if rc == 0:
        if target_dir:
            site.addsitedir(target_dir)
        importlib.invalidate_caches()
        if audio_dsp.refresh_scipy():
            _scipy_install_state["status_message"] = "SciPy installed and active."
            _scipy_install_state["error_message"] = ""
        else:
            _scipy_install_state["status_message"] = ""
            _scipy_install_state["error_message"] = (
                "SciPy installed, but scipy.signal import failed. Restart Blender."
            )
    else:
        _scipy_install_state["status_message"] = ""
        detail = f": {tail}" if tail else ""
        _scipy_install_state["error_message"] = f"SciPy install failed (exit {rc}){detail}"

    _redraw_view3d()
    return None


def _cleanup_scipy_installer() -> None:
    """Cancel active SciPy install timer and terminate running subprocess on unregister."""
    bpy_mod = _get_bpy()
    timers = getattr(getattr(bpy_mod, "app", None), "timers", None) if bpy_mod else None
    if timers is not None:
        try:
            if timers.is_registered(_poll_scipy_install):
                timers.unregister(_poll_scipy_install)
        except Exception:
            pass

    proc = _scipy_install_state.get("process")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
        except Exception:
            pass

    _close_install_log()
    _scipy_install_state["process"] = None
    _scipy_install_state["running"] = False
    _scipy_install_state["status_message"] = ""
    _scipy_install_state["error_message"] = ""


if bpy is not None:

    class KuchiSyncSettings(bpy.types.PropertyGroup):
        """Scene-level settings for KuchiSync lip-sync generation."""

        shape_a: bpy.props.StringProperty(
            name="A",
            description="Shape Key name mapped to vowel A",
            default="",
        )
        shape_i: bpy.props.StringProperty(
            name="I",
            description="Shape Key name mapped to vowel I",
            default="",
        )
        shape_u: bpy.props.StringProperty(
            name="U",
            description="Shape Key name mapped to vowel U",
            default="",
        )
        shape_e: bpy.props.StringProperty(
            name="E",
            description="Shape Key name mapped to vowel E",
            default="",
        )
        shape_o: bpy.props.StringProperty(
            name="O",
            description="Shape Key name mapped to vowel O",
            default="",
        )

        strength_a: bpy.props.FloatProperty(
            name="Strength A",
            description="Peak multiplier for vowel A (0.0 disables, above 1.0 expands slider_max)",
            min=0.0,
            max=1.5,
            default=1.0,
            precision=2,
            step=5,
            subtype="FACTOR",
        )
        strength_i: bpy.props.FloatProperty(
            name="Strength I",
            description="Peak multiplier for vowel I (0.0 disables, above 1.0 expands slider_max)",
            min=0.0,
            max=1.5,
            default=1.0,
            precision=2,
            step=5,
            subtype="FACTOR",
        )
        strength_u: bpy.props.FloatProperty(
            name="Strength U",
            description="Peak multiplier for vowel U (0.0 disables, above 1.0 expands slider_max)",
            min=0.0,
            max=1.5,
            default=1.0,
            precision=2,
            step=5,
            subtype="FACTOR",
        )
        strength_e: bpy.props.FloatProperty(
            name="Strength E",
            description="Peak multiplier for vowel E (0.0 disables, above 1.0 expands slider_max)",
            min=0.0,
            max=1.5,
            default=1.0,
            precision=2,
            step=5,
            subtype="FACTOR",
        )
        strength_o: bpy.props.FloatProperty(
            name="Strength O",
            description="Peak multiplier for vowel O (0.0 disables, above 1.0 expands slider_max)",
            min=0.0,
            max=1.5,
            default=1.0,
            precision=2,
            step=5,
            subtype="FACTOR",
        )

        audio_filepath: bpy.props.StringProperty(
            name="Audio File",
            description="Dialogue audio file (.wav, .mp3, .ogg, .flac)",
            default="",
            subtype="FILE_PATH",
        )
        language: bpy.props.EnumProperty(
            name="Language",
            description="Phoneme centroid and G2P rules language",
            items=[
                ("INDONESIAN", "Indonesian", "Indonesian vowels and syllable rules"),
                ("ENGLISH", "English", "English vowels and pronunciation rules"),
            ],
            default="INDONESIAN",
        )
        start_frame: bpy.props.IntProperty(
            name="Start Frame",
            description="Timeline frame where the audio clip begins",
            default=1,
            min=0,
        )
        add_to_vse: bpy.props.BoolProperty(
            name="Add Audio to VSE",
            description="Insert audio strip into Video Sequencer and enable Audio Sync",
            default=True,
        )
        clear_existing_keys: bpy.props.BoolProperty(
            name="Clear Existing LipSync Keys First",
            description="Clear prior keyframes on mapped vowel Shape Keys within the generated clip range",
            default=True,
        )

        detection_mode: bpy.props.EnumProperty(
            name="Detection Mode",
            description="Select pure-audio formant detection or transcript-guided alignment",
            items=[
                ("PURE_AUDIO", "Pure Audio (Formant)", "Track F1/F2 formants directly from audio"),
                ("TRANSCRIPT", "Audio + Transcript", "Align spoken transcript vowels to audio islands"),
            ],
            default="PURE_AUDIO",
        )
        voice_type: bpy.props.EnumProperty(
            name="Voice Type",
            description="Vocal tract scaling preset for Pure Audio formant classification",
            items=[
                ("HIGH_CHILD", "High / Child", "Higher F1/F2 formant scale"),
                ("FEMALE_STANDARD", "Female / Standard", "Neutral F1/F2 formant scale"),
                ("MALE_DEEP", "Male / Deep", "Lower F1/F2 formant scale"),
            ],
            default="FEMALE_STANDARD",
        )
        silence_threshold_db: bpy.props.FloatProperty(
            name="Silence Threshold (dB)",
            description="Shared loudness threshold in dBFS separating speech from silence",
            min=-60.0,
            max=-10.0,
            default=-38.0,
            precision=1,
            step=50,
        )
        transcript_text: bpy.props.StringProperty(
            name="Transcript",
            description="Inline dialogue text for Audio + Transcript mode",
            default="",
        )
        transcript_datablock: bpy.props.PointerProperty(
            name="Text Datablock",
            description="Optional Blender Text datablock for multi-line transcripts",
            type=bpy.types.Text,
        )

        max_mouth_open: bpy.props.FloatProperty(
            name="Max Mouth Open",
            description="Global mouth amplitude multiplier before per-vowel strength",
            min=0.1,
            max=1.5,
            default=1.0,
            precision=2,
            step=5,
        )
        min_phoneme_frames: bpy.props.IntProperty(
            name="Min Phoneme Hold",
            description="Minimum phoneme segment duration in frames for Pure Audio filtering",
            min=1,
            max=10,
            default=3,
        )
        transition_frames: bpy.props.IntProperty(
            name="Transition Speed",
            description="Attack and release ramp length in frames",
            min=1,
            max=6,
            default=2,
        )
        co_articulation: bpy.props.FloatProperty(
            name="Co-Articulation Overlap",
            description="Cross-fade overlap ratio between consecutive distinct vowels",
            min=0.0,
            max=0.5,
            default=0.25,
            precision=2,
            step=5,
            subtype="FACTOR",
        )
        interpolation_mode: bpy.props.EnumProperty(
            name="Keyframe Interpolation",
            description="Interpolation mode assigned to generated Shape Key keyframes",
            items=[
                ("BEZIER", "Bezier (Auto Clamped)", "Smooth Bezier curves with Auto Clamped handles"),
                ("LINEAR", "Linear", "Linear transitions between keyframes"),
                ("CONSTANT", "Constant (Stepped)", "Stepped holds for limited animation"),
            ],
            default="BEZIER",
        )

    class KUCHISYNC_OT_auto_detect_shapekeys(bpy.types.Operator):
        """Scan active Mesh Shape Keys and populate A, I, U, E, O slots."""

        bl_idname = "lipsync.auto_detect_shapekeys"
        bl_label = "Auto-Detect Shape Keys"
        bl_description = "Scan active Mesh Shape Keys and match A, I, U, E, O vowel slots"
        bl_options = {"REGISTER", "UNDO"}

        @classmethod
        def poll(cls, context) -> bool:
            scene = getattr(context, "scene", None)
            obj = getattr(context, "active_object", None)
            return (
                scene is not None
                and hasattr(scene, "kuchisync")
                and _has_valid_mesh_with_shapekeys(obj)
            )

        def execute(self, context):
            obj = getattr(context, "active_object", None)
            if not _has_valid_mesh_with_shapekeys(obj):
                self.report({"ERROR"}, "Select a Mesh object with Shape Keys first.")
                return {"CANCELLED"}

            key_names = _key_block_names(obj.data.shape_keys.key_blocks)
            detected = keyframe_gen.detect_shapekeys(key_names)
            props = context.scene.kuchisync
            for vowel in keyframe_gen.VOWELS:
                setattr(props, f"shape_{vowel.lower()}", detected.get(vowel, ""))

            if not detected:
                self.report(
                    {"WARNING"},
                    "No A-I-U-E-O Shape Keys matched automatically. Select them manually.",
                )
            else:
                self.report({"INFO"}, f"Mapped {len(detected)} of 5 vowel Shape Keys.")
            return {"FINISHED"}

    class KUCHISYNC_OT_browse_audio(bpy.types.Operator, ImportHelper):
        """Open Blender file browser to select a dialogue audio file."""

        bl_idname = "kuchisync.browse_audio"
        bl_label = "Select Audio File"
        bl_description = "Choose a dialogue audio file (.wav, .mp3, .ogg, .flac)"
        bl_options = {"REGISTER", "UNDO"}

        filename_ext = ""
        filter_glob: bpy.props.StringProperty(
            default="*.wav;*.mp3;*.ogg;*.flac",
            options={"HIDDEN"},
        )

        def invoke(self, context, event):
            props = getattr(getattr(context, "scene", None), "kuchisync", None)
            if props is not None and not getattr(self, "filepath", ""):
                current = str(getattr(props, "audio_filepath", "") or "").strip()
                resolved = resolve_audio_path(current) if current else ""
                if resolved and os.path.exists(resolved):
                    self.filepath = resolved
            return super().invoke(context, event)

        def execute(self, context):
            scene = getattr(context, "scene", None)
            if scene is None or not hasattr(scene, "kuchisync"):
                self.report({"ERROR"}, "KuchiSync settings are unavailable.")
                return {"CANCELLED"}
            scene.kuchisync.audio_filepath = self.filepath
            return {"FINISHED"}

    class KUCHISYNC_OT_set_start_frame(bpy.types.Operator):
        """Sync lip-sync Start Frame to the active timeline frame."""

        bl_idname = "kuchisync.set_start_frame"
        bl_label = "Use Current Frame"
        bl_description = "Set Start Frame to the current scene playhead frame"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            scene = getattr(context, "scene", None)
            if scene is None or not hasattr(scene, "kuchisync"):
                self.report({"ERROR"}, "KuchiSync settings are unavailable.")
                return {"CANCELLED"}
            scene.kuchisync.start_frame = max(0, int(getattr(scene, "frame_current", 1)))
            return {"FINISHED"}

    class KUCHISYNC_OT_load_transcript(bpy.types.Operator, ImportHelper):
        """Load UTF-8 dialogue transcript text from a .txt file."""

        bl_idname = "kuchisync.load_transcript"
        bl_label = "Load .TXT File"
        bl_description = "Load dialogue transcript text from a UTF-8 .txt file"
        bl_options = {"REGISTER", "UNDO"}

        filename_ext = ".txt"
        filter_glob: bpy.props.StringProperty(
            default="*.txt",
            options={"HIDDEN"},
        )

        def execute(self, context):
            scene = getattr(context, "scene", None)
            if scene is None or not hasattr(scene, "kuchisync"):
                self.report({"ERROR"}, "KuchiSync settings are unavailable.")
                return {"CANCELLED"}

            resolved = resolve_audio_path(self.filepath)
            if not resolved or not os.path.isfile(resolved):
                self.report({"ERROR"}, f"Transcript file not found: {self.filepath}")
                return {"CANCELLED"}

            try:
                with open(resolved, "r", encoding="utf-8-sig") as handle:
                    content = handle.read()
            except UnicodeDecodeError as exc:
                self.report({"ERROR"}, f"Transcript file must be UTF-8 encoded: {exc}")
                return {"CANCELLED"}
            except OSError as exc:
                self.report({"ERROR"}, f"Failed to read transcript file: {exc}")
                return {"CANCELLED"}

            props = scene.kuchisync
            props.transcript_text = content
            props.transcript_datablock = None
            self.report({"INFO"}, f"Loaded transcript ({len(content)} characters).")
            return {"FINISHED"}

    class KUCHISYNC_OT_generate_lipsync(bpy.types.Operator):
        """Analyze audio and write sparse A-I-U-E-O Shape Key keyframes."""

        bl_idname = "kuchisync.generate_lipsync"
        bl_label = "Generate Auto Lip-Sync"
        bl_description = "Generate sparse A-I-U-E-O Shape Key keyframes from audio"
        bl_options = {"REGISTER", "UNDO"}

        @classmethod
        def poll(cls, context) -> bool:
            scene = getattr(context, "scene", None)
            return scene is not None and hasattr(scene, "kuchisync")

        def execute(self, context):
            scene = getattr(context, "scene", None)
            if scene is None or not hasattr(scene, "kuchisync"):
                self.report({"ERROR"}, "KuchiSync settings are unavailable.")
                return {"CANCELLED"}

            props = scene.kuchisync
            obj = getattr(context, "active_object", None)
            ok, msg = validate_generation_context(obj, props)
            if not ok:
                self.report({"ERROR"}, msg)
                return {"CANCELLED"}

            audio_path = resolve_audio_path(props.audio_filepath)
            fps = _scene_fps(scene)

            try:
                clip = audio_dsp.load_audio(audio_path, target_sr=audio_dsp.TARGET_SAMPLE_RATE)
            except audio_dsp.AudioLoadError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            key_blocks = obj.data.shape_keys.key_blocks
            mapped_shapes, vowel_strengths = _collect_rig_mapping(props, key_blocks)

            start_frame = int(props.start_frame)
            transition_frames = int(props.transition_frames)
            min_phoneme_frames = int(props.min_phoneme_frames)

            if props.detection_mode == "TRANSCRIPT":
                transcript = get_effective_transcript(props)
                try:
                    syllables = transcript_align.align_transcript(
                        clip.samples,
                        clip.sample_rate,
                        transcript,
                        language=props.language,
                        silence_threshold_db=float(props.silence_threshold_db),
                        fps=fps,
                        start_frame=start_frame,
                    )
                except transcript_align.TranscriptError as exc:
                    self.report({"ERROR"}, str(exc))
                    return {"CANCELLED"}
                preserve_syllables = True
            else:
                frames = audio_dsp.analyze_audio_frames(
                    clip.samples,
                    sr=clip.sample_rate,
                    silence_threshold_db=float(props.silence_threshold_db),
                    voice_type=props.voice_type,
                    language=props.language,
                )
                syllables = keyframe_gen.build_mode_a_segments(
                    frames,
                    fps=fps,
                    start_frame=start_frame,
                    min_phoneme_frames=min_phoneme_frames,
                )
                preserve_syllables = False

            if props.clear_existing_keys:
                clear_start = max(0, start_frame - transition_frames)
                clear_end = math.ceil(start_frame + clip.duration_sec * fps) + transition_frames
                keyframe_gen.clear_lipsync_keyframes(
                    obj,
                    sorted(set(mapped_shapes.values())),
                    frame_range=(clear_start, clear_end),
                )

            scheduled = keyframe_gen.schedule_keyframes(
                syllables=syllables,
                mapped_shapes=mapped_shapes,
                vowel_strengths=vowel_strengths,
                start_frame=start_frame,
                max_mouth_open=float(props.max_mouth_open),
                min_phoneme_frames=min_phoneme_frames,
                transition_frames=transition_frames,
                co_articulation=float(props.co_articulation),
                preserve_syllables=preserve_syllables,
            )
            written = keyframe_gen.apply_keyframes_to_object(
                obj,
                scheduled,
                interpolation_mode=props.interpolation_mode,
            )

            if props.add_to_vse:
                try:
                    keyframe_gen.add_audio_to_vse(scene, audio_path, start_frame)
                except Exception as exc:
                    self.report(
                        {"WARNING"},
                        f"Generated {written} keyframes, but VSE audio insert failed: {exc}",
                    )
                    return {"FINISHED"}

            if written == 0:
                self.report(
                    {"WARNING"},
                    "No voiced speech exceeded the silence threshold; 0 keyframes generated.",
                )
            else:
                self.report(
                    {"INFO"},
                    f"Generated {written} lip-sync keyframes across {len(scheduled)} Shape Keys.",
                )
            return {"FINISHED"}

    class KUCHISYNC_OT_clear_lipsync(bpy.types.Operator):
        """Remove all keyframes on mapped vowel Shape Keys across the entire timeline."""

        bl_idname = "kuchisync.clear_lipsync"
        bl_label = "Clear Lip-Sync Keyframes"
        bl_description = "Clear all keyframes on mapped A, I, U, E, O Shape Keys across the timeline"
        bl_options = {"REGISTER", "UNDO"}

        @classmethod
        def poll(cls, context) -> bool:
            scene = getattr(context, "scene", None)
            obj = getattr(context, "active_object", None)
            return (
                scene is not None
                and hasattr(scene, "kuchisync")
                and _has_valid_mesh_with_shapekeys(obj)
            )

        def execute(self, context):
            scene = getattr(context, "scene", None)
            if scene is None or not hasattr(scene, "kuchisync"):
                self.report({"ERROR"}, "KuchiSync settings are unavailable.")
                return {"CANCELLED"}
            obj = getattr(context, "active_object", None)
            if not _has_valid_mesh_with_shapekeys(obj):
                self.report({"ERROR"}, "Select a Mesh object with Shape Keys first.")
                return {"CANCELLED"}

            props = scene.kuchisync
            target_names = sorted(
                {
                    str(getattr(props, f"shape_{v.lower()}", "") or "").strip()
                    for v in keyframe_gen.VOWELS
                }
                - {""}
            )
            if not target_names:
                self.report({"WARNING"}, "No vowel Shape Keys are mapped to clear.")
                return {"CANCELLED"}

            removed = keyframe_gen.clear_lipsync_keyframes(obj, target_names, frame_range=None)
            self.report({"INFO"}, f"Cleared {removed} lip-sync keyframes.")
            return {"FINISHED"}

    class KUCHISYNC_OT_install_scipy(bpy.types.Operator):
        """Install optional SciPy package into Blender's user scripts/modules folder."""

        bl_idname = "kuchisync.install_scipy"
        bl_label = "Install SciPy Automatically"
        bl_description = "Download and install optional SciPy DSP dependency into user modules"
        bl_options = {"REGISTER"}

        @classmethod
        def poll(cls, context) -> bool:
            del context
            return not bool(_scipy_install_state.get("running", False))

        def _fail(self, msg: str) -> set[str]:
            _scipy_install_state["error_message"] = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        def execute(self, context):
            del context
            if audio_dsp.has_scipy():
                _scipy_install_state["status_message"] = "SciPy is already installed."
                _scipy_install_state["error_message"] = ""
                self.report({"INFO"}, "SciPy is already installed.")
                return {"FINISHED"}

            if _scipy_install_state.get("running"):
                self.report({"INFO"}, "SciPy installation is already running.")
                return {"CANCELLED"}

            bpy_mod = _get_bpy()
            if bpy_mod is None:
                return self._fail("Blender runtime is unavailable.")

            if not getattr(bpy_mod.app, "online_access", True):
                return self._fail(
                    "Online access is disabled in Blender Preferences (System > Network)."
                )

            target_dir = bpy_mod.utils.user_resource("SCRIPTS", path="modules", create=True)
            if not target_dir:
                return self._fail(
                    "Could not locate writable Blender user scripts/modules directory."
                )

            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError as exc:
                return self._fail(f"Target modules directory is not writable: {exc}")

            if not os.access(target_dir, os.W_OK):
                return self._fail(f"Target modules directory is read-only: {target_dir}")

            _close_install_log()
            log_file = None
            try:
                log_file = tempfile.TemporaryFile(mode="w+b")
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "scipy", "--target", target_dir],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            except OSError as exc:
                if log_file is not None:
                    try:
                        log_file.close()
                    except Exception:
                        pass
                _close_install_log()
                return self._fail(f"Failed to start pip subprocess: {exc}")

            _scipy_install_state["running"] = True
            _scipy_install_state["process"] = proc
            _scipy_install_state["log_file"] = log_file
            _scipy_install_state["target_dir"] = target_dir
            _scipy_install_state["status_message"] = "Installing SciPy..."
            _scipy_install_state["error_message"] = ""

            try:
                bpy_mod.app.timers.register(_poll_scipy_install, first_interval=0.5)
            except Exception as exc:
                _cleanup_scipy_installer()
                return self._fail(f"Failed to register SciPy install timer: {exc}")

            _redraw_view3d()
            self.report({"INFO"}, "Installing SciPy in background...")
            return {"FINISHED"}

    class _KuchiSyncBasePanel(bpy.types.Panel):
        """Shared panel location metadata for View3D > UI > LipSync."""

        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "LipSync"

        @classmethod
        def poll(cls, context) -> bool:
            scene = getattr(context, "scene", None)
            return scene is not None and hasattr(scene, "kuchisync")

    class KUCHISYNC_PT_rig_shapekeys(_KuchiSyncBasePanel):
        """Panel 1: Target Mesh validation, Shape Key auto-detect, and vowel strength slots."""

        bl_idname = "KUCHISYNC_PT_rig_shapekeys"
        bl_label = "Target Rig & Shape Keys"
        bl_order = 0

        def draw(self, context):
            layout = self.layout
            props = context.scene.kuchisync
            obj = getattr(context, "active_object", None)

            target_ok, target_msg = _validate_target_object(obj)
            if not target_ok:
                box = layout.box()
                if obj is not None:
                    box.alert = True
                box.label(
                    text=target_msg,
                    icon=safe_icon("INFO" if obj is None else "ERROR"),
                )
                return

            shape_keys = obj.data.shape_keys
            key_blocks = shape_keys.key_blocks

            layout.label(
                text=f"Target: {obj.name}",
                icon=safe_icon("MESH_DATA", "OBJECT_DATA"),
            )
            layout.operator(
                "lipsync.auto_detect_shapekeys",
                text="Auto-Detect Shape Keys",
                icon=safe_icon("VIEWZOOM"),
            )

            slots_box = layout.box()
            header = slots_box.split(factor=0.22, align=True)
            header.label(text="Vowel")
            header_right = header.split(factor=0.62, align=True)
            header_right.label(text="Shape Key")
            header_right.label(text="Strength")

            mapped_count = 0
            for vowel in keyframe_gen.VOWELS:
                v_low = vowel.lower()
                shape_prop = f"shape_{v_low}"
                strength_prop = f"strength_{v_low}"
                shape_name = str(getattr(props, shape_prop, "") or "").strip()
                is_mapped = bool(shape_name and _key_block_exists(key_blocks, shape_name))
                if is_mapped:
                    mapped_count += 1

                row = slots_box.row(align=True)
                if is_mapped:
                    status_icon = safe_icon("CHECKMARK")
                else:
                    status_icon = safe_icon("ERROR")
                    row.alert = bool(shape_name)

                left = row.split(factor=0.22, align=True)
                left.label(text=vowel, icon=status_icon)

                right = left.split(factor=0.62, align=True)
                right.prop_search(props, shape_prop, shape_keys, "key_blocks", text="")
                strength_col = right.column(align=True)
                strength_col.enabled = is_mapped
                strength_col.prop(props, strength_prop, text="", slider=True)

            if mapped_count == 0:
                note = layout.box()
                note.alert = True
                note.label(
                    text="0 of 5 mapped: assign at least 1 Shape Key.",
                    icon=safe_icon("ERROR"),
                )
            elif mapped_count < 5:
                note = layout.box()
                note.label(
                    text=f"Mapped {mapped_count} of 5 vowels (unmapped use nearest fallback).",
                    icon=safe_icon("INFO"),
                )
            else:
                layout.label(
                    text="All 5 vowel Shape Keys mapped.",
                    icon=safe_icon("CHECKMARK"),
                )

    class KUCHISYNC_PT_audio_timing(_KuchiSyncBasePanel):
        """Panel 2: Audio clip picker, language, start frame sync, and VSE/clearing options."""

        bl_idname = "KUCHISYNC_PT_audio_timing"
        bl_label = "Audio & Timing"
        bl_order = 1

        def draw(self, context):
            layout = self.layout
            props = context.scene.kuchisync

            audio_row = layout.row(align=True)
            audio_row.prop(props, "audio_filepath", text="Audio")
            audio_row.operator(
                "kuchisync.browse_audio",
                text="",
                icon=safe_icon("FILE_FOLDER"),
            )

            raw_audio = str(getattr(props, "audio_filepath", "") or "").strip()
            if raw_audio:
                resolved = resolve_audio_path(raw_audio)
                if not resolved or not os.path.isfile(resolved):
                    err_row = layout.row()
                    err_row.alert = True
                    err_row.label(
                        text="Audio file not found on disk.",
                        icon=safe_icon("ERROR"),
                    )

            layout.prop(props, "language", text="Language")

            sf_row = layout.row(align=True)
            sf_row.prop(props, "start_frame", text="Start Frame")
            sf_row.operator(
                "kuchisync.set_start_frame",
                text="",
                icon=safe_icon("TIME"),
            )

            layout.prop(props, "add_to_vse")
            layout.prop(props, "clear_existing_keys")

    class KUCHISYNC_PT_detection_engine(_KuchiSyncBasePanel):
        """Panel 3: Mode selection, shared silence threshold, and voice/transcript inputs."""

        bl_idname = "KUCHISYNC_PT_detection_engine"
        bl_label = "Detection Engine"
        bl_order = 2

        def draw(self, context):
            layout = self.layout
            props = context.scene.kuchisync

            layout.prop(props, "detection_mode", text="Mode")
            layout.prop(props, "silence_threshold_db", slider=True)

            if props.detection_mode == "PURE_AUDIO":
                layout.prop(props, "voice_type", text="Voice Type")
                return

            datablock = getattr(props, "transcript_datablock", None)
            using_db = bool(datablock is not None and _read_text_datablock(datablock).strip())

            inline_row = layout.row()
            inline_row.enabled = not using_db
            inline_row.prop(props, "transcript_text", text="Transcript")
            layout.prop(props, "transcript_datablock", text="Text Block")
            layout.operator(
                "kuchisync.load_transcript",
                text="Load .TXT File",
                icon=safe_icon("FILE_FOLDER"),
            )

            effective = get_effective_transcript(props)
            summary_box = layout.box()
            if not effective:
                summary_box.alert = True
                summary_box.label(
                    text="Transcript empty: enter text, pick Text Block, or load .TXT.",
                    icon=safe_icon("ERROR"),
                )
                return

            line_count = len(effective.splitlines())
            source_label = (
                f"Text '{getattr(datablock, 'name', 'Text')}'"
                if using_db
                else "Inline text"
            )

            if len(effective) > MAX_LIVE_TRANSCRIPT_PREVIEW_CHARS:
                summary_box.label(
                    text=f"{source_label}: {line_count} line(s), {len(effective)} chars.",
                    icon=safe_icon("INFO"),
                )
                return

            try:
                phrases = transcript_align.extract_vowels(effective, props.language)
                vowel_count = sum(len(ph) for ph in phrases)
                summary_box.label(
                    text=f"{source_label}: {line_count} line(s), {vowel_count} vowel(s).",
                    icon=safe_icon("CHECKMARK"),
                )
            except transcript_align.TranscriptError:
                summary_box.alert = True
                summary_box.label(
                    text=f"{source_label}: {line_count} line(s), 0 vowel targets.",
                    icon=safe_icon("ERROR"),
                )

    class KUCHISYNC_PT_animation_curve(_KuchiSyncBasePanel):
        """Panel 4: Amplitude, phoneme hold, transition, co-articulation, and curve controls."""

        bl_idname = "KUCHISYNC_PT_animation_curve"
        bl_label = "Animation & Curve Refinement"
        bl_order = 3

        def draw(self, context):
            layout = self.layout
            props = context.scene.kuchisync

            layout.prop(props, "max_mouth_open", slider=True)
            layout.prop(props, "min_phoneme_frames")
            layout.prop(props, "transition_frames")
            layout.prop(props, "co_articulation", slider=True)
            layout.prop(props, "interpolation_mode", text="Interpolation")

    class KUCHISYNC_PT_execution(_KuchiSyncBasePanel):
        """Panel 5: Readiness check, Generate/Clear actions, and optional SciPy installer."""

        bl_idname = "KUCHISYNC_PT_execution"
        bl_label = "Execution & Utilities"
        bl_order = 4

        def draw(self, context):
            layout = self.layout
            props = context.scene.kuchisync
            obj = getattr(context, "active_object", None)

            ok, status_msg = validate_generation_context(obj, props)
            status_box = layout.box()
            if ok:
                status_box.label(text=status_msg, icon=safe_icon("CHECKMARK"))
            else:
                status_box.alert = True
                status_box.label(text=status_msg, icon=safe_icon("ERROR"))

            gen_row = layout.row()
            gen_row.enabled = ok
            gen_row.scale_y = 1.5
            gen_row.operator(
                "kuchisync.generate_lipsync",
                text="Generate Auto Lip-Sync",
                icon=safe_icon("PLAY"),
            )

            clr_row = layout.row()
            clr_row.enabled = _has_valid_mesh_with_shapekeys(obj)
            clr_row.operator(
                "kuchisync.clear_lipsync",
                text="Clear Lip-Sync Keyframes",
                icon=safe_icon("TRASH"),
            )

            if audio_dsp.has_scipy():
                layout.label(
                    text="DSP Backend: SciPy active",
                    icon=safe_icon("CHECKMARK"),
                )
            else:
                scipy_box = layout.box()
                scipy_box.label(
                    text="DSP Backend: Pure NumPy (SciPy optional)",
                    icon=safe_icon("INFO"),
                )
                is_running = bool(_scipy_install_state.get("running", False))
                err_msg = str(_scipy_install_state.get("error_message") or "")
                if is_running:
                    scipy_box.label(
                        text="Installing SciPy...",
                        icon=safe_icon("TIME"),
                    )
                elif err_msg:
                    err_row = scipy_box.row()
                    err_row.alert = True
                    err_row.label(
                        text=err_msg[:90],
                        icon=safe_icon("ERROR"),
                    )

                btn_row = scipy_box.row()
                btn_row.enabled = not is_running
                btn_row.operator(
                    "kuchisync.install_scipy",
                    text="Install SciPy Automatically",
                    icon=safe_icon("IMPORT", "FILE_REFRESH"),
                )

    _CLASSES = (
        KuchiSyncSettings,
        KUCHISYNC_OT_auto_detect_shapekeys,
        KUCHISYNC_OT_browse_audio,
        KUCHISYNC_OT_set_start_frame,
        KUCHISYNC_OT_load_transcript,
        KUCHISYNC_OT_generate_lipsync,
        KUCHISYNC_OT_clear_lipsync,
        KUCHISYNC_OT_install_scipy,
        KUCHISYNC_PT_rig_shapekeys,
        KUCHISYNC_PT_audio_timing,
        KUCHISYNC_PT_detection_engine,
        KUCHISYNC_PT_animation_curve,
        KUCHISYNC_PT_execution,
    )
else:
    _CLASSES = ()


def register() -> None:
    """Register KuchiSync property group, operators, and N-panels inside Blender."""
    if bpy is None:
        return
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kuchisync = bpy.props.PointerProperty(type=KuchiSyncSettings)


def unregister() -> None:
    """Cancel active SciPy install tasks and unregister KuchiSync Blender classes."""
    _cleanup_scipy_installer()
    if bpy is None:
        return
    if hasattr(bpy.types.Scene, "kuchisync"):
        del bpy.types.Scene.kuchisync
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
