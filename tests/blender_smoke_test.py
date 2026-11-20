# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless Blender smoke test for KuchiSync across Blender 4.4.3, 4.5 LTS, and 5.2 LTS."""

import importlib
import math
import pathlib
import shutil
import subprocess
import tempfile
import warnings
import wave

import addon_utils
import bpy
import numpy as np


EXT_MODULE = "bl_ext.user_default.kuchisync"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "dist" / "kuchisync.zip"

REQUIRED_ICONS = (
    "CHECKMARK",
    "ERROR",
    "INFO",
    "VIEWZOOM",
    "FILE_FOLDER",
    "TIME",
    "PLAY",
    "TRASH",
    "IMPORT",
    "MESH_DATA",
)

FALLBACK_ICONS = (
    "OBJECT_DATA",
    "FILE_REFRESH",
)

PANEL_IDS = (
    "KUCHISYNC_PT_rig_shapekeys",
    "KUCHISYNC_PT_audio_timing",
    "KUCHISYNC_PT_detection_engine",
    "KUCHISYNC_PT_animation_curve",
    "KUCHISYNC_PT_execution",
)


class _LayoutSpy:
    """Minimal UILayout probe to exercise Panel.draw() against real Blender RNA data."""

    def __init__(
        self,
        calls: list[tuple[str, dict]] | None = None,
        parent: "_LayoutSpy | None" = None,
    ) -> None:
        self.calls = calls if calls is not None else []
        self._parent = parent
        self.alert = False
        self.enabled = True
        self.scale_y = 1.0

    def _effective_alert(self) -> bool:
        if self._parent is not None:
            return self.alert or self._parent._effective_alert()
        return self.alert

    def _child(self) -> "_LayoutSpy":
        return _LayoutSpy(self.calls, parent=self)

    def box(self) -> "_LayoutSpy":
        return self._child()

    def row(self, align: bool = False) -> "_LayoutSpy":
        del align
        return self._child()

    def column(self, align: bool = False) -> "_LayoutSpy":
        del align
        return self._child()

    def split(self, factor: float = 0.5, align: bool = False) -> "_LayoutSpy":
        del factor, align
        return self._child()

    def label(self, text: str = "", icon: str = "NONE") -> None:
        self.calls.append(("label", {"text": text, "icon": icon, "alert": self._effective_alert()}))

    def prop(self, data, prop_name: str, text: str = "", slider: bool = False, icon: str = "NONE") -> None:
        del slider
        assert hasattr(data, prop_name), f"Missing property {prop_name!r} on {data!r}"
        self.calls.append(("prop", {"prop": prop_name, "text": text, "icon": icon}))

    def prop_search(
        self,
        data,
        prop_name: str,
        search_data,
        search_prop: str,
        text: str = "",
        icon: str = "NONE",
    ) -> None:
        assert hasattr(data, prop_name), f"Missing property {prop_name!r} on {data!r}"
        assert hasattr(search_data, search_prop), f"Missing search prop {search_prop!r}"
        self.calls.append(("prop_search", {"prop": prop_name, "text": text, "icon": icon}))

    def operator(self, op_idname: str, text: str = "", icon: str = "NONE") -> dict:
        cat, op_name = op_idname.split(".", 1)
        assert hasattr(getattr(bpy.ops, cat), op_name), f"Unregistered operator {op_idname}"
        self.calls.append(("operator", {"op": op_idname, "text": text, "icon": icon}))
        return {}


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _synth_formant_segment(
    f1: float,
    f2: float,
    dur_sec: float,
    sr: int = 16000,
    amp: float = 0.9,
) -> np.ndarray:
    n = int(round(sr * dur_sec))
    t = np.arange(n, dtype=np.float64) / float(sr)
    sig = amp * (0.45 * np.sin(2.0 * np.pi * f1 * t) + 0.30 * np.sin(2.0 * np.pi * f2 * t))
    fade = min(int(0.01 * sr), max(1, n // 10))
    if fade > 0 and n > 2 * fade:
        win = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        sig[:fade] *= win
        sig[-fade:] *= win[::-1]
    return np.clip(sig, -0.95, 0.95).astype(np.float32)


def _write_wav_16bit(path: pathlib.Path, samples: np.ndarray, sr: int = 16000) -> None:
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _build_mode_a_wav(path: pathlib.Path, sr: int = 16000) -> float:
    gap = np.zeros(int(0.10 * sr), dtype=np.float32)
    vowels = (
        (820.0, 1280.0),
        (330.0, 2300.0),
        (360.0, 830.0),
        (540.0, 1950.0),
        (540.0, 920.0),
    )
    parts = [np.zeros(int(0.12 * sr), dtype=np.float32)]
    for f1, f2 in vowels:
        parts.append(_synth_formant_segment(f1, f2, 0.30, sr=sr, amp=0.9))
        parts.append(gap)
    audio = np.concatenate(parts)
    _write_wav_16bit(path, audio, sr=sr)
    return float(audio.size) / float(sr)


def _build_mode_b_wav(path: pathlib.Path, sr: int = 16000) -> float:
    # Island 1 models "saya makan" (4 consecutive A pulses separated by soft dips < 120 ms).
    dip = _synth_formant_segment(820.0, 1280.0, 0.06, sr=sr, amp=0.12)
    pulse_a = _synth_formant_segment(820.0, 1280.0, 0.16, sr=sr, amp=0.95)
    island_1 = np.concatenate([pulse_a, dip, pulse_a, dip, pulse_a, dip, pulse_a])

    # Pause > 120 ms cleanly splits phrase 1 ("saya makan") from phrase 2 ("halo!").
    pause = np.zeros(int(0.28 * sr), dtype=np.float32)

    # Island 2 models "halo" (A -> O).
    pulse_o = _synth_formant_segment(540.0, 920.0, 0.22, sr=sr, amp=0.95)
    island_2 = np.concatenate([pulse_a, dip, pulse_o])

    lead = np.zeros(int(0.10 * sr), dtype=np.float32)
    tail = np.zeros(int(0.10 * sr), dtype=np.float32)
    audio = np.concatenate([lead, island_1, pause, island_2, tail])
    _write_wav_16bit(path, audio, sr=sr)
    return float(audio.size) / float(sr)


def _ensure_extension_enabled() -> object:
    module_names = {mod.__name__ for mod in addon_utils.modules(refresh=True)}
    if EXT_MODULE not in module_names:
        _assert(ZIP_PATH.is_file(), f"Missing packaged extension zip: {ZIP_PATH}")
        if hasattr(bpy.ops, "extensions") and hasattr(bpy.ops.extensions, "package_install_files"):
            res = bpy.ops.extensions.package_install_files(
                filepath=str(ZIP_PATH),
                repo="user_default",
                enable_on_install=True,
            )
            _assert("FINISHED" in res, f"package_install_files failed: {res}")
        module_names = {mod.__name__ for mod in addon_utils.modules(refresh=True)}

    _assert(EXT_MODULE in module_names, f"{EXT_MODULE} not found in installed extensions")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = bpy.ops.preferences.addon_enable(module=EXT_MODULE)
        _assert("FINISHED" in res, f"addon_enable failed: {res}")

    relevant_warnings = [
        w for w in caught
        if "kuchisync" in str(getattr(w, "filename", "")).lower()
        or "kuchisync" in str(w.message).lower()
    ]
    _assert(not relevant_warnings, f"Unexpected addon warnings: {[str(w.message) for w in relevant_warnings]}")

    _assert(hasattr(bpy.types.Scene, "kuchisync"), "Scene.kuchisync property missing after enable")
    _assert(hasattr(bpy.context.scene, "kuchisync"), "scene.kuchisync missing on active scene")
    return importlib.import_module(EXT_MODULE)


def _check_audio_decoders(kuchisync_mod, tmp_dir: pathlib.Path, wav_path: pathlib.Path) -> None:
    audio_dsp = kuchisync_mod.audio_dsp
    try:
        import aud
        aud_available = hasattr(aud, "Sound")
    except Exception:
        aud_available = False

    print(f"[audio] aud_available={aud_available}")
    clip_wav = audio_dsp.load_audio(str(wav_path), target_sr=16000)
    _assert(clip_wav.sample_rate == 16000 and clip_wav.samples.size > 0, "Failed decoding WAV")
    if aud_available:
        aud_samples = audio_dsp._decode_with_aud(wav_path, 16000)
        _assert(aud_samples.size > 0 and np.all(np.isfinite(aud_samples)), "aud WAV decode failed")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[audio] ffmpeg not found on PATH; skipping OGG/MP3 encode check")
        return

    for ext, codec in ((".ogg", "libvorbis"), (".mp3", "libmp3lame")):
        out_path = tmp_dir / f"sample{ext}"
        proc = subprocess.run(
            [ffmpeg_bin, "-y", "-v", "error", "-i", str(wav_path), "-c:a", codec, str(out_path)],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not out_path.is_file():
            print(f"[audio] skipping {ext} encode (codec {codec} unavailable)")
            continue
        clip = audio_dsp.load_audio(str(out_path), target_sr=16000)
        _assert(clip.sample_rate == 16000 and clip.samples.size > 0, f"load_audio failed for {ext}")
        if aud_available:
            aud_dec = audio_dsp._decode_with_aud(out_path, 16000)
            _assert(aud_dec.size > 0 and np.all(np.isfinite(aud_dec)), f"aud failed for {ext}")
        print(f"[audio] decoded {ext}: {clip.samples.size} samples")


def _create_test_head() -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    _assert(obj is not None and obj.type == "MESH", "Failed to create test Mesh object")
    obj.name = "KuchiSyncSmokeHead"

    for name in ("Basis", "vrc.v_aa", "い", "mouth_u", "viseme_eh", "お", "Blink", "Angry"):
        obj.shape_key_add(name=name, from_mix=False)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _fcurve_by_shape(keyframe_gen, shape_keys) -> dict[str, object]:
    result = {}
    for fc in keyframe_gen.iter_shape_key_fcurves(shape_keys):
        name = keyframe_gen._extract_key_block_name(getattr(fc, "data_path", ""))
        if name:
            result[name] = fc
    return result


def _vse_sound_strips(scene) -> list:
    se = getattr(scene, "sequence_editor", None)
    if se is None:
        return []
    container = getattr(se, "strips", None)
    if container is None:
        container = getattr(se, "sequences", None)
    if container is None:
        return []
    return [s for s in container if getattr(s, "type", None) == "SOUND"]


def _strip_start_frame(keyframe_gen, strip) -> int:
    return int(round(keyframe_gen._strip_start(strip, -1.0)))


def _verify_icons_and_panels(kuchisync_mod) -> None:
    ui_panel = kuchisync_mod.ui_panel
    prop_fn = bpy.types.UILayout.bl_rna.functions["prop"]
    enum_items = prop_fn.parameters["icon"].enum_items

    for icon_name in REQUIRED_ICONS + FALLBACK_ICONS:
        resolved = ui_panel.safe_icon(icon_name)
        _assert(resolved == icon_name, f"Icon {icon_name!r} resolved to {resolved!r}")
        _assert(icon_name in enum_items, f"Icon {icon_name!r} missing from UILayout enum_items")

    props = bpy.context.scene.kuchisync
    saved_mode = props.detection_mode
    saved_shape_i = props.shape_i
    saved_audio = props.audio_filepath
    saved_active = bpy.context.view_layer.objects.active
    try:
        for mode in ("PURE_AUDIO", "TRANSCRIPT"):
            props.detection_mode = mode
            for panel_id in PANEL_IDS:
                panel_cls = getattr(bpy.types, panel_id, None)
                _assert(panel_cls is not None, f"Panel {panel_id} is not registered")
                _assert(panel_cls.poll(bpy.context), f"Panel {panel_id}.poll() returned False")
                panel_inst = type("_FakePanel", (), {"layout": _LayoutSpy()})()
                panel_cls.draw(panel_inst, bpy.context)
                for _call_kind, payload in panel_inst.layout.calls:
                    icon_val = payload.get("icon", "NONE")
                    if icon_val != "NONE":
                        _assert(icon_val in enum_items, f"Panel {panel_id} emitted unknown icon {icon_val}")

        # Exercise partial mapping, unknown Shape Key row.alert, missing audio row.alert, and missing-object UI branches.
        props.shape_i = ""
        rig_cls = getattr(bpy.types, "KUCHISYNC_PT_rig_shapekeys")
        audio_cls = getattr(bpy.types, "KUCHISYNC_PT_audio_timing")
        exec_cls = getattr(bpy.types, "KUCHISYNC_PT_execution")
        rig_inst = type("_FakePanel", (), {"layout": _LayoutSpy()})()
        exec_inst = type("_FakePanel", (), {"layout": _LayoutSpy()})()
        rig_cls.draw(rig_inst, bpy.context)
        exec_cls.draw(exec_inst, bpy.context)

        props.shape_i = "nonexistent_morph"
        props.audio_filepath = "/nonexistent/kuchisync_missing.wav"
        rig_inst_err = type("_FakePanel", (), {"layout": _LayoutSpy()})()
        audio_inst_err = type("_FakePanel", (), {"layout": _LayoutSpy()})()
        rig_cls.draw(rig_inst_err, bpy.context)
        audio_cls.draw(audio_inst_err, bpy.context)
        _assert(
            any(
                kind == "label" and payload.get("text") == "I" and payload.get("alert")
                for kind, payload in rig_inst_err.layout.calls
            ),
            "Expected Panel 1 row.alert for shape_i='nonexistent_morph'",
        )
        _assert(
            any(
                kind == "label"
                and "not found" in payload.get("text", "").lower()
                and payload.get("alert")
                for kind, payload in audio_inst_err.layout.calls
            ),
            "Expected Panel 2 row.alert for missing audio_filepath",
        )

        bpy.context.view_layer.objects.active = None
        rig_inst_none = type("_FakePanel", (), {"layout": _LayoutSpy()})()
        exec_inst_none = type("_FakePanel", (), {"layout": _LayoutSpy()})()
        rig_cls.draw(rig_inst_none, bpy.context)
        exec_cls.draw(exec_inst_none, bpy.context)
    finally:
        bpy.context.view_layer.objects.active = saved_active
        props.audio_filepath = saved_audio
        props.shape_i = saved_shape_i
        props.detection_mode = saved_mode
    print("[ui] verified icons and drew all 5 panels cleanly across UI states")


def run_smoke_test() -> None:
    print(f"[env] Blender {bpy.app.version_string} ({bpy.app.version})")
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        _run_smoke_checks()

    relevant_warnings = [
        f"{getattr(w, 'filename', '')}:{getattr(w, 'lineno', 0)}: {w.category.__name__}: {w.message}"
        for w in captured_warnings
        if any(
            token in str(getattr(w, "filename", "")).lower() or token in str(w.message).lower()
            for token in ("kuchisync", "blender_smoke_test")
        )
    ]
    _assert(not relevant_warnings, f"Unexpected warnings detected: {relevant_warnings}")
    print("ALL BLENDER SMOKE CHECKS PASSED")


def _run_smoke_checks() -> None:
    kuchisync_mod = _ensure_extension_enabled()
    keyframe_gen = kuchisync_mod.keyframe_gen

    with tempfile.TemporaryDirectory(prefix="kuchisync_audio_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        mode_a_wav = tmp_dir / "mode_a_vowels.wav"
        mode_b_wav = tmp_dir / "mode_b_saya_makan_halo.wav"
        dur_a = _build_mode_a_wav(mode_a_wav, sr=16000)
        dur_b = _build_mode_b_wav(mode_b_wav, sr=16000)

        _check_audio_decoders(kuchisync_mod, tmp_dir, mode_a_wav)

        obj = _create_test_head()
        scene = bpy.context.scene
        scene.render.fps = 24
        scene.render.fps_base = 1.0
        props = scene.kuchisync

        res = bpy.ops.lipsync.auto_detect_shapekeys()
        _assert("FINISHED" in res, f"auto_detect_shapekeys failed: {res}")
        _assert(props.shape_a == "vrc.v_aa", f"shape_a={props.shape_a!r}")
        _assert(props.shape_i == "い", f"shape_i={props.shape_i!r}")
        _assert(props.shape_u == "mouth_u", f"shape_u={props.shape_u!r}")
        _assert(props.shape_e == "viseme_eh", f"shape_e={props.shape_e!r}")
        _assert(props.shape_o == "お", f"shape_o={props.shape_o!r}")
        print("[detect] A=vrc.v_aa I=い U=mouth_u E=viseme_eh O=お")

        shape_keys = obj.data.shape_keys
        kb_blink = shape_keys.key_blocks["Blink"]
        kb_blink.value = 1.0
        kb_blink.keyframe_insert(data_path="value", frame=1)

        kb_aa = shape_keys.key_blocks["vrc.v_aa"]
        kb_aa.value = 0.5
        kb_aa.keyframe_insert(data_path="value", frame=200)
        fc_aa_pre = _fcurve_by_shape(keyframe_gen, shape_keys)["vrc.v_aa"]
        for kp in fc_aa_pre.keyframe_points:
            if abs(float(kp.co[0]) - 200.0) <= 1e-4:
                kp.interpolation = "CONSTANT"
        fc_aa_pre.update()

        props.strength_a = 1.1
        props.strength_o = 0.7
        props.add_to_vse = True
        props.clear_existing_keys = True
        props.start_frame = 10
        props.language = "INDONESIAN"
        props.silence_threshold_db = -38.0

        # Mode A: PURE_AUDIO with BEZIER (AUTO_CLAMPED)
        props.detection_mode = "PURE_AUDIO"
        props.audio_filepath = str(mode_a_wav)
        props.interpolation_mode = "BEZIER"
        res = bpy.ops.kuchisync.generate_lipsync()
        _assert("FINISHED" in res, f"Mode A generate_lipsync failed: {res}")

        fcurves = _fcurve_by_shape(keyframe_gen, shape_keys)
        total_audio_frames_a = math.ceil(dur_a * 24.0)
        for v_shape in ("vrc.v_aa", "い", "mouth_u", "viseme_eh", "お"):
            _assert(v_shape in fcurves, f"Missing F-Curve for {v_shape} after Mode A")
            kps = list(fcurves[v_shape].keyframe_points)
            in_clip = [kp for kp in kps if 1 <= float(kp.co[0]) <= 180]
            _assert(len(in_clip) >= 3, f"Expected >= 3 keys on {v_shape}, got {len(in_clip)}")
            _assert(
                len(in_clip) < total_audio_frames_a,
                f"Expected sparse keys (< {total_audio_frames_a}) on {v_shape}, got {len(in_clip)}",
            )
            for kp in in_clip:
                _assert(kp.interpolation == "BEZIER", f"Expected BEZIER on {v_shape} at {kp.co[0]}")
                _assert(kp.handle_left_type == "AUTO_CLAMPED", f"Expected AUTO_CLAMPED left handle on {v_shape}")
                _assert(kp.handle_right_type == "AUTO_CLAMPED", f"Expected AUTO_CLAMPED right handle on {v_shape}")

        _assert(kb_aa.slider_max >= 1.1 - 1e-5, f"Expected slider_max >= 1.1, got {kb_aa.slider_max}")
        aa_clip_vals = [float(kp.co[1]) for kp in fcurves["vrc.v_aa"].keyframe_points if float(kp.co[0]) < 180]
        o_clip_vals = [float(kp.co[1]) for kp in fcurves["お"].keyframe_points if float(kp.co[0]) < 180]
        _assert(max(aa_clip_vals) > 1.0, f"Expected strength_a=1.1 peak > 1.0, got {max(aa_clip_vals)}")
        _assert(0.0 < max(o_clip_vals) <= 0.7 + 1e-4, f"Expected strength_o=0.7 peak <= 0.7, got {max(o_clip_vals)}")

        blink_kps = list(fcurves["Blink"].keyframe_points)
        _assert(
            len(blink_kps) == 1 and abs(float(blink_kps[0].co[0]) - 1.0) < 1e-4 and abs(float(blink_kps[0].co[1]) - 1.0) < 1e-4,
            "Blink keyframe at frame 1 was altered by Mode A",
        )
        aa_200 = [kp for kp in fcurves["vrc.v_aa"].keyframe_points if abs(float(kp.co[0]) - 200.0) <= 1e-4]
        _assert(
            len(aa_200) == 1 and abs(float(aa_200[0].co[1]) - 0.5) < 1e-4 and aa_200[0].interpolation == "CONSTANT",
            "Out-of-clip frame 200 CONSTANT keyframe on vrc.v_aa was altered by Mode A",
        )

        _assert(scene.sync_mode == "AUDIO_SYNC", f"Expected sync_mode AUDIO_SYNC, got {scene.sync_mode}")
        sound_strips = _vse_sound_strips(scene)
        _assert(len(sound_strips) == 1, f"Expected 1 VSE SOUND strip after Mode A, got {len(sound_strips)}")
        _assert(
            any(_strip_start_frame(keyframe_gen, s) == 10 for s in sound_strips),
            "Expected VSE SOUND strip starting at frame 10",
        )
        print("[mode_a] PURE_AUDIO sparse BEZIER + VSE + AUDIO_SYNC + scoped clear verified")

        # Mode B: TRANSCRIPT ("saya makan, halo!") with LINEAR and CONSTANT at frame 80
        props.detection_mode = "TRANSCRIPT"
        props.start_frame = 80
        props.audio_filepath = str(mode_b_wav)
        props.transcript_text = "saya makan, halo!"
        total_audio_frames_b = math.ceil(dur_b * 24.0)

        for interp_mode in ("LINEAR", "CONSTANT"):
            props.interpolation_mode = interp_mode
            res = bpy.ops.kuchisync.generate_lipsync()
            _assert("FINISHED" in res, f"Mode B ({interp_mode}) failed: {res}")

            fcurves = _fcurve_by_shape(keyframe_gen, shape_keys)
            aa_kps = list(fcurves["vrc.v_aa"].keyframe_points)
            aa_in_clip = [kp for kp in aa_kps if 75.0 <= float(kp.co[0]) < 180.0]
            o_in_clip = [kp for kp in fcurves["お"].keyframe_points if 75.0 <= float(kp.co[0]) < 180.0]

            _assert(
                3 <= len(aa_in_clip) < total_audio_frames_b,
                f"Expected sparse Mode B keys on vrc.v_aa (< {total_audio_frames_b}), got {len(aa_in_clip)}",
            )
            _assert(len(o_in_clip) >= 3, f"Expected Mode B keys on お, got {len(o_in_clip)}")
            for shape_tag, kps_check in (("vrc.v_aa", aa_in_clip), ("お", o_in_clip)):
                for kp in kps_check:
                    _assert(
                        kp.interpolation == interp_mode,
                        f"Expected {interp_mode} on {shape_tag} at frame {kp.co[0]:.2f}, got {kp.interpolation}",
                    )

            # Verify "saya makan" generates multiple articulated peaks and valley dips (not 1 flat hold).
            vals = [float(kp.co[1]) for kp in aa_in_clip]
            dedup_vals: list[float] = []
            for v in vals:
                if not dedup_vals or abs(v - dedup_vals[-1]) > 1e-6:
                    dedup_vals.append(v)
            peak_count = sum(
                1
                for i in range(1, len(dedup_vals) - 1)
                if dedup_vals[i] > 0.5
                and dedup_vals[i] > dedup_vals[i - 1]
                and dedup_vals[i] > dedup_vals[i + 1]
            )
            dip_count = sum(
                1
                for i in range(1, len(dedup_vals) - 1)
                if 0.0 < dedup_vals[i] < 0.5
                and dedup_vals[i] < dedup_vals[i - 1]
                and dedup_vals[i] < dedup_vals[i + 1]
            )
            _assert(peak_count >= 4, f"Expected >= 4 articulated peaks for 'saya makan, halo!', got {peak_count} ({vals})")
            _assert(dip_count >= 2, f"Expected >= 2 valley dips for 'saya makan', got {dip_count} ({vals})")

            aa_200 = [kp for kp in aa_kps if abs(float(kp.co[0]) - 200.0) <= 1e-4]
            _assert(
                len(aa_200) == 1 and aa_200[0].interpolation == "CONSTANT",
                f"Out-of-clip frame 200 lost during Mode B ({interp_mode})",
            )
            blink_kps = list(fcurves["Blink"].keyframe_points)
            _assert(len(blink_kps) == 1, f"Blink keyframe lost during Mode B ({interp_mode})")
        sound_strips_b = _vse_sound_strips(scene)
        _assert(
            len(sound_strips_b) == 2,
            f"Expected 2 deduplicated VSE SOUND strips after Mode B re-run, got {len(sound_strips_b)}",
        )
        print("[mode_b] TRANSCRIPT LINEAR & CONSTANT + valley articulation + VSE dedupe verified")

        _verify_icons_and_panels(kuchisync_mod)

        # Clear all mapped vowel keys across timeline while leaving Blink untouched.
        res = bpy.ops.kuchisync.clear_lipsync()
        _assert("FINISHED" in res, f"clear_lipsync failed: {res}")
        fcurves_after = _fcurve_by_shape(keyframe_gen, shape_keys)
        for v_shape in ("vrc.v_aa", "い", "mouth_u", "viseme_eh", "お"):
            if v_shape in fcurves_after:
                rem = len(fcurves_after[v_shape].keyframe_points)
                _assert(rem == 0, f"Expected 0 remaining keys on {v_shape}, got {rem}")
        _assert(
            "Blink" in fcurves_after and len(fcurves_after["Blink"].keyframe_points) == 1,
            "Blink keyframe was removed by clear_lipsync",
        )
        print("[clear] clear_lipsync removed mapped vowels and kept Blink intact")

    # Verify clean disable / re-enable cycle.
    res = bpy.ops.preferences.addon_disable(module=EXT_MODULE)
    _assert("FINISHED" in res, f"addon_disable failed: {res}")
    _assert(not hasattr(bpy.types.Scene, "kuchisync"), "Scene.kuchisync persisted after disable")
    res = bpy.ops.preferences.addon_enable(module=EXT_MODULE)
    _assert("FINISHED" in res, f"addon_enable after disable failed: {res}")
    _assert(hasattr(bpy.types.Scene, "kuchisync"), "Scene.kuchisync missing after re-enable")
    print("[lifecycle] disable/re-enable cycle passed")


if __name__ == "__main__":
    run_smoke_test()
