# SPDX-License-Identifier: GPL-3.0-or-later

import importlib
import pathlib
import re
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave
import zipfile

import numpy as np

from kuchisync import ui_panel

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_vowel_wav(
    path: pathlib.Path,
    segments: list[tuple[tuple[float, float], float]],
    sr: int = 16000,
) -> None:
    """Write a 16-bit mono WAV containing synthetic formant vowel segments."""
    chunks = []
    for (f1, f2), dur in segments:
        n = int(sr * dur)
        t = np.arange(n, dtype=np.float64) / sr
        sig = 0.45 * np.sin(2.0 * np.pi * f1 * t) + 0.30 * np.sin(2.0 * np.pi * f2 * t)
        chunks.append(sig)
    full = np.clip(np.concatenate(chunks), -0.95, 0.95)
    pcm = (full * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class _FakeRecordingLayout:
    """Minimal UILayout recorder verifying panel draw paths and widget states."""

    def __init__(self, log: list[dict] | None = None, parent: "_FakeRecordingLayout | None" = None):
        self.enabled = True
        self.alert = False
        self.scale_y = 1.0
        self._parent = parent
        self._log = log if log is not None else []

    def _effective_enabled(self) -> bool:
        if self._parent is not None:
            return self.enabled and self._parent._effective_enabled()
        return self.enabled

    def _effective_alert(self) -> bool:
        if self._parent is not None:
            return self.alert or self._parent._effective_alert()
        return self.alert

    def _child(self) -> "_FakeRecordingLayout":
        return _FakeRecordingLayout(self._log, parent=self)

    def row(self, align: bool = False) -> "_FakeRecordingLayout":
        del align
        return self._child()

    def column(self, align: bool = False) -> "_FakeRecordingLayout":
        del align
        return self._child()

    def box(self) -> "_FakeRecordingLayout":
        return self._child()

    def split(self, factor: float = 0.5, align: bool = False) -> "_FakeRecordingLayout":
        del factor, align
        return self._child()

    def label(self, text: str = "", icon: str = "NONE") -> None:
        self._log.append(
            {
                "kind": "label",
                "text": text,
                "icon": icon,
                "alert": self._effective_alert(),
                "enabled": self._effective_enabled(),
            }
        )

    def prop(self, data, prop_name: str, text=None, slider: bool = False, expand: bool = False) -> None:
        del data, text, slider, expand
        self._log.append(
            {
                "kind": "prop",
                "prop": prop_name,
                "enabled": self._effective_enabled(),
                "alert": self._effective_alert(),
            }
        )

    def prop_search(self, data, prop_name: str, search_data, search_prop: str, text: str = "") -> None:
        del data, search_data, search_prop, text
        self._log.append(
            {
                "kind": "prop_search",
                "prop": prop_name,
                "enabled": self._effective_enabled(),
                "alert": self._effective_alert(),
            }
        )

    def operator(self, operator_id: str, text: str | None = None, icon: str = "NONE") -> dict:
        entry = {
            "kind": "operator",
            "operator": operator_id,
            "text": text,
            "icon": icon,
            "enabled": self._effective_enabled(),
            "alert": self._effective_alert(),
            "scale_y": self.scale_y,
        }
        self._log.append(entry)
        return entry

    def separator(self) -> None:
        pass


def _build_mock_bpy(icon_names: set[str] | None = None):
    """Build a bare ModuleType('bpy') without __spec__ plus bpy_extras mock."""
    valid_icons = set(
        icon_names
        or {
            "NONE",
            "CHECKMARK",
            "ERROR",
            "INFO",
            "VIEWZOOM",
            "FILE_FOLDER",
            "TIME",
            "PLAY",
            "TRASH",
            "IMPORT",
            "FILE_REFRESH",
            "MESH_DATA",
            "OBJECT_DATA",
        }
    )

    bpy_mod = types.ModuleType("bpy")
    # Leave __spec__ unset intentionally to verify bare mock compatibility (Task 1 deferred minor).

    def make_prop(**kwargs):
        return ("PROP", kwargs)

    bpy_mod.props = types.SimpleNamespace(
        StringProperty=make_prop,
        FloatProperty=make_prop,
        IntProperty=make_prop,
        BoolProperty=make_prop,
        EnumProperty=make_prop,
        PointerProperty=make_prop,
    )

    enum_items = {name: types.SimpleNamespace(identifier=name) for name in valid_icons}

    class FakePropertyGroup:
        pass

    class FakeOperator:
        reports: list[tuple[set[str], str]] = []

        def report(self, level: set[str], message: str) -> None:
            self.reports.append((set(level), str(message)))

    class FakePanel:
        pass

    class FakeScene:
        pass

    class FakeText:
        pass

    class FakeUILayout:
        bl_rna = types.SimpleNamespace(
            properties={},
            functions={
                "prop": types.SimpleNamespace(
                    parameters={"icon": types.SimpleNamespace(enum_items=enum_items)}
                ),
                "label": types.SimpleNamespace(
                    parameters={"icon": types.SimpleNamespace(enum_items=enum_items)}
                ),
            },
        )

    bpy_mod.types = types.SimpleNamespace(
        PropertyGroup=FakePropertyGroup,
        Operator=FakeOperator,
        Panel=FakePanel,
        Scene=FakeScene,
        Text=FakeText,
        UILayout=FakeUILayout,
    )

    registered_classes: list[type] = []

    def register_class(cls):
        if cls not in registered_classes:
            registered_classes.append(cls)

    def unregister_class(cls):
        if cls not in registered_classes:
            raise RuntimeError(f"{cls.__name__} not registered")
        registered_classes.remove(cls)

    resources_root = tempfile.mkdtemp(prefix="kuchisync_res_")

    def user_resource(resource_type: str, path: str = "", create: bool = False):
        del resource_type
        target = pathlib.Path(resources_root) / path
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return str(target)

    bpy_mod.utils = types.SimpleNamespace(
        register_class=register_class,
        unregister_class=unregister_class,
        user_resource=user_resource,
        _registered=registered_classes,
        _resources_root=resources_root,
    )

    active_timers: dict[object, float] = {}

    class FakeTimers:
        @staticmethod
        def register(fn, first_interval: float = 0.0):
            active_timers[fn] = first_interval

        @staticmethod
        def is_registered(fn) -> bool:
            return fn in active_timers

        @staticmethod
        def unregister(fn) -> None:
            active_timers.pop(fn, None)

        @staticmethod
        def active():
            return active_timers

    wm_areas = [types.SimpleNamespace(type="VIEW_3D", tag_redraw=mock.MagicMock())]
    wm_windows = [types.SimpleNamespace(screen=types.SimpleNamespace(areas=wm_areas))]
    wm = types.SimpleNamespace(windows=wm_windows)

    bpy_mod.app = types.SimpleNamespace(online_access=True, timers=FakeTimers)
    bpy_mod.context = types.SimpleNamespace(window_manager=wm)
    bpy_mod.data = types.SimpleNamespace(filepath="", window_managers=[wm])

    def abspath(p: str) -> str:
        if p.startswith("//") and bpy_mod.data.filepath:
            base = pathlib.Path(bpy_mod.data.filepath).parent
            return str(base / p[2:])
        return p

    bpy_mod.path = types.SimpleNamespace(abspath=abspath)

    bpy_extras_mod = types.ModuleType("bpy_extras")
    io_utils_mod = types.ModuleType("bpy_extras.io_utils")

    class FakeImportHelper:
        filepath: str = ""
        invoke_calls: list[str] = []

        def invoke(self, context, event):
            del context, event
            self.invoke_calls.append(self.filepath)
            return {"RUNNING_MODAL"}

    io_utils_mod.ImportHelper = FakeImportHelper
    bpy_extras_mod.io_utils = io_utils_mod

    return bpy_mod, bpy_extras_mod, io_utils_mod


class TestUiLogicAndHygiene(unittest.TestCase):
    def test_validate_generation_context_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = pathlib.Path(tmp) / "line.wav"
            audio.write_bytes(b"RIFF0000WAVE")

            def make_props(**kw):
                base = dict(
                    shape_a="mouth_a",
                    shape_i="",
                    shape_u="",
                    shape_e="",
                    shape_o="",
                    audio_filepath=str(audio),
                    detection_mode="PURE_AUDIO",
                    transcript_text="",
                    transcript_datablock=None,
                )
                base.update(kw)
                return types.SimpleNamespace(**base)

            mesh_obj = types.SimpleNamespace(
                type="MESH",
                data=types.SimpleNamespace(
                    shape_keys=types.SimpleNamespace(key_blocks={"mouth_a": object()})
                ),
            )
            armature_obj = types.SimpleNamespace(type="ARMATURE", data=None)
            no_sk_obj = types.SimpleNamespace(
                type="MESH",
                data=types.SimpleNamespace(shape_keys=None),
            )

            ok, _ = ui_panel.validate_generation_context(None, make_props())
            self.assertFalse(ok)

            ok, _ = ui_panel.validate_generation_context(armature_obj, make_props())
            self.assertFalse(ok)

            ok, _ = ui_panel.validate_generation_context(no_sk_obj, make_props())
            self.assertFalse(ok)

            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(shape_a=""))
            self.assertFalse(ok)
            self.assertIn("Shape Key", msg)

            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(shape_a="missing_key"))
            self.assertFalse(ok)
            self.assertIn("Shape Key", msg)

            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(audio_filepath=""))
            self.assertFalse(ok)
            self.assertIn("audio", msg.lower())

            ok, msg = ui_panel.validate_generation_context(
                mesh_obj, make_props(audio_filepath=str(pathlib.Path(tmp) / "absent.wav"))
            )
            self.assertFalse(ok)
            self.assertIn("audio", msg.lower())

            ok, msg = ui_panel.validate_generation_context(mesh_obj, make_props(detection_mode="TRANSCRIPT"))
            self.assertFalse(ok)
            self.assertIn("transcript", msg.lower())

            ok, _ = ui_panel.validate_generation_context(
                mesh_obj, make_props(detection_mode="TRANSCRIPT", transcript_text="saya makan")
            )
            self.assertTrue(ok)

            text_block = types.SimpleNamespace(as_string=lambda: "halo dunia")
            ok, _ = ui_panel.validate_generation_context(
                mesh_obj, make_props(detection_mode="TRANSCRIPT", transcript_datablock=text_block)
            )
            self.assertTrue(ok)

            ok, _ = ui_panel.validate_generation_context(mesh_obj, make_props())
            self.assertTrue(ok)

            # Verify // Blender relative audio paths resolve when bpy.path.abspath is present.
            fake_bpy = types.SimpleNamespace(
                path=types.SimpleNamespace(abspath=lambda p: str(audio) if p == "//line.wav" else p)
            )
            with mock.patch.object(ui_panel, "bpy", fake_bpy):
                ok, _ = ui_panel.validate_generation_context(mesh_obj, make_props(audio_filepath="//line.wav"))
                self.assertTrue(ok)

    def test_get_effective_transcript_and_resolve_audio_path(self):
        self.assertEqual(
            ui_panel.get_effective_transcript(
                types.SimpleNamespace(transcript_text="   ", transcript_datablock=None)
            ),
            "",
        )
        self.assertEqual(
            ui_panel.get_effective_transcript(
                types.SimpleNamespace(transcript_text=" halo ", transcript_datablock=None)
            ),
            "halo",
        )
        tb = types.SimpleNamespace(as_string=lambda: "baris satu\nbaris dua")
        self.assertEqual(
            ui_panel.get_effective_transcript(
                types.SimpleNamespace(transcript_text="inline", transcript_datablock=tb)
            ),
            "baris satu\nbaris dua",
        )
        empty_tb = types.SimpleNamespace(as_string=lambda: "   \n  ")
        self.assertEqual(
            ui_panel.get_effective_transcript(
                types.SimpleNamespace(transcript_text="fallback", transcript_datablock=empty_tb)
            ),
            "fallback",
        )
        self.assertEqual(ui_panel.resolve_audio_path("  "), "")

    def test_safe_icon_fallback_behavior(self):
        with mock.patch.object(ui_panel, "bpy", None):
            with mock.patch.dict(sys.modules, {"bpy": None}):
                self.assertEqual(ui_panel.safe_icon("CHECKMARK"), "NONE")
                self.assertEqual(ui_panel.safe_icon("CHECKMARK", fallback="INFO"), "INFO")

        icon_items = {"CHECKMARK": object(), "INFO": object(), "NONE": object()}
        fake_bpy_prop_fn = types.SimpleNamespace(
            types=types.SimpleNamespace(
                UILayout=types.SimpleNamespace(
                    bl_rna=types.SimpleNamespace(
                        properties={},
                        functions={
                            "prop": types.SimpleNamespace(
                                parameters={"icon": types.SimpleNamespace(enum_items=icon_items)}
                            )
                        },
                    )
                )
            )
        )
        with mock.patch.object(ui_panel, "bpy", fake_bpy_prop_fn):
            self.assertEqual(ui_panel.safe_icon("CHECKMARK"), "CHECKMARK")
            self.assertEqual(ui_panel.safe_icon("MISSING_ICON"), "NONE")
            self.assertEqual(ui_panel.safe_icon("MISSING_ICON", fallback="INFO"), "INFO")
            self.assertEqual(ui_panel.safe_icon("MISSING_ICON", fallback="NOPE"), "NONE")

        # Fallback to bl_rna.functions["label"].parameters["icon"].enum_items when "prop" lacks it.
        fake_bpy_label_fn = types.SimpleNamespace(
            types=types.SimpleNamespace(
                UILayout=types.SimpleNamespace(
                    bl_rna=types.SimpleNamespace(
                        properties={},
                        functions={
                            "label": types.SimpleNamespace(
                                parameters={"icon": types.SimpleNamespace(enum_items=icon_items)}
                            )
                        },
                    )
                )
            )
        )
        with mock.patch.object(ui_panel, "bpy", fake_bpy_label_fn):
            self.assertEqual(ui_panel.safe_icon("CHECKMARK"), "CHECKMARK")

        # Real Blender UILayout.bl_rna.properties does not carry "icon"; mock with only properties fails safely.
        fake_bpy_props_only = types.SimpleNamespace(
            types=types.SimpleNamespace(
                UILayout=types.SimpleNamespace(
                    bl_rna=types.SimpleNamespace(
                        properties={"icon": types.SimpleNamespace(enum_items=icon_items)},
                        functions={},
                    )
                )
            )
        )
        with mock.patch.object(ui_panel, "bpy", fake_bpy_props_only):
            self.assertEqual(ui_panel.safe_icon("CHECKMARK"), "NONE")

    def test_antislop_code_and_copy_hygiene(self):
        banned_buzz = (
            "AI Powered",
            "Next Generation",
            "Revolutionary",
            "Seamless",
            "Cutting Edge",
            "Intelligent",
            "Ultimate",
            "Powerful",
            "Effortless",
        )
        emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
        banner_re = re.compile(r"#\s*[=\-]{4,}")
        for py in sorted(list((ROOT / "kuchisync").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))):
            src = py.read_text(encoding="utf-8")
            self.assertTrue(src.startswith("# SPDX-License-Identifier: GPL-3.0-or-later"), str(py))
            self.assertNotIn("\u2014", src, f"Em dash found in {py}")
            self.assertIsNone(emoji_re.search(src), f"Emoji character found in {py}")
            self.assertIsNone(banner_re.search(src), f"Decorative banner comment found in {py}")
            self.assertNotIn("TODO", src, f"TODO found in {py}")
            for word in banned_buzz:
                pattern = rf"\b{re.escape(word.lower())}\b"
                self.assertIsNone(
                    re.search(pattern, src.lower()),
                    f"Buzzword {word} found in {py}",
                )

    def test_mock_bpy_registration_operators_panels_and_scipy_installer(self):
        bpy_mod, bpy_extras_mod, io_utils_mod = _build_mock_bpy()
        saved_modules = {
            name: sys.modules.get(name)
            for name in ("bpy", "bpy_extras", "bpy_extras.io_utils")
        }
        try:
            sys.modules["bpy"] = bpy_mod
            sys.modules["bpy_extras"] = bpy_extras_mod
            sys.modules["bpy_extras.io_utils"] = io_utils_mod

            import kuchisync

            importlib.reload(kuchisync)
            kuchisync.register()
            try:
                registered = bpy_mod.utils._registered
                op_ids = {getattr(c, "bl_idname", "") for c in registered if hasattr(c, "execute")}
                self.assertEqual(
                    op_ids,
                    {
                        "lipsync.auto_detect_shapekeys",
                        "kuchisync.browse_audio",
                        "kuchisync.set_start_frame",
                        "kuchisync.load_transcript",
                        "kuchisync.generate_lipsync",
                        "kuchisync.clear_lipsync",
                        "kuchisync.install_scipy",
                    },
                )
                panel_classes = [c for c in registered if hasattr(c, "bl_space_type")]
                self.assertEqual(
                    {c.__name__ for c in panel_classes},
                    {
                        "KUCHISYNC_PT_rig_shapekeys",
                        "KUCHISYNC_PT_audio_timing",
                        "KUCHISYNC_PT_detection_engine",
                        "KUCHISYNC_PT_animation_curve",
                        "KUCHISYNC_PT_execution",
                    },
                )
                for p_cls in panel_classes:
                    self.assertEqual(p_cls.bl_space_type, "VIEW_3D")
                    self.assertEqual(p_cls.bl_region_type, "UI")
                    self.assertEqual(p_cls.bl_category, "LipSync")

                self.assertTrue(hasattr(bpy_mod.types.Scene, "kuchisync"))
                settings_cls = next(c for c in registered if c.__name__ == "KuchiSyncSettings")
                ann = settings_cls.__annotations__
                for v in ("a", "i", "u", "e", "o"):
                    self.assertIn(f"shape_{v}", ann)
                    self.assertIn(f"strength_{v}", ann)
                    s_kw = ann[f"strength_{v}"][1]
                    self.assertEqual(s_kw["min"], 0.0)
                    self.assertEqual(s_kw["max"], 1.5)
                    self.assertEqual(s_kw["default"], 1.0)
                    self.assertEqual(s_kw["subtype"], "FACTOR")
                self.assertEqual(ann["silence_threshold_db"][1]["min"], -60.0)
                self.assertEqual(ann["silence_threshold_db"][1]["max"], -10.0)
                self.assertEqual(ann["silence_threshold_db"][1]["default"], -38.0)

                # Build realistic fake scene, mesh, slotted action, and VSE container.
                class FakeKP:
                    def __init__(self, frame, value):
                        self.co = [float(frame), float(value)]
                        self.interpolation = "BEZIER"
                        self.handle_left_type = "AUTO"
                        self.handle_right_type = "AUTO"

                class FakeKPs(list):
                    def remove(self, kp, fast=False):
                        del fast
                        super().remove(kp)

                class FakeFCurve:
                    def __init__(self, data_path, points=None):
                        self.data_path = data_path
                        self.keyframe_points = FakeKPs(points or [])

                    def update(self):
                        pass

                fcurves_map = {
                    'key_blocks["Blink"].value': FakeFCurve('key_blocks["Blink"].value', [FakeKP(5, 1.0)]),
                    'key_blocks["vrc.v_aa"].value': FakeFCurve(
                        'key_blocks["vrc.v_aa"].value', [FakeKP(10, 0.7), FakeKP(200, 0.8)]
                    ),
                }

                class FakeKeyBlock:
                    def __init__(self, name):
                        self.name = name
                        self.value = 0.0
                        self.slider_max = 1.0

                    def keyframe_insert(self, data_path="value", frame=1):
                        dp = f'key_blocks["{self.name}"].{data_path}'
                        fc = fcurves_map.setdefault(dp, FakeFCurve(dp))
                        fc.keyframe_points.append(FakeKP(frame, self.value))

                class FakeKeyBlocks(dict):
                    def __iter__(self):
                        return iter(self.values())

                kb_dict = FakeKeyBlocks(
                    {
                        n: FakeKeyBlock(n)
                        for n in ("Basis", "Blink", "Angry", "vrc.v_aa", "い", "mouth_u", "viseme_eh", "お")
                    }
                )
                bag = types.SimpleNamespace(fcurves=list(fcurves_map.values()))
                strip = types.SimpleNamespace(channelbag=lambda _slot: FakeBagView())

                class FakeBagView:
                    @property
                    def fcurves(self):
                        return list(fcurves_map.values())

                del bag
                action = types.SimpleNamespace(
                    slots=["slot0"],
                    layers=[types.SimpleNamespace(strips=[strip])],
                )
                sk = types.SimpleNamespace(
                    key_blocks=kb_dict,
                    animation_data=types.SimpleNamespace(action=action, action_slot="slot0"),
                )
                mesh_obj = types.SimpleNamespace(
                    name="Character",
                    type="MESH",
                    data=types.SimpleNamespace(shape_keys=sk),
                )

                class FakeStrips(list):
                    def new_sound(self, name, filepath, channel, frame_start):
                        item = types.SimpleNamespace(
                            name=name,
                            sound=types.SimpleNamespace(filepath=filepath),
                            channel=channel,
                            frame_start=frame_start,
                            type="SOUND",
                        )
                        self.append(item)
                        return item

                props = types.SimpleNamespace(
                    shape_a="",
                    shape_i="",
                    shape_u="",
                    shape_e="",
                    shape_o="",
                    strength_a=1.2,
                    strength_i=1.0,
                    strength_u=1.0,
                    strength_e=1.0,
                    strength_o=0.7,
                    audio_filepath="",
                    language="INDONESIAN",
                    start_frame=1,
                    add_to_vse=True,
                    clear_existing_keys=True,
                    detection_mode="PURE_AUDIO",
                    voice_type="FEMALE_STANDARD",
                    silence_threshold_db=-38.0,
                    transcript_text="",
                    transcript_datablock=None,
                    max_mouth_open=1.0,
                    min_phoneme_frames=3,
                    transition_frames=2,
                    co_articulation=0.25,
                    interpolation_mode="BEZIER",
                )
                scene = types.SimpleNamespace(
                    frame_current=10,
                    render=types.SimpleNamespace(fps=24, fps_base=1.0),
                    sync_mode="NONE",
                    sequence_editor=types.SimpleNamespace(strips=FakeStrips()),
                    kuchisync=props,
                )
                ctx = types.SimpleNamespace(active_object=mesh_obj, scene=scene)

                op_by_id = {
                    getattr(c, "bl_idname", ""): c
                    for c in registered
                    if hasattr(c, "execute")
                }

                # Auto-detect operator
                auto_op = op_by_id["lipsync.auto_detect_shapekeys"]()
                auto_op.reports = []
                self.assertEqual(auto_op.execute(ctx), {"FINISHED"})
                self.assertEqual(props.shape_a, "vrc.v_aa")
                self.assertEqual(props.shape_i, "い")
                self.assertEqual(props.shape_u, "mouth_u")
                self.assertEqual(props.shape_e, "viseme_eh")
                self.assertEqual(props.shape_o, "お")

                # Sync start frame operator
                sf_op = op_by_id["kuchisync.set_start_frame"]()
                self.assertEqual(sf_op.execute(ctx), {"FINISHED"})
                self.assertEqual(props.start_frame, 10)

                with tempfile.TemporaryDirectory() as tmp:
                    wav_path = pathlib.Path(tmp) / "test.wav"
                    _write_vowel_wav(
                        wav_path,
                        [((820.0, 1280.0), 0.25), ((550.0, 920.0), 0.25)],
                    )
                    txt_path = pathlib.Path(tmp) / "line.txt"
                    txt_path.write_text("saya makan", encoding="utf-8")

                    load_op = op_by_id["kuchisync.load_transcript"]()
                    load_op.filepath = str(txt_path)
                    load_op.reports = []
                    self.assertEqual(load_op.execute(ctx), {"FINISHED"})
                    self.assertEqual(props.transcript_text, "saya makan")

                    props.audio_filepath = str(wav_path)

                    # Browse audio operator (invoke + execute)
                    browse_op = op_by_id["kuchisync.browse_audio"]()
                    browse_op.reports = []
                    browse_op.filepath = ""
                    bpy_extras_mod.io_utils.ImportHelper.invoke_calls.clear()
                    self.assertEqual(browse_op.invoke(ctx, None), {"RUNNING_MODAL"})
                    self.assertEqual(browse_op.filepath, str(wav_path))
                    self.assertEqual(bpy_extras_mod.io_utils.ImportHelper.invoke_calls, [str(wav_path)])
                    picked_wav = pathlib.Path(tmp) / "picked.wav"
                    browse_op.filepath = str(picked_wav)
                    self.assertEqual(browse_op.execute(ctx), {"FINISHED"})
                    self.assertEqual(props.audio_filepath, str(picked_wav))
                    props.audio_filepath = str(wav_path)

                    gen_op = op_by_id["kuchisync.generate_lipsync"]()
                    gen_op.reports = []
                    self.assertEqual(gen_op.execute(ctx), {"FINISHED"})

                    # Blink key and out-of-range vrc.v_aa key at frame 200 remain; scoped frame 10 replaced/cleared.
                    self.assertEqual(len(fcurves_map['key_blocks["Blink"].value'].keyframe_points), 1)
                    a_frames = [kp.co[0] for kp in fcurves_map['key_blocks["vrc.v_aa"].value'].keyframe_points]
                    self.assertIn(200.0, a_frames)
                    self.assertGreaterEqual(kb_dict["vrc.v_aa"].slider_max, 1.2)
                    self.assertEqual(scene.sync_mode, "AUDIO_SYNC")
                    self.assertEqual(len(scene.sequence_editor.strips), 1)

                    # Mode B test
                    props.detection_mode = "TRANSCRIPT"
                    self.assertEqual(gen_op.execute(ctx), {"FINISHED"})

                    # Corrupt audio reports ERROR cleanly
                    bad_wav = pathlib.Path(tmp) / "bad.wav"
                    bad_wav.write_bytes(b"not a valid wave file")
                    props.audio_filepath = str(bad_wav)
                    gen_op.reports = []
                    self.assertEqual(gen_op.execute(ctx), {"CANCELLED"})
                    self.assertTrue(any("ERROR" in lvl for lvl, _ in gen_op.reports))

                # Clear entire timeline operator
                clr_op = op_by_id["kuchisync.clear_lipsync"]()
                clr_op.reports = []
                self.assertEqual(clr_op.execute(ctx), {"FINISHED"})
                self.assertEqual(len(fcurves_map['key_blocks["vrc.v_aa"].value'].keyframe_points), 0)
                self.assertEqual(len(fcurves_map['key_blocks["Blink"].value'].keyframe_points), 1)

                # Panel draw checks across modes and states
                panel_by_name = {c.__name__: c for c in panel_classes}

                # Panel 1: empty object, partial mapping, and full mapping
                log_p1_empty = []
                panel_by_name["KUCHISYNC_PT_rig_shapekeys"].draw(
                    types.SimpleNamespace(layout=_FakeRecordingLayout(log_p1_empty)),
                    types.SimpleNamespace(active_object=None, scene=scene),
                )
                self.assertTrue(any("No active object" in e.get("text", "") for e in log_p1_empty))

                log_p1_full = []
                panel_by_name["KUCHISYNC_PT_rig_shapekeys"].draw(
                    types.SimpleNamespace(layout=_FakeRecordingLayout(log_p1_full)), ctx
                )
                self.assertEqual(
                    [e["prop"] for e in log_p1_full if e["kind"] == "prop_search"],
                    ["shape_a", "shape_i", "shape_u", "shape_e", "shape_o"],
                )
                for v in ("a", "i", "u", "e", "o"):
                    self.assertTrue(
                        any(e.get("prop") == f"strength_{v}" for e in log_p1_full)
                    )

                # Panel 1: partial mapping (1..4 vowels) info box + missing Shape Key row.alert state
                saved_shapes = (props.shape_i, props.shape_u, props.shape_e, props.shape_o)
                try:
                    props.shape_i = "nonexistent_key"
                    props.shape_u = ""
                    props.shape_e = ""
                    props.shape_o = ""
                    log_p1_partial = []
                    panel_by_name["KUCHISYNC_PT_rig_shapekeys"].draw(
                        types.SimpleNamespace(layout=_FakeRecordingLayout(log_p1_partial)), ctx
                    )
                    self.assertTrue(
                        any(
                            e["kind"] == "label"
                            and "Mapped 1 of 5 vowels" in e["text"]
                            and e["icon"] == "INFO"
                            and not e["alert"]
                            for e in log_p1_partial
                        )
                    )
                    vowel_labels = {
                        e["text"]: e
                        for e in log_p1_partial
                        if e["kind"] == "label" and e["text"] in ("A", "I", "U", "E", "O")
                    }
                    self.assertEqual(vowel_labels["A"]["icon"], "CHECKMARK")
                    self.assertFalse(vowel_labels["A"]["alert"])
                    self.assertEqual(vowel_labels["I"]["icon"], "ERROR")
                    self.assertTrue(vowel_labels["I"]["alert"])
                    self.assertEqual(vowel_labels["U"]["icon"], "ERROR")
                    self.assertFalse(vowel_labels["U"]["alert"])
                finally:
                    props.shape_i, props.shape_u, props.shape_e, props.shape_o = saved_shapes

                # Panel 2 and Panel 4 draws
                log_p2 = []
                panel_by_name["KUCHISYNC_PT_audio_timing"].draw(
                    types.SimpleNamespace(layout=_FakeRecordingLayout(log_p2)), ctx
                )
                self.assertTrue(any(e.get("prop") == "audio_filepath" for e in log_p2))
                self.assertTrue(any(e.get("operator") == "kuchisync.set_start_frame" for e in log_p2))

                log_p4 = []
                panel_by_name["KUCHISYNC_PT_animation_curve"].draw(
                    types.SimpleNamespace(layout=_FakeRecordingLayout(log_p4)), ctx
                )
                self.assertEqual(
                    [e["prop"] for e in log_p4 if e["kind"] == "prop"],
                    [
                        "max_mouth_open",
                        "min_phoneme_frames",
                        "transition_frames",
                        "co_articulation",
                        "interpolation_mode",
                    ],
                )

                # Panel 3 in Mode A and Mode B both render shared silence_threshold_db
                props.detection_mode = "PURE_AUDIO"
                log_mode_a = []
                panel_by_name["KUCHISYNC_PT_detection_engine"].draw(
                    types.SimpleNamespace(layout=_FakeRecordingLayout(log_mode_a)), ctx
                )
                drawn_props_a = {e["prop"] for e in log_mode_a if e["kind"] == "prop"}
                self.assertIn("silence_threshold_db", drawn_props_a)
                self.assertIn("voice_type", drawn_props_a)

                props.detection_mode = "TRANSCRIPT"
                log_mode_b = []
                panel_by_name["KUCHISYNC_PT_detection_engine"].draw(
                    types.SimpleNamespace(layout=_FakeRecordingLayout(log_mode_b)), ctx
                )
                drawn_props_b = {e["prop"] for e in log_mode_b if e["kind"] == "prop"}
                self.assertIn("silence_threshold_db", drawn_props_b)
                self.assertIn("transcript_text", drawn_props_b)
                self.assertIn("transcript_datablock", drawn_props_b)

                log_exec = []
                with mock.patch("kuchisync.audio_dsp.has_scipy", return_value=False):
                    panel_by_name["KUCHISYNC_PT_execution"].draw(
                        types.SimpleNamespace(layout=_FakeRecordingLayout(log_exec)), ctx
                    )
                gen_entries = [e for e in log_exec if e.get("operator") == "kuchisync.generate_lipsync"]
                self.assertEqual(len(gen_entries), 1)
                self.assertEqual(gen_entries[0]["scale_y"], 1.5)
                self.assertFalse(gen_entries[0]["enabled"])  # bad.wav doesn't exist after temp dir exit

                # SciPy installer offline gate & subprocess timer lifecycle + unregister cleanup
                scipy_op = op_by_id["kuchisync.install_scipy"]()
                scipy_op.reports = []
                bpy_mod.app.online_access = False
                with mock.patch("kuchisync.audio_dsp.has_scipy", return_value=False):
                    self.assertEqual(scipy_op.execute(ctx), {"CANCELLED"})
                    self.assertTrue(any("ERROR" in lvl for lvl, _ in scipy_op.reports))

                bpy_mod.app.online_access = True
                fake_proc = mock.MagicMock()
                fake_proc.poll.return_value = None
                with (
                    mock.patch("kuchisync.audio_dsp.has_scipy", return_value=False),
                    mock.patch("kuchisync.ui_panel.subprocess.Popen", return_value=fake_proc) as popen_mock,
                    mock.patch("kuchisync.ui_panel.site.addsitedir") as addsitedir_mock,
                    mock.patch("kuchisync.audio_dsp.refresh_scipy", return_value=True) as refresh_mock,
                ):
                    self.assertEqual(scipy_op.execute(ctx), {"FINISHED"})
                    popen_args = popen_mock.call_args[0][0]
                    self.assertEqual(popen_args[:5], [sys.executable, "-m", "pip", "install", "scipy"])
                    self.assertEqual(popen_args[5], "--target")
                    self.assertEqual(len(bpy_mod.app.timers.active()), 1)

                    # Panel 5 shows running state while process is alive
                    log_running = []
                    panel_by_name["KUCHISYNC_PT_execution"].draw(
                        types.SimpleNamespace(layout=_FakeRecordingLayout(log_running)), ctx
                    )
                    self.assertTrue(any("Installing SciPy" in e.get("text", "") for e in log_running))
                    install_btns = [e for e in log_running if e.get("operator") == "kuchisync.install_scipy"]
                    self.assertFalse(install_btns[0]["enabled"])

                    # Poll returns 0 -> activates target dir, refreshes scipy, stops timer
                    fake_proc.poll.return_value = 0
                    timer_cb = next(iter(bpy_mod.app.timers.active()))
                    self.assertIsNone(timer_cb())
                    addsitedir_mock.assert_called_once()
                    refresh_mock.assert_called_once()

                    # Test non-zero exit code + _read_install_log_tail() error display
                    def popen_failing(*args, **kwargs):
                        del args
                        out_fh = kwargs["stdout"]
                        out_fh.write(b"Collecting scipy\nERROR: No matching distribution found\n")
                        out_fh.flush()
                        fail_proc = mock.MagicMock()
                        fail_proc.poll.return_value = 1
                        return fail_proc

                    with mock.patch("kuchisync.ui_panel.subprocess.Popen", side_effect=popen_failing):
                        self.assertEqual(scipy_op.execute(ctx), {"FINISHED"})
                        timer_cb = next(iter(bpy_mod.app.timers.active()))
                        self.assertIsNone(timer_cb())
                    log_err = []
                    panel_by_name["KUCHISYNC_PT_execution"].draw(
                        types.SimpleNamespace(layout=_FakeRecordingLayout(log_err)), ctx
                    )
                    self.assertTrue(
                        any(
                            e["kind"] == "label"
                            and e["icon"] == "ERROR"
                            and e["alert"]
                            and "exit 1" in e["text"]
                            and "No matching distribution found" in e["text"]
                            for e in log_err
                        )
                    )

                    # Read-only target directory aborts without spawning pip
                    scipy_op.reports = []
                    with (
                        mock.patch("kuchisync.ui_panel.os.access", return_value=False),
                        mock.patch("kuchisync.ui_panel.subprocess.Popen") as ro_popen,
                    ):
                        self.assertEqual(scipy_op.execute(ctx), {"CANCELLED"})
                        ro_popen.assert_not_called()
                        self.assertTrue(any("read-only" in msg.lower() for _, msg in scipy_op.reports))

                    # Popen OSError closes temporary log handle cleanly without ResourceWarning leak
                    created_logs: list = []
                    orig_temp_file = tempfile.TemporaryFile

                    def spy_temp_file(*args, **kwargs):
                        fh = orig_temp_file(*args, **kwargs)
                        created_logs.append(fh)
                        return fh

                    scipy_op.reports = []
                    with (
                        mock.patch("kuchisync.ui_panel.tempfile.TemporaryFile", side_effect=spy_temp_file),
                        mock.patch("kuchisync.ui_panel.subprocess.Popen", side_effect=OSError("no pip")),
                    ):
                        self.assertEqual(scipy_op.execute(ctx), {"CANCELLED"})
                    self.assertEqual(len(created_logs), 1)
                    self.assertTrue(created_logs[0].closed)

                    # Test unregister() cleanup while process & timer are active
                    fake_proc.poll.return_value = None
                    scipy_op.execute(ctx)
                    self.assertEqual(len(bpy_mod.app.timers.active()), 1)
                    kuchisync.unregister()
                    fake_proc.terminate.assert_called_once()
                    self.assertEqual(len(bpy_mod.app.timers.active()), 0)
            finally:
                if hasattr(bpy_mod.types.Scene, "kuchisync"):
                    kuchisync.unregister()
        finally:
            shutil.rmtree(bpy_mod.utils._resources_root, ignore_errors=True)
            for name, orig in saved_modules.items():
                if orig is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = orig
            import kuchisync

            importlib.reload(ui_panel)
            importlib.reload(kuchisync)

    def test_package_addon_builds_root_extension_zip(self):
        from scripts import package_addon

        expected_root_files = {
            "__init__.py",
            "audio_dsp.py",
            "blender_manifest.toml",
            "keyframe_gen.py",
            "transcript_align.py",
            "ui_panel.py",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_zip = package_addon.build_extension_zip(pathlib.Path(tmp) / "kuchisync.zip")
            self.assertTrue(out_zip.is_file())
            with zipfile.ZipFile(out_zip, "r") as zf:
                names = set(zf.namelist())
                self.assertEqual(names, expected_root_files)
                self.assertFalse(any("__pycache__" in n or n.endswith(".pyc") for n in names))

        dist_zip = ROOT / "dist" / "kuchisync.zip"
        self.assertTrue(dist_zip.is_file(), "dist/kuchisync.zip missing")
        with zipfile.ZipFile(dist_zip, "r") as zf:
            names = set(zf.namelist())
            self.assertEqual(names, expected_root_files)
            self.assertFalse(any("__pycache__" in n or n.endswith(".pyc") for n in names))


if __name__ == "__main__":
    unittest.main()
