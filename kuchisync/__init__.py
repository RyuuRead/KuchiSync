# SPDX-License-Identifier: GPL-3.0-or-later

import importlib
import importlib.util
import sys

bl_info = {
    "name": "KuchiSync",
    "author": "RyuuRead",
    "version": (1, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > UI > LipSync",
    "description": "Sparse A-I-U-E-O Shape Key lip-sync from audio and transcripts",
    "category": "Animation",
}

if importlib.util.find_spec("bpy") is not None:
    if "bpy" in locals():
        for _submodule in ("audio_dsp", "transcript_align", "keyframe_gen", "ui_panel"):
            _module_name = f"{__name__}.{_submodule}"
            if _module_name in sys.modules:
                importlib.reload(sys.modules[_module_name])
    import bpy

    if importlib.util.find_spec(f"{__name__}.ui_panel") is not None:
        from . import ui_panel
    else:
        ui_panel = None
else:
    ui_panel = None


def register() -> None:
    """Register add-on classes and scene properties when running inside Blender."""
    if ui_panel is not None:
        ui_panel.register()


def unregister() -> None:
    """Unregister add-on classes and clean up scene properties."""
    if ui_panel is not None:
        ui_panel.unregister()
