# SPDX-License-Identifier: GPL-3.0-or-later

import importlib
import importlib.util
import sys

bl_info = {
    "name": "KuchiSync",
    "author": "RyuuRead",
    "version": (0, 2, 0),
    "blender": (4, 4, 0),
    "location": "View3D > UI > KuchiSync",
    "description": "Sparse A-I-U-E-O Shape Key lip-sync from audio and transcripts",
    "category": "Animation",
}

def _has_module(module_name: str) -> bool:
    """Check module availability safely even if bare mocks without __spec__ exist."""
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    return importlib.util.find_spec(module_name) is not None


if _has_module("bpy"):
    if "bpy" in locals() or any(
        f"{__name__}.{_sub}" in sys.modules
        for _sub in ("audio_dsp", "transcript_align", "keyframe_gen", "ui_panel")
    ):
        for _submodule in ("audio_dsp", "transcript_align", "keyframe_gen", "ui_panel"):
            _module_name = f"{__name__}.{_submodule}"
            if _module_name in sys.modules:
                importlib.reload(sys.modules[_module_name])
    import bpy

    if _has_module(f"{__name__}.ui_panel"):
        from . import ui_panel as _ui_panel
    else:
        _ui_panel = None
else:
    bpy = None
    _ui_panel = None


def register() -> None:
    """Register add-on classes and scene properties when running inside Blender."""
    if _ui_panel is not None:
        _ui_panel.register()


def unregister() -> None:
    """Unregister add-on classes and clean up scene properties."""
    if _ui_panel is not None:
        _ui_panel.unregister()
