# KuchiSync UI Direction (Blender N-Panel)

## Design Read

Reading this as: native Blender 3D Viewport N-Panel utility for character animators, in a calm workbench tool style, dial ENERGY 1 / RHYTHM 1 / MOTION 1.

- **ENERGY 1 (Calm):** Native Blender theme palette, zero marketing badges, zero glow or decorative borders. Only the primary generation button uses taller height (`scale_y = 1.5`) as the single visual focal point.
- **RHYTHM 1 (Consistent):** Five sequential N-Panel sections matching the animator's workflow from top to bottom (`Target Rig & Shape Keys` -> `Audio & Timing` -> `Detection Engine` -> `Animation & Curve Refinement` -> `Execution & Utilities`).
- **MOTION 1 (Static):** Standard Blender widget hover/active/disabled states only; no continuous timers or pulsing indicators except status polling while an explicit background SciPy installation is running.

## Purpose Decisions (antislop R-31)

- **Color & Contrast (R-25, R-29, antislop-human):** Inherit Blender's active UI theme for text, backgrounds, and focus outlines instead of hardcoding custom colors. Never signal status by red/green hue alone: every Shape Key mapping slot and readiness check pairs an icon shape (`CHECKMARK` for mapped and valid, `ERROR` for unmapped/missing) with explicit text labels or alert cues.
- **Icons (R-04):** Every Blender icon matches its literal action or state (`VIEWZOOM` for auto-detect, `FILE_FOLDER` for audio/transcript file pickers, `TIME` for syncing Start Frame to playhead, `PLAY` for generating keyframes, `TRASH` for clearing keyframes, `CHECKMARK`/`ERROR`/`INFO` for state rows), wrapped with a `safe_icon()` RNA enum lookup so Blender 4.4-5.2 icon differences never crash UI drawing. No sparkle, magic, star, or emoji characters anywhere in labels or tooltips.
- **Mapping Row Composition (Faceit / LipSync Pro + Rhubarb reference):** Each vowel (`A`, `I`, `U`, `E`, `O`) uses one compact aligned row: `[Status Icon] [Vowel] [Shape Key prop_search] [Strength 0.0-1.5]`. Placing the per-vowel multiplier directly next to each Shape Key slot prevents modal dialogs and lets animators balance exaggerated or subtle morphs in place. Partial mappings (1-4 vowels) stay runnable via automatic nearest-vowel fallback, with a clear informational note in Panel 1.
- **Detection Engine Composition:** `Detection Mode` (`Pure Audio (Formant)` vs `Audio + Transcript`) sits at the top, followed by the shared `Silence Threshold (dB)` slider used by both modes, then mode-specific controls (`Voice Type` in Pure Audio; inline `Transcript` string, optional `bpy.types.Text` datablock selector, `Load .TXT File`, and multi-line/vowel count summary in Transcript mode).
- **UI States (R-27, C-4):**
  - **Empty / Invalid Target:** When no active object exists, active object is not `MESH`, or mesh has no `shape_keys`, display an explicit guidance row naming the missing condition and disable `Generate Auto Lip-Sync`.
  - **Missing Inputs:** If 0 vowel Shape Keys are mapped, audio path is empty/missing on disk, or Transcript mode has no text, display the exact missing item in `Execution & Utilities` before running.
  - **Running / Loading:** While `Install SciPy Automatically` runs in a background subprocess, disable the install trigger and display live progress text (`Installing SciPy...`).
  - **Error:** Report audio decode failures, unsupported files, empty transcripts, or offline/read-only pip errors clearly through Blender's `self.report({'ERROR'}, ...)` and panel feedback.
- **Copywriting (R-02, R-15, R-16):** Functional, concrete labels (`Auto-Detect Shape Keys`, `Generate Auto Lip-Sync`, `Clear Lip-Sync Keyframes`, `Load .TXT File`, `Install SciPy Automatically`). Zero buzzwords (`AI Powered`, `Revolutionary`, `Seamless`) and zero em dashes.
