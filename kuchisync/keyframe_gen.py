# SPDX-License-Identifier: GPL-3.0-or-later

from collections.abc import Collection, Sequence
from dataclasses import dataclass
import os
import pathlib
import re

from . import audio_dsp
from .audio_dsp import FormantFrame, SIGMA_F1, SIGMA_F2, VOWEL_CENTROIDS
from .transcript_align import AlignedSyllable

VOWELS: tuple[str, ...] = audio_dsp.VOWELS

_EXACT_VOWEL_ALIASES: dict[str, tuple[str, ...]] = {
    "A": ("a", "aa", "ah", "vrc.v_aa", "mouth_a", "viseme_a", "viseme_aa", "mth_a", "あ", "ア", "ぁ", "ァ"),
    "I": ("i", "ii", "ih", "ee", "vrc.v_ih", "mouth_i", "viseme_i", "viseme_ih", "mth_i", "い", "イ", "ぃ", "ィ"),
    "U": ("u", "uu", "ou", "oo", "vrc.v_ou", "mouth_u", "viseme_u", "viseme_ou", "mth_u", "う", "ウ", "ぅ", "ゥ"),
    "E": ("e", "ee", "eh", "vrc.v_e", "mouth_e", "viseme_e", "viseme_eh", "mth_e", "え", "エ", "ぇ", "ェ"),
    "O": ("o", "oo", "oh", "vrc.v_oh", "mouth_o", "viseme_o", "viseme_oh", "mth_o", "お", "オ", "ぉ", "ォ"),
}

_CANONICAL_PREFIXES: tuple[str, ...] = ("vrc.v_", "viseme_", "mouth_", "mth_")
_AMBIGUOUS_EXACT_ALIASES: frozenset[str] = frozenset(("ee", "oo"))
_SMALL_KANA: frozenset[str] = frozenset(("ぁ", "ァ", "ぃ", "ィ", "ぅ", "ゥ", "ぇ", "ェ", "ぉ", "ォ"))
_RIG_HINT_TOKENS: frozenset[str] = frozenset(("vrc", "viseme", "mouth", "mth", "lip", "lipsync"))

_ASCII_BOUNDED_REGEX: dict[str, re.Pattern[str]] = {
    vowel: re.compile(
        r"(?<![a-z])(?:"
        + "|".join(
            re.escape(alias)
            for alias in aliases
            if alias.isascii() and "." not in alias and "_" not in alias
        )
        + r")(?![a-z])"
    )
    for vowel, aliases in _EXACT_VOWEL_ALIASES.items()
}

_KANA_ALIASES: dict[str, tuple[str, ...]] = {
    vowel: tuple(alias for alias in aliases if not alias.isascii())
    for vowel, aliases in _EXACT_VOWEL_ALIASES.items()
}

_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")
_TOKEN_SPLIT_RE = re.compile(r"[_\.\-\s]+")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_KEY_BLOCK_PATH_RE = re.compile(r"""^key_blocks\[(["'])(.*)\1\]\.value$""")

_VALID_INTERPOLATIONS: frozenset[str] = frozenset(("BEZIER", "LINEAR", "CONSTANT"))


@dataclass
class KeyframeEvent:
    """Single sparse Shape Key keyframe at an integer Blender timeline frame."""

    frame: int
    value: float


@dataclass
class _MappedSegment:
    """Syllable mapped onto a target Shape Key with computed peak amplitude."""

    shape_name: str
    start_frame: float
    end_frame: float
    peak: float
    valley_dip_before: bool


@dataclass
class _SegmentPlan:
    """Mutable sparse Attack-Sustain-Release frame plan for one segment."""

    shape_name: str
    s: float
    e: float
    i_start: int
    i_end: int
    peak: float
    valley_dip_before: bool
    pre_zero: int
    attack: int
    hold: tuple[int, float] | None
    release_zero: int
    omit_pre_zero: bool = False
    omit_release: bool = False
    bridge_event: tuple[int, float] | None = None

    @property
    def last_active_frame(self) -> int:
        """Return frame index of the final non-zero hold or attack key."""
        if self.hold is not None:
            return self.hold[0]
        return self.attack


def _score_candidate(name: str, lower_name: str, vowel: str) -> tuple[int, int] | None:
    """Return (priority_score, preference_tiebreak) for matching a Shape Key name to a vowel."""
    aliases = _EXACT_VOWEL_ALIASES[vowel]
    if lower_name in aliases:
        if lower_name in _AMBIGUOUS_EXACT_ALIASES:
            pref = 40
        elif lower_name.startswith(_CANONICAL_PREFIXES):
            pref = 90
        elif lower_name in _SMALL_KANA:
            pref = 70
        else:
            pref = 80
        return (100, pref)

    camel_split = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", name).lower()
    raw_tokens = [tok for tok in _TOKEN_SPLIT_RE.split(camel_split) if tok]
    if raw_tokens:
        has_rig_hint = any(tok in _RIG_HINT_TOKENS for tok in raw_tokens)
        best_token_pref = -1
        for tok in raw_tokens:
            stripped = _TRAILING_DIGITS_RE.sub("", tok)
            for candidate_tok, exact_bonus in ((tok, 5), (stripped, 0)):
                if not candidate_tok or candidate_tok not in aliases:
                    continue
                pref = 10 if candidate_tok in _AMBIGUOUS_EXACT_ALIASES else 30
                if has_rig_hint:
                    pref += 15
                pref += exact_bonus
                if pref > best_token_pref:
                    best_token_pref = pref
        if best_token_pref >= 0:
            return (70, best_token_pref)

    if _ASCII_BOUNDED_REGEX[vowel].search(lower_name):
        return (40, 20)

    for kana in _KANA_ALIASES[vowel]:
        if kana in name:
            pref = 10 if kana in _SMALL_KANA else 25
            return (40, pref)

    return None


def detect_shapekeys(key_names: Sequence[str]) -> dict[str, str]:
    """Auto-detect A/I/U/E/O Shape Key names using scored priority and greedy uniqueness."""
    candidates: list[tuple[int, int, int, int, int, str, str]] = []
    for idx, raw_name in enumerate(key_names):
        name = str(raw_name)
        lower_name = name.strip().lower()
        if not lower_name or lower_name == "basis":
            continue
        for v_idx, vowel in enumerate(VOWELS):
            scored = _score_candidate(name, lower_name, vowel)
            if scored is None:
                continue
            score, pref = scored
            candidates.append((score, pref, -len(name), -v_idx, -idx, vowel, name))

    candidates.sort(reverse=True)
    assigned: dict[str, str] = {}
    used_names: set[str] = set()
    for _score, _pref, _neg_len, _neg_v, _neg_idx, vowel, name in candidates:
        if vowel in assigned or name in used_names:
            continue
        assigned[vowel] = name
        used_names.add(name)
        if len(assigned) == len(VOWELS):
            break

    return {vowel: assigned[vowel] for vowel in VOWELS if vowel in assigned}


def resolve_vowel_fallback(vowel: str, mapped_vowels: Collection[str]) -> str:
    """Resolve an unmapped vowel to the nearest mapped vowel in normalized (F1, F2) space."""
    normalized_set = {str(v).strip().upper() for v in mapped_vowels}
    candidates = [v for v in VOWELS if v in normalized_set]
    if not candidates:
        raise ValueError("mapped_vowels must contain at least one valid vowel")

    target = str(vowel).strip().upper()
    if target in candidates:
        return target
    if target not in VOWELS:
        raise ValueError(f"Unknown vowel: {vowel}")

    centroids = VOWEL_CENTROIDS["INDONESIAN"]
    f1_t, f2_t = centroids[target]

    def _norm_dist(cand: str) -> tuple[float, int]:
        f1_c, f2_c = centroids[cand]
        d2 = ((f1_t - f1_c) / SIGMA_F1) ** 2 + ((f2_t - f2_c) / SIGMA_F2) ** 2
        return (d2, VOWELS.index(cand))

    return min(candidates, key=_norm_dist)


def _estimate_hop_sec(frames: Sequence[FormantFrame]) -> float:
    """Estimate frame hop step in seconds from consecutive FormantFrame timestamps."""
    deltas = [
        frames[i + 1].time_sec - frames[i].time_sec
        for i in range(len(frames) - 1)
        if frames[i + 1].time_sec > frames[i].time_sec
    ]
    if not deltas:
        return audio_dsp.HOP_MS / 1000.0
    deltas.sort()
    median_dt = deltas[len(deltas) // 2]
    if median_dt <= 0.0 or median_dt > 0.1:
        return audio_dsp.HOP_MS / 1000.0
    return float(median_dt)


def _merge_close_identical_segments(
    segments: Sequence[AlignedSyllable],
    fps: float,
    max_gap_frames: float = 1.0,
    max_gap_sec: float = 0.015,
) -> list[AlignedSyllable]:
    """Merge adjacent segments with the same vowel separated by <= 1 frame or <= 15ms."""
    merged: list[AlignedSyllable] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap_frames = seg.start_frame - prev.end_frame
            gap_sec = gap_frames / fps
            if seg.vowel == prev.vowel and (
                gap_frames <= max_gap_frames + 1e-6 or gap_sec <= max_gap_sec + 1e-6
            ):
                prev.end_frame = max(prev.end_frame, seg.end_frame)
                prev.peak_rms_db = max(prev.peak_rms_db, seg.peak_rms_db)
                continue
        merged.append(
            AlignedSyllable(
                vowel=seg.vowel,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                peak_rms_db=seg.peak_rms_db,
                valley_dip_before=False,
            )
        )
    return merged


def build_mode_a_segments(
    frames: Sequence[FormantFrame],
    fps: float,
    start_frame: int,
    min_phoneme_frames: int,
) -> list[AlignedSyllable]:
    """Aggregate voiced FormantFrames into merged and spike-filtered Mode A segments."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive (got {fps})")
    if not frames:
        return []

    hop_sec = _estimate_hop_sec(frames)
    raw_segments: list[AlignedSyllable] = []

    cur_vowel: str | None = None
    cur_start_t = 0.0
    cur_end_t = 0.0
    cur_peak_db = -120.0

    def _flush_run() -> None:
        nonlocal cur_vowel
        if cur_vowel is None:
            return
        raw_segments.append(
            AlignedSyllable(
                vowel=cur_vowel,
                start_frame=float(start_frame) + cur_start_t * float(fps),
                end_frame=float(start_frame) + cur_end_t * float(fps),
                peak_rms_db=cur_peak_db,
                valley_dip_before=False,
            )
        )
        cur_vowel = None

    for frame in frames:
        voiced_vowel = frame.vowel.upper() if (frame.is_voiced and frame.vowel) else None
        if voiced_vowel not in VOWELS:
            _flush_run()
            continue

        if cur_vowel == voiced_vowel:
            cur_end_t = frame.time_sec + hop_sec
            if frame.rms_db > cur_peak_db:
                cur_peak_db = frame.rms_db
        else:
            _flush_run()
            cur_vowel = voiced_vowel
            cur_start_t = frame.time_sec
            cur_end_t = frame.time_sec + hop_sec
            cur_peak_db = frame.rms_db

    _flush_run()

    if not raw_segments:
        return []

    # Bridge single-hop unvoiced dropouts inside steady vowels first.
    segments = _merge_close_identical_segments(raw_segments, fps=float(fps))

    # Drop sub-frame 1-2 hop transient glitches before final duration filtering so steady runs heal.
    min_frames = max(0.0, float(min_phoneme_frames))
    glitch_cutoff = min(1.0, min_frames)
    if glitch_cutoff > 0.0:
        segments = [s for s in segments if (s.end_frame - s.start_frame) + 1e-6 >= glitch_cutoff]
        segments = _merge_close_identical_segments(segments, fps=float(fps))

    if min_frames > 0.0:
        segments = [s for s in segments if (s.end_frame - s.start_frame) + 1e-6 >= min_frames]
        segments = _merge_close_identical_segments(segments, fps=float(fps))

    return segments


def _compute_energy_scales(syllables: Sequence[AlignedSyllable]) -> list[float]:
    """Scale each syllable's peak amplitude in [0.65, 1.0] from voiced utterance RMS range."""
    if not syllables:
        return []
    rms_vals = [float(s.peak_rms_db) for s in syllables]
    rms_min = min(rms_vals)
    rms_max = max(rms_vals)
    span = rms_max - rms_min
    if span <= 0.5:
        return [1.0] * len(syllables)
    scales: list[float] = []
    for val in rms_vals:
        rel = min(1.0, max(0.0, (val - rms_min) / span))
        scales.append(0.65 + 0.35 * rel)
    return scales


def _merge_adjacent_same_shape(segments: Sequence[_MappedSegment]) -> list[_MappedSegment]:
    """Merge consecutive Mode A segments mapped onto the same Shape Key within <= 1 frame."""
    merged: list[_MappedSegment] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            if seg.shape_name == prev.shape_name and (seg.start_frame - prev.end_frame) <= 1.0 + 1e-6:
                prev.end_frame = max(prev.end_frame, seg.end_frame)
                prev.peak = max(prev.peak, seg.peak)
                continue
        merged.append(
            _MappedSegment(
                shape_name=seg.shape_name,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                peak=seg.peak,
                valley_dip_before=seg.valley_dip_before,
            )
        )
    return merged


def _build_initial_plan(seg: _MappedSegment, transition_frames: int) -> _SegmentPlan:
    """Create initial sparse Attack-Sustain-Release frame positions for one segment."""
    t_frames = max(0, int(transition_frames))
    i_start = max(0, int(round(seg.start_frame)))
    i_end = max(i_start + 1, int(round(seg.end_frame)))

    pre_zero = max(0, i_start - t_frames)
    attack = i_start + 1

    hold: tuple[int, float] | None = None
    if (seg.end_frame - seg.start_frame) > 5.0 and (i_end - 1) > attack:
        hold = (i_end - 1, round(seg.peak * 0.9, 6))

    last_active = hold[0] if hold is not None else attack
    release_zero = max(last_active + 1, i_end + t_frames)

    return _SegmentPlan(
        shape_name=seg.shape_name,
        s=seg.start_frame,
        e=seg.end_frame,
        i_start=i_start,
        i_end=i_end,
        peak=seg.peak,
        valley_dip_before=seg.valley_dip_before,
        pre_zero=pre_zero,
        attack=attack,
        hold=hold,
        release_zero=release_zero,
    )


def _shift_attack_after(plan: _SegmentPlan, min_frame: int) -> None:
    """Advance a plan's attack (and drop/adjust hold) so it stays strictly after min_frame."""
    if plan.attack >= min_frame:
        return
    plan.attack = min_frame
    if plan.hold is not None and plan.hold[0] <= plan.attack:
        plan.hold = None
    if plan.release_zero <= plan.last_active_frame:
        plan.release_zero = plan.last_active_frame + 1


def _resolve_distinct_pair(
    prev: _SegmentPlan,
    curr: _SegmentPlan,
    transition_frames: int,
    co_art: float,
) -> None:
    """Apply Co-Articulation cross-fade or clean separation between distinct Shape Keys."""
    t_frames = max(0, int(transition_frames))
    gap = curr.s - prev.e
    mid = int(round(0.5 * (prev.i_end + curr.i_start)))

    if co_art <= 0.0:
        if gap <= float(t_frames) + 1e-6 or prev.release_zero > curr.pre_zero:
            prev.release_zero = max(prev.last_active_frame + 1, min(prev.release_zero, mid))
            curr.pre_zero = min(curr.attack - 1, max(curr.pre_zero, mid, prev.release_zero))
        return

    if gap > float(t_frames) + 1e-6:
        return

    overlap = max(1, round(float(t_frames) * co_art * 2.0))
    prev.release_zero = max(
        prev.last_active_frame + 1,
        min(prev.i_end + t_frames, max(prev.i_end, mid) + overlap),
    )
    desired_pre = max(prev.i_start, min(curr.i_start, mid) - overlap)
    curr.pre_zero = max(0, min(curr.attack - 1, max(curr.i_start - t_frames, desired_pre)))


def _clamp_same_shape_boundary(prev: _SegmentPlan, curr: _SegmentPlan) -> None:
    """Collapse crossed release/pre-zero frames onto a single midpoint zero."""
    mid = int(round(0.5 * (prev.i_end + curr.i_start)))
    prev.release_zero = max(prev.last_active_frame + 1, min(prev.release_zero, mid))
    curr.pre_zero = min(curr.attack - 1, max(curr.pre_zero, prev.release_zero))


def _resolve_same_shape_adjacent(
    prev: _SegmentPlan,
    curr: _SegmentPlan,
    transition_frames: int,
    preserve_syllables: bool,
) -> None:
    """Handle consecutive segments sharing a Shape Key (valley dip, bridge, or single zero)."""
    t_frames = max(0, int(transition_frames))
    gap = curr.s - prev.e

    if gap <= float(t_frames) + 1e-6:
        if preserve_syllables and curr.valley_dip_before:
            dip_val = 0.35 * min(prev.peak, curr.peak)
            dip_frame = int(round(0.5 * (prev.i_end + curr.i_start)))
            if prev.hold is not None and dip_frame <= prev.hold[0]:
                prev.hold = None
            if dip_frame <= prev.last_active_frame:
                dip_frame = prev.last_active_frame + 1
            if dip_frame >= curr.attack:
                _shift_attack_after(curr, dip_frame + 1)
            prev.omit_release = True
            curr.omit_pre_zero = True
            prev.bridge_event = (dip_frame, dip_val)
            return

        if preserve_syllables and not curr.valley_dip_before:
            # Remapped distinct vowels (e.g. E -> I) flow without snapping shut to 0.0.
            if prev.hold is not None and prev.hold[0] >= curr.attack:
                prev.hold = None
            if prev.last_active_frame >= curr.attack:
                _shift_attack_after(curr, prev.last_active_frame + 1)
            prev.omit_release = True
            curr.omit_pre_zero = True
            return

        _clamp_same_shape_boundary(prev, curr)
        return

    if prev.release_zero >= curr.pre_zero:
        _clamp_same_shape_boundary(prev, curr)


def schedule_keyframes(
    syllables: Sequence[AlignedSyllable],
    mapped_shapes: dict[str, str],
    vowel_strengths: dict[str, float],
    start_frame: int,
    max_mouth_open: float,
    min_phoneme_frames: int,
    transition_frames: int,
    co_articulation: float,
    preserve_syllables: bool,
) -> dict[str, list[KeyframeEvent]]:
    """Build sparse, co-articulated Attack-Sustain-Release keyframes per Shape Key."""
    del start_frame
    mapped_norm = {v: str(mapped_shapes.get(v, "") or "").strip() for v in VOWELS}
    mapped_vowels = [v for v in VOWELS if mapped_norm[v]]
    if not mapped_vowels or float(max_mouth_open) <= 0.0 or not syllables:
        return {}

    energy_scales = _compute_energy_scales(syllables)
    min_dur = float(max(0, min_phoneme_frames))
    mouth_open = float(max_mouth_open)
    co_art = min(0.5, max(0.0, float(co_articulation)))

    mapped_segments: list[_MappedSegment] = []
    for syl, e_scale in zip(syllables, energy_scales):
        dur = float(syl.end_frame) - float(syl.start_frame)
        if not preserve_syllables:
            if dur + 1e-6 < min_dur:
                continue
        elif dur <= 0.0:
            continue

        orig_vowel = str(syl.vowel).strip().upper()
        if orig_vowel not in VOWELS:
            continue

        target_vowel = (
            orig_vowel
            if mapped_norm[orig_vowel]
            else resolve_vowel_fallback(orig_vowel, mapped_vowels)
        )
        shape_name = mapped_norm[target_vowel]
        strength = float(vowel_strengths.get(target_vowel, 1.0))
        if not shape_name or strength <= 0.0:
            continue

        peak = min(1.5, max(0.0, e_scale * mouth_open * strength))
        if peak <= 0.0:
            continue

        mapped_segments.append(
            _MappedSegment(
                shape_name=shape_name,
                start_frame=float(syl.start_frame),
                end_frame=float(syl.end_frame),
                peak=peak,
                valley_dip_before=bool(syl.valley_dip_before),
            )
        )

    if not mapped_segments:
        return {}

    mapped_segments.sort(key=lambda seg: (seg.start_frame, seg.end_frame))
    if not preserve_syllables:
        mapped_segments = _merge_adjacent_same_shape(mapped_segments)

    plans = [_build_initial_plan(seg, transition_frames) for seg in mapped_segments]

    for idx in range(len(plans) - 1):
        prev = plans[idx]
        curr = plans[idx + 1]
        if prev.shape_name != curr.shape_name:
            _resolve_distinct_pair(prev, curr, transition_frames, co_art)
        else:
            _resolve_same_shape_adjacent(prev, curr, transition_frames, preserve_syllables)

    # Prevent same-shape keyframe collisions across short intervening vowels (e.g. A -> I -> A).
    last_by_shape: dict[str, _SegmentPlan] = {}
    for plan in plans:
        prev_same = last_by_shape.get(plan.shape_name)
        if prev_same is not None and not prev_same.omit_release and not plan.omit_pre_zero:
            if prev_same.release_zero >= plan.pre_zero:
                _clamp_same_shape_boundary(prev_same, plan)
        last_by_shape[plan.shape_name] = plan

    events_by_shape: dict[str, dict[int, float]] = {}
    for plan in plans:
        frame_map = events_by_shape.setdefault(plan.shape_name, {})
        if not plan.omit_pre_zero:
            frame_map.setdefault(plan.pre_zero, 0.0)
        frame_map[plan.attack] = max(frame_map.get(plan.attack, 0.0), plan.peak)
        if plan.hold is not None:
            frame_map[plan.hold[0]] = plan.hold[1]
        if plan.bridge_event is not None:
            frame_map[plan.bridge_event[0]] = plan.bridge_event[1]
        if not plan.omit_release:
            frame_map.setdefault(plan.release_zero, 0.0)

    out: dict[str, list[KeyframeEvent]] = {}
    for shape_name, frame_map in events_by_shape.items():
        if not any(val > 0.0 for val in frame_map.values()):
            continue
        out[shape_name] = [
            KeyframeEvent(frame=frame, value=float(frame_map[frame]))
            for frame in sorted(frame_map)
        ]
    return out


def _ensure_action_slot_bound(anim_data, action) -> None:
    """Bind a compatible ActionSlot on Slotted Actions when action_slot is unset."""
    if not hasattr(anim_data, "action_slot") or anim_data.action_slot is not None:
        return
    slots = getattr(action, "slots", None)
    if not slots:
        return
    chosen = None
    for slot in slots:
        if getattr(slot, "target_id_type", None) in ("KEY", "UNSPECIFIED", None):
            chosen = slot
            break
    if chosen is None:
        try:
            chosen = slots[0]
        except Exception:
            return
    try:
        anim_data.action_slot = chosen
    except (AttributeError, RuntimeError, TypeError):
        pass


def iter_shape_key_fcurves(shape_keys) -> list:
    """Return F-Curves on shape_keys across Blender 4.4-5.2+ Slotted and legacy Actions."""
    if shape_keys is None:
        return []
    anim_data = getattr(shape_keys, "animation_data", None)
    if anim_data is None:
        return []
    action = getattr(anim_data, "action", None)
    if action is None:
        return []

    _ensure_action_slot_bound(anim_data, action)
    slot = getattr(anim_data, "action_slot", None)

    if slot is not None:
        try:
            from bpy_extras import anim_utils
        except Exception:
            anim_utils = None
        if anim_utils is not None and callable(
            getattr(anim_utils, "action_get_channelbag_for_slot", None)
        ):
            try:
                bag = anim_utils.action_get_channelbag_for_slot(action, slot)
            except (AttributeError, TypeError, RuntimeError):
                bag = None
            if bag is not None and hasattr(bag, "fcurves"):
                return list(bag.fcurves)

    layers = getattr(action, "layers", None)
    if layers:
        found: list = []
        for layer in layers:
            for strip in getattr(layer, "strips", ()):
                bag = None
                if slot is not None and callable(getattr(strip, "channelbag", None)):
                    try:
                        bag = strip.channelbag(slot)
                    except (AttributeError, TypeError, RuntimeError):
                        bag = None
                if bag is not None and hasattr(bag, "fcurves"):
                    found.extend(bag.fcurves)
                    continue
                for cand_bag in getattr(strip, "channelbags", ()):
                    if slot is None or getattr(cand_bag, "slot", None) in (None, slot):
                        if hasattr(cand_bag, "fcurves"):
                            found.extend(cand_bag.fcurves)
        if found:
            return found

    legacy_fcurves = getattr(action, "fcurves", None)
    if legacy_fcurves is not None:
        return list(legacy_fcurves)
    return []


def _extract_key_block_name(data_path: str) -> str | None:
    """Extract Shape Key block name from an F-Curve data_path such as key_blocks[\"A\"].value."""
    match = _KEY_BLOCK_PATH_RE.match(str(data_path))
    if not match:
        return None
    return re.sub(r"\\(.)", r"\1", match.group(2))


def clear_lipsync_keyframes(
    obj,
    target_shape_names: Collection[str],
    frame_range: tuple[int, int] | None = None,
) -> int:
    """Delete keyframes on target Shape Keys (scoped by frame_range if given), leaving others intact."""
    shape_keys = getattr(getattr(obj, "data", None), "shape_keys", None)
    if shape_keys is None:
        return 0
    names_set = {str(name) for name in target_shape_names if name}
    if not names_set:
        return 0

    f_min = float(frame_range[0]) if frame_range is not None else None
    f_max = float(frame_range[1]) if frame_range is not None else None
    removed = 0

    for fc in iter_shape_key_fcurves(shape_keys):
        kb_name = _extract_key_block_name(getattr(fc, "data_path", ""))
        if kb_name not in names_set:
            continue
        kps = getattr(fc, "keyframe_points", None)
        if kps is None:
            continue

        if f_min is None or f_max is None:
            count = len(kps)
            if count > 0:
                kps.clear()
                if callable(getattr(fc, "update", None)):
                    fc.update()
                removed += count
            continue

        removed_here = 0
        for idx in range(len(kps) - 1, -1, -1):
            kp = kps[idx]
            if (f_min - 1e-6) <= float(kp.co[0]) <= (f_max + 1e-6):
                kps.remove(kp, fast=True)
                removed_here += 1
        if removed_here:
            if callable(getattr(fc, "update", None)):
                fc.update()
            removed += removed_here

    return removed


def apply_keyframes_to_object(
    obj,
    scheduled: dict[str, list[KeyframeEvent]],
    interpolation_mode: str = "BEZIER",
) -> int:
    """Write sparse keyframes onto Shape Key blocks and configure F-Curve interpolation/handles."""
    mode = str(interpolation_mode or "BEZIER").strip().upper()
    if mode not in _VALID_INTERPOLATIONS:
        raise ValueError(f"Unsupported interpolation mode: {interpolation_mode}")

    shape_keys = getattr(getattr(obj, "data", None), "shape_keys", None)
    if shape_keys is None:
        return 0

    if getattr(shape_keys, "animation_data", None) is None and callable(
        getattr(shape_keys, "animation_data_create", None)
    ):
        shape_keys.animation_data_create()

    key_blocks = getattr(shape_keys, "key_blocks", None)
    if key_blocks is None:
        return 0

    written = 0
    written_frames_by_shape: dict[str, set[int]] = {}
    for shape_name, events in scheduled.items():
        if not events:
            continue
        kb = key_blocks.get(shape_name)
        if kb is None:
            continue

        max_val = max(float(ev.value) for ev in events)
        if hasattr(kb, "slider_max") and max_val > float(kb.slider_max):
            kb.slider_max = min(1.5, max(float(kb.slider_max), round(max_val + 0.05, 2)))

        frame_set = written_frames_by_shape.setdefault(shape_name, set())
        for ev in events:
            frame_i = int(ev.frame)
            kb.value = float(ev.value)
            kb.keyframe_insert(data_path="value", frame=frame_i)
            frame_set.add(frame_i)
            written += 1

    if written_frames_by_shape:
        for fc in iter_shape_key_fcurves(shape_keys):
            kb_name = _extract_key_block_name(getattr(fc, "data_path", ""))
            target_frames = written_frames_by_shape.get(kb_name)
            if not target_frames:
                continue
            updated_any = False
            for kp in getattr(fc, "keyframe_points", ()):
                frame_f = float(kp.co[0])
                frame_i = int(round(frame_f))
                if abs(frame_f - frame_i) <= 1e-4 and frame_i in target_frames:
                    kp.interpolation = mode
                    if mode == "BEZIER":
                        kp.handle_left_type = "AUTO_CLAMPED"
                        kp.handle_right_type = "AUTO_CLAMPED"
                    updated_any = True
            if updated_any and callable(getattr(fc, "update", None)):
                fc.update()

    return written


def _normalize_audio_path(filepath: str) -> str:
    """Resolve Blender // relative paths if bpy is present and normalize slashes."""
    text = str(filepath)
    try:
        import bpy
    except ModuleNotFoundError:
        bpy = None
    if bpy is not None and hasattr(bpy, "path") and callable(getattr(bpy.path, "abspath", None)):
        text = bpy.path.abspath(text)
    return os.path.normpath(os.path.abspath(text))


def _resolve_vse_container(sequence_editor):
    """Return VSE strip collection across Blender 5.0+ (strips) and 4.4-4.5 (sequences)."""
    for attr in ("strips", "sequences"):
        cand = getattr(sequence_editor, attr, None)
        if cand is not None and callable(getattr(cand, "new_sound", None)):
            return cand
    return None


def _find_open_vse_channel(strips: Sequence, frame_start: int) -> int:
    """Choose the lowest VSE channel >= 1 free at frame_start."""
    occupied: set[int] = set()
    for strip in strips:
        ch = int(getattr(strip, "channel", 1))
        s_start = float(getattr(strip, "frame_start", 0))
        s_end = getattr(strip, "frame_final_end", None)
        if s_end is None:
            if int(round(s_start)) == int(frame_start):
                occupied.add(ch)
        elif s_start <= float(frame_start) < float(s_end):
            occupied.add(ch)

    channel = 1
    while channel in occupied:
        channel += 1
    return channel


def add_audio_to_vse(scene, filepath: str, frame_start: int) -> None:
    """Insert an audio clip into the VSE (4.4-5.2+ compatible) and enable AUDIO_SYNC."""
    if scene is None:
        raise ValueError("scene must not be None")
    if not filepath:
        return

    if getattr(scene, "sequence_editor", None) is None and callable(
        getattr(scene, "sequence_editor_create", None)
    ):
        scene.sequence_editor_create()

    se = getattr(scene, "sequence_editor", None)
    if se is None:
        raise RuntimeError("Scene sequence_editor could not be initialized")

    container = _resolve_vse_container(se)
    if container is None:
        raise RuntimeError("Sequence editor does not expose strips.new_sound or sequences.new_sound")

    target_norm = _normalize_audio_path(filepath)
    existing = list(container)
    for strip in existing:
        if getattr(strip, "type", None) != "SOUND":
            continue
        sound_obj = getattr(strip, "sound", None)
        strip_path = getattr(sound_obj, "filepath", "") if sound_obj is not None else ""
        if not strip_path:
            continue
        if (
            _normalize_audio_path(strip_path) == target_norm
            and int(round(float(getattr(strip, "frame_start", -1)))) == int(frame_start)
        ):
            if hasattr(scene, "sync_mode"):
                scene.sync_mode = "AUDIO_SYNC"
            return

    channel = _find_open_vse_channel(existing, int(frame_start))
    strip_name = f"KuchiSync_{pathlib.Path(filepath).stem}"[:60]
    container.new_sound(
        name=strip_name,
        filepath=str(filepath),
        channel=channel,
        frame_start=int(frame_start),
    )
    if hasattr(scene, "sync_mode"):
        scene.sync_mode = "AUDIO_SYNC"
