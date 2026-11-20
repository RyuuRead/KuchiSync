# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kuchisync import keyframe_gen
from kuchisync.audio_dsp import FormantFrame
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


def _make_frame(time_sec: float, vowel: str | None, rms_db: float = -16.0, is_voiced: bool = True) -> FormantFrame:
    return FormantFrame(
        time_sec=time_sec,
        f1=800.0 if is_voiced else None,
        f2=1250.0 if is_voiced else None,
        rms_db=rms_db,
        zcr=0.08,
        is_voiced=is_voiced,
        vowel=vowel if is_voiced else None,
        scores={},
    )


class TestKeyframeGen(unittest.TestCase):
    def test_auto_detect_alphabet_kana_and_no_false_positives(self):
        names = ["Basis", "Angry", "Happy", "Blink", "vrc.v_aa", "い", "mouth_u", "viseme_eh", "お"]
        detected = keyframe_gen.detect_shapekeys(names)
        self.assertEqual(detected, {"A": "vrc.v_aa", "I": "い", "U": "mouth_u", "E": "viseme_eh", "O": "お"})

    def test_auto_detect_ambiguous_ee_oo_greedy_and_bounds(self):
        names = ["basis", "Smile", "Sad", "Mouth_Open", "vrc.v_ih", "ee", "mouth_o", "oo", "face_A_01"]
        detected = keyframe_gen.detect_shapekeys(names)
        self.assertEqual(detected["A"], "face_A_01")
        self.assertEqual(detected["I"], "vrc.v_ih")
        self.assertEqual(detected["E"], "ee")
        self.assertEqual(detected["O"], "mouth_o")
        self.assertEqual(detected["U"], "oo")
        self.assertEqual(len(set(detected.values())), 5)

        kana_substr = ["Basis", "口あ", "リップい", "顔ア", "Blink"]
        detected_kana = keyframe_gen.detect_shapekeys(kana_substr)
        self.assertEqual(detected_kana["A"], "口あ")
        self.assertEqual(detected_kana["I"], "リップい")
        self.assertNotIn("O", detected_kana)

    def test_small_kana_skipped_in_substring_tier_but_kept_at_exact_and_token_tiers(self):
        # Partial rigs without い/え must not match small kana inside ウィンク or フェイス.
        no_i_rig = ["Basis", "あ", "う", "お", "ウィンク"]
        det_no_i = keyframe_gen.detect_shapekeys(no_i_rig)
        self.assertNotIn("I", det_no_i)
        self.assertNotIn("ウィンク", det_no_i.values())

        no_e_rig = ["Basis", "あ", "い", "う", "お", "ウィンク", "フェイス"]
        det_no_e = keyframe_gen.detect_shapekeys(no_e_rig)
        self.assertNotIn("E", det_no_e)
        self.assertNotIn("ウィンク", det_no_e.values())
        self.assertNotIn("フェイス", det_no_e.values())

        no_ie_rig = ["Basis", "あ", "う", "お", "ウィンク", "フェア"]
        det_no_ie = keyframe_gen.detect_shapekeys(no_ie_rig)
        self.assertEqual(det_no_ie, {"A": "あ", "U": "う", "O": "お"})

        # Exact small-kana names remain at tier 100.
        exact_small = ["Basis", "ぁ", "ィ", "ぅ", "ェ", "ぉ"]
        self.assertEqual(
            keyframe_gen.detect_shapekeys(exact_small),
            {"A": "ぁ", "I": "ィ", "U": "ぅ", "E": "ェ", "O": "ぉ"},
        )

        # Separator-delimited small-kana tokens remain at tier 70.
        token_small = ["Basis", "lip_ぁ", "mouth_ィ", "mth_ゥ_01", "v_ぇ", "face_ぉ"]
        self.assertEqual(
            keyframe_gen.detect_shapekeys(token_small),
            {"A": "lip_ぁ", "I": "mouth_ィ", "U": "mth_ゥ_01", "E": "v_ぇ", "O": "face_ぉ"},
        )

    def test_partial_mapping_fallback(self):
        self.assertEqual(keyframe_gen.resolve_vowel_fallback("E", ["A", "I", "U", "O"]), "I")
        self.assertEqual(keyframe_gen.resolve_vowel_fallback("O", ["A", "I", "U", "E"]), "U")
        with self.assertRaises(ValueError):
            keyframe_gen.resolve_vowel_fallback("A", [])

    def test_build_mode_a_segments_merges_and_drops_micro_spikes(self):
        # 24 fps -> 1 frame = 41.67ms; 10ms hop -> 3 frames = 125ms (>= 13 hops).
        frames = []
        # 0.00..0.14s: 15 hops of A with a 1-hop unvoiced dropout at index 7
        for idx in range(15):
            if idx == 7:
                frames.append(_make_frame(idx * 0.01, None, rms_db=-45.0, is_voiced=False))
            else:
                frames.append(_make_frame(idx * 0.01, "A", rms_db=-18.0 if idx != 3 else -12.0, is_voiced=True))
        # 0.15..0.16s: 2-hop micro-spike of E (< 3 frames)
        frames.append(_make_frame(0.15, "E", rms_db=-15.0, is_voiced=True))
        frames.append(_make_frame(0.16, "E", rms_db=-15.0, is_voiced=True))
        # 0.17..0.31s: 15 hops of A -> adjacent to first A after micro-spike removal (gap < 1 frame)
        for idx in range(17, 32):
            frames.append(_make_frame(idx * 0.01, "A", rms_db=-16.0, is_voiced=True))
        # 0.32..0.41s: 10 hops of silent gap (2.4 frames)
        for idx in range(32, 42):
            frames.append(_make_frame(idx * 0.01, None, rms_db=-60.0, is_voiced=False))
        # 0.42..0.46s: 5 hops of U (1.2 frames < 3 frames -> dropped)
        for idx in range(42, 47):
            frames.append(_make_frame(idx * 0.01, "U", rms_db=-14.0, is_voiced=True))
        # 0.47..0.62s: 16 hops of I (3.84 frames >= 3 -> kept)
        for idx in range(47, 63):
            frames.append(_make_frame(idx * 0.01, "I", rms_db=-17.0, is_voiced=True))

        segments = keyframe_gen.build_mode_a_segments(frames, fps=24.0, start_frame=1, min_phoneme_frames=3)
        self.assertEqual([s.vowel for s in segments], ["A", "I"])
        self.assertAlmostEqual(segments[0].start_frame, 1.0, places=2)
        self.assertAlmostEqual(segments[0].peak_rms_db, -12.0, places=2)
        self.assertGreater(segments[0].end_frame - segments[0].start_frame, 7.0)
        self.assertGreater(segments[1].end_frame - segments[1].start_frame, 3.0)

        with self.assertRaises(ValueError):
            keyframe_gen.build_mode_a_segments(frames, fps=0.0, start_frame=1, min_phoneme_frames=3)

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

    def test_schedule_zero_coarticulation_partial_remap_energy_and_muted_vowel(self):
        syllables = [
            AlignedSyllable("E", 5.0, 10.0, -24.0, False),
            AlignedSyllable("O", 10.0, 15.0, -12.0, False),
            AlignedSyllable("U", 20.0, 25.0, -12.0, False),
        ]
        mapped = {"A": "mouth_a", "I": "mouth_i", "U": "mouth_u"}
        strengths = {"A": 1.0, "I": 1.0, "U": 0.0}
        scheduled = keyframe_gen.schedule_keyframes(
            syllables, mapped, strengths, start_frame=1, max_mouth_open=1.0,
            min_phoneme_frames=3, transition_frames=2, co_articulation=0.0, preserve_syllables=False
        )
        # E remaps to I (quieter -> 0.65), O remaps to U (strength 0 -> skipped).
        self.assertNotIn("mouth_u", scheduled)
        self.assertIn("mouth_i", scheduled)
        i_vals = [k.value for k in scheduled["mouth_i"]]
        self.assertAlmostEqual(max(i_vals), 0.65, places=2)

        # Distinct vowels with co_articulation=0.0 do not cross-fade overlap (gap <= T and T < gap < 2*T).
        for gap_frames in (0.0, 3.0):
            distinct_syl = [
                AlignedSyllable("A", 5.0, 10.0, -16.0, False),
                AlignedSyllable("I", 10.0 + gap_frames, 15.0 + gap_frames, -16.0, False),
            ]
            no_co = keyframe_gen.schedule_keyframes(
                distinct_syl, {"A": "mouth_a", "I": "mouth_i"}, {"A": 1.0, "I": 1.0},
                start_frame=1, max_mouth_open=1.0, min_phoneme_frames=3,
                transition_frames=2, co_articulation=0.0, preserve_syllables=True
            )
            self.assertLessEqual(
                max(k.frame for k in no_co["mouth_a"]),
                min(k.frame for k in no_co["mouth_i"]),
                msg=f"Unexpected overlap for co_articulation=0.0 at gap={gap_frames}",
            )

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

    def test_mode_b_pause_same_vowel_closes_to_zero_while_remapped_distinct_bridges(self):
        # "a, a" across a brief pause (gap <= transition_frames, valley_dip_before=False) closes to 0.0.
        pause_same = [
            AlignedSyllable("A", 2.0, 6.0, -16.0, False),
            AlignedSyllable("A", 7.0, 11.0, -16.0, False),
        ]
        sched_pause = keyframe_gen.schedule_keyframes(
            pause_same,
            {"A": "mouth_a"},
            {"A": 1.0},
            start_frame=1,
            max_mouth_open=1.0,
            min_phoneme_frames=3,
            transition_frames=2,
            co_articulation=0.25,
            preserve_syllables=True,
        )
        events = sched_pause["mouth_a"]
        peak_idxs = [idx for idx, ev in enumerate(events) if ev.value > 0.5]
        self.assertEqual(len(peak_idxs), 2)
        between_vals = [events[i].value for i in range(peak_idxs[0] + 1, peak_idxs[1])]
        self.assertIn(0.0, between_vals)

        # Distinct remapped vowels ("e" + "a" onto mouth_a) still bridge cleanly without 0.0.
        remapped_distinct = [
            AlignedSyllable("E", 2.0, 6.0, -16.0, False),
            AlignedSyllable("A", 7.0, 11.0, -16.0, False),
        ]
        sched_bridge = keyframe_gen.schedule_keyframes(
            remapped_distinct,
            {"A": "mouth_a"},
            {"A": 1.0},
            start_frame=1,
            max_mouth_open=1.0,
            min_phoneme_frames=3,
            transition_frames=2,
            co_articulation=0.25,
            preserve_syllables=True,
        )
        bridge_events = sched_bridge["mouth_a"]
        bridge_peaks = [idx for idx, ev in enumerate(bridge_events) if ev.value > 0.5]
        self.assertEqual(len(bridge_peaks), 2)
        bridge_between = [bridge_events[i].value for i in range(bridge_peaks[0] + 1, bridge_peaks[1])]
        self.assertEqual(bridge_between, [])

    def test_slotted_action_writer_scoped_clear_and_vse_sync_mode(self):
        existing_kp = _FakeKP(5, 0.8)
        existing_kp.interpolation = "CONSTANT"
        existing_kp.handle_left_type = "FREE"
        existing_kp.handle_right_type = "VECTOR"
        fcurves_map = {
            'key_blocks["Blink"].value': _FakeFCurve('key_blocks["Blink"].value', [_FakeKP(5, 1.0)]),
            'key_blocks["mouth_a"].value': _FakeFCurve(
                'key_blocks["mouth_a"].value', [existing_kp, _FakeKP(50, 0.9)]
            ),
        }

        class FakeBag:
            @property
            def fcurves(self):
                return list(fcurves_map.values())

        class FakeStrip:
            def channelbag(self, _slot):
                return FakeBag()

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
        # Pre-existing keyframe at frame 5 keeps its original interpolation and handles.
        preserved_kp = fcurves_map['key_blocks["mouth_a"].value'].keyframe_points[0]
        self.assertEqual(preserved_kp.co[0], 5.0)
        self.assertEqual(preserved_kp.interpolation, "CONSTANT")
        self.assertEqual(preserved_kp.handle_left_type, "FREE")
        self.assertEqual(preserved_kp.handle_right_type, "VECTOR")

        created = []

        class FakeStrips(list):
            def new_sound(self, name, filepath, channel, frame_start):
                s = type(
                    "S",
                    (),
                    {
                        "name": name,
                        "sound": type("SND", (), {"filepath": filepath})(),
                        "channel": channel,
                        "frame_start": frame_start,
                        "type": "SOUND",
                    },
                )()
                created.append(s)
                self.append(s)
                return s

        class FakeScene:
            sync_mode = "NONE"
            sequence_editor = type("SE", (), {"strips": FakeStrips()})()

            def sequence_editor_create(self):
                pass

        scene = FakeScene()
        keyframe_gen.add_audio_to_vse(scene, "/tmp/test.wav", 10)
        self.assertEqual(scene.sync_mode, "AUDIO_SYNC")
        self.assertEqual(len(created), 1)
        # Re-adding identical clip at same frame avoids creating a duplicate strip.
        keyframe_gen.add_audio_to_vse(scene, "/tmp/test.wav", 10)
        self.assertEqual(len(created), 1)

    def test_legacy_fcurves_linear_constant_full_clear_and_vse_sequences_fallback(self):
        fc_a = _FakeFCurve('key_blocks["mouth_a"].value', [_FakeKP(2, 0.5), _FakeKP(10, 1.0)])
        fc_blink = _FakeFCurve('key_blocks["Blink"].value', [_FakeKP(4, 1.0)])
        fcurves_map = {
            'key_blocks["mouth_a"].value': fc_a,
            'key_blocks["Blink"].value': fc_blink,
        }

        class LegacyAction:
            fcurves = [fc_a, fc_blink]

        class LegacyAnim:
            action = LegacyAction()

        kb_a = _FakeKeyBlock("mouth_a", fcurves_map)
        kb_blink = _FakeKeyBlock("Blink", fcurves_map)

        class LegacySK:
            animation_data = LegacyAnim()
            key_blocks = {"mouth_a": kb_a, "Blink": kb_blink}

        class LegacyObj:
            data = type("D", (), {"shape_keys": LegacySK()})()

        # Full timeline clear (frame_range=None) clears mouth_a completely, preserves Blink.
        removed_all = keyframe_gen.clear_lipsync_keyframes(LegacyObj(), ["mouth_a"], frame_range=None)
        self.assertEqual(removed_all, 2)
        self.assertEqual(len(fc_a.keyframe_points), 0)
        self.assertEqual(len(fc_blink.keyframe_points), 1)
        self.assertTrue(fc_a.updated)

        # LINEAR and CONSTANT interpolation modes preserve Blink interpolation.
        keyframe_gen.apply_keyframes_to_object(
            LegacyObj(), {"mouth_a": [keyframe_gen.KeyframeEvent(8, 0.9)]}, "LINEAR"
        )
        self.assertEqual(fc_a.keyframe_points[-1].interpolation, "LINEAR")
        self.assertEqual(fc_blink.keyframe_points[0].interpolation, "BEZIER")

        keyframe_gen.apply_keyframes_to_object(
            LegacyObj(), {"mouth_a": [keyframe_gen.KeyframeEvent(12, 0.0)]}, "CONSTANT"
        )
        self.assertEqual(fc_a.keyframe_points[-1].interpolation, "CONSTANT")

        # VSE legacy sequences fallback
        created = []

        class FakeSequences(list):
            def new_sound(self, name, filepath, channel, frame_start):
                s = type(
                    "S",
                    (),
                    {
                        "name": name,
                        "sound": type("SND", (), {"filepath": filepath})(),
                        "channel": channel,
                        "frame_start": frame_start,
                        "type": "SOUND",
                    },
                )()
                created.append(s)
                self.append(s)
                return s

        class LegacyScene:
            sync_mode = "NONE"
            sequence_editor = None

            def sequence_editor_create(self):
                self.sequence_editor = type("SE", (), {"sequences": FakeSequences()})()

        scene = LegacyScene()
        keyframe_gen.add_audio_to_vse(scene, "/tmp/legacy.ogg", 5)
        self.assertEqual(scene.sync_mode, "AUDIO_SYNC")
        self.assertEqual(len(created), 1)

    def test_unbound_slot_channelbags_and_aia_collision_prevention(self):
        # A -> I (short) -> A with max co-articulation does not produce crossed zeros on mouth_a.
        syllables = [
            AlignedSyllable("A", 2.0, 6.0, -16.0, False),
            AlignedSyllable("I", 6.0, 8.0, -16.0, False),
            AlignedSyllable("A", 8.0, 12.0, -16.0, False),
        ]
        scheduled = keyframe_gen.schedule_keyframes(
            syllables,
            {"A": "mouth_a", "I": "mouth_i"},
            {"A": 1.0, "I": 1.0},
            start_frame=1,
            max_mouth_open=1.5,
            min_phoneme_frames=3,
            transition_frames=2,
            co_articulation=0.5,
            preserve_syllables=True,
        )
        a_events = scheduled["mouth_a"]
        a_frames = [e.frame for e in a_events]
        self.assertEqual(a_frames, sorted(a_frames))
        self.assertEqual(len(a_frames), len(set(a_frames)))
        # Ensure mouth_a reaches 0.0 between the two A peaks.
        peak_indices = [idx for idx, ev in enumerate(a_events) if ev.value > 1.0]
        self.assertEqual(len(peak_indices), 2)
        between_vals = [a_events[i].value for i in range(peak_indices[0] + 1, peak_indices[1])]
        self.assertIn(0.0, between_vals)

        # Unbound action_slot binds automatically and reads strip.channelbags.
        fc_a = _FakeFCurve('key_blocks["mouth_a"].value', [_FakeKP(3, 0.5)])
        slot_key = type("Slot", (), {"target_id_type": "KEY"})()
        bag = type("Bag", (), {"slot": slot_key, "fcurves": [fc_a]})()
        strip = type("Strip", (), {"channelbags": [bag]})()
        layer = type("Layer", (), {"strips": [strip]})()
        action = type("Action", (), {"slots": [slot_key], "layers": [layer]})()
        anim = type("Anim", (), {"action": action, "action_slot": None})()
        sk = type("SK", (), {"animation_data": anim, "key_blocks": {}})()

        fcurves = keyframe_gen.iter_shape_key_fcurves(sk)
        self.assertEqual(len(fcurves), 1)
        self.assertIs(anim.action_slot, slot_key)

        with self.assertRaises(ValueError):
            keyframe_gen.apply_keyframes_to_object(type("Obj", (), {})(), {}, "INVALID_MODE")

    def test_clear_lipsync_keyframes_multi_delete_revalidates_indices_on_realloc(self):
        class ReallocProxy:
            def __init__(self, backing_kp, gen, container):
                self._kp = backing_kp
                self._gen = gen
                self._container = container

            @property
            def co(self):
                if self._gen != self._container.gen:
                    raise RuntimeError("Stale PointerRNA reference after reallocation")
                return self._kp.co

        class ReallocKPs:
            def __init__(self, items):
                self._items = list(items)
                self.gen = 0

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return ReallocProxy(self._items[idx], self.gen, self)

            def __iter__(self):
                snapshot_gen = self.gen
                for idx in range(len(self._items)):
                    yield ReallocProxy(self._items[idx], snapshot_gen, self)

            def remove(self, proxy, fast=False):
                del fast
                if proxy._gen != self.gen:
                    raise RuntimeError("Keyframe not found in F-Curve")
                self._items.remove(proxy._kp)
                self.gen += 1

        fc_a = _FakeFCurve('key_blocks["mouth_a"].value')
        fc_a.keyframe_points = ReallocKPs(
            [_FakeKP(5, 0.4), _FakeKP(20, 0.8), _FakeKP(25, 1.0), _FakeKP(30, 0.2), _FakeKP(50, 0.9)]
        )
        action = type("Action", (), {"fcurves": [fc_a]})()
        anim = type("Anim", (), {"action": action})()
        sk = type("SK", (), {"animation_data": anim, "key_blocks": {}})()
        obj = type("Obj", (), {"data": type("D", (), {"shape_keys": sk})()})()

        removed = keyframe_gen.clear_lipsync_keyframes(obj, ["mouth_a"], frame_range=(20, 30))
        self.assertEqual(removed, 3)
        self.assertEqual([kp.co[0] for kp in fc_a.keyframe_points], [5.0, 50.0])
        self.assertTrue(fc_a.updated)

    def test_vse_5_2_content_start_and_right_handle_without_frame_start(self):
        created = []

        class StripV52:
            def __init__(self, filepath, channel, start, end):
                self.type = "SOUND"
                self.sound = type("SND", (), {"filepath": filepath})()
                self.channel = channel
                self.content_start = float(start)
                self.right_handle = float(end)

        class StripsV52(list):
            def new_sound(self, name, filepath, channel, frame_start):
                del name
                s = StripV52(filepath, channel, frame_start, frame_start + 30)
                created.append(s)
                self.append(s)
                return s

        existing = StripV52("/tmp/first.wav", channel=1, start=10, end=50)
        strips = StripsV52([existing])
        scene = type(
            "Scene52",
            (),
            {
                "sync_mode": "NONE",
                "sequence_editor": type("SE", (), {"strips": strips})(),
            },
        )()

        # Re-adding /tmp/first.wav at frame 10 hits content_start dedupe without creating a strip.
        keyframe_gen.add_audio_to_vse(scene, "/tmp/first.wav", 10)
        self.assertEqual(len(created), 0)
        self.assertEqual(scene.sync_mode, "AUDIO_SYNC")

        # Adding a different file overlapping frame 10 picks open channel 2 via content_start/right_handle.
        keyframe_gen.add_audio_to_vse(scene, "/tmp/second.wav", 15)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].channel, 2)

    def test_vse_left_trimmed_strips_use_visual_start_for_channel_selection(self):
        # Blender 5.2+ left_handle > content_start frees earlier frames on channel 1.
        created_52 = []

        class Trimmed52:
            def __init__(self, filepath, channel, content_start, left_handle, right_handle):
                self.type = "SOUND"
                self.sound = type("SND", (), {"filepath": filepath})()
                self.channel = channel
                self.content_start = float(content_start)
                self.left_handle = float(left_handle)
                self.right_handle = float(right_handle)

        class Strips52(list):
            def new_sound(self, name, filepath, channel, frame_start):
                del name
                s = Trimmed52(filepath, channel, frame_start, frame_start, frame_start + 20)
                created_52.append(s)
                self.append(s)
                return s

        trimmed_52 = Trimmed52("/tmp/orig.wav", channel=1, content_start=10, left_handle=30, right_handle=50)
        scene_52 = type(
            "Scene52Trim",
            (),
            {"sync_mode": "NONE", "sequence_editor": type("SE", (), {"strips": Strips52([trimmed_52])})()},
        )()
        keyframe_gen.add_audio_to_vse(scene_52, "/tmp/early.wav", 15)
        self.assertEqual(created_52[-1].channel, 1)
        keyframe_gen.add_audio_to_vse(scene_52, "/tmp/overlap.wav", 35)
        self.assertEqual(created_52[-1].channel, 2)

        # Blender 4.4-5.1 frame_final_start > frame_start also frees earlier frames on channel 1.
        created_45 = []

        class Trimmed45:
            def __init__(self, filepath, channel, frame_start, final_start, final_end):
                self.type = "SOUND"
                self.sound = type("SND", (), {"filepath": filepath})()
                self.channel = channel
                self.frame_start = float(frame_start)
                self.frame_final_start = float(final_start)
                self.frame_final_end = float(final_end)

        class Strips45(list):
            def new_sound(self, name, filepath, channel, frame_start):
                del name
                s = Trimmed45(filepath, channel, frame_start, frame_start, frame_start + 20)
                created_45.append(s)
                self.append(s)
                return s

        trimmed_45 = Trimmed45("/tmp/orig.wav", channel=1, frame_start=10, final_start=30, final_end=50)
        scene_45 = type(
            "Scene45Trim",
            (),
            {"sync_mode": "NONE", "sequence_editor": type("SE", (), {"sequences": Strips45([trimmed_45])})()},
        )()
        keyframe_gen.add_audio_to_vse(scene_45, "/tmp/early45.wav", 15)
        self.assertEqual(created_45[-1].channel, 1)
        keyframe_gen.add_audio_to_vse(scene_45, "/tmp/overlap45.wav", 35)
        self.assertEqual(created_45[-1].channel, 2)


if __name__ == "__main__":
    unittest.main()
