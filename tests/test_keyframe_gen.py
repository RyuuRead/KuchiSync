# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

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

        # Distinct vowels with co_articulation=0.0 do not cross-fade overlap.
        distinct_syl = [
            AlignedSyllable("A", 5.0, 10.0, -16.0, False),
            AlignedSyllable("I", 10.0, 15.0, -16.0, False),
        ]
        no_co = keyframe_gen.schedule_keyframes(
            distinct_syl, {"A": "mouth_a", "I": "mouth_i"}, {"A": 1.0, "I": 1.0},
            start_frame=1, max_mouth_open=1.0, min_phoneme_frames=3,
            transition_frames=2, co_articulation=0.0, preserve_syllables=True
        )
        self.assertLessEqual(max(k.frame for k in no_co["mouth_a"]), min(k.frame for k in no_co["mouth_i"]))

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
        fcurves_map = {
            'key_blocks["Blink"].value': _FakeFCurve('key_blocks["Blink"].value', [_FakeKP(5, 1.0)]),
            'key_blocks["mouth_a"].value': _FakeFCurve(
                'key_blocks["mouth_a"].value', [_FakeKP(5, 0.8), _FakeKP(50, 0.9)]
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


if __name__ == "__main__":
    unittest.main()
