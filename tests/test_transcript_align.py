# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kuchisync import transcript_align


def _synth_formant_tone(
    f1: float,
    f2: float,
    dur: float,
    sr: int = 16000,
    amp: float = 1.0,
) -> np.ndarray:
    t = np.linspace(0.0, dur, int(round(sr * dur)), endpoint=False)
    sig = amp * (0.4 * np.sin(2.0 * np.pi * f1 * t) + 0.25 * np.sin(2.0 * np.pi * f2 * t))
    return sig.astype(np.float32)


class TestTranscriptAlign(unittest.TestCase):
    def test_indonesian_g2p_diphthongs_and_stress_weights(self):
        phrases = transcript_align.extract_vowels("saya makan pisau amboi!", "INDONESIAN")
        self.assertEqual(len(phrases), 1)
        vowels = [t.vowel for t in phrases[0]]
        self.assertEqual(vowels, ["A", "A", "A", "A", "I", "A", "U", "A", "O", "I"])
        self.assertTrue(phrases[0][1].same_as_prev_in_word)
        self.assertGreater(phrases[0][0].weight, 1.0)

    def test_english_g2p_silent_e_open_monosyllables_and_lexicon(self):
        phrases = transcript_align.extract_vowels("make time hope, be you.", "ENGLISH")
        self.assertEqual(len(phrases), 2)
        self.assertEqual([t.vowel for t in phrases[0]], ["E", "A", "I", "O"])
        self.assertEqual([t.vowel for t in phrases[1]], ["I", "U"])

    def test_empty_transcript_raises_transcript_error(self):
        with self.assertRaises(transcript_align.TranscriptError):
            transcript_align.extract_vowels("... !!!", "INDONESIAN")
        with self.assertRaises(transcript_align.TranscriptError):
            transcript_align.extract_vowels("   \n\t  ", "ENGLISH")

    def test_align_transcript_places_phrases_across_speech_islands(self):
        sr = 16000
        t1 = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
        p1 = 0.4 * np.sin(2 * np.pi * 800 * t1) + 0.25 * np.sin(2 * np.pi * 1250 * t1)
        gap = np.zeros(int(sr * 0.35))
        t2 = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
        p2 = 0.4 * np.sin(2 * np.pi * 330 * t2) + 0.25 * np.sin(2 * np.pi * 2250 * t2)
        audio = np.concatenate([p1, gap, p2]).astype(np.float32)

        aligned = transcript_align.align_transcript(
            audio, sr, "a, i", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned], ["A", "I"])
        self.assertLess(aligned[0].end_frame, 12.0)
        self.assertGreater(aligned[1].start_frame, 16.0)

    def test_consecutive_same_vowel_marks_valley_dip(self):
        sr = 16000

        def tone(dur, amp):
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            return amp * (0.4 * np.sin(2 * np.pi * 820 * t) + 0.25 * np.sin(2 * np.pi * 1280 * t))

        audio = np.concatenate([tone(0.18, 1.0), tone(0.06, 0.15), tone(0.18, 1.0)]).astype(np.float32)
        aligned = transcript_align.align_transcript(
            audio, sr, "saya", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned], ["A", "A"])
        self.assertTrue(aligned[1].valley_dip_before)

    def test_silent_or_empty_audio_returns_empty_list(self):
        sr = 16000
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        empty = np.zeros(0, dtype=np.float32)
        self.assertEqual(transcript_align.detect_speech_islands(silence, sr, -38.0), [])
        self.assertEqual(transcript_align.detect_speech_islands(empty, sr, -38.0), [])
        self.assertEqual(
            transcript_align.align_transcript(silence, sr, "halo", "INDONESIAN", -38.0, 24.0, 1),
            [],
        )
        self.assertEqual(
            transcript_align.align_transcript(empty, sr, "halo", "INDONESIAN", -38.0, 24.0, 1),
            [],
        )
        with self.assertRaises(transcript_align.TranscriptError):
            transcript_align.align_transcript(silence, sr, "...", "INDONESIAN", -38.0, 24.0, 1)

    def test_island_phrase_count_mismatch(self):
        sr = 16000
        # Case 1: 1 phrase spoken across 2 islands (more islands than phrases)
        a1 = _synth_formant_tone(820.0, 1280.0, 0.30, sr=sr)
        gap = np.zeros(int(sr * 0.30), dtype=np.float32)
        i2 = _synth_formant_tone(330.0, 2280.0, 0.30, sr=sr)
        audio_two = np.concatenate([a1, gap, i2]).astype(np.float32)
        aligned_two = transcript_align.align_transcript(
            audio_two, sr, "saya ini", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned_two], ["A", "A", "I", "I"])
        self.assertLess(aligned_two[1].end_frame, 11.0)
        self.assertGreater(aligned_two[2].start_frame, 13.0)
        for syl in aligned_two:
            self.assertGreater(syl.end_frame, syl.start_frame)

        # Case 2: 2 comma-separated phrases spoken in 1 continuous island (fewer islands than phrases)
        continuous = np.concatenate([
            _synth_formant_tone(820.0, 1280.0, 0.25, sr=sr),
            _synth_formant_tone(330.0, 2280.0, 0.25, sr=sr),
        ]).astype(np.float32)
        aligned_one = transcript_align.align_transcript(
            continuous, sr, "a, i", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=10
        )
        self.assertEqual([s.vowel for s in aligned_one], ["A", "I"])
        self.assertGreaterEqual(aligned_one[0].start_frame, 10.0)
        self.assertGreater(aligned_one[1].start_frame, aligned_one[0].start_frame)
        self.assertGreater(aligned_one[1].end_frame, aligned_one[1].start_frame)

    def test_english_g2p_suffixes_ed_es_le_and_unseen_digraphs(self):
        # Silent -ed vs -ted/-ded, syllabic -le, -es sibilants vs silent, qu-, unseen digraphs
        samples = {
            "baked": ["E"],
            "walked": ["O"],
            "waited": ["E", "I"],
            "faded": ["E", "I"],
            "table": ["E", "A"],
            "little": ["I", "A"],
            "maple": ["E", "A"],
            "bottle": ["O", "A"],
            "coin": ["O", "I"],
            "pulled": ["U"],
            "called": ["O"],
            "walled": ["O"],
            "planned": ["A"],
            "stopped": ["A"],
            "boxes": ["O", "I"],
            "makes": ["E"],
            "queen": ["I"],
            "screen": ["I"],
            "spoon": ["U"],
            "cloud": ["A", "U"],
            "train": ["E"],
            "voice": ["O", "I"],
            "boat": ["O"],
            "thief": ["I"],
        }
        for word, expected in samples.items():
            phrases = transcript_align.extract_vowels(word, "ENGLISH")
            self.assertEqual(len(phrases), 1, word)
            self.assertEqual([t.vowel for t in phrases[0]], expected, f"Failed for {word}")

        # Verify lexicon coverage for common irregular words
        phrases = transcript_align.extract_vowels("you are through enough people", "ENGLISH")
        self.assertEqual(len(phrases), 1)
        self.assertEqual([t.vowel for t in phrases[0]], ["U", "A", "U", "I", "A", "I", "A"])

    def test_cross_word_same_vowel_and_asymmetric_valley_alignment(self):
        sr = 16000
        # Asymmetric dips at 0.12s (frame 3.88), 0.28s (frame 7.72), 0.44s (frame 11.56)
        audio = np.concatenate([
            _synth_formant_tone(820.0, 1280.0, 0.09, sr=sr, amp=1.0),
            _synth_formant_tone(820.0, 1280.0, 0.06, sr=sr, amp=0.15),
            _synth_formant_tone(820.0, 1280.0, 0.10, sr=sr, amp=1.0),
            _synth_formant_tone(820.0, 1280.0, 0.06, sr=sr, amp=0.15),
            _synth_formant_tone(820.0, 1280.0, 0.10, sr=sr, amp=1.0),
            _synth_formant_tone(820.0, 1280.0, 0.06, sr=sr, amp=0.15),
            _synth_formant_tone(820.0, 1280.0, 0.12, sr=sr, amp=1.0),
        ]).astype(np.float32)
        aligned = transcript_align.align_transcript(
            audio, sr, "saya makan", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned], ["A", "A", "A", "A"])
        self.assertFalse(aligned[0].valley_dip_before)
        self.assertTrue(aligned[1].valley_dip_before)
        self.assertTrue(aligned[2].valley_dip_before)
        self.assertTrue(aligned[3].valley_dip_before)
        self.assertAlmostEqual(aligned[0].end_frame, 3.88, delta=0.8)
        self.assertAlmostEqual(aligned[1].end_frame, 7.72, delta=0.8)
        self.assertAlmostEqual(aligned[2].end_frame, 11.56, delta=0.8)

    def test_detect_speech_islands_bridges_micro_pauses_and_drops_short_glitches(self):
        sr = 16000
        # 1) 80ms micro-pause bridged (<= 120ms); 20ms click dropped next to 200ms vowel
        click = _synth_formant_tone(820.0, 1280.0, 0.02, sr=sr)
        long_gap = np.zeros(int(sr * 0.25), dtype=np.float32)
        seg1 = _synth_formant_tone(820.0, 1280.0, 0.20, sr=sr)
        micro_gap = np.zeros(int(sr * 0.08), dtype=np.float32)
        seg2 = _synth_formant_tone(330.0, 2280.0, 0.20, sr=sr)
        audio = np.concatenate([click, long_gap, seg1, micro_gap, seg2]).astype(np.float32)

        islands = transcript_align.detect_speech_islands(audio, sr, silence_threshold_db=-38.0)
        self.assertEqual(len(islands), 1)
        self.assertGreaterEqual(islands[0][0], 20)

        # 2) Tiny pulse (< 30ms) preserved when it is the only sound in clip
        only_pulse = np.concatenate([
            np.zeros(int(sr * 0.05), dtype=np.float32),
            _synth_formant_tone(820.0, 1280.0, 0.02, sr=sr),
            np.zeros(int(sr * 0.05), dtype=np.float32),
        ]).astype(np.float32)
        pulse_islands = transcript_align.detect_speech_islands(only_pulse, sr, silence_threshold_db=-38.0)
        self.assertEqual(len(pulse_islands), 1)

    def test_indonesian_g2p_digraphs_and_phrase_final_weights(self):
        phrases = transcript_align.extract_vowels("nyanyi khusus syarat, makanan", "INDONESIAN")
        self.assertEqual(len(phrases), 2)
        self.assertEqual([t.vowel for t in phrases[0]], ["A", "I", "U", "U", "A", "A"])
        # In "khusus", second U has same_as_prev_in_word = True
        self.assertFalse(phrases[0][2].same_as_prev_in_word)
        self.assertTrue(phrases[0][3].same_as_prev_in_word)
        # In "syarat", both A's -> second A has same_as_prev_in_word = True
        self.assertTrue(phrases[0][5].same_as_prev_in_word)
        # In "makanan" (phrase 1): ma (0.90), ka (penult 1.35), nan (phrase-final 1.25)
        self.assertEqual([t.vowel for t in phrases[1]], ["A", "A", "A"])
        self.assertLess(phrases[1][0].weight, 1.0)
        self.assertGreater(phrases[1][1].weight, phrases[1][2].weight)
        self.assertAlmostEqual(phrases[1][2].weight, 1.25, places=2)
        self.assertEqual(phrases[1][0].phrase_index, 1)

    def test_align_transcript_resamples_44100_and_speaker_normalizes_shifted_voice(self):
        sr = 44100
        # Child-shifted A (1.16x) followed by child-shifted I (1.18x)
        audio = np.concatenate([
            _synth_formant_tone(825.0 * 1.16, 1275.0 * 1.16, 0.25, sr=sr),
            _synth_formant_tone(325.0 * 1.16, 2300.0 * 1.18, 0.35, sr=sr),
        ]).astype(np.float32)
        aligned = transcript_align.align_transcript(
            audio, sr, "a i", "INDONESIAN", silence_threshold_db=-38.0, fps=24.0, start_frame=1
        )
        self.assertEqual([s.vowel for s in aligned], ["A", "I"])
        # Boundary should sit reasonably near 0.25s (~frame 7.0) rather than drifting wildly
        self.assertGreater(aligned[0].end_frame, 4.5)
        self.assertLess(aligned[0].end_frame, 9.5)
        self.assertAlmostEqual(aligned[0].end_frame, aligned[1].start_frame, places=4)

        # Verify align_transcript passes normalized language to compute_speaker_normalized_scores.
        eng_audio = np.concatenate([
            _synth_formant_tone(780.0, 1320.0, 0.20, sr=16000),
            _synth_formant_tone(325.0, 2300.0, 0.20, sr=16000),
        ]).astype(np.float32)
        with mock.patch(
            "kuchisync.transcript_align.audio_dsp.compute_speaker_normalized_scores",
            wraps=transcript_align.audio_dsp.compute_speaker_normalized_scores,
        ) as norm_spy:
            eng_aligned = transcript_align.align_transcript(
                eng_audio, 16000, "pa see", "en", silence_threshold_db=-38.0, fps=24.0, start_frame=1
            )
        self.assertEqual([s.vowel for s in eng_aligned], ["A", "I"])
        self.assertEqual(norm_spy.call_args.kwargs.get("language"), "ENGLISH")


if __name__ == "__main__":
    unittest.main()
