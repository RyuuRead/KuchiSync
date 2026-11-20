# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kuchisync import audio_dsp


def _synth_vowel_wav(
    path: pathlib.Path,
    formants_hz: tuple[float, float],
    sr: int = 44100,
    dur: float = 0.25,
    sampwidth: int = 2,
    channels: int = 2,
    gain: float = 1.0,
) -> None:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    f1, f2 = formants_hz
    sig = gain * (0.45 * np.sin(2 * np.pi * f1 * t) + 0.30 * np.sin(2 * np.pi * f2 * t))
    sig = np.clip(sig, -0.95, 0.95)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        if sampwidth == 1:
            pcm = np.clip(np.round(sig * 127.0 + 128.0), 0, 255).astype(np.uint8)
            if channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1).ravel()
            wf.writeframes(pcm.tobytes())
        elif sampwidth == 2:
            pcm = (sig * 32767.0).astype("<i2")
            if channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1).ravel()
            wf.writeframes(pcm.tobytes())
        elif sampwidth == 3:
            ints = (sig * 8388607.0).astype(np.int32)
            raw = bytearray()
            for v in ints:
                b = struct.pack("<i", int(v))[:3]
                raw.extend(b * channels)
            wf.writeframes(bytes(raw))
        elif sampwidth == 4:
            pcm = (sig * 2147483647.0).astype("<i4")
            if channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1).ravel()
            wf.writeframes(pcm.tobytes())


class TestAudioDsp(unittest.TestCase):
    def test_load_24bit_stereo_wav_and_classify_vowels(self):
        expected = {
            "A": (820.0, 1280.0),
            "I": (330.0, 2300.0),
            "U": (360.0, 830.0),
            "E": (540.0, 1950.0),
            "O": (540.0, 920.0),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for vowel, pair in expected.items():
                wav_path = pathlib.Path(tmp) / f"{vowel}.wav"
                _synth_vowel_wav(wav_path, pair, sr=44100, sampwidth=3, channels=2)
                clip = audio_dsp.load_audio(str(wav_path), target_sr=16000)
                self.assertEqual(clip.sample_rate, 16000)
                frames = audio_dsp.analyze_audio_frames(
                    clip.samples, 16000, -38.0, "FEMALE_STANDARD", "INDONESIAN"
                )
                voiced_vowels = [f.vowel for f in frames if f.is_voiced and f.vowel]
                self.assertTrue(voiced_vowels, f"No voiced frames for {vowel}")
                majority = max(set(voiced_vowels), key=voiced_vowels.count)
                self.assertEqual(majority, vowel)

    def test_pcm_bit_depths_preserve_true_rms_db(self):
        sr = 16000
        dur = 0.2
        t = np.arange(int(sr * dur), dtype=np.float64) / sr
        amp = 0.1  # RMS = 0.1 / sqrt(2) -> ~ -23.01 dB
        expected_db = 20.0 * np.log10(amp / np.sqrt(2.0))
        sig = amp * np.sin(2 * np.pi * 1000.0 * t)
        with tempfile.TemporaryDirectory() as tmp:
            for width in (1, 2, 3, 4):
                p = pathlib.Path(tmp) / f"tone_{width}b.wav"
                with wave.open(str(p), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(width)
                    wf.setframerate(sr)
                    if width == 1:
                        wf.writeframes(np.round(sig * 127.0 + 128.0).astype(np.uint8).tobytes())
                    elif width == 2:
                        wf.writeframes((sig * 32767.0).astype("<i2").tobytes())
                    elif width == 3:
                        raw = bytearray()
                        for v in (sig * 8388607.0).astype(np.int32):
                            raw.extend(struct.pack("<i", int(v))[:3])
                        wf.writeframes(bytes(raw))
                    elif width == 4:
                        wf.writeframes((sig * 2147483647.0).astype("<i4").tobytes())
                clip = audio_dsp.load_audio(str(p), target_sr=16000)
                rms = float(np.sqrt(np.mean(clip.samples.astype(np.float64) ** 2)))
                actual_db = 20.0 * np.log10(max(rms, 1e-10))
                self.assertAlmostEqual(actual_db, expected_db, delta=0.6)

    def test_silence_and_sibilant_suppression(self):
        sr = 16000
        silence = np.zeros(int(sr * 0.2), dtype=np.float32)
        rng = np.random.default_rng(7)
        sibilant = (
            0.015 * np.where(np.arange(int(sr * 0.2)) % 2 == 0, 1.0, -1.0)
            + 0.005 * rng.standard_normal(int(sr * 0.2))
        ).astype(np.float32)
        frames_s = audio_dsp.analyze_audio_frames(silence, sr, -38.0)
        frames_sib = audio_dsp.analyze_audio_frames(sibilant, sr, -38.0)
        self.assertTrue(all(not f.is_voiced for f in frames_s))
        self.assertTrue(all(not f.is_voiced for f in frames_sib))

    def test_voice_type_shift_high_child_and_male_deep(self):
        sr = 16000
        dur = 0.25
        t = np.arange(int(sr * dur), dtype=np.float64) / sr
        cases = [
            ("HIGH_CHILD", (1.16, 1.18), {"A": (825.0, 1275.0), "I": (325.0, 2300.0), "U": (365.0, 825.0), "E": (550.0, 1950.0), "O": (550.0, 925.0)}),
            ("MALE_DEEP", (0.85, 0.87), {"A": (825.0, 1275.0), "I": (325.0, 2300.0), "U": (365.0, 825.0), "E": (550.0, 1950.0), "O": (550.0, 925.0)}),
        ]
        for voice_type, (sf1, sf2), centroids in cases:
            for vowel, (f1, f2) in centroids.items():
                sig = (
                    0.45 * np.sin(2 * np.pi * (f1 * sf1) * t)
                    + 0.30 * np.sin(2 * np.pi * (f2 * sf2) * t)
                ).astype(np.float32)
                frames = audio_dsp.analyze_audio_frames(
                    sig, sr, silence_threshold_db=-38.0, voice_type=voice_type, language="INDONESIAN"
                )
                voiced = [f.vowel for f in frames if f.is_voiced and f.vowel]
                self.assertTrue(voiced, f"No voiced frames for {voice_type}/{vowel}")
                majority = max(set(voiced), key=voiced.count)
                self.assertEqual(majority, vowel, f"Mismatch for {voice_type}/{vowel}: got {majority}")

    def test_speaker_normalized_scores_across_shifted_utterances(self):
        sr = 16000
        dur = 0.12
        t = np.arange(int(sr * dur), dtype=np.float64) / sr
        centroids = {"A": (825.0, 1275.0), "I": (325.0, 2300.0), "U": (365.0, 825.0), "E": (550.0, 1950.0), "O": (550.0, 925.0)}
        for sf1, sf2 in ((1.16, 1.18), (0.85, 0.87)):
            chunks = []
            labels = []
            for vowel, (f1, f2) in centroids.items():
                chunk = (
                    0.45 * np.sin(2 * np.pi * (f1 * sf1) * t)
                    + 0.30 * np.sin(2 * np.pi * (f2 * sf2) * t)
                ).astype(np.float32)
                chunks.append(chunk)
                labels.append(vowel)
            full_audio = np.concatenate(chunks)
            frames = audio_dsp.analyze_audio_frames(
                full_audio, sr, silence_threshold_db=-38.0, voice_type="FEMALE_STANDARD"
            )
            norm_scores = audio_dsp.compute_speaker_normalized_scores(frames)
            self.assertEqual(len(norm_scores), len(frames))
            chunk_frames = len(frames) // len(labels)
            for idx, expected_vowel in enumerate(labels):
                sl = norm_scores[idx * chunk_frames + 2 : (idx + 1) * chunk_frames - 2]
                preds = [max(sc, key=sc.get) for sc in sl]
                majority = max(set(preds), key=preds.count)
                self.assertEqual(majority, expected_vowel)

            # Single-vowel and 2-vowel shifted utterances resist collapsing onto neighboring vowels.
            for subset in (("A",), ("I",), ("A", "I"), ("U", "E")):
                sub_chunks = []
                for v in subset:
                    f1, f2 = centroids[v]
                    sub_chunks.append(
                        (
                            0.45 * np.sin(2 * np.pi * (f1 * sf1) * t)
                            + 0.30 * np.sin(2 * np.pi * (f2 * sf2) * t)
                        ).astype(np.float32)
                    )
                sub_audio = np.concatenate(sub_chunks)
                sub_frames = audio_dsp.analyze_audio_frames(
                    sub_audio, sr, silence_threshold_db=-38.0, voice_type="FEMALE_STANDARD"
                )
                sub_norm = audio_dsp.compute_speaker_normalized_scores(sub_frames)
                sub_chunk_frames = len(sub_frames) // len(subset)
                for idx, expected_vowel in enumerate(subset):
                    sl = sub_norm[idx * sub_chunk_frames + 2 : (idx + 1) * sub_chunk_frames - 2]
                    preds = [max(sc, key=sc.get) for sc in sl]
                    majority = max(set(preds), key=preds.count)
                    self.assertEqual(
                        majority,
                        expected_vowel,
                        f"Mismatch for subset={subset} scale={(sf1, sf2)} at {expected_vowel}",
                    )

    def test_bandpass_vocal_attenuates_out_of_band_frequencies(self):
        sr = 16000
        t = np.arange(int(sr * 0.5), dtype=np.float64) / sr
        low_tone = np.sin(2 * np.pi * 80.0 * t)
        mid_tone = np.sin(2 * np.pi * 1000.0 * t)
        high_tone = np.sin(2 * np.pi * 5500.0 * t)
        filtered_low = audio_dsp.bandpass_vocal(low_tone.astype(np.float32), sr)
        filtered_mid = audio_dsp.bandpass_vocal(mid_tone.astype(np.float32), sr)
        filtered_high = audio_dsp.bandpass_vocal(high_tone.astype(np.float32), sr)

        def rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

        self.assertLess(rms(filtered_low), 0.1 * rms(low_tone))
        self.assertGreater(rms(filtered_mid), 0.85 * rms(mid_tone))
        self.assertLess(rms(filtered_high), 0.1 * rms(high_tone))

    def test_aud_contract_specs_and_2d_data(self):
        fake_aud = types.ModuleType("aud")

        class FakeSound:
            specs = (48000.0, 2)

            def rechannel(self, _c):
                return self

            def resample(self, _r, high_quality=True):
                return self

            def cache(self):
                return self

            def data(self):
                return np.ones((1600, 1), dtype=np.float32) * 0.1

        fake_aud.Sound = types.SimpleNamespace(file=lambda _p: FakeSound())
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "fake.ogg"
            p.write_bytes(b"OggSdummy")
            old = sys.modules.get("aud")
            sys.modules["aud"] = fake_aud
            try:
                clip = audio_dsp.load_audio(str(p), target_sr=16000)
                self.assertEqual(clip.sample_rate, 16000)
                self.assertEqual(clip.samples.ndim, 1)
                self.assertEqual(len(clip.samples), 1600)
            finally:
                if old is None:
                    sys.modules.pop("aud", None)
                else:
                    sys.modules["aud"] = old

    def test_aud_fallback_without_chaining_resamples_from_specs(self):
        fake_aud = types.ModuleType("aud")

        class RawOnlySound:
            specs = (32000.0, 2)

            def data(self):
                return np.full((3200, 2), 0.2, dtype=np.float32)

        fake_aud.Sound = types.SimpleNamespace(file=lambda _p: RawOnlySound())
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "raw.flac"
            p.write_bytes(b"fLaCdummy")
            old = sys.modules.get("aud")
            sys.modules["aud"] = fake_aud
            try:
                clip = audio_dsp.load_audio(str(p), target_sr=16000)
                self.assertEqual(clip.sample_rate, 16000)
                self.assertEqual(clip.samples.ndim, 1)
                self.assertEqual(len(clip.samples), 1600)
                self.assertAlmostEqual(float(np.median(clip.samples)), 0.2, places=2)
            finally:
                if old is None:
                    sys.modules.pop("aud", None)
                else:
                    sys.modules["aud"] = old

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_ffmpeg_ogg_and_mp3_decode(self):
        ffmpeg_bin = shutil.which("ffmpeg")
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = pathlib.Path(tmp) / "src.wav"
            _synth_vowel_wav(wav_path, (820.0, 1280.0), sr=44100, dur=0.25, sampwidth=2, channels=1)

            old_aud = sys.modules.get("aud")
            sys.modules["aud"] = None
            try:
                ogg_path = pathlib.Path(tmp) / "test.ogg"
                res_ogg = subprocess.run(
                    [ffmpeg_bin, "-y", "-v", "error", "-i", str(wav_path), "-c:a", "libvorbis", str(ogg_path)],
                    capture_output=True,
                    check=False,
                )
                if res_ogg.returncode == 0:
                    clip_ogg = audio_dsp.load_audio(str(ogg_path), target_sr=16000)
                    self.assertEqual(clip_ogg.sample_rate, 16000)
                    self.assertGreater(len(clip_ogg.samples), 3000)
                    frames = audio_dsp.analyze_audio_frames(clip_ogg.samples, 16000, -38.0)
                    voiced = [f.vowel for f in frames if f.is_voiced and f.vowel]
                    self.assertTrue(voiced)
                    self.assertEqual(max(set(voiced), key=voiced.count), "A")

                mp3_path = pathlib.Path(tmp) / "test.mp3"
                res_mp3 = subprocess.run(
                    [ffmpeg_bin, "-y", "-v", "error", "-i", str(wav_path), "-c:a", "libmp3lame", str(mp3_path)],
                    capture_output=True,
                    check=False,
                )
                if res_mp3.returncode == 0:
                    clip_mp3 = audio_dsp.load_audio(str(mp3_path), target_sr=16000)
                    self.assertEqual(clip_mp3.sample_rate, 16000)
                    self.assertGreater(len(clip_mp3.samples), 3000)
                    frames = audio_dsp.analyze_audio_frames(clip_mp3.samples, 16000, -38.0)
                    voiced = [f.vowel for f in frames if f.is_voiced and f.vowel]
                    self.assertTrue(voiced)
                    self.assertEqual(max(set(voiced), key=voiced.count), "A")

                if res_ogg.returncode != 0 and res_mp3.returncode != 0:
                    self.skipTest("ffmpeg has neither libvorbis nor libmp3lame encoder")
            finally:
                if old_aud is None:
                    sys.modules.pop("aud", None)
                else:
                    sys.modules["aud"] = old_aud

    def test_missing_and_corrupt_files_raise_audio_load_error(self):
        with self.assertRaises(audio_dsp.AudioLoadError):
            audio_dsp.load_audio("/nonexistent/path/audio.wav")

        with tempfile.TemporaryDirectory() as tmp:
            empty_wav = pathlib.Path(tmp) / "empty.wav"
            empty_wav.write_bytes(b"")
            with self.assertRaises(audio_dsp.AudioLoadError):
                audio_dsp.load_audio(str(empty_wav))

            corrupt_wav = pathlib.Path(tmp) / "corrupt.wav"
            corrupt_wav.write_bytes(b"NOT_A_VALID_RIFF_FILE")
            with self.assertRaises(audio_dsp.AudioLoadError):
                audio_dsp.load_audio(str(corrupt_wav))

    def test_has_scipy_and_refresh_scipy_return_bool(self):
        self.assertIsInstance(audio_dsp.has_scipy(), bool)
        self.assertIsInstance(audio_dsp.refresh_scipy(), bool)

    def test_english_centroids_and_short_or_empty_inputs(self):
        self.assertEqual(audio_dsp.analyze_audio_frames(np.empty(0, dtype=np.float32), 16000), [])
        self.assertEqual(audio_dsp.compute_speaker_normalized_scores([]), [])

        sr = 16000
        short = np.zeros(120, dtype=np.float32)
        short_frames = audio_dsp.analyze_audio_frames(short, sr, -38.0)
        self.assertEqual(len(short_frames), 1)
        self.assertFalse(short_frames[0].is_voiced)

        t = np.arange(int(sr * 0.2), dtype=np.float64) / sr
        eng_a = (0.45 * np.sin(2 * np.pi * 780.0 * t) + 0.30 * np.sin(2 * np.pi * 1320.0 * t)).astype(
            np.float32
        )
        frames = audio_dsp.analyze_audio_frames(eng_a, sr, -38.0, "FEMALE_STANDARD", "ENGLISH")
        voiced = [f.vowel for f in frames if f.is_voiced and f.vowel]
        self.assertTrue(voiced)
        self.assertEqual(max(set(voiced), key=voiced.count), "A")

        # Verify compute_speaker_normalized_scores selects language-specific centroids.
        eng_frame = audio_dsp.FormantFrame(
            time_sec=0.0,
            f1=780.0,
            f2=1320.0,
            rms_db=-16.0,
            zcr=0.05,
            is_voiced=True,
            vowel="A",
            scores={},
        )
        en_scores = audio_dsp.compute_speaker_normalized_scores([eng_frame] * 4, language="ENGLISH")
        id_scores = audio_dsp.compute_speaker_normalized_scores([eng_frame] * 4, language="INDONESIAN")
        self.assertGreater(en_scores[0]["A"], id_scores[0]["A"])

        with mock.patch(
            "kuchisync.audio_dsp._estimate_vocal_tract_scale",
            wraps=audio_dsp._estimate_vocal_tract_scale,
        ) as scale_spy:
            audio_dsp.compute_speaker_normalized_scores([eng_frame] * 6, language="ENGLISH")
        self.assertEqual(scale_spy.call_args[0][2], audio_dsp.VOWEL_CENTROIDS["ENGLISH"])

    def test_decode_with_ffmpeg_passes_devnull_stdin(self):
        fake_samples = np.ones(8, dtype="<f4").tobytes()
        fake_proc = types.SimpleNamespace(returncode=0, stdout=fake_samples, stderr=b"")
        with (
            mock.patch("kuchisync.audio_dsp.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch("kuchisync.audio_dsp.subprocess.run", return_value=fake_proc) as run_mock,
        ):
            decoded = audio_dsp._decode_with_ffmpeg(pathlib.Path("/tmp/clip.ogg"), 16000)
        self.assertEqual(len(decoded), 8)
        self.assertEqual(run_mock.call_args.kwargs.get("stdin"), subprocess.DEVNULL)

    def test_dynamic_scipy_detection_and_dsp_dispatch(self):
        calls = []
        fake_scipy = types.ModuleType("scipy")
        fake_signal = types.ModuleType("scipy.signal")
        fake_io = types.ModuleType("scipy.io")
        fake_wavfile = types.ModuleType("scipy.io.wavfile")

        def fake_resample_poly(x, up, down):
            calls.append(("resample_poly", up, down))
            return np.zeros(max(1, int(round(len(x) * up / down))), dtype=np.float64)

        def fake_butter(order, freqs, btype="bandpass", fs=16000.0, output="sos"):
            calls.append(("butter", order, freqs))
            return np.zeros((2, 6), dtype=np.float64)

        def fake_sosfiltfilt(_sos, x):
            calls.append(("sosfiltfilt", len(x)))
            return x

        fake_signal.resample_poly = fake_resample_poly
        fake_signal.butter = fake_butter
        fake_signal.sosfiltfilt = fake_sosfiltfilt
        fake_wavfile.read = lambda _p: (16000, np.ones(160, dtype=np.float32) * 0.25)
        fake_scipy.signal = fake_signal
        fake_io.wavfile = fake_wavfile
        fake_scipy.io = fake_io

        keys = ("scipy", "scipy.signal", "scipy.io", "scipy.io.wavfile")
        saved = {k: sys.modules.get(k) for k in keys}
        present = {k: (k in sys.modules) for k in keys}
        try:
            sys.modules["scipy"] = fake_scipy
            sys.modules["scipy.signal"] = fake_signal
            sys.modules["scipy.io"] = fake_io
            sys.modules["scipy.io.wavfile"] = fake_wavfile

            self.assertTrue(audio_dsp.refresh_scipy())
            self.assertTrue(audio_dsp.has_scipy())

            res = audio_dsp.resample_mono(np.ones(480, dtype=np.float32), 48000, 16000)
            self.assertEqual(len(res), 160)
            bp = audio_dsp.bandpass_vocal(np.ones(128, dtype=np.float32), 16000)
            self.assertEqual(len(bp), 128)
            self.assertEqual([c[0] for c in calls], ["resample_poly", "butter", "sosfiltfilt"])

            with tempfile.TemporaryDirectory() as tmp:
                float_wav = pathlib.Path(tmp) / "float.wav"
                float_wav.write_bytes(b"RIFF0000WAVEfakefloat")
                old_aud = sys.modules.get("aud")
                sys.modules["aud"] = None
                try:
                    clip = audio_dsp.load_audio(str(float_wav), target_sr=16000)
                    self.assertEqual(len(clip.samples), 160)
                    self.assertAlmostEqual(float(clip.samples[0]), 0.25, places=5)
                finally:
                    if old_aud is None:
                        sys.modules.pop("aud", None)
                    else:
                        sys.modules["aud"] = old_aud

            sys.modules["scipy"] = None
            self.assertFalse(audio_dsp.has_scipy())
            self.assertFalse(audio_dsp.refresh_scipy())
        finally:
            for k in keys:
                if present[k]:
                    sys.modules[k] = saved[k]
                else:
                    sys.modules.pop(k, None)
            audio_dsp.refresh_scipy()

    def test_broad_damped_pole_is_filtered_from_formants(self):
        sr = 16000
        n = 400
        y = np.zeros(n, dtype=np.float64)
        y[150] = 1.0
        for freq_hz, bw_hz in ((330.0, 90.0), (1200.0, 950.0), (2300.0, 110.0)):
            radius = float(np.exp(-np.pi * bw_hz / sr))
            theta = 2.0 * float(np.pi) * freq_hz / sr
            a1 = -2.0 * radius * float(np.cos(theta))
            a2 = radius * radius
            out = np.zeros_like(y)
            for i in range(n):
                val = y[i]
                if i >= 1:
                    val -= a1 * out[i - 1]
                if i >= 2:
                    val -= a2 * out[i - 2]
                out[i] = val
            y = out

        # Invert pre-emphasis (y[i] = x[i] - 0.97 * x[i - 1]) so LPC sees the exact 3-pole envelope.
        x = np.zeros(n, dtype=np.float64)
        x[0] = y[0]
        for i in range(1, n):
            x[i] = y[i] + 0.97 * x[i - 1]
        x = (0.3 * x / np.max(np.abs(x))).astype(np.float32)

        frames = audio_dsp.analyze_audio_frames(x, sr=sr, silence_threshold_db=-38.0)
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertTrue(frame.is_voiced)
        self.assertIsNotNone(frame.f1)
        self.assertIsNotNone(frame.f2)
        self.assertAlmostEqual(frame.f1, 330.0, delta=30.0)
        self.assertAlmostEqual(frame.f2, 2300.0, delta=60.0)
        self.assertEqual(frame.vowel, "I")


if __name__ == "__main__":
    unittest.main()
