# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
import importlib
import math
import pathlib
import shutil
import subprocess
import sys
import wave

import numpy as np

VOWELS: tuple[str, ...] = ("A", "I", "U", "E", "O")
TARGET_SAMPLE_RATE: int = 16000
FRAME_MS: float = 25.0
HOP_MS: float = 10.0
PRE_EMPHASIS: float = 0.97
LPC_ORDER: int = 16
MIN_FORMANT_HZ: float = 200.0
MAX_FORMANT_HZ: float = 3500.0
MAX_BANDWIDTH_HZ: float = 600.0
SIGMA_F1: float = 130.0
SIGMA_F2: float = 260.0

VOICE_TYPE_SCALES: dict[str, tuple[float, float]] = {
    "HIGH_CHILD": (1.16, 1.18),
    "FEMALE_STANDARD": (1.00, 1.00),
    "MALE_DEEP": (0.85, 0.87),
}

VOWEL_CENTROIDS: dict[str, dict[str, tuple[float, float]]] = {
    "INDONESIAN": {
        "A": (825.0, 1275.0),
        "I": (325.0, 2300.0),
        "U": (365.0, 825.0),
        "E": (550.0, 1950.0),
        "O": (550.0, 925.0),
    },
    "ENGLISH": {
        "A": (780.0, 1320.0),
        "I": (325.0, 2300.0),
        "U": (365.0, 825.0),
        "E": (550.0, 1950.0),
        "O": (550.0, 925.0),
    },
}

_scipy_checked: bool = False
_scipy_signal = None
_scipy_wavfile = None


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded into mono float32 samples."""


@dataclass
class AudioClip:
    """Decoded mono float32 audio clip resampled to the analysis sample rate."""

    samples: np.ndarray
    sample_rate: int
    duration_sec: float


@dataclass
class FormantFrame:
    """Per-frame acoustic features, VAD state, tracked formants, and vowel scores."""

    time_sec: float
    f1: float | None
    f2: float | None
    rms_db: float
    zcr: float
    is_voiced: bool
    vowel: str | None
    scores: dict[str, float]


def _clear_stale_scipy_namespace() -> None:
    """Remove an empty namespace placeholder if SciPy was scanned before pip install."""
    mod = sys.modules.get("scipy")
    if mod is None:
        return
    if getattr(mod, "__file__", None) is not None or hasattr(mod, "signal"):
        return
    for name in list(sys.modules):
        if name == "scipy" or name.startswith("scipy."):
            sys.modules.pop(name, None)


def refresh_scipy() -> bool:
    """Re-probe optional scipy and scipy.signal modules after runtime installation."""
    global _scipy_checked, _scipy_signal, _scipy_wavfile
    importlib.invalidate_caches()
    _scipy_checked = True

    if "scipy" in sys.modules and sys.modules["scipy"] is None:
        _scipy_signal = None
        _scipy_wavfile = None
        return False

    for attempt in range(2):
        try:
            importlib.import_module("scipy")
            signal_mod = importlib.import_module("scipy.signal")
            if not (
                hasattr(signal_mod, "resample_poly")
                and hasattr(signal_mod, "butter")
                and hasattr(signal_mod, "sosfiltfilt")
            ):
                raise ImportError("scipy.signal is missing required DSP functions")
            try:
                wavfile_mod = importlib.import_module("scipy.io.wavfile")
            except Exception:
                wavfile_mod = None
            _scipy_signal = signal_mod
            _scipy_wavfile = wavfile_mod
            return True
        except Exception:
            if attempt == 0:
                _clear_stale_scipy_namespace()
                importlib.invalidate_caches()
                continue
            _scipy_signal = None
            _scipy_wavfile = None
            return False
    return False


def has_scipy() -> bool:
    """Return True when SciPy DSP functions are available without blocking UI draw."""
    global _scipy_signal, _scipy_wavfile
    if "scipy" in sys.modules and sys.modules["scipy"] is None:
        return False
    if not _scipy_checked:
        return refresh_scipy()
    if _scipy_signal is not None:
        current_sig = sys.modules.get("scipy.signal")
        if current_sig is _scipy_signal and sys.modules.get("scipy") is not None:
            return True
        _scipy_signal = None
        _scipy_wavfile = None
        return refresh_scipy()
    if sys.modules.get("scipy.signal") is not None:
        return refresh_scipy()
    return False


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    """Convert an arbitrary 1D/2D audio array into a contiguous mono float32 vector."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 2:
        if arr.shape[1] == 0:
            return np.empty(0, dtype=np.float32)
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.mean(axis=1, dtype=np.float32)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _fft_lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Apply a zero-phase FFT low-pass filter with a raised-cosine transition band."""
    n = len(x)
    if n < 4 or cutoff_hz >= sr * 0.5:
        return x.astype(np.float64, copy=False)

    pad = min(n - 1, max(16, int(0.02 * sr)))
    x_pad = np.pad(x.astype(np.float64), pad, mode="reflect")
    n_fft = 1 << (len(x_pad) - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))

    f_pass = cutoff_hz * 0.9
    f_stop = min(sr * 0.495, cutoff_hz * 1.1)
    mask = np.ones_like(freqs)
    if f_stop > f_pass:
        trans = (freqs > f_pass) & (freqs < f_stop)
        mask[trans] = 0.5 * (
            1.0 + np.cos(np.pi * (freqs[trans] - f_pass) / (f_stop - f_pass))
        )
        mask[freqs >= f_stop] = 0.0
    else:
        mask[freqs > cutoff_hz] = 0.0

    y = np.fft.irfft(np.fft.rfft(x_pad, n=n_fft) * mask, n=n_fft)
    return y[pad : pad + n]


def resample_mono(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio from orig_sr to target_sr using SciPy or FFT anti-aliasing."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive (got {orig_sr} -> {target_sr})")

    mono = _to_mono_float32(samples)
    if mono.size == 0 or orig_sr == target_sr:
        return mono

    if has_scipy() and _scipy_signal is not None:
        gcd = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd
        if max(up, down) <= 1000:
            res = _scipy_signal.resample_poly(mono.astype(np.float64), up, down)
            return np.ascontiguousarray(res, dtype=np.float32)

    cutoff = min(float(orig_sr), float(target_sr)) * 0.45
    filtered = _fft_lowpass(mono.astype(np.float64), int(orig_sr), cutoff)
    n = len(filtered)
    out_len = max(1, int(round(n * float(target_sr) / float(orig_sr))))
    if n == 1:
        return np.full(out_len, float(filtered[0]), dtype=np.float32)

    t_orig = np.arange(n, dtype=np.float64) / float(orig_sr)
    t_new = np.arange(out_len, dtype=np.float64) / float(target_sr)
    out = np.interp(t_new, t_orig, filtered)
    return np.ascontiguousarray(out, dtype=np.float32)


def bandpass_vocal(
    samples: np.ndarray,
    sr: int = 16000,
    low_hz: float = 300.0,
    high_hz: float = 3000.0,
) -> np.ndarray:
    """Zero-phase bandpass filter isolating vocal energy in [low_hz, high_hz]."""
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive (got {sr})")

    mono = _to_mono_float32(samples)
    if mono.size == 0:
        return mono

    nyq = 0.5 * float(sr)
    low = max(1.0, float(low_hz))
    high = min(nyq * 0.98, float(high_hz))
    if low >= high or mono.size < 4:
        return mono

    if has_scipy() and _scipy_signal is not None and mono.size > 32:
        sos = _scipy_signal.butter(4, [low, high], btype="bandpass", fs=float(sr), output="sos")
        filtered = _scipy_signal.sosfiltfilt(sos, mono.astype(np.float64))
        return np.ascontiguousarray(filtered, dtype=np.float32)

    pad = min(mono.size - 1, max(16, int(0.02 * sr)))
    x_pad = np.pad(mono.astype(np.float64), pad, mode="reflect")
    n_fft = 1 << (len(x_pad) - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))

    low_stop = max(0.0, low - max(30.0, 0.25 * low))
    high_stop = min(nyq, high + max(100.0, 0.15 * high))

    mask = np.zeros_like(freqs)
    pass_band = (freqs >= low) & (freqs <= high)
    mask[pass_band] = 1.0

    if low > low_stop:
        low_skirt = (freqs > low_stop) & (freqs < low)
        mask[low_skirt] = 0.5 * (
            1.0 - np.cos(np.pi * (freqs[low_skirt] - low_stop) / (low - low_stop))
        )
    if high_stop > high:
        high_skirt = (freqs > high) & (freqs < high_stop)
        mask[high_skirt] = 0.5 * (
            1.0 + np.cos(np.pi * (freqs[high_skirt] - high) / (high_stop - high))
        )

    y = np.fft.irfft(np.fft.rfft(x_pad, n=n_fft) * mask, n=n_fft)
    return np.ascontiguousarray(y[pad : pad + mono.size], dtype=np.float32)


def _decode_pcm_bytes(raw: bytes, sampwidth: int, nchannels: int) -> np.ndarray:
    """Decode unsigned 8-bit or signed 16/24/32-bit little-endian PCM into mono float32."""
    frame_bytes = nchannels * sampwidth
    usable = (len(raw) // frame_bytes) * frame_bytes
    if usable == 0:
        raise AudioLoadError("WAV file contains no complete sample frames")
    raw = raw[:usable]

    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        # Place 24-bit LE bytes into MSBs of int32 so >> 8 performs exact sign extension.
        padded = np.zeros((triplets.shape[0], 4), dtype=np.uint8)
        padded[:, 1:4] = triplets
        signed = padded.view("<i4").ravel() >> 8
        data = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = (np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0).astype(
            np.float32
        )
    else:
        raise AudioLoadError(f"Unsupported PCM sample width: {sampwidth * 8} bits")

    if nchannels > 1:
        data = data.reshape(-1, nchannels).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def _decode_wav_stdlib(path: pathlib.Path, target_sr: int) -> np.ndarray:
    """Decode PCM WAV files via stdlib wave without external dependencies."""
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        if nchannels <= 0 or framerate <= 0 or nframes <= 0:
            raise AudioLoadError("WAV header declares empty or invalid stream parameters")
        raw = wf.readframes(nframes)

    mono = _decode_pcm_bytes(raw, sampwidth, nchannels)
    if framerate != target_sr:
        return resample_mono(mono, framerate, target_sr)
    return mono


def _decode_with_aud(path: pathlib.Path, target_sr: int) -> np.ndarray:
    """Decode audio inside Blender using Audaspace (aud.Sound)."""
    try:
        import aud  # type: ignore[import-not-found]
    except Exception as exc:
        raise AudioLoadError(f"aud module unavailable: {exc}") from exc

    if aud is None or not hasattr(aud, "Sound"):
        raise AudioLoadError("aud.Sound unavailable")

    sound = aud.Sound.file(str(path))
    specs = getattr(sound, "specs", None)
    orig_sr = int(round(specs[0])) if specs and len(specs) > 0 and specs[0] > 0 else target_sr

    try:
        processed = sound.rechannel(1).resample(int(target_sr), True).cache()
        mono = _to_mono_float32(np.asarray(processed.data(), dtype=np.float32))
    except Exception:
        mono = _to_mono_float32(np.asarray(sound.data(), dtype=np.float32))
        if orig_sr != target_sr and mono.size > 0:
            mono = resample_mono(mono, orig_sr, target_sr)

    if mono.size == 0:
        raise AudioLoadError("aud returned zero audio samples")
    return mono


def _decode_with_ffmpeg(path: pathlib.Path, target_sr: int) -> np.ndarray:
    """Decode compressed or non-PCM audio using ffmpeg raw float32 stdout pipe."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise AudioLoadError("ffmpeg executable not found on PATH")

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLoadError(f"ffmpeg failed ({proc.returncode}): {err_msg or 'unknown error'}")

    usable = (len(proc.stdout) // 4) * 4
    if usable == 0:
        raise AudioLoadError("ffmpeg decoded zero audio samples")

    return np.frombuffer(proc.stdout[:usable], dtype="<f4").copy()


def _decode_with_scipy_wavfile(path: pathlib.Path, target_sr: int) -> np.ndarray:
    """Fallback WAV reader via scipy.io.wavfile for IEEE float or extended WAV files."""
    if not has_scipy() or _scipy_wavfile is None:
        raise AudioLoadError("scipy.io.wavfile unavailable")

    orig_sr, raw = _scipy_wavfile.read(str(path))
    arr = np.asarray(raw)
    if arr.size == 0:
        raise AudioLoadError("scipy.io.wavfile returned empty array")

    if arr.dtype == np.uint8:
        data = (arr.astype(np.float32) - 128.0) / 128.0
    elif arr.dtype == np.int16:
        data = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        data = (arr.astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        data = arr.astype(np.float32)

    mono = _to_mono_float32(data)
    if int(orig_sr) != target_sr:
        return resample_mono(mono, int(orig_sr), target_sr)
    return mono


def load_audio(filepath: str, target_sr: int = 16000) -> AudioClip:
    """Load and resample an audio file to mono float32 at target_sr."""
    if not filepath or not str(filepath).strip():
        raise AudioLoadError("Audio filepath is empty")
    if target_sr <= 0:
        raise AudioLoadError(f"Target sample rate must be positive (got {target_sr})")

    path = pathlib.Path(filepath)
    if not path.is_file():
        raise AudioLoadError(f"Audio file not found: {filepath}")
    if path.stat().st_size == 0:
        raise AudioLoadError(f"Audio file is empty: {filepath}")

    decoders = []
    if path.suffix.lower() == ".wav":
        decoders.append(("wave", _decode_wav_stdlib))
    decoders.append(("aud", _decode_with_aud))
    decoders.append(("ffmpeg", _decode_with_ffmpeg))
    if path.suffix.lower() == ".wav":
        decoders.append(("scipy.io.wavfile", _decode_with_scipy_wavfile))

    errors: list[str] = []
    for label, fn in decoders:
        try:
            samples = fn(path, int(target_sr))
            if samples.size == 0:
                raise AudioLoadError(f"{label} returned zero samples")
            if not np.all(np.isfinite(samples)):
                raise AudioLoadError(f"{label} produced non-finite samples")
            mono = np.ascontiguousarray(samples, dtype=np.float32)
            duration_sec = float(mono.size) / float(target_sr)
            return AudioClip(samples=mono, sample_rate=int(target_sr), duration_sec=duration_sec)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise AudioLoadError(f"Unable to decode audio '{filepath}': {'; '.join(errors)}")


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Solve Yule-Walker autocorrelation equations for LPC polynomial coefficients."""
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    err = float(r[0])
    if err <= 1e-20:
        return a

    for m in range(1, order + 1):
        acc = float(r[m]) + float(np.dot(a[1:m], r[m - 1 : 0 : -1]))
        k = -acc / err
        if abs(k) >= 1.0:
            k = math.copysign(0.9999, k)
        a_prev = a[1:m].copy()
        a[1:m] = a_prev + k * a_prev[::-1]
        a[m] = k
        err *= 1.0 - k * k
        if err <= 1e-20:
            break
    return a


def _extract_formants(xw: np.ndarray, sr: int, order: int = LPC_ORDER) -> tuple[float | None, float | None]:
    """Estimate lowest two vocal tract resonances (F1, F2) from LPC polynomial roots."""
    n = len(xw)
    r = np.array([np.dot(xw[: n - k], xw[k:]) for k in range(order + 1)], dtype=np.float64)
    if r[0] <= 1e-18:
        return None, None

    r[0] *= 1.0001
    a = _levinson_durbin(r, order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.0]
    if roots.size == 0:
        return None, None

    scale = float(sr) / (2.0 * math.pi)
    freqs = np.arctan2(np.imag(roots), np.real(roots)) * scale
    mags = np.maximum(np.abs(roots), 1e-12)
    bws = -(float(sr) / math.pi) * np.log(mags)

    valid = sorted(
        float(freq)
        for freq, bw in zip(freqs, bws)
        if MIN_FORMANT_HZ <= freq <= MAX_FORMANT_HZ and bw <= MAX_BANDWIDTH_HZ
    )
    if len(valid) >= 2:
        return valid[0], valid[1]
    if len(valid) == 1:
        return valid[0], None
    return None, None


def _score_formants(
    f1: float,
    f2: float,
    centroids: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Compute normalized Gaussian softmax scores for (f1, f2) across A-I-U-E-O."""
    log_likes = {
        v: -0.5 * (((f1 - c1) / SIGMA_F1) ** 2 + ((f2 - c2) / SIGMA_F2) ** 2)
        for v, (c1, c2) in centroids.items()
    }
    max_ll = max(log_likes.values())
    exps = {v: math.exp(val - max_ll) for v, val in log_likes.items()}
    total = sum(exps.values())
    if total <= 0.0:
        return {v: 0.2 for v in VOWELS}
    return {v: exps[v] / total for v in VOWELS}


def _unvoiced_frame(
    time_sec: float,
    rms_db: float,
    zcr: float,
    zero_scores: dict[str, float],
) -> FormantFrame:
    """Build an unvoiced/silence FormantFrame with zeroed vowel scores."""
    return FormantFrame(
        time_sec=time_sec,
        f1=None,
        f2=None,
        rms_db=rms_db,
        zcr=zcr,
        is_voiced=False,
        vowel=None,
        scores=dict(zero_scores),
    )


def analyze_audio_frames(
    samples: np.ndarray,
    sr: int = 16000,
    silence_threshold_db: float = -38.0,
    voice_type: str = "FEMALE_STANDARD",
    language: str = "INDONESIAN",
) -> list[FormantFrame]:
    """Run 25ms/10ms VAD and order-16 LPC formant tracking over mono audio."""
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive (got {sr})")

    mono = _to_mono_float32(samples)
    if mono.size == 0:
        return []

    if sr != TARGET_SAMPLE_RATE:
        mono = resample_mono(mono, sr, TARGET_SAMPLE_RATE)

    work_sr = TARGET_SAMPLE_RATE
    frame_len = int(round(work_sr * FRAME_MS / 1000.0))
    hop_len = int(round(work_sr * HOP_MS / 1000.0))

    orig_len = mono.size
    if orig_len < frame_len:
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[:orig_len] = mono.astype(np.float64)
        work = padded
        starts = [0]
    else:
        work = mono.astype(np.float64)
        starts = list(range(0, orig_len - frame_len + 1, hop_len))

    emph = np.empty_like(work)
    emph[0] = work[0]
    emph[1:] = work[1:] - PRE_EMPHASIS * work[:-1]

    window = np.hamming(frame_len)
    fft_freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(work_sr))
    mid_mask = (fft_freqs >= 200.0) & (fft_freqs <= 3000.0)
    high_mask = (fft_freqs >= 3500.0) & (fft_freqs <= 8000.0)

    scale_f1, scale_f2 = VOICE_TYPE_SCALES.get(str(voice_type).upper(), (1.00, 1.00))
    centroids = VOWEL_CENTROIDS.get(str(language).upper(), VOWEL_CENTROIDS["INDONESIAN"])
    zero_scores = {v: 0.0 for v in VOWELS}

    results: list[FormantFrame] = []
    for start in starts:
        raw_frame = work[start : start + frame_len]
        if orig_len < frame_len:
            time_sec = 0.5 * float(orig_len) / float(work_sr)
        else:
            time_sec = (float(start) + 0.5 * float(frame_len)) / float(work_sr)

        rms = float(np.sqrt(np.mean(raw_frame * raw_frame)))
        rms_db = 20.0 * math.log10(max(rms, 1e-10))
        zcr = float(np.mean((raw_frame[:-1] * raw_frame[1:]) < 0.0))

        if rms_db < silence_threshold_db:
            results.append(_unvoiced_frame(time_sec, rms_db, zcr, zero_scores))
            continue

        if zcr > 0.35:
            spec = np.abs(np.fft.rfft(raw_frame * window)) ** 2
            mid_energy = float(np.sum(spec[mid_mask]))
            high_energy = float(np.sum(spec[high_mask]))
            if rms_db < silence_threshold_db + 12.0 or high_energy > 1.5 * max(mid_energy, 1e-18):
                results.append(_unvoiced_frame(time_sec, rms_db, zcr, zero_scores))
                continue

        xw = emph[start : start + frame_len] * window
        f1, f2 = _extract_formants(xw, work_sr, LPC_ORDER)

        if f1 is not None and f2 is not None:
            scores = _score_formants(f1 / scale_f1, f2 / scale_f2, centroids)
            vowel = max(VOWELS, key=scores.__getitem__)
            results.append(
                FormantFrame(
                    time_sec=time_sec,
                    f1=f1,
                    f2=f2,
                    rms_db=rms_db,
                    zcr=zcr,
                    is_voiced=True,
                    vowel=vowel,
                    scores=scores,
                )
            )
        elif f1 is not None:
            results.append(
                FormantFrame(
                    time_sec=time_sec,
                    f1=f1,
                    f2=None,
                    rms_db=rms_db,
                    zcr=zcr,
                    is_voiced=True,
                    vowel=None,
                    scores={v: 0.2 for v in VOWELS},
                )
            )
        else:
            results.append(_unvoiced_frame(time_sec, rms_db, zcr, zero_scores))

    return results


def _estimate_vocal_tract_scale(
    f1_vals: np.ndarray,
    f2_vals: np.ndarray,
    centroids: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Find bounded vocal-tract scaling (k1, k2) aligning utterance formants to neutral vowels."""
    n = f1_vals.size
    if n < 5:
        return 1.0, 1.0

    f1_col = f1_vals[:, None]
    f2_col = f2_vals[:, None]
    c1 = np.array([centroids[v][0] for v in VOWELS], dtype=np.float64)[None, :]
    c2 = np.array([centroids[v][1] for v in VOWELS], dtype=np.float64)[None, :]

    best_obj = -1e18
    best_scale = (1.0, 1.0)
    # Search coupled F1/F2 scaling so single-vowel utterances cannot collapse onto another vowel.
    for k1 in np.linspace(0.78, 1.28, 26):
        for dk in (-0.06, -0.03, 0.0, 0.03, 0.06):
            k2 = float(k1) + dk
            d2 = (((f1_col / float(k1)) - c1) / SIGMA_F1) ** 2 + (
                ((f2_col / k2) - c2) / SIGMA_F2
            ) ** 2
            ll = np.maximum(-0.5 * np.min(d2, axis=1), -12.0)
            prior = 0.05 * n * (((float(k1) - 1.0) / 0.2) ** 2 + ((k2 - 1.0) / 0.2) ** 2)
            obj = float(np.sum(ll)) - prior
            if obj > best_obj:
                best_obj = obj
                best_scale = (float(k1), k2)

    return best_scale


def compute_speaker_normalized_scores(frames: list[FormantFrame]) -> list[dict[str, float]]:
    """Compute speaker-normalized Mode B vowel likelihoods without requiring Voice Type."""
    if not frames:
        return []

    centroids = VOWEL_CENTROIDS["INDONESIAN"]
    voiced_indices: list[int] = []
    f1_list: list[float] = []
    f2_list: list[float] = []

    for idx, frame in enumerate(frames):
        if frame.is_voiced and frame.f1 is not None and frame.f2 is not None:
            voiced_indices.append(idx)
            f1_list.append(frame.f1)
            f2_list.append(frame.f2)

    scale_f1, scale_f2 = _estimate_vocal_tract_scale(
        np.asarray(f1_list, dtype=np.float64),
        np.asarray(f2_list, dtype=np.float64),
        centroids,
    )

    out: list[dict[str, float]] = [dict(frame.scores) for frame in frames]
    for idx in voiced_indices:
        frame = frames[idx]
        if frame.f1 is not None and frame.f2 is not None:
            out[idx] = _score_formants(frame.f1 / scale_f1, frame.f2 / scale_f2, centroids)

    return out
