"""Utility helpers for converting MP3 files to 16kHz mono WAV files."""

import argparse
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import wave
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# ``ProductionVAD`` is lightweight (numpy-only) and validates that the audio we
# produced is compatible with the voice AI stack shipped in this repository.
from production_vad import ProductionVAD


def _is_module_available(module_name: str) -> bool:
    """Return ``True`` when the provided module can be imported."""

    return importlib.util.find_spec(module_name) is not None


def _ensure_packages_installed(packages: Sequence[str]) -> None:
    """Install any missing packages from the provided ``packages`` list."""

    missing = [pkg for pkg in packages if not _is_module_available(pkg)]
    if not missing:
        return

    print(f"Installing missing packages: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def _convert_with_moviepy(
    mp3_files: Sequence[str],
    input_dir: str,
    output_dir: str,
    overwrite: bool,
) -> List[str]:
    from moviepy.editor import AudioFileClip

    print("Using moviepy for conversion...")
    print()

    converted: List[str] = []
    for i, mp3_file in enumerate(mp3_files):
        mp3_path = os.path.join(input_dir, mp3_file)
        wav_file = mp3_file.replace(".mp3", ".wav")
        wav_path = os.path.join(output_dir, wav_file)

        print(f"  [{i + 1:2d}/{len(mp3_files)}] {mp3_file} -> {wav_file}")

        if os.path.exists(wav_path) and not overwrite:
            print("    - already exists, skipping (use --overwrite to force)")
            converted.append(wav_file)
            continue

        audio = AudioFileClip(mp3_path)
        audio_array = audio.to_soundarray(fps=16000)

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        audio_int16 = (audio_array * 32767).astype(np.int16)

        with wave.open(wav_path, "w") as wav_file_obj:
            wav_file_obj.setnchannels(1)
            wav_file_obj.setsampwidth(2)
            wav_file_obj.setframerate(16000)
            wav_file_obj.writeframes(audio_int16.tobytes())

        audio.close()
        converted.append(wav_file)

    return converted


def _convert_with_librosa(
    mp3_files: Sequence[str],
    input_dir: str,
    output_dir: str,
    overwrite: bool,
) -> List[str]:
    import librosa
    import soundfile as sf

    print("Using librosa for conversion...")
    print()

    converted: List[str] = []
    for i, mp3_file in enumerate(mp3_files):
        mp3_path = os.path.join(input_dir, mp3_file)
        wav_file = mp3_file.replace(".mp3", ".wav")
        wav_path = os.path.join(output_dir, wav_file)

        print(f"  [{i + 1:2d}/{len(mp3_files)}] {mp3_file} -> {wav_file}")

        if os.path.exists(wav_path) and not overwrite:
            print("    - already exists, skipping (use --overwrite to force)")
            converted.append(wav_file)
            continue

        audio, _ = librosa.load(mp3_path, sr=16000, mono=True)
        sf.write(wav_path, audio, 16000)
        converted.append(wav_file)

    return converted


def _convert_with_ffmpeg(
    mp3_files: Sequence[str],
    input_dir: str,
    output_dir: str,
    overwrite: bool,
) -> List[str]:
    """Use a locally installed ffmpeg executable for conversion."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg executable not found on PATH")

    print("Using ffmpeg CLI for conversion...")
    print()

    converted: List[str] = []
    for i, mp3_file in enumerate(mp3_files):
        mp3_path = os.path.join(input_dir, mp3_file)
        wav_file = mp3_file.replace(".mp3", ".wav")
        wav_path = os.path.join(output_dir, wav_file)

        print(f"  [{i + 1:2d}/{len(mp3_files)}] {mp3_file} -> {wav_file}")

        if os.path.exists(wav_path) and not overwrite:
            print("    - already exists, skipping (use --overwrite to force)")
            converted.append(wav_file)
            continue

        process = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                mp3_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                wav_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="ignore").strip()
            message = stderr or f"ffmpeg exited with code {process.returncode}"
            raise RuntimeError(message)

        converted.append(wav_file)

    return converted


def _validate_outputs(
    wav_files: Sequence[str],
    output_dir: str,
    expected_sample_rate: int = 16000,
) -> List[Dict[str, float]]:
    """Validate WAV files and report metadata for voice AI compatibility."""

    print("Running output validation (format + duration checks)...")
    print()

    results: List[Dict[str, float]] = []
    for wav_file in wav_files:
        wav_path = os.path.join(output_dir, wav_file)

        with wave.open(wav_path, "rb") as wav_obj:
            sample_rate = wav_obj.getframerate()
            channels = wav_obj.getnchannels()
            sample_width = wav_obj.getsampwidth()
            frames = wav_obj.getnframes()

        duration_sec = frames / sample_rate if sample_rate else 0.0

        info = {
            "file": wav_file,
            "sample_rate": float(sample_rate),
            "channels": float(channels),
            "sample_width_bytes": float(sample_width),
            "duration_sec": duration_sec,
        }
        results.append(info)

        status = "OK"
        if sample_rate != expected_sample_rate:
            status = f"Unexpected sample rate ({sample_rate} Hz)"
        elif channels != 1:
            status = f"Unexpected channels ({channels})"
        elif sample_width != 2:
            status = f"Unexpected sample width ({sample_width} bytes)"

        print(
            f"  - {wav_file:30s} | sr={sample_rate:5d}Hz | "
            f"channels={channels} | width={sample_width} bytes | "
            f"duration={duration_sec:6.2f}s | {status}"
        )

    print()
    return results


def _run_voice_ai_check(
    wav_files: Sequence[str],
    output_dir: str,
    expected_sample_rate: int = 16000,
) -> List[Dict[str, float]]:
    """Run ``ProductionVAD`` on the WAV files to ensure speech is detected."""

    vad = ProductionVAD(sample_rate=expected_sample_rate)
    frame_samples = vad.frame_samples

    print("Running ProductionVAD compatibility check...")
    print()

    summaries: List[Dict[str, float]] = []
    for wav_file in wav_files:
        wav_path = os.path.join(output_dir, wav_file)

        with wave.open(wav_path, "rb") as wav_obj:
            sample_rate = wav_obj.getframerate()
            channels = wav_obj.getnchannels()
            audio_int16 = wav_obj.readframes(wav_obj.getnframes())

        if sample_rate != expected_sample_rate or channels != 1:
            print(
                f"  - {wav_file:30s} | skipped (format mismatch: "
                f"sr={sample_rate}Hz, channels={channels})"
            )
            summaries.append(
                {
                    "file": wav_file,
                    "speech_ratio": math.nan,
                    "mean_confidence": math.nan,
                    "max_confidence": math.nan,
                    "speech_frames": 0.0,
                    "total_frames": 0.0,
                }
            )
            continue

        audio = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32)
        if audio.size == 0:
            print(f"  - {wav_file:30s} | empty audio stream")
            summaries.append(
                {
                    "file": wav_file,
                    "speech_ratio": 0.0,
                    "mean_confidence": 0.0,
                    "max_confidence": 0.0,
                    "speech_frames": 0.0,
                    "total_frames": 0.0,
                }
            )
            continue

        audio = audio / 32768.0

        total_frames = 0
        speech_frames = 0
        confidences: List[float] = []

        for start in range(0, len(audio), frame_samples):
            frame = audio[start : start + frame_samples]
            if frame.size == 0:
                break

            result = vad.detect_frame(frame)
            total_frames += 1
            confidences.append(float(result.get("confidence", 0.0)))
            if result.get("is_speech"):
                speech_frames += 1

        speech_ratio = speech_frames / total_frames if total_frames else 0.0
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        max_confidence = float(np.max(confidences)) if confidences else 0.0

        summaries.append(
            {
                "file": wav_file,
                "speech_ratio": speech_ratio,
                "mean_confidence": mean_confidence,
                "max_confidence": max_confidence,
                "speech_frames": float(speech_frames),
                "total_frames": float(total_frames),
            }
        )

        print(
            f"  - {wav_file:30s} | speech frames: {speech_frames:4d}/"
            f"{total_frames:4d} | speech ratio: {speech_ratio:5.2f} | "
            f"mean conf: {mean_confidence:4.2f} | max conf: {max_confidence:4.2f}"
        )

    print()
    return summaries


def convert_mp3_to_wav(
    input_dir: str = "test_audio_mp3",
    output_dir: str = "test_audio",
    preferred_method: Optional[str] = None,
    auto_install: bool = False,
    overwrite: bool = False,
    skip_validation: bool = False,
    validate_only: bool = False,
    run_vad_check: bool = False,
) -> List[str]:
    """Convert MP3 files inside ``input_dir`` and save WAV files into ``output_dir``."""

    print("=" * 80)
    print("CONVERTING MP3 TO WAV (16kHz MONO)")
    print("=" * 80)
    print()

    os.makedirs(output_dir, exist_ok=True)

    if validate_only:
        wav_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".wav"))
        if not wav_files:
            print(f"No WAV files found in {output_dir}/ to validate.")
            return []

        print("Validation-only mode: skipping conversion and checking existing WAVs.")
        print("-" * 80)

        if not skip_validation:
            _validate_outputs(wav_files, output_dir)
        else:
            print("Skipping validation as requested.")
            print()

        if run_vad_check:
            _run_voice_ai_check(wav_files, output_dir)

        print("=" * 80)
        return list(wav_files)

    try:
        mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]
    except FileNotFoundError:
        print(f"Input directory '{input_dir}' does not exist.")
        return []

    if not mp3_files:
        print(f"No MP3 files found in {input_dir}/")
        return []

    print(f"Found {len(mp3_files)} MP3 files")
    print("-" * 80)

    methods: List[str] = []
    if preferred_method:
        methods.append(preferred_method)
    methods.extend(["moviepy", "librosa", "ffmpeg"])

    ordered_methods = []
    for method in methods:
        if method not in ordered_methods:
            ordered_methods.append(method)

    converters = {
        "moviepy": (
            _convert_with_moviepy,
            ["moviepy", "numpy"],
        ),
        "librosa": (
            _convert_with_librosa,
            ["librosa", "soundfile"],
        ),
        "ffmpeg": (
            _convert_with_ffmpeg,
            (),
        ),
    }

    errors: dict[str, str] = {}

    for method in ordered_methods:
        if method not in converters:
            continue

        converter, packages = converters[method]

        if method == "ffmpeg" and shutil.which("ffmpeg") is None:
            errors[method] = "ffmpeg executable not found on PATH"
            continue

        if auto_install:
            try:
                _ensure_packages_installed(packages)
            except subprocess.CalledProcessError as exc:
                errors[method] = f"Failed to install dependencies ({exc})"
                continue

        try:
            converted = converter(mp3_files, input_dir, output_dir, overwrite)
            print()
            print("-" * 80)
            print(f"Processed {len(converted)} files into {output_dir}/")
            print("Format: 16kHz, 16-bit, Mono WAV")
            print()

            if not skip_validation:
                _validate_outputs(converted, output_dir)
            else:
                print("Skipping validation as requested.")
                print()

            if run_vad_check:
                _run_voice_ai_check(converted, output_dir)

            print("=" * 80)
            return list(converted)
        except ImportError as exc:
            module_name = getattr(exc, "name", str(exc))
            errors[method] = f"Missing dependency: {module_name}"
        except Exception as exc:  # noqa: BLE001
            errors[method] = str(exc)

    print("ERROR: No suitable audio conversion backend is available.")
    for method, message in errors.items():
        print(f"  - {method}: {message}")

    print()
    print("Please install dependencies for one of the supported methods:")
    print("  pip install moviepy numpy        # MoviePy backend")
    print("  pip install librosa soundfile    # Librosa backend")
    print("  Install ffmpeg and ensure it is on PATH")
    return []


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MP3 files to 16kHz mono WAV using available backends.",
    )
    parser.add_argument(
        "--input-dir",
        default="test_audio_mp3",
        help="Directory containing MP3 files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_audio",
        help="Destination directory for WAV files (default: %(default)s)",
    )
    parser.add_argument(
        "--prefer-method",
        choices=["moviepy", "librosa", "ffmpeg"],
        help="Preferred backend to use when available.",
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Automatically install missing dependencies via pip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert files even if the WAV already exists.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip WAV format validation after conversion.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip conversion and only validate existing WAV files.",
    )
    parser.add_argument(
        "--run-vad-check",
        action="store_true",
        help=(
            "Run the ProductionVAD voice AI on the WAV files to confirm "
            "speech is detected."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        converted = convert_mp3_to_wav(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            preferred_method=args.prefer_method,
            auto_install=args.auto_install,
            overwrite=args.overwrite,
            skip_validation=args.skip_validation,
            validate_only=args.validate_only,
            run_vad_check=args.run_vad_check,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed: {exc}")
        return 1

    if converted:
        print()
        print("SUCCESS! Audio files ready for testing.")
        print()
        print("NEXT STEPS:")
        print(f"1. Check {args.output_dir}/ directory for WAV files")
        print("2. Run: python test_with_real_audio.py")
        print()
        return 0

    print()
    print("Conversion failed. See error messages above.")
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
