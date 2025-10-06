"""Evaluate Excellence VAD quality on real (or converted) audio data.

The script runs the full Excellence VAD pipeline – including the optional
intent-classifier fusion – against a directory of WAV files and reports how
well the model predicts turn endings compared to the baseline ProductionVAD.

By default it consumes the gTTS-generated German fixtures that ship with the
repository (``test_audio``).  Supplying ``--audio-dir`` allows evaluation on
any directory of 16kHz mono WAV files, while ``--metadata`` can provide a JSON
file that maps filenames to their text transcripts and expected decisions.

Usage
-----
::

    python evaluate_excellence_real_audio.py \
        --audio-dir ./test_audio \
        --metadata ./fixtures.json

The metadata file should contain a mapping from filename (without extension) to
an object with ``text`` and ``expected`` fields, where ``expected`` is either
``"interrupt"`` (the AI should yield the turn) or ``"wait"`` (the AI should
keep speaking).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.io import wavfile

from excellence_vad import ExcellenceVAD
from intent_classifier_german import IntentClassifierGerman
from production_vad import ProductionVAD

# ---------------------------------------------------------------------------
# Default metadata describing the shipped gTTS fixtures.
# ---------------------------------------------------------------------------

DEFAULT_METADATA: Dict[str, Dict[str, str]] = {
    "complete_sentence_1": {
        "text": "Das Hotel hat fünfzig Zimmer",
        "expected": "interrupt",
    },
    "complete_sentence_2": {
        "text": "Vielen Dank für Ihren Anruf",
        "expected": "interrupt",
    },
    "complete_sentence_3": {
        "text": "Guten Tag, wie kann ich Ihnen helfen",
        "expected": "interrupt",
    },
    "complete_question": {
        "text": "Haben Sie noch weitere Fragen",
        "expected": "interrupt",
    },
    "complete_confirmation": {
        "text": "Ja, das ist korrekt",
        "expected": "interrupt",
    },
    "complete_polite": {
        "text": "Sehr gerne, ich helfe Ihnen",
        "expected": "interrupt",
    },
    "complete_with_number": {
        "text": "Der Preis beträgt zweihundert Euro",
        "expected": "interrupt",
    },
    "incomplete_hesitation": {
        "text": "Ich möchte Ihnen sagen dass",
        "expected": "wait",
    },
    "incomplete_preposition": {
        "text": "Ich gehe zur",
        "expected": "wait",
    },
    "incomplete_conjunction": {
        "text": "Das Zimmer ist verfügbar und",
        "expected": "wait",
    },
}


def _load_metadata(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None:
        return DEFAULT_METADATA

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata: Dict[str, Dict[str, str]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Metadata entry for {key!r} must be an object")
        if "expected" not in value:
            raise ValueError(f"Metadata entry for {key!r} missing 'expected'")
        if value["expected"] not in {"interrupt", "wait"}:
            raise ValueError(
                "Metadata 'expected' must be either 'interrupt' or 'wait'"
            )
        metadata[key] = {
            "text": value.get("text", ""),
            "expected": value["expected"],
        }
    return metadata


def _load_audio(sample_path: Path) -> np.ndarray:
    sr, audio = wavfile.read(sample_path)
    if sr != 16000:
        raise ValueError(
            f"{sample_path.name}: expected 16kHz audio but found {sr}Hz."
        )
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return audio


def _run_excellence(
    audio: np.ndarray,
    text: str,
    vad: ExcellenceVAD,
) -> Dict:
    frame_size = vad.prosody_detector.frame_samples
    silence = np.zeros(frame_size, dtype=np.float32)
    vad.reset()

    result = None
    for start in range(0, len(audio), frame_size):
        frame = audio[start : start + frame_size]
        if len(frame) < frame_size:
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[: len(frame)] = frame
            frame = padded
        result = vad.process_frame(silence, frame, text)

    # Flush a handful of silent frames so the prosody detector can decay.
    for _ in range(5):
        result = vad.process_frame(silence, np.zeros(frame_size, dtype=np.float32), text)

    assert result is not None, "No frames processed"
    return result


def _run_production(audio: np.ndarray, detector: ProductionVAD) -> Dict:
    frame_size = detector.frame_samples
    result = None

    for start in range(0, len(audio), frame_size):
        frame = audio[start : start + frame_size]
        if len(frame) < frame_size:
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[: len(frame)] = frame
            frame = padded
        result = detector.detect_frame(frame)

    for _ in range(5):
        result = detector.detect_frame(np.zeros(frame_size, dtype=np.float32))

    assert result is not None, "No frames processed"
    return result


@dataclass
class DetectorSummary:
    """Aggregate metrics for a detector evaluated over multiple samples."""

    accuracy: float
    precision: float
    recall: float
    counts: Dict[str, int]
    total: int


def _confusion_counts(records: Iterable[Tuple[bool, bool]]) -> Dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for expected, predicted in records:
        if expected and predicted:
            counts["tp"] += 1
        elif expected and not predicted:
            counts["fn"] += 1
        elif not expected and predicted:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return counts


def _summarize_records(records: List[Tuple[bool, bool]]) -> DetectorSummary:
    if not records:
        return DetectorSummary(0.0, 0.0, 0.0, {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, 0)

    counts = _confusion_counts(records)
    total = len(records)
    accuracy = (counts["tp"] + counts["tn"]) / total
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    return DetectorSummary(accuracy, precision, recall, counts, total)


def evaluate(audio_dir: Path, metadata: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    classifier = IntentClassifierGerman()
    excellence = ExcellenceVAD(
        sample_rate=16000,
        turn_end_threshold=0.75,
        intent_classifier=classifier,
    )
    production = ProductionVAD(sample_rate=16000)

    excellence_records: List[Tuple[bool, bool]] = []
    production_records: List[Tuple[bool, bool]] = []
    excellence_turn_end_scores: List[float] = []
    decision_margins: List[float] = []

    print("Evaluating audio directory:", audio_dir)
    print()

    skipped = 0

    for key, meta in sorted(metadata.items()):
        wav_path = audio_dir / f"{key}.wav"
        if not wav_path.exists():
            print(f"  SKIP {key:30s} (missing {wav_path.name})")
            skipped += 1
            continue

        audio = _load_audio(wav_path)
        text = meta.get("text", "") or ""
        expected_ready = meta["expected"] == "interrupt"

        excellence_result = _run_excellence(audio, text, excellence)
        production_result = _run_production(audio, production)

        predicted_ready = (
            excellence_result["turn_end_prob"] >= excellence.turn_end_threshold
        )
        production_ready = (
            production_result is not None and not production_result["is_speech"]
        )

        excellence_records.append((expected_ready, predicted_ready))
        production_records.append((expected_ready, production_ready))

        excellence_turn_end_scores.append(excellence_result["turn_end_prob"])
        if expected_ready:
            decision_margins.append(
                excellence_result["turn_end_prob"] - excellence.turn_end_threshold
            )
        else:
            decision_margins.append(
                excellence.turn_end_threshold - excellence_result["turn_end_prob"]
            )

        print(f"  {key:30s} text={text!r}")
        production_state = "ready" if production_ready else "speaking"
        print(
            f"    expected={meta['expected']:<9s} "
            f"excellence={excellence_result['turn_end_prob']:.3f} "
            f"production={production_state}"
        )
        print(
            f"    decision=\"{'interrupt' if predicted_ready else 'wait'}\" "
            f"intent={excellence_result.get('intent_type')} "
            f"gap_ms={excellence_result.get('intent_expected_gap_ms')}"
        )
        print()

    excellence_summary = _summarize_records(excellence_records)
    production_summary = _summarize_records(production_records)

    def _print_summary(name: str, summary: DetectorSummary) -> None:
        if summary.total == 0:
            print(f"No evaluation records for {name}")
            return
        counts = summary.counts
        print(
            f"{name} accuracy: {summary.accuracy * 100:.1f}% "
            f"({counts['tp'] + counts['tn']}/{summary.total})"
        )
        print(
            f"  precision={summary.precision * 100:.1f}% "
            f"recall={summary.recall * 100:.1f}% "
            f"tp={counts['tp']} fp={counts['fp']} fn={counts['fn']} tn={counts['tn']}"
        )
        print()

    print("SUMMARY")
    print("-" * 60)
    _print_summary("Excellence", excellence_summary)
    _print_summary("Production", production_summary)

    if excellence_turn_end_scores:
        avg_score = mean(excellence_turn_end_scores)
        avg_margin = mean(decision_margins)
        print(f"Average Excellence turn-end probability: {avg_score:.3f}")
        print(f"Average decision margin: {avg_margin:.3f}")
        print()
    else:
        avg_score = 0.0
        avg_margin = 0.0

    return {
        "samples_evaluated": len(excellence_records),
        "samples_skipped": skipped,
        "excellence": asdict(excellence_summary),
        "production": asdict(production_summary),
        "avg_excellence_turn_end_prob": avg_score,
        "avg_decision_margin": avg_margin,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("test_audio"),
        help="Directory containing 16kHz mono WAV files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON file describing transcripts/expectations.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write JSON metrics summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(args.metadata)
    if not args.audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {args.audio_dir}")
    summary = evaluate(args.audio_dir, metadata)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
