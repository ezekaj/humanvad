"""Benchmark Excellence VAD quality using semantic intent cues."""

from pathlib import Path
from typing import Dict
import unittest

import numpy as np
from scipy.io import wavfile

from excellence_vad import ExcellenceVAD
from intent_classifier_german import IntentClassifierGerman
from production_vad import ProductionVAD


FIXTURE_METADATA: Dict[str, Dict[str, str]] = {
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


def _load_fixture(name: str) -> np.ndarray:
    sample_path = Path("test_audio") / f"{name}.wav"
    sr, audio = wavfile.read(sample_path)
    assert sr == 16000, "Fixtures are expected to be 16kHz"
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return audio


def _run_excellence(audio: np.ndarray, text: str, vad: ExcellenceVAD) -> Dict:
    vad.reset()
    frame_size = vad.prosody_detector.frame_samples
    silence_frame = np.zeros(frame_size, dtype=np.float32)
    results = []

    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size:
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[: len(frame)] = frame
            frame = padded
        results.append(vad.process_frame(silence_frame, frame, text))

    for _ in range(5):
        results.append(vad.process_frame(silence_frame, np.zeros(frame_size, dtype=np.float32), text))

    return results[-1]


def _run_production(audio: np.ndarray) -> Dict:
    prod_vad = ProductionVAD(sample_rate=16000)
    frame_size = prod_vad.frame_samples
    result = None

    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size:
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[: len(frame)] = frame
            frame = padded
        result = prod_vad.detect_frame(frame)

    for _ in range(5):
        result = prod_vad.detect_frame(np.zeros(frame_size, dtype=np.float32))

    return result


class ExcellenceIntentBenchmarkTest(unittest.TestCase):
    def test_intent_classifier_alignment(self) -> None:
        classifier = IntentClassifierGerman()
        question = classifier.classify("Haben Sie noch weitere Fragen")
        self.assertEqual("question", question.intent_type)
        self.assertTrue(question.is_fpp)
        self.assertLessEqual(question.expected_gap_ms, 200)

        incomplete = classifier.classify("Das Zimmer ist verfügbar und")
        self.assertIn(incomplete.intent_type, {"unknown", "statement"})
        self.assertLessEqual(incomplete.confidence, 0.6)

    def test_excellence_outperforms_production_turn_end(self) -> None:
        classifier = IntentClassifierGerman()
        excellence = ExcellenceVAD(
            sample_rate=16000,
            turn_end_threshold=0.75,
            intent_classifier=classifier,
        )

        excellence_correct = 0
        production_correct = 0
        total = 0

        for name, metadata in FIXTURE_METADATA.items():
            audio = _load_fixture(name)
            text = metadata["text"]
            expected_ready = metadata["expected"] == "interrupt"

            excellence_result = _run_excellence(audio, text, excellence)
            predicted_ready = excellence_result["turn_end_prob"] >= excellence.turn_end_threshold

            production_result = _run_production(audio)
            production_ready = production_result is not None and not production_result["is_speech"]

            excellence_correct += int(predicted_ready == expected_ready)
            production_correct += int(production_ready == expected_ready)
            total += 1

            if metadata["expected"] == "interrupt":
                self.assertGreaterEqual(
                    excellence_result["turn_end_prob"],
                    0.6,
                    f"Low readiness for {name}",
                )
            else:
                self.assertLess(
                    excellence_result["turn_end_prob"],
                    excellence.turn_end_threshold,
                    f"False positive for {name}",
                )

            self.assertIsNotNone(excellence_result["intent_type"])
            self.assertIsNotNone(excellence_result["intent_expected_gap_ms"])

        excellence_accuracy = excellence_correct / total
        production_accuracy = production_correct / total

        self.assertGreater(excellence_accuracy, 0.8)
        self.assertGreaterEqual(excellence_accuracy - production_accuracy, 0.3)


if __name__ == "__main__":
    unittest.main()
