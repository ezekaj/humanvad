"""
Disfluency Detector - Feature #5
Detects hesitations and incomplete thoughts in German speech
"""

from typing import Dict


class DisfluencyDetector:
    """
    Lightweight hesitation detection for German speech

    Distinguishes between:
    - Incomplete thoughts with hesitations (äh, ähm, pauses)
    - Complete utterances ready for turn-taking
    """

    def __init__(self):
        # German fillers (most common in spontaneous speech)
        # Weight: 0.0-1.0 (higher = stronger hesitation signal)
        self.fillers = {
            'äh': 0.8,      # Strong hesitation signal (like "uh")
            'ähm': 0.9,     # Very strong (longer planning, like "um")
            'ehm': 0.8,     # Variant spelling
            'also': 0.6,    # Moderate (discourse marker "so", "well")
            'mh': 0.5,      # Weak minimal filler
            'hm': 0.5,      # Weak minimal filler
            'öh': 0.7,      # Regional variant
            'ähem': 0.8,    # Variant
        }

        # Pause tracking
        self.pause_threshold_ms = 200  # Pauses >200ms indicate planning

    def detect_hesitation(
        self,
        text: str,
        pause_ms: float = 0
    ) -> Dict[str, float]:
        """
        Detect hesitation probability in speech segment

        Args:
            text: Text segment (from ASR or AI output)
            pause_ms: Duration of pause/silence in milliseconds

        Returns:
            dict with:
                - hesitation_prob: 0.0-1.0 (0 = fluent, 1 = strong hesitation)
                - detected_fillers: List of detected filler words
                - pause_detected: Boolean if pause exceeded threshold
        """

        hesitation_prob = 0.0
        detected_fillers = []
        pause_detected = False

        if not text:
            return {
                'hesitation_prob': 0.0,
                'detected_fillers': [],
                'pause_detected': False
            }

        text_lower = text.lower().strip()

        # 1. Check for filler words
        for filler, weight in self.fillers.items():
            if filler in text_lower:
                detected_fillers.append(filler)
                hesitation_prob = max(hesitation_prob, weight)

        # 2. Check for pause
        if pause_ms > self.pause_threshold_ms:
            pause_detected = True
            # Linear increase: 200ms=0.0, 700ms=0.7
            pause_prob = min((pause_ms - self.pause_threshold_ms) / 500, 0.7)
            hesitation_prob = max(hesitation_prob, pause_prob)

        # 3. Combine pause + filler (stronger signal)
        # Pause + filler = very strong hesitation indicator
        if pause_detected and len(detected_fillers) > 0:
            hesitation_prob = min(hesitation_prob * 1.5, 1.0)

        # 4. Detect repair patterns (false start)
        # Examples: "zum... zur", "ich... wir", short fragments
        if '...' in text:
            hesitation_prob = max(hesitation_prob, 0.6)
        elif ' ' not in text.strip() and len(text.strip()) > 0 and len(text.strip()) < 5:
            # Very short utterances often indicate false starts
            hesitation_prob = max(hesitation_prob, 0.4)

        return {
            'hesitation_prob': hesitation_prob,
            'detected_fillers': detected_fillers,
            'pause_detected': pause_detected
        }

    def adjust_completion_probability(
        self,
        semantic_prob: float,
        hesitation_prob: float
    ) -> float:
        """
        Adjust semantic completion probability based on hesitation

        High hesitation → Lower completion probability
        (Incomplete thought, should wait)

        Args:
            semantic_prob: Original semantic completion probability (0.0-1.0)
            hesitation_prob: Detected hesitation probability (0.0-1.0)

        Returns:
            Adjusted semantic completion probability
        """

        if hesitation_prob < 0.1:
            # No hesitation - keep original
            return semantic_prob

        # Strong hesitation reduces completion probability
        # hesitation_prob=0.6 → reduction factor=0.7 (30% reduction)
        # hesitation_prob=0.9 → reduction factor=0.55 (45% reduction)
        reduction_factor = 1.0 - (hesitation_prob * 0.5)

        adjusted_prob = semantic_prob * reduction_factor

        # Don't drop below 0.1 (always leave some probability)
        return max(adjusted_prob, 0.1)


def demo():
    """Demo hesitation detection"""
    print("=" * 80)
    print(" DISFLUENCY DETECTOR - FEATURE #5")
    print("=" * 80)
    print()
    print("Detects hesitations vs completions in German speech:")
    print()

    detector = DisfluencyDetector()

    test_cases = [
        # (text, pause_ms, expected_behavior)
        ("äh", 0, "Strong hesitation (filler)"),
        ("ähm", 300, "Very strong hesitation (filler + pause)"),
        ("Ich möchte Ihnen sagen dass", 0, "Fluent incomplete"),
        ("Das Hotel hat 50 Zimmer.", 0, "Fluent complete"),
        ("also", 0, "Moderate hesitation (discourse marker)"),
        ("zum... zur", 0, "Repair pattern (false start)"),
        ("", 400, "Long pause only"),
        ("Das ist alles", 0, "Fluent complete"),
    ]

    print("Test Cases:")
    print("-" * 80)
    for text, pause_ms, description in test_cases:
        result = detector.detect_hesitation(text, pause_ms)
        print(f"\nText: '{text}'")
        print(f"Pause: {pause_ms}ms")
        print(f"Description: {description}")
        print(f"Hesitation Probability: {result['hesitation_prob']:.2f}")
        print(f"Detected Fillers: {result['detected_fillers']}")
        print(f"Pause Detected: {result['pause_detected']}")

        # Test adjustment
        original_semantic = 0.8
        adjusted = detector.adjust_completion_probability(
            original_semantic,
            result['hesitation_prob']
        )
        print(f"Semantic Adjustment: {original_semantic:.2f} -> {adjusted:.2f}")

    print()
    print("=" * 80)
    print(" LATENCY TEST")
    print("=" * 80)
    print()

    import time
    import numpy as np

    times = []
    for _ in range(1000):
        start = time.perf_counter()
        detector.detect_hesitation("ähm", 300)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    print(f"Iterations: 1000")
    print(f"Average: {np.mean(times):.4f}ms")
    print(f"Median: {np.percentile(times, 50):.4f}ms")
    print(f"p95: {np.percentile(times, 95):.4f}ms")
    print(f"Max: {np.max(times):.4f}ms")
    print()
    print("[PASS] Target: <1ms" if np.mean(times) < 1.0 else "[FAIL] Target: <1ms")


if __name__ == "__main__":
    demo()
