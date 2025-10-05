"""
Benchmark Intent Classifier Speed

Measure latency in milliseconds
"""

import sys
sys.path.append('../human-speech-detection')

import time
from intent_classifier_german import IntentClassifierGerman


def benchmark_speed():
    """Benchmark classification speed"""

    classifier = IntentClassifierGerman()

    # Test cases - mix of different complexity
    test_cases = [
        "Guten Tag",
        "Haben Sie ein Zimmer frei?",
        "Möchten Sie das Zimmer buchen?",
        "Ich brauche ein Taxi zum Flughafen",
        "Äh, ich denke... vielleicht morgen?",
        "Das Zimmer ist verfügbar",
        "Vielen Dank für Ihren Anruf",
        "Auf Wiederhören",
        "Können Sie mir helfen?",
        "Ja, bitte",
        "Nein, danke",
        "Wie viele Personen?",
        "Am Montag",
        "Das macht 180 Euro pro Nacht",
        "Ist Frühstück inklusive?",
        "Natürlich, was benötigen Sie?",
        "Grüß Gott",
        "Wo kann ich parken?",
        "Ich hätte gern ein ruhiges Zimmer",
        "Selbstverständlich",
    ]

    print("=" * 80)
    print(" INTENT CLASSIFIER SPEED BENCHMARK")
    print("=" * 80)
    print()

    # Warmup (first call may be slower)
    for _ in range(10):
        classifier.classify("Warmup")

    # Benchmark each test case
    individual_times = []

    print("Individual Classifications:")
    print("-" * 80)

    for text in test_cases:
        start = time.perf_counter()
        result = classifier.classify(text)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        individual_times.append(elapsed_ms)

        print(f"{elapsed_ms:6.3f}ms - \"{text[:40]}\"")

    print()

    # Batch benchmark (1000 classifications)
    print("Batch Benchmark (1000 classifications):")
    print("-" * 80)

    batch_start = time.perf_counter()
    for _ in range(1000):
        for text in test_cases[:5]:  # Use first 5 test cases
            classifier.classify(text)
    batch_end = time.perf_counter()

    total_classifications = 5000
    batch_time_ms = (batch_end - batch_start) * 1000
    avg_per_classification = batch_time_ms / total_classifications

    print(f"Total time: {batch_time_ms:.1f}ms")
    print(f"Classifications: {total_classifications}")
    print(f"Average: {avg_per_classification:.3f}ms per classification")
    print()

    # Statistics
    print("=" * 80)
    print(" STATISTICS")
    print("=" * 80)
    print()

    min_time = min(individual_times)
    max_time = max(individual_times)
    avg_time = sum(individual_times) / len(individual_times)
    median_time = sorted(individual_times)[len(individual_times) // 2]

    print(f"Min:     {min_time:.3f}ms")
    print(f"Max:     {max_time:.3f}ms")
    print(f"Average: {avg_time:.3f}ms")
    print(f"Median:  {median_time:.3f}ms")
    print()

    # Real-time capability check
    print("=" * 80)
    print(" REAL-TIME PERFORMANCE")
    print("=" * 80)
    print()

    # Audio frame timing (10ms frames at 16kHz)
    frame_duration_ms = 10

    print(f"Audio frame duration: {frame_duration_ms}ms (10ms @ 16kHz)")
    print(f"Classification time:  {avg_time:.3f}ms")
    print(f"Overhead ratio:       {(avg_time/frame_duration_ms)*100:.1f}%")
    print()

    if avg_time < frame_duration_ms:
        print("[EXCELLENT] Can process faster than real-time!")
        print(f"            Can handle {frame_duration_ms/avg_time:.1f}x real-time speed")
    else:
        print("[WARNING] Slower than real-time")
        print(f"          {avg_time/frame_duration_ms:.1f}x slower than frame rate")

    print()

    # Combined with VAD timing
    print("=" * 80)
    print(" INTEGRATION WITH VAD")
    print("=" * 80)
    print()

    vad_latency_ms = 0.43  # From excellence_vad_german.py benchmark
    total_latency = vad_latency_ms + avg_time

    print(f"VAD latency:         {vad_latency_ms:.2f}ms")
    print(f"Intent latency:      {avg_time:.3f}ms")
    print(f"Total pipeline:      {total_latency:.3f}ms")
    print()

    budget_ms = 200  # User requirement: under 200ms
    remaining_budget = budget_ms - total_latency

    print(f"Latency budget:      {budget_ms}ms")
    print(f"Used:                {total_latency:.3f}ms ({(total_latency/budget_ms)*100:.1f}%)")
    print(f"Remaining:           {remaining_budget:.2f}ms")
    print()

    if total_latency < budget_ms:
        print(f"[OK] Under budget by {remaining_budget:.2f}ms")
    else:
        print(f"[FAIL] Over budget by {abs(remaining_budget):.2f}ms")

    print()

    return {
        'min': min_time,
        'max': max_time,
        'avg': avg_time,
        'median': median_time,
        'batch_avg': avg_per_classification,
        'total_with_vad': total_latency
    }


if __name__ == "__main__":
    results = benchmark_speed()

    print()
    print("=" * 80)
    print(f"FINAL: {results['avg']:.3f}ms average classification time")
    print(f"       {results['total_with_vad']:.3f}ms total with VAD")
    print("=" * 80)
