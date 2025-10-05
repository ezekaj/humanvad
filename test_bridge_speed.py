"""
Benchmark Memory-VAD Bridge Speed
"""

import sys
import time
import numpy as np
from memory_vad_bridge import MemoryVADBridge


def benchmark():
    bridge = MemoryVADBridge(embedding_dim=64, min_observations=5)

    # Warmup
    for _ in range(10):
        bridge.observe_turn_taking(
            speaker_id="warmup",
            intent_type="question",
            intent_subtype="wh_question",
            is_fpp=True,
            expected_gap_ms=100,
            actual_gap_ms=105,
            prosody_features={'final_f0_slope': 0.0, 'speech_rate': 5.0, 'duration': 1000},
            context="test"
        )

    print("=" * 80)
    print(" MEMORY-VAD BRIDGE SPEED BENCHMARK")
    print("=" * 80)
    print()

    # Benchmark observe_turn_taking (critical path)
    times_observe = []
    for i in range(100):
        start = time.perf_counter()
        bridge.observe_turn_taking(
            speaker_id=f"Guest_{i%10}",  # 10 different speakers
            intent_type=["question", "statement", "request"][i % 3],
            intent_subtype="test",
            is_fpp=True,
            expected_gap_ms=150,
            actual_gap_ms=160 + np.random.randint(-20, 20),
            prosody_features={'final_f0_slope': np.random.randn(), 'speech_rate': 5.0, 'duration': 1000},
            context="benchmark"
        )
        end = time.perf_counter()
        times_observe.append((end - start) * 1000)

    # Benchmark predict_next_gap (fast path)
    times_predict = []
    for i in range(100):
        start = time.perf_counter()
        bridge.predict_next_gap(f"Guest_{i%10}", "question")
        end = time.perf_counter()
        times_predict.append((end - start) * 1000)

    # Benchmark profile retrieval
    times_summary = []
    for i in range(100):
        start = time.perf_counter()
        bridge.get_speaker_summary(f"Guest_{i%10}")
        end = time.perf_counter()
        times_summary.append((end - start) * 1000)

    print("RESULTS:")
    print("-" * 80)
    print(f"observe_turn_taking() - Full learning update:")
    print(f"  Average: {np.mean(times_observe):.3f}ms")
    print(f"  Min: {np.min(times_observe):.3f}ms")
    print(f"  Max: {np.max(times_observe):.3f}ms")
    print()

    print(f"predict_next_gap() - Fast prediction:")
    print(f"  Average: {np.mean(times_predict):.3f}ms")
    print(f"  Min: {np.min(times_predict):.3f}ms")
    print(f"  Max: {np.max(times_predict):.3f}ms")
    print()

    print(f"get_speaker_summary() - Profile retrieval:")
    print(f"  Average: {np.mean(times_summary):.3f}ms")
    print(f"  Min: {np.min(times_summary):.3f}ms")
    print(f"  Max: {np.max(times_summary):.3f}ms")
    print()

    # Total pipeline latency
    total_latency = np.mean(times_observe)
    print("=" * 80)
    print(" INTEGRATION LATENCY")
    print("=" * 80)
    print()

    print("Complete VAD + Intent + Memory pipeline:")
    vad_latency = 0.43  # ms (from excellence_vad)
    intent_latency = 0.027  # ms (from intent_classifier)
    memory_latency = total_latency  # ms (from bridge)

    total = vad_latency + intent_latency + memory_latency
    print(f"  VAD:    {vad_latency:.3f}ms")
    print(f"  Intent: {intent_latency:.3f}ms")
    print(f"  Memory: {memory_latency:.3f}ms")
    print(f"  ---")
    print(f"  TOTAL:  {total:.3f}ms")
    print()

    budget_ms = 200
    print(f"Latency budget: {budget_ms}ms")
    print(f"Used: {total:.3f}ms ({(total/budget_ms)*100:.1f}%)")
    print(f"Remaining: {budget_ms - total:.2f}ms")
    print()

    if total < 10:
        print(f"[EXCELLENT] Ultra-low latency! {(10/total):.1f}x faster than 10ms target")
    elif total < budget_ms:
        print(f"[OK] Within budget")
    else:
        print(f"[WARNING] Over budget")

    print()
    print("=" * 80)


if __name__ == "__main__":
    benchmark()
