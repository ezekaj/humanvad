"""
Benchmark Turn-End Predictor Speed
"""

import time
import numpy as np
from turn_end_predictor import TurnEndPredictor


def benchmark():
    print("=" * 80)
    print(" TURN-END PREDICTOR SPEED BENCHMARK")
    print("=" * 80)
    print()

    # Test both modes
    predictor_stub = TurnEndPredictor(use_lstm=False)
    predictor_lstm = TurnEndPredictor(use_lstm=True, lookahead_ms=300)

    # Warmup
    for _ in range(20):
        predictor_stub.predict(0.5, {'f0_slope': 0, 'energy': 50, 'speech_rate': 5})
        predictor_lstm.predict(0.5, {'f0_slope': 0, 'energy': 50, 'speech_rate': 5})

    # Benchmark stub mode
    times_stub = []
    for i in range(100):
        start = time.perf_counter()
        predictor_stub.predict(
            current_vad_prob=0.5 + np.random.rand() * 0.3,
            prosody_features={'f0_slope': np.random.randn(), 'energy': 50, 'speech_rate': 5.0},
            intent_type='question'
        )
        end = time.perf_counter()
        times_stub.append((end - start) * 1000)

    # Benchmark LSTM mode (heuristic)
    times_lstm = []
    for i in range(100):
        start = time.perf_counter()
        predictor_lstm.predict(
            current_vad_prob=0.5 + np.random.rand() * 0.3,
            prosody_features={'f0_slope': np.random.randn(), 'energy': 50, 'speech_rate': 5.0},
            intent_type='question',
            speaker_profile={'avg_turn_gap_ms': 200, 'interruption_tolerance': 0.75}
        )
        end = time.perf_counter()
        times_lstm.append((end - start) * 1000)

    print("RESULTS:")
    print("-" * 80)
    print(f"Stage 1 (Stub - No Prediction):")
    print(f"  Average: {np.mean(times_stub):.3f}ms")
    print(f"  Min: {np.min(times_stub):.3f}ms")
    print(f"  Max: {np.max(times_stub):.3f}ms")
    print()

    print(f"Stage 2 (LSTM Heuristic - 300ms Lookahead):")
    print(f"  Average: {np.mean(times_lstm):.3f}ms")
    print(f"  Min: {np.min(times_lstm):.3f}ms")
    print(f"  Max: {np.max(times_lstm):.3f}ms")
    print()

    # Full pipeline with prediction
    print("=" * 80)
    print(" FULL PIPELINE WITH PREDICTION")
    print("=" * 80)
    print()

    vad_latency = 0.43
    intent_latency = 0.027
    memory_latency = 5.231
    predictor_latency = np.mean(times_lstm)

    total = vad_latency + intent_latency + memory_latency + predictor_latency

    print("Complete pipeline:")
    print(f"  VAD:       {vad_latency:.3f}ms")
    print(f"  Intent:    {intent_latency:.3f}ms")
    print(f"  Memory:    {memory_latency:.3f}ms")
    print(f"  Predictor: {predictor_latency:.3f}ms")
    print(f"  ---")
    print(f"  TOTAL:     {total:.3f}ms")
    print()

    budget = 200
    print(f"Latency budget: {budget}ms")
    print(f"Used: {total:.3f}ms ({(total/budget)*100:.1f}%)")
    print(f"Remaining: {budget - total:.2f}ms")
    print()

    if total < 10:
        print(f"[EXCELLENT] Ultra-low latency!")
    elif total < 20:
        print(f"[VERY GOOD] <20ms total")
    elif total < budget:
        print(f"[OK] Within budget")
    else:
        print(f"[WARNING] Over budget")

    print()

    # Lookahead benefit analysis
    print("=" * 80)
    print(" LOOKAHEAD BENEFIT")
    print("=" * 80)
    print()

    lookahead_ms = 300
    print(f"Prediction lookahead: {lookahead_ms}ms")
    print(f"Processing time: {total:.1f}ms")
    print(f"Effective advance warning: {lookahead_ms - total:.1f}ms")
    print()

    if (lookahead_ms - total) > 200:
        print("[EXCELLENT] >200ms advance warning - can interrupt smoothly")
    elif (lookahead_ms - total) > 100:
        print("[GOOD] >100ms advance warning - decent interruption time")
    else:
        print("[FAIR] <100ms advance warning - limited benefit")

    print()
    print("=" * 80)


if __name__ == "__main__":
    benchmark()
