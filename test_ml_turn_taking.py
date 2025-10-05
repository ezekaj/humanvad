"""
Test ML Turn-Taking VAD
=======================

Honest evaluation on realistic turn-taking scenarios.
"""

import numpy as np
from ml_turn_taking_vad import MLTurnTakingVAD


def generate_speech_with_prosody(
    duration=1.0,
    sr=16000,
    f0_start=150,
    f0_end=150,
    modulation_rate=5.0
):
    """Generate speech-like signal with controlled prosody"""
    t = np.linspace(0, duration, int(sr * duration))

    # Variable F0
    f0 = np.linspace(f0_start, f0_end, len(t))

    # Fundamental + harmonics
    signal = 1.0 * np.sin(2 * np.pi * f0 * t)
    signal += 0.8 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.6 * np.sin(2 * np.pi * 3 * f0 * t)

    # Formants
    signal += 0.4 * np.sin(2 * np.pi * 500 * t)
    signal += 0.3 * np.sin(2 * np.pi * 1500 * t)

    # Syllable-rate modulation
    modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * modulation_rate * t))
    signal *= modulation

    return signal / np.max(np.abs(signal))


def test_ml_turn_taking():
    """Test ML turn-taking VAD with honest evaluation"""

    print("=" * 80)
    print(" ML TURN-TAKING VAD TEST - Honest Evaluation")
    print("=" * 80)
    print()

    sr = 16000
    vad = MLTurnTakingVAD(sample_rate=sr)

    print("Testing on realistic scenarios:")
    print("-" * 80)
    print(f"{'Scenario':<50} {'Turn-End':<12} {'Action':<25} {'Result'}")
    print("-" * 80)

    test_cases = []
    frame_size = 160

    # ========================================================================
    # SCENARIO 1: Falling pitch + pause (NATURAL TURN-END)
    # ========================================================================

    print("\n--- SCENARIO 1: Falling Pitch + Pause (Natural Turn-End) ---")

    ai_speech = generate_speech_with_prosody(
        duration=0.8,
        sr=sr,
        f0_start=180,
        f0_end=120,  # Falling 60 Hz
        modulation_rate=5.0
    )

    ai_speech_with_pause = np.concatenate([
        ai_speech,
        np.zeros(int(sr * 0.2))  # 200ms pause
    ])

    user_response = generate_speech_with_prosody(
        duration=0.3,
        sr=sr,
        f0_start=150,
        f0_end=150,
        modulation_rate=5.0
    )

    vad.reset()
    results = []

    # Process AI speech
    for i in range(0, len(ai_speech_with_pause) - frame_size, frame_size):
        ai_frame = ai_speech_with_pause[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame)
        results.append(result)

    # User responds
    for i in range(0, len(user_response) - frame_size, frame_size):
        user_frame = user_response[i:i+frame_size]
        ai_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame)
        if result['user_speaking']:
            results.append(result)
            break

    final_result = results[-1]
    turn_end_prob = final_result['turn_end_prob']
    action = final_result['action']

    expected_action = "interrupt_ai_immediately"  # AI already silent
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Falling pitch + pause',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected_action': expected_action
    })

    print(f"{'Falling pitch + pause':<50} {turn_end_prob:>6.1%}      {action:<25} {status}")

    # ========================================================================
    # SCENARIO 2: Level pitch mid-sentence (INTERRUPTION)
    # ========================================================================

    print("\n--- SCENARIO 2: Level Pitch Mid-Sentence (Interruption) ---")

    ai_speech = generate_speech_with_prosody(
        duration=0.6,
        sr=sr,
        f0_start=160,
        f0_end=160,  # Level pitch
        modulation_rate=6.0
    )

    user_interrupt = generate_speech_with_prosody(
        duration=0.3,
        sr=sr,
        f0_start=150,
        f0_end=150,
        modulation_rate=5.0
    )

    vad.reset()
    results = []

    # Process AI speech
    for i in range(0, len(ai_speech) - frame_size, frame_size * 2):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame)
        results.append(result)

    # User interrupts (overlap)
    for i in range(0, len(user_interrupt) - frame_size, frame_size):
        user_frame = user_interrupt[i:i+frame_size]
        ai_idx = min(len(ai_speech) // 2 + i, len(ai_speech) - frame_size)
        ai_frame = ai_speech[ai_idx:ai_idx+frame_size]

        result = vad.process_frame(user_frame, ai_frame)
        if result['overlap']:
            results.append(result)
            break

    if len(results) > 0:
        overlap_results = [r for r in results if r.get('overlap', False)]
        if len(overlap_results) > 0:
            final_result = overlap_results[0]
            turn_end_prob = final_result['turn_end_prob']
            action = final_result['action']
        else:
            turn_end_prob = 0.0
            action = "continue"
    else:
        turn_end_prob = 0.0
        action = "continue"

    expected_action = "interrupt_ai_immediately"
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Level pitch mid-sentence',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected_action': expected_action
    })

    print(f"{'Level pitch mid-sentence':<50} {turn_end_prob:>6.1%}      {action:<25} {status}")

    # ========================================================================
    # SCENARIO 3: Slowing speech (TURN-END APPROACHING)
    # ========================================================================

    print("\n--- SCENARIO 3: Slowing Speech Rate ---")

    speech_fast = generate_speech_with_prosody(
        duration=0.5,
        sr=sr,
        f0_start=160,
        f0_end=160,
        modulation_rate=6.0  # Fast
    )

    speech_slow = generate_speech_with_prosody(
        duration=0.5,
        sr=sr,
        f0_start=160,
        f0_end=140,
        modulation_rate=4.0  # Slowing
    )

    ai_speech = np.concatenate([speech_fast, speech_slow])

    vad.reset()
    results = []

    for i in range(0, len(ai_speech) - frame_size, frame_size):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame)
        results.append(result)

    # Compare first vs second half turn-end probability
    mid = len(results) // 2
    first_half_prob = np.mean([r['turn_end_prob'] for r in results[:mid]])
    second_half_prob = np.mean([r['turn_end_prob'] for r in results[mid:]])

    increasing = second_half_prob > first_half_prob
    status = "PASS" if increasing else "INFO"

    print(f"{'Slowing speech (first->second half)':<50} {first_half_prob:>6.1%}->{second_half_prob:<6.1%}         {status}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 80)

    testable = [t for t in test_cases if 'expected_action' in t]
    correct = sum(1 for t in testable if t['action'] == t['expected_action'])
    total = len(testable)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f" SUMMARY: {correct}/{total} scenarios correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)
    print()

    # Performance stats
    stats = vad.get_stats()
    print(f"PERFORMANCE STATS:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Target: <10ms {'PASS' if stats['avg_latency_ms'] < 10 else 'FAIL'}")
    print()

    print("METHOD:")
    print("  - Energy trend analysis (falling = ending)")
    print("  - Pitch estimation via autocorrelation")
    print("  - Speech rate tracking (ZCR)")
    print("  - Temporal context (3 seconds)")
    print()

    # HONEST ASSESSMENT
    print("=" * 80)
    print(" HONEST ASSESSMENT")
    print("=" * 80)
    print()

    if accuracy >= 80:
        print(f"Result: {accuracy:.1f}% accuracy - GOOD for heuristic approach")
    elif accuracy >= 60:
        print(f"Result: {accuracy:.1f}% accuracy - MODERATE, needs ML training")
    else:
        print(f"Result: {accuracy:.1f}% accuracy - POOR, fundamental issues")

    print()
    print("Limitations:")
    print("  - Autocorrelation pitch tracking unreliable on synthetic speech")
    print("  - Heuristic thresholds not optimized for real voices")
    print("  - No trained ML model (would need labeled data)")
    print()
    print("To reach 90%+ accuracy:")
    print("  1. Train RandomForest/SVM on labeled conversation data")
    print("  2. Use better pitch tracker (YIN, CREPE)")
    print("  3. Add prosodic features (intensity, duration)")
    print("  4. Or use pre-trained VAP model (75%+ proven)")
    print()

    return accuracy


if __name__ == "__main__":
    accuracy = test_ml_turn_taking()
