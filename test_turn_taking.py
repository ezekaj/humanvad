"""
Test Turn-Taking VAD with Prosodic Scenarios
=============================================

Tests human-like turn-taking detection:
1. Falling pitch at turn-end (should NOT interrupt - natural response)
2. Level pitch mid-sentence (SHOULD interrupt - user cutting in)
3. Speech rate changes
4. Natural pauses vs interruptions
"""

import numpy as np
from turn_taking_vad import TurnTakingVAD


def generate_speech_with_prosody(
    duration=1.0,
    sr=16000,
    f0_start=150,
    f0_end=150,
    modulation_rate=5.0
):
    """
    Generate speech-like signal with controlled prosody

    Args:
        f0_start: Starting pitch (Hz)
        f0_end: Ending pitch (Hz) - for falling/rising contours
        modulation_rate: Syllable rate (Hz)
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Variable F0 (pitch contour)
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


def test_turn_end_scenarios():
    """
    Test turn-taking detection with realistic prosodic scenarios
    """
    print("=" * 80)
    print(" TURN-TAKING VAD TEST - Prosodic Scenarios")
    print("=" * 80)
    print()

    sr = 16000
    vad = TurnTakingVAD(sample_rate=sr, turn_end_threshold=0.65)

    print("Testing turn-taking prediction with prosodic cues:")
    print("-" * 80)
    print(f"{'Scenario':<45} {'Turn-End Prob':<15} {'Should Interrupt':<18} {'Result'}")
    print("-" * 80)

    test_cases = []

    # ========================================================================
    # SCENARIO 1: Falling pitch at sentence end (NATURAL TURN-END)
    # ========================================================================
    # Expected: HIGH turn-end probability, DON'T interrupt (natural response)

    print("\n--- SCENARIO 1: Falling Pitch (Natural Turn-End) ---")
    speech_falling = generate_speech_with_prosody(
        duration=0.8,
        sr=sr,
        f0_start=180,
        f0_end=120,  # Falling pitch (turn-ending cue)
        modulation_rate=5.0
    )

    vad.reset()
    frame_size = 160  # 10ms frames

    # Process speech with falling pitch
    turn_end_probs = []
    for i in range(0, len(speech_falling) - frame_size, frame_size):
        frame = speech_falling[i:i+frame_size]
        result = vad.process_frame(frame)

        if result['is_speaking']:
            turn_end_probs.append(result['turn_end_prob'])

    avg_turn_end_prob = np.mean(turn_end_probs[-20:]) if len(turn_end_probs) >= 20 else 0
    should_interrupt = avg_turn_end_prob < 0.65

    test_cases.append({
        'name': 'Falling pitch (natural turn-end)',
        'turn_end_prob': avg_turn_end_prob,
        'should_interrupt': should_interrupt,
        'expected_interrupt': False,  # DON'T interrupt (natural response)
        'pitch_pattern': result.get('pitch_pattern', 'unknown')
    })

    correct = (should_interrupt == False)
    status = "PASS" if correct else "FAIL"
    print(f"{'Falling pitch (natural turn-end)':<45} {avg_turn_end_prob:>6.1%}         {str(should_interrupt):<18} {status}")

    # ========================================================================
    # SCENARIO 2: Level pitch mid-sentence (USER INTERRUPTION)
    # ========================================================================
    # Expected: LOW turn-end probability, SHOULD interrupt (user cutting in)

    print("\n--- SCENARIO 2: Level Pitch Mid-Sentence (Interruption) ---")
    speech_level = generate_speech_with_prosody(
        duration=0.5,
        sr=sr,
        f0_start=160,
        f0_end=160,  # Level pitch (turn-holding)
        modulation_rate=6.0
    )

    vad.reset()
    turn_end_probs = []

    for i in range(0, len(speech_level) - frame_size, frame_size):
        frame = speech_level[i:i+frame_size]
        result = vad.process_frame(frame)

        if result['is_speaking']:
            turn_end_probs.append(result['turn_end_prob'])

    avg_turn_end_prob = np.mean(turn_end_probs[-10:]) if len(turn_end_probs) >= 10 else 0
    should_interrupt = avg_turn_end_prob < 0.65

    test_cases.append({
        'name': 'Level pitch mid-sentence (interruption)',
        'turn_end_prob': avg_turn_end_prob,
        'should_interrupt': should_interrupt,
        'expected_interrupt': True,  # SHOULD interrupt
        'pitch_pattern': result.get('pitch_pattern', 'unknown')
    })

    correct = (should_interrupt == True)
    status = "PASS" if correct else "FAIL"
    print(f"{'Level pitch mid-sentence (interruption)':<45} {avg_turn_end_prob:>6.1%}         {str(should_interrupt):<18} {status}")

    # ========================================================================
    # SCENARIO 3: Rising pitch (QUESTION)
    # ========================================================================
    # Expected: MEDIUM turn-end probability (question invites response but not definitive)

    print("\n--- SCENARIO 3: Rising Pitch (Question) ---")
    speech_rising = generate_speech_with_prosody(
        duration=0.6,
        sr=sr,
        f0_start=140,
        f0_end=200,  # Rising pitch (question)
        modulation_rate=5.5
    )

    vad.reset()
    turn_end_probs = []

    for i in range(0, len(speech_rising) - frame_size, frame_size):
        frame = speech_rising[i:i+frame_size]
        result = vad.process_frame(frame)

        if result['is_speaking']:
            turn_end_probs.append(result['turn_end_prob'])

    avg_turn_end_prob = np.mean(turn_end_probs[-15:]) if len(turn_end_probs) >= 15 else 0
    should_interrupt = avg_turn_end_prob < 0.65

    test_cases.append({
        'name': 'Rising pitch (question)',
        'turn_end_prob': avg_turn_end_prob,
        'should_interrupt': should_interrupt,
        'expected_interrupt': None,  # Ambiguous (depends on context)
        'pitch_pattern': result.get('pitch_pattern', 'unknown')
    })

    # For rising pitch, we don't test pass/fail since it's context-dependent
    print(f"{'Rising pitch (question)':<45} {avg_turn_end_prob:>6.1%}         {str(should_interrupt):<18} INFO")

    # ========================================================================
    # SCENARIO 4: Slowing speech rate (TURN-END APPROACHING)
    # ========================================================================
    # Expected: Increasing turn-end probability as speech slows

    print("\n--- SCENARIO 4: Slowing Speech Rate ---")
    # Generate speech with decreasing modulation rate (slowing down)
    speech_slowing = np.concatenate([
        generate_speech_with_prosody(duration=0.3, sr=sr, f0_start=160, f0_end=160, modulation_rate=6.0),
        generate_speech_with_prosody(duration=0.3, sr=sr, f0_start=160, f0_end=140, modulation_rate=4.0)
    ])

    vad.reset()
    turn_end_probs = []

    for i in range(0, len(speech_slowing) - frame_size, frame_size):
        frame = speech_slowing[i:i+frame_size]
        result = vad.process_frame(frame)

        if result['is_speaking']:
            turn_end_probs.append(result['turn_end_prob'])

    # Compare first half vs second half
    first_half = np.mean(turn_end_probs[:len(turn_end_probs)//2]) if len(turn_end_probs) > 10 else 0
    second_half = np.mean(turn_end_probs[len(turn_end_probs)//2:]) if len(turn_end_probs) > 10 else 0

    increasing = second_half > first_half

    test_cases.append({
        'name': 'Slowing speech rate',
        'turn_end_prob': second_half,
        'should_interrupt': second_half < 0.65,
        'expected_interrupt': None,  # Check if probability increases
        'pitch_pattern': result.get('pitch_pattern', 'unknown')
    })

    status = "PASS" if increasing else "FAIL"
    print(f"{'Slowing speech (first->second half)':<45} {first_half:>6.1%}->{second_half:<6.1%}  {'N/A':<18} {status}")

    # ========================================================================
    # SCENARIO 5: Pause after falling pitch (DEFINITE TURN-END)
    # ========================================================================
    # Expected: VERY HIGH turn-end probability

    print("\n--- SCENARIO 5: Falling Pitch + Pause ---")
    speech_with_pause = np.concatenate([
        generate_speech_with_prosody(duration=0.5, sr=sr, f0_start=170, f0_end=110, modulation_rate=5.0),
        np.zeros(int(sr * 0.2))  # 200ms pause
    ])

    vad.reset()
    turn_end_probs = []

    for i in range(0, len(speech_with_pause) - frame_size, frame_size):
        frame = speech_with_pause[i:i+frame_size]
        result = vad.process_frame(frame)

        # Track turn-end probability during speech portion
        if i < int(sr * 0.5):  # Before pause
            if result['is_speaking']:
                turn_end_probs.append(result['turn_end_prob'])

    avg_turn_end_prob = np.mean(turn_end_probs[-10:]) if len(turn_end_probs) >= 10 else 0
    should_interrupt = avg_turn_end_prob < 0.65

    test_cases.append({
        'name': 'Falling pitch + pause',
        'turn_end_prob': avg_turn_end_prob,
        'should_interrupt': should_interrupt,
        'expected_interrupt': False,  # DON'T interrupt (clear turn-end)
        'pitch_pattern': result.get('pitch_pattern', 'unknown')
    })

    correct = (should_interrupt == False)
    status = "PASS" if correct else "FAIL"
    print(f"{'Falling pitch + pause':<45} {avg_turn_end_prob:>6.1%}         {str(should_interrupt):<18} {status}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 80)

    # Count results (only scenarios with expected values)
    testable = [t for t in test_cases if t['expected_interrupt'] is not None]
    correct = sum(1 for t in testable if t['should_interrupt'] == t['expected_interrupt'])
    total = len(testable)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f" SUMMARY: {correct}/{total} scenarios correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)
    for tc in test_cases:
        print(f"\n{tc['name']}:")
        print(f"  Turn-end probability: {tc['turn_end_prob']:.1%}")
        print(f"  Pitch pattern: {tc['pitch_pattern']}")
        print(f"  Should interrupt: {tc['should_interrupt']}")
        if tc['expected_interrupt'] is not None:
            print(f"  Expected: {tc['expected_interrupt']}")
            print(f"  Result: {'PASS' if tc['should_interrupt'] == tc['expected_interrupt'] else 'FAIL'}")

    print()
    print("=" * 80)
    print()

    # Performance stats
    stats = vad.get_stats()
    print(f"PERFORMANCE STATS:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Target: <10ms {'PASS' if stats['avg_latency_ms'] < 10 else 'FAIL'}")
    print()

    print("KEY INSIGHTS:")
    print("  - Falling pitch = Turn-end cue → DON'T interrupt (natural response)")
    print("  - Level pitch = Turn-holding → INTERRUPT (user cutting in)")
    print("  - Rising pitch = Question → Context-dependent")
    print("  - Slowing rate = Turn-end approaching → Increase turn-end probability")
    print()

    if accuracy >= 80:
        print(f"TEST PASSED: {accuracy:.1f}% accuracy on turn-taking scenarios")
    else:
        print(f"TEST NEEDS IMPROVEMENT: {accuracy:.1f}% accuracy (target: 80%+)")

    print()
    return accuracy


if __name__ == "__main__":
    accuracy = test_turn_end_scenarios()
