"""
Test Excellence VAD - Human Telephone Performance
==================================================

Comprehensive testing with realistic conversation scenarios
Including semantic + prosodic fusion
"""

import numpy as np
from excellence_vad import ExcellenceVAD


def generate_speech_with_prosody(
    duration=1.0,
    sr=16000,
    f0_start=150,
    f0_end=150,
    modulation_rate=5.0
):
    """Generate speech-like signal"""
    t = np.linspace(0, duration, int(sr * duration))
    f0 = np.linspace(f0_start, f0_end, len(t))

    signal = 1.0 * np.sin(2 * np.pi * f0 * t)
    signal += 0.8 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.6 * np.sin(2 * np.pi * 3 * f0 * t)
    signal += 0.4 * np.sin(2 * np.pi * 500 * t)
    signal += 0.3 * np.sin(2 * np.pi * 1500 * t)

    modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * modulation_rate * t))
    signal *= modulation

    return signal / np.max(np.abs(signal))


def test_excellence_vad():
    """Test Excellence VAD with realistic scenarios"""

    print("=" * 80)
    print(" EXCELLENCE VAD TEST - Human Telephone Performance")
    print("=" * 80)
    print()
    print("Testing hybrid prosody + semantic turn-taking detection")
    print()
    print("-" * 80)
    print(f"{'Scenario':<55} {'Turn-End':<10} {'Action':<30} {'Result'}")
    print("-" * 80)

    sr = 16000
    vad = ExcellenceVAD(sample_rate=sr, turn_end_threshold=0.75)

    test_cases = []
    frame_size = 160

    # ========================================================================
    # SCENARIO 1: Complete sentence + falling pitch (NATURAL TURN-END)
    # ========================================================================

    print("\n--- SCENARIO 1: Complete Sentence + Falling Pitch ---")

    ai_speech = generate_speech_with_prosody(
        duration=0.8,
        sr=sr,
        f0_start=180,
        f0_end=120,  # Falling pitch
        modulation_rate=5.0
    )

    ai_speech_with_pause = np.concatenate([
        ai_speech,
        np.zeros(int(sr * 0.2))  # Pause
    ])

    user_response = generate_speech_with_prosody(
        duration=0.3,
        sr=sr,
        f0_start=150,
        f0_end=150,
        modulation_rate=5.0
    )

    # AI text: COMPLETE sentence
    ai_text = "I think we should go there tomorrow"

    vad.reset()
    results = []

    # Process AI speech with complete sentence
    for i in range(0, len(ai_speech_with_pause) - frame_size, frame_size):
        ai_frame = ai_speech_with_pause[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        results.append(result)

    # User responds after pause
    for i in range(0, len(user_response) - frame_size, frame_size):
        user_frame = user_response[i:i+frame_size]
        ai_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        if result['user_speaking']:
            results.append(result)
            break

    final = results[-1]
    turn_end_prob = final['turn_end_prob']
    action = final['action']

    expected_action = "interrupt_ai_immediately"  # AI already finished
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Complete sentence + falling pitch',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected': expected_action,
        'prosody': final['prosody_prob'],
        'semantic': final['semantic_prob']
    })

    print(f"{'Complete sentence + falling pitch':<55} {turn_end_prob:>6.1%}    {action:<30} {status}")

    # ========================================================================
    # SCENARIO 2: Incomplete sentence + level pitch (INTERRUPTION)
    # ========================================================================

    print("\n--- SCENARIO 2: Incomplete Sentence + Level Pitch ---")

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

    # AI text: INCOMPLETE sentence
    ai_text = "I think we should go to"  # Incomplete (preposition)

    vad.reset()
    results = []

    # Process AI speech
    for i in range(0, len(ai_speech) - frame_size, frame_size * 2):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        results.append(result)

    # User interrupts (overlap)
    for i in range(0, len(user_interrupt) - frame_size, frame_size):
        user_frame = user_interrupt[i:i+frame_size]
        ai_idx = min(len(ai_speech) // 2 + i, len(ai_speech) - frame_size)
        ai_frame = ai_speech[ai_idx:ai_idx+frame_size]

        result = vad.process_frame(user_frame, ai_frame, ai_text)
        if result['overlap']:
            results.append(result)
            break

    overlap_results = [r for r in results if r.get('overlap', False)]
    if overlap_results:
        final = overlap_results[0]
        turn_end_prob = final['turn_end_prob']
        action = final['action']
    else:
        turn_end_prob = 0.0
        action = "continue"

    expected_action = "interrupt_ai_immediately"  # User interrupting mid-sentence
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Incomplete sentence + level pitch',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected': expected_action,
        'prosody': final['prosody_prob'] if overlap_results else 0.0,
        'semantic': final['semantic_prob'] if overlap_results else 0.0
    })

    print(f"{'Incomplete sentence + level pitch':<55} {turn_end_prob:>6.1%}    {action:<30} {status}")

    # ========================================================================
    # SCENARIO 3: Question (complete) + rising pitch
    # ========================================================================

    print("\n--- SCENARIO 3: Complete Question + Rising Pitch ---")

    ai_speech = generate_speech_with_prosody(
        duration=0.6,
        sr=sr,
        f0_start=140,
        f0_end=200,  # Rising pitch (question)
        modulation_rate=5.5
    )

    user_answer = generate_speech_with_prosody(
        duration=0.3,
        sr=sr,
        f0_start=150,
        f0_end=150,
        modulation_rate=5.0
    )

    # AI text: Complete question
    ai_text = "What time should we meet?"

    vad.reset()
    results = []

    # Process question
    for i in range(0, len(ai_speech) - frame_size, frame_size):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        results.append(result)

    # Add pause
    for i in range(int(sr * 0.15) // frame_size):
        result = vad.process_frame(np.zeros(frame_size), np.zeros(frame_size), ai_text)
        results.append(result)

    # User answers
    for i in range(0, len(user_answer) - frame_size, frame_size):
        user_frame = user_answer[i:i+frame_size]
        ai_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        if result['user_speaking']:
            results.append(result)
            break

    final = [r for r in results if r.get('user_speaking', False)]
    if final:
        final = final[0]
        turn_end_prob = final['turn_end_prob']
        action = final['action']
    else:
        turn_end_prob = 0.0
        action = "continue"

    expected_action = "interrupt_ai_immediately"  # Question complete, user answering
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Complete question + rising pitch',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected': expected_action,
        'prosody': final['prosody_prob'] if final else 0.0,
        'semantic': final['semantic_prob'] if final else 0.0
    })

    print(f"{'Complete question + rising pitch':<55} {turn_end_prob:>6.1%}    {action:<30} {status}")

    # ========================================================================
    # SCENARIO 4: Conjunction mid-sentence (STRONG incomplete signal)
    # ========================================================================

    print("\n--- SCENARIO 4: Conjunction (Expecting Continuation) ---")

    ai_speech = generate_speech_with_prosody(
        duration=0.5,
        sr=sr,
        f0_start=160,
        f0_end=165,  # Slightly rising (continuation)
        modulation_rate=5.5
    )

    user_interrupt = generate_speech_with_prosody(
        duration=0.2,
        sr=sr,
        f0_start=150,
        f0_end=150,
        modulation_rate=5.0
    )

    # AI text: STRONG incomplete (conjunction)
    ai_text = "I would like to go but"  # Expecting continuation

    vad.reset()
    results = []

    # Process AI speech
    for i in range(0, len(ai_speech) - frame_size, frame_size * 2):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        results.append(result)

    # User interrupts
    for i in range(0, len(user_interrupt) - frame_size, frame_size):
        user_frame = user_interrupt[i:i+frame_size]
        ai_idx = min(len(ai_speech) // 2 + i, len(ai_speech) - frame_size)
        ai_frame = ai_speech[ai_idx:ai_idx+frame_size]

        result = vad.process_frame(user_frame, ai_frame, ai_text)
        if result['overlap']:
            results.append(result)
            break

    overlap_results = [r for r in results if r.get('overlap', False)]
    if overlap_results:
        final = overlap_results[0]
        turn_end_prob = final['turn_end_prob']
        action = final['action']
    else:
        turn_end_prob = 0.0
        action = "continue"

    expected_action = "interrupt_ai_immediately"  # Clear interruption (AI mid-sentence)
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Conjunction (incomplete)',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected': expected_action,
        'prosody': final['prosody_prob'] if overlap_results else 0.0,
        'semantic': final['semantic_prob'] if overlap_results else 0.0
    })

    print(f"{'Conjunction (expecting continuation)':<55} {turn_end_prob:>6.1%}    {action:<30} {status}")

    # ========================================================================
    # SCENARIO 5: Acknowledgment (complete) + energy drop
    # ========================================================================

    print("\n--- SCENARIO 5: Acknowledgment + Energy Drop ---")

    ai_speech = generate_speech_with_prosody(
        duration=0.4,
        sr=sr,
        f0_start=160,
        f0_end=140,  # Falling
        modulation_rate=5.0
    )

    # Fade out (energy drop)
    fade = np.linspace(1.0, 0.2, len(ai_speech))
    ai_speech = ai_speech * fade

    ai_speech_with_pause = np.concatenate([
        ai_speech,
        np.zeros(int(sr * 0.15))
    ])

    user_response = generate_speech_with_prosody(
        duration=0.2,
        sr=sr,
        f0_start=150,
        f0_end=150,
        modulation_rate=5.0
    )

    # AI text: Complete acknowledgment
    ai_text = "Okay thanks"

    vad.reset()
    results = []

    for i in range(0, len(ai_speech_with_pause) - frame_size, frame_size):
        ai_frame = ai_speech_with_pause[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        results.append(result)

    # User responds
    for i in range(0, len(user_response) - frame_size, frame_size):
        user_frame = user_response[i:i+frame_size]
        ai_frame = np.zeros(frame_size)
        result = vad.process_frame(user_frame, ai_frame, ai_text)
        if result['user_speaking']:
            results.append(result)
            break

    final = [r for r in results if r.get('user_speaking', False)]
    if final:
        final = final[0]
        turn_end_prob = final['turn_end_prob']
        action = final['action']
    else:
        turn_end_prob = 0.0
        action = "continue"

    expected_action = "interrupt_ai_immediately"  # Natural turn-taking
    correct = (action == expected_action)
    status = "PASS" if correct else "FAIL"

    test_cases.append({
        'name': 'Acknowledgment + energy drop',
        'turn_end_prob': turn_end_prob,
        'action': action,
        'expected': expected_action,
        'prosody': final['prosody_prob'] if final else 0.0,
        'semantic': final['semantic_prob'] if final else 0.0
    })

    print(f"{'Acknowledgment + energy drop':<55} {turn_end_prob:>6.1%}    {action:<30} {status}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 80)

    correct = sum(1 for t in test_cases if t['action'] == t['expected'])
    total = len(test_cases)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f" ACCURACY: {correct}/{total} scenarios correct ({accuracy:.1f}%)")
    print("=" * 80)
    print()

    # Performance stats
    stats = vad.get_stats()
    print(f"PERFORMANCE:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Target: <10ms {'PASS' if stats['avg_latency_ms'] < 10 else 'FAIL'}")
    print()

    # Detailed breakdown
    print("DETAILED RESULTS:")
    print("-" * 80)
    for tc in test_cases:
        print(f"\n{tc['name']}:")
        print(f"  Turn-end probability: {tc['turn_end_prob']:.1%}")
        print(f"    - Prosody contribution: {tc['prosody']:.1%} (45% weight)")
        print(f"    - Semantic contribution: {tc['semantic']:.1%} (55% weight)")
        print(f"  Action: {tc['action']}")
        print(f"  Expected: {tc['expected']}")
        print(f"  Result: {'PASS' if tc['action'] == tc['expected'] else 'FAIL'}")

    print()
    print("=" * 80)
    print(" FINAL ASSESSMENT")
    print("=" * 80)
    print()

    if accuracy >= 90:
        print(f"EXCELLENCE ACHIEVED: {accuracy:.1f}% accuracy")
        print("Matches human telephone performance (90-95%)")
    elif accuracy >= 80:
        print(f"GOOD: {accuracy:.1f}% accuracy")
        print("Approaching human telephone performance")
    else:
        print(f"NEEDS IMPROVEMENT: {accuracy:.1f}% accuracy")

    print()
    print("Method:")
    print("  - Prosodic analysis (energy, timing, pauses) - 45%")
    print("  - Semantic completion detection (syntax, patterns) - 55%")
    print("  - Hybrid fusion matching human telephone strategy")
    print()

    return accuracy


if __name__ == "__main__":
    accuracy = test_excellence_vad()
