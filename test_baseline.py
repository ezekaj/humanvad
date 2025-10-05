"""
Test Pure Baseline (No Features)
Verify baseline behavior without any added features
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman


def test_pure_baseline():
    """
    Test baseline Excellence VAD with NO features enabled
    """

    print("=" * 80)
    print(" PURE BASELINE TEST (No Features)")
    print("=" * 80)
    print()
    print("Testing Excellence VAD German with:")
    print("  - Prosody analysis (ProductionVAD) - 45%")
    print("  - German semantic patterns - 55%")
    print("  - NO additional features")
    print()
    print("-" * 80)
    print()

    # Initialize pure baseline (all features disabled)
    vad = ExcellenceVADGerman(
        use_disfluency_detection=False  # Pure baseline
    )

    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Mid-Sentence Hesitation',
            'ai_text': 'Ich möchte Ihnen... ähm... sagen dass',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',
            'description': 'AI hesitates - should NOT allow interrupt'
        },
        {
            'name': 'Scenario 2: Natural Completion',
            'ai_text': 'Das Hotel hat 50 Zimmer.',
            'user_interrupts': True,
            'expected_action': 'interrupt_ai_immediately',
            'description': 'Complete sentence - ALLOW interrupt'
        },
        {
            'name': 'Scenario 3: Repair Pattern',
            'ai_text': 'Ich gehe zum... zur Schule',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',
            'description': 'Self-correction - should NOT allow interrupt'
        },
        {
            'name': 'Scenario 4: Multiple Fillers',
            'ai_text': 'Das ist... äh... also...',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',
            'description': 'Multiple fillers - DON\'T interrupt'
        },
        {
            'name': 'Scenario 5: Fluent Completion',
            'ai_text': 'Vielen Dank für Ihren Anruf.',
            'user_interrupts': True,
            'expected_action': 'interrupt_ai_immediately',
            'description': 'Fluent complete - ALLOW interrupt'
        },
    ]

    correct = 0
    total = len(scenarios)

    for i, scenario in enumerate(scenarios, 1):
        print(f"{scenario['name']}")
        print(f"  Description: {scenario['description']}")
        print(f"  AI Text: \"{scenario['ai_text']}\"")
        print(f"  User Interrupts: {scenario['user_interrupts']}")
        print(f"  Expected Action: {scenario['expected_action']}")
        print()

        # Generate test audio frames
        user_frame = np.random.randn(160) * 0.3 if scenario['user_interrupts'] else np.zeros(160)
        ai_frame = np.random.randn(160) * 0.3

        # Test baseline
        result = vad.process_frame(
            user_frame=user_frame,
            ai_frame=ai_frame,
            ai_text=scenario['ai_text']
        )

        # Check correctness
        is_correct = result['action'] == scenario['expected_action']
        if is_correct:
            correct += 1

        # Print detailed results
        print(f"  Result:")
        print(f"    Action: {result['action']}")
        if 'reasoning' in result:
            print(f"    Reasoning: {result['reasoning']}")
        print(f"    Turn-End Prob: {result['turn_end_prob']:.3f}")
        print(f"    Prosody Prob: {result['prosody_prob']:.3f}")
        print(f"    Semantic Prob: {result['semantic_prob']:.3f}")
        if 'semantic_reason' in result:
            print(f"    Semantic Reason: {result['semantic_reason']}")
        print(f"    User Speaking: {result['user_speaking']}")
        print(f"    AI Speaking: {result['ai_speaking']}")
        if 'overlap' in result:
            print(f"    Overlap: {result['overlap']}")
        print(f"    Latency: {result['latency_ms']:.4f}ms")
        print(f"    Correct: {'YES' if is_correct else 'NO'}")
        print()
        print("-" * 80)
        print()

    # Final results
    accuracy = (correct / total) * 100

    print()
    print("=" * 80)
    print(" BASELINE RESULTS")
    print("=" * 80)
    print()
    print(f"Total Scenarios: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()

    # Performance stats
    stats = vad.get_stats()
    if stats:
        print("Performance Stats:")
        print(f"  Average Latency: {stats['avg_latency_ms']:.4f}ms")
        print(f"  P50 Latency: {stats['p50_latency_ms']:.4f}ms")
        print(f"  P95 Latency: {stats['p95_latency_ms']:.4f}ms")
        print(f"  Max Latency: {stats['max_latency_ms']:.4f}ms")
        print()

    print("=" * 80)

    # Detailed diagnosis
    print()
    print("=" * 80)
    print(" DIAGNOSIS")
    print("=" * 80)
    print()

    if accuracy == 40.0:
        print("PATTERN DETECTED: 40% accuracy (2/5 scenarios)")
        print()
        print("This matches all previous feature tests.")
        print()
        print("Likely issues:")
        print("  1. Random noise not detected as speech by ProductionVAD")
        print("  2. ai_speaking=False causes logic to skip overlap conditions")
        print("  3. Falls through to 'user_speaking' branch")
        print("  4. Always returns 'interrupt_ai_immediately'")
        print("  5. Matches expected for Scenarios 2 & 5 only")
        print()
        print("Recommendation:")
        print("  - Generate real speech audio (not random noise)")
        print("  - Verify ProductionVAD detects speech correctly")
        print("  - Test with actual German voice recordings")
    elif accuracy > 60:
        print("UNEXPECTED: Baseline accuracy >60%")
        print("This is higher than all previous tests.")
        print("Need to investigate what changed.")
    else:
        print(f"Baseline accuracy: {accuracy:.1f}%")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test_pure_baseline()
