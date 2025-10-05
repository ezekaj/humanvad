"""
Benchmark: Feature #7 - Adaptive Thresholds (Re-test with FIXED baseline)
Compares fixed baseline (80%) vs adaptive threshold enhancement
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman


def test_with_adaptive_thresholds():
    """
    Re-test Feature #7 with FIXED baseline (now 80% accurate)

    Previous test: 40% → 40% (0% improvement)
    Expected now: 80% → ??? with fixed baseline
    """

    print("=" * 80)
    print(" FEATURE #7: ADAPTIVE THRESHOLDS RE-TEST (Fixed Baseline)")
    print("=" * 80)
    print()
    print("Baseline Status: FIXED (inverted logic corrected, semantic detector working)")
    print("Baseline Accuracy: 80% (4/5 scenarios)")
    print()
    print("Testing:")
    print("  Control: Fixed threshold 0.60")
    print("  Feature #7: Adaptive threshold (adjusts based on confidence + semantics)")
    print()
    print("-" * 80)
    print()

    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Mid-Sentence Hesitation',
            'ai_text': 'Ich möchte Ihnen... ähm... sagen dass',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',
            'description': 'Hesitation + dass → Low semantic (0.20) → Should WAIT'
        },
        {
            'name': 'Scenario 2: Natural Completion',
            'ai_text': 'Das Hotel hat 50 Zimmer.',
            'user_interrupts': True,
            'expected_action': 'interrupt_ai_immediately',
            'description': 'Complete sentence → High semantic (0.90) → Should INTERRUPT'
        },
        {
            'name': 'Scenario 3: Repair Pattern',
            'ai_text': 'Ich gehe zum... zur Schule',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',
            'description': 'Ellipsis → Low semantic (0.20) → Should WAIT'
        },
        {
            'name': 'Scenario 4: Multiple Fillers',
            'ai_text': 'Das ist... äh... also...',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',
            'description': 'Fillers → Low semantic (0.20) → Should WAIT'
        },
        {
            'name': 'Scenario 5: Fluent Completion',
            'ai_text': 'Vielen Dank für Ihren Anruf.',
            'user_interrupts': True,
            'expected_action': 'interrupt_ai_immediately',
            'description': 'Fluent complete → High semantic (0.90) → Should INTERRUPT'
        },
    ]

    # Test both versions
    print("Testing Fixed Baseline (Control - 80% expected):")
    print("-" * 80)

    baseline_vad = ExcellenceVADGerman(
        use_disfluency_detection=False,
        turn_end_threshold=0.60  # Fixed baseline
    )

    baseline_correct = 0
    for scenario in scenarios:
        user_frame = np.random.randn(160) * 0.3 if scenario['user_interrupts'] else np.zeros(160)
        ai_frame = np.random.randn(160) * 0.3

        result = baseline_vad.process_frame(
            user_frame=user_frame,
            ai_frame=ai_frame,
            ai_text=scenario['ai_text']
        )

        correct = result['action'] == scenario['expected_action']
        baseline_correct += correct

        status = "OK" if correct else "FAIL"
        print(f"  {status:4} {scenario['name']}: {result['action'][:20]:20} (prob={result['turn_end_prob']:.3f})")

    baseline_accuracy = (baseline_correct / len(scenarios)) * 100
    print()
    print(f"Baseline Accuracy: {baseline_accuracy:.1f}% ({baseline_correct}/{len(scenarios)})")
    print()
    print("=" * 80)
    print()

    # Now test with adaptive thresholds - need to manually integrate
    print("NOTE: Adaptive thresholds require integration into excellence_vad_german.py")
    print("Current test shows baseline performance only.")
    print()
    print("To enable Feature #7:")
    print("  1. Import AdaptiveThresholdManager in excellence_vad_german.py")
    print("  2. Add use_adaptive_threshold parameter")
    print("  3. Replace fixed threshold with adaptive_threshold_manager.get_threshold()")
    print()
    print("Expected improvement:")
    print("  - High semantic prob (0.90) → Threshold lowered to 0.50 → INTERRUPT")
    print("  - Low semantic prob (0.20) → Threshold raised to 0.70 → WAIT")
    print("  - Could improve Scenario 1 (currently fails due to user_speaking=False)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    test_with_adaptive_thresholds()
