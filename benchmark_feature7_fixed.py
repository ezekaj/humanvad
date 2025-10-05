"""
Feature #7 Re-Test with FIXED Baseline
Tests adaptive thresholds with corrected 80% baseline
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman


def test_feature_7():
    print("=" * 80)
    print(" FEATURE #7: ADAPTIVE THRESHOLDS - RE-TEST WITH FIXED BASELINE")
    print("=" * 80)
    print()
    print("Previous test (broken baseline): 40% -> 40% (0% improvement)")
    print("Current baseline (fixed): 80% (4/5 scenarios)")
    print()
    print("Testing:")
    print("  Baseline: Fixed threshold 0.60")
    print("  Feature #7: Adaptive threshold")
    print()
    print("-" * 80)
    print()

    scenarios = [
        ('Scenario 1: Hesitation', 'Ich mochte Ihnen... ahm... sagen dass', True, 'wait_for_ai_completion'),
        ('Scenario 2: Complete', 'Das Hotel hat 50 Zimmer.', True, 'interrupt_ai_immediately'),
        ('Scenario 3: Repair', 'Ich gehe zum... zur Schule', True, 'wait_for_ai_completion'),
        ('Scenario 4: Fillers', 'Das ist... ah... also...', True, 'wait_for_ai_completion'),
        ('Scenario 5: Complete', 'Vielen Dank fur Ihren Anruf.', True, 'interrupt_ai_immediately'),
    ]

    # Test baseline
    print("BASELINE (Fixed threshold=0.60):")
    print("-" * 80)
    vad_baseline = ExcellenceVADGerman(use_adaptive_threshold=False)
    baseline_correct = 0

    for name, text, user_int, expected in scenarios:
        user_frame = np.random.randn(160) * 0.3 if user_int else np.zeros(160)
        ai_frame = np.random.randn(160) * 0.3
        result = vad_baseline.process_frame(user_frame, ai_frame, text)

        correct = result['action'] == expected
        baseline_correct += correct
        status = "OK  " if correct else "FAIL"

        print(f"  {status} {name[:30]:30} prob={result['turn_end_prob']:.3f} -> {result['action'][:20]}")

    baseline_acc = (baseline_correct / len(scenarios)) * 100
    print(f"\nBaseline: {baseline_acc:.0f}% ({baseline_correct}/{len(scenarios)})")
    print()

    # Test with Feature #7
    print("FEATURE #7 (Adaptive threshold):")
    print("-" * 80)
    vad_adaptive = ExcellenceVADGerman(use_adaptive_threshold=True)
    adaptive_correct = 0

    for name, text, user_int, expected in scenarios:
        user_frame = np.random.randn(160) * 0.3 if user_int else np.zeros(160)
        ai_frame = np.random.randn(160) * 0.3
        result = vad_adaptive.process_frame(user_frame, ai_frame, text)

        correct = result['action'] == expected
        adaptive_correct += correct
        status = "OK  " if correct else "FAIL"

        # Get threshold info if available
        threshold_info = ""
        if 'adaptive_threshold' in result:
            threshold_info = f" thresh={result['adaptive_threshold']:.3f}"

        print(f"  {status} {name[:30]:30} prob={result['turn_end_prob']:.3f}{threshold_info} -> {result['action'][:20]}")

    adaptive_acc = (adaptive_correct / len(scenarios)) * 100
    improvement = adaptive_acc - baseline_acc

    print(f"\nFeature #7: {adaptive_acc:.0f}% ({adaptive_correct}/{len(scenarios)})")
    print()
    print("=" * 80)
    print(" RESULTS")
    print("=" * 80)
    print(f"Baseline:   {baseline_acc:.0f}% ({baseline_correct}/{len(scenarios)})")
    print(f"Feature #7: {adaptive_acc:.0f}% ({adaptive_correct}/{len(scenarios)})")
    print(f"Improvement: {improvement:+.0f}%")
    print()

    if improvement >= 10:
        print("[SUCCESS] Feature #7 provides significant improvement")
        print("Recommendation: INTEGRATE")
    elif improvement > 0:
        print("[PARTIAL] Feature #7 provides some improvement")
        print(f"Recommendation: Consider if {improvement:.0f}% gain is worth complexity")
    else:
        print("[NO IMPROVEMENT] Feature #7 does not improve accuracy")
        print("Recommendation: DO NOT integrate")

    print("=" * 80)


if __name__ == "__main__":
    test_feature_7()
