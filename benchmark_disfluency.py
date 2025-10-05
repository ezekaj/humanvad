"""
Benchmark: Feature #5 - Disfluency Detection
Compares baseline vs disfluency-enhanced turn-taking detection
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman


def benchmark_disfluency_detection():
    """
    Test Feature #5 (Disfluency Detection) vs Baseline

    Same 5 scenarios as previous benchmarks
    """

    print("=" * 80)
    print(" FEATURE #5: DISFLUENCY DETECTION BENCHMARK")
    print("=" * 80)
    print()
    print("Comparing:")
    print("  Baseline: No disfluency detection")
    print("  Feature #5: Hesitation-aware turn-taking")
    print()
    print("-" * 80)
    print()

    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Mid-Sentence Hesitation',
            'ai_text': 'Ich möchte Ihnen... ähm... sagen dass',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',  # DON'T interrupt hesitation
            'description': 'AI hesitates with "ähm" - should NOT allow interrupt'
        },
        {
            'name': 'Scenario 2: Natural Completion',
            'ai_text': 'Das Hotel hat 50 Zimmer.',
            'user_interrupts': True,
            'expected_action': 'interrupt_ai_immediately',  # ALLOW interrupt
            'description': 'Complete sentence, no hesitation - ALLOW interrupt'
        },
        {
            'name': 'Scenario 3: Repair Pattern',
            'ai_text': 'Ich gehe zum... zur Schule',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',  # DON'T interrupt repair
            'description': 'Self-correction detected - should NOT allow interrupt'
        },
        {
            'name': 'Scenario 4: Multiple Fillers',
            'ai_text': 'Das ist... äh... also...',
            'user_interrupts': True,
            'expected_action': 'wait_for_ai_completion',  # DON'T interrupt
            'description': 'Multiple fillers - strong hesitation - DON\'T interrupt'
        },
        {
            'name': 'Scenario 5: Fluent Completion',
            'ai_text': 'Vielen Dank für Ihren Anruf.',
            'user_interrupts': True,
            'expected_action': 'interrupt_ai_immediately',  # ALLOW interrupt
            'description': 'Fluent, complete - ALLOW interrupt'
        },
    ]

    # Initialize VAD instances
    vad_baseline = ExcellenceVADGerman(
        use_disfluency_detection=False
    )
    vad_disfluency = ExcellenceVADGerman(
        use_disfluency_detection=True
    )

    # Test both versions
    baseline_correct = 0
    disfluency_correct = 0

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

        # Test Baseline
        result_baseline = vad_baseline.process_frame(
            user_frame=user_frame,
            ai_frame=ai_frame,
            ai_text=scenario['ai_text']
        )

        # Test Feature #5
        result_disfluency = vad_disfluency.process_frame(
            user_frame=user_frame,
            ai_frame=ai_frame,
            ai_text=scenario['ai_text']
        )

        # Check correctness
        baseline_match = result_baseline['action'] == scenario['expected_action']
        disfluency_match = result_disfluency['action'] == scenario['expected_action']

        if baseline_match:
            baseline_correct += 1
        if disfluency_match:
            disfluency_correct += 1

        # Print results
        print(f"  Baseline:")
        print(f"    Action: {result_baseline['action']}")
        print(f"    Turn-End Prob: {result_baseline['turn_end_prob']:.2f}")
        print(f"    Semantic Prob: {result_baseline['semantic_prob']:.2f}")
        print(f"    Correct: {'YES' if baseline_match else 'NO'}")
        print()

        print(f"  Feature #5 (Disfluency):")
        print(f"    Action: {result_disfluency['action']}")
        print(f"    Turn-End Prob: {result_disfluency['turn_end_prob']:.2f}")
        print(f"    Semantic Prob (adjusted): {result_disfluency['semantic_prob']:.2f}")
        if 'hesitation_prob' in result_disfluency:
            print(f"    Hesitation Prob: {result_disfluency['hesitation_prob']:.2f}")
            print(f"    Detected Fillers: {result_disfluency['detected_fillers']}")
        print(f"    Correct: {'YES' if disfluency_match else 'NO'}")
        print()
        print("-" * 80)
        print()

    # Final results
    total_scenarios = len(scenarios)
    baseline_accuracy = (baseline_correct / total_scenarios) * 100
    disfluency_accuracy = (disfluency_correct / total_scenarios) * 100
    improvement = disfluency_accuracy - baseline_accuracy

    print()
    print("=" * 80)
    print(" RESULTS")
    print("=" * 80)
    print()
    print(f"Baseline Accuracy (No Disfluency): {baseline_accuracy:.1f}% ({baseline_correct}/{total_scenarios})")
    print(f"Feature #5 Accuracy (Disfluency): {disfluency_accuracy:.1f}% ({disfluency_correct}/{total_scenarios})")
    print()
    print(f"Improvement: {improvement:+.1f}%")
    print()

    # Verdict
    if improvement >= 15:
        print("[SUCCESS] Feature #5 provides significant improvement (>15%)")
        print("Recommendation: INTEGRATE into Excellence VAD")
    elif improvement > 0:
        print("[PARTIAL SUCCESS] Feature #5 provides some improvement")
        print(f"Recommendation: Consider integration if {improvement:.1f}% is acceptable")
    else:
        print("[FAILURE] Feature #5 does not improve accuracy")
        print("Recommendation: DO NOT integrate")

    print()
    print("=" * 80)

    # Performance check
    avg_latency_baseline = np.mean(vad_baseline.processing_times)
    avg_latency_disfluency = np.mean(vad_disfluency.processing_times)
    latency_overhead = avg_latency_disfluency - avg_latency_baseline

    print()
    print("Performance:")
    print(f"  Baseline Latency: {avg_latency_baseline:.4f}ms")
    print(f"  Feature #5 Latency: {avg_latency_disfluency:.4f}ms")
    print(f"  Overhead: +{latency_overhead:.4f}ms")
    print()

    if latency_overhead < 1.0:
        print("[PASS] Latency overhead <1ms")
    else:
        print("[FAIL] Latency overhead >1ms")

    print()
    print("=" * 80)


if __name__ == "__main__":
    benchmark_disfluency_detection()
