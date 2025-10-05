"""
Test Excellence VAD v2.0 - LLM-Based Turn-Taking Detection
===========================================================

Comprehensive test suite comparing:
1. v1.0 (regex) vs v2.0 (LLM forward model)
2. v2.0 performance on diverse scenarios
3. Human-like prediction accuracy

Target: 95%+ accuracy matching human telephone performance
"""

import numpy as np
from excellence_vad_v2 import ExcellenceVADv2
import time


def generate_realistic_audio(duration, f0_start, f0_end, modulation_rate, energy_decay=1.0, sr=16000):
    """
    Generate realistic audio with natural prosody

    Args:
        duration: Length in seconds
        f0_start: Starting pitch (Hz)
        f0_end: Ending pitch (Hz) - falling = complete, rising = question
        modulation_rate: Amplitude modulation frequency
        energy_decay: Energy drop at end (1.0 = no decay, 0.5 = 50% drop)
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Pitch contour (fundamental frequency)
    f0 = np.linspace(f0_start, f0_end, len(t))

    # Rich harmonic signal (like human voice)
    signal = np.sin(2 * np.pi * f0 * t)          # Fundamental
    signal += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)  # Second harmonic
    signal += 0.3 * np.sin(2 * np.pi * 3 * f0 * t)  # Third harmonic
    signal += 0.1 * np.sin(2 * np.pi * 500 * t)     # Formant 1
    signal += 0.1 * np.sin(2 * np.pi * 1500 * t)    # Formant 2

    # Amplitude modulation (syllable rhythm)
    modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * modulation_rate * t))
    signal *= modulation

    # Energy decay at end (natural sentence ending)
    if energy_decay < 1.0:
        decay_envelope = np.linspace(1.0, energy_decay, len(t))
        signal *= decay_envelope

    # Normalize
    return signal / np.max(np.abs(signal))


def test_semantic_predictions():
    """Test LLM semantic predictions vs regex"""
    print("=" * 80)
    print(" SEMANTIC PREDICTION COMPARISON: LLM vs Regex")
    print("=" * 80)
    print()

    # Test cases with ground truth
    test_cases = [
        # (text, expected_complete, description)
        ("I think we should go there tomorrow", True, "Complete declarative with time"),
        ("I think we should go there and", False, "Incomplete with conjunction"),
        ("What time is it?", True, "Complete question"),
        ("What time is", False, "Incomplete question"),
        ("Yes", True, "Single word acknowledgment"),
        ("Let me", False, "Incomplete auxiliary"),
        ("I am going to the store", True, "Complete action statement"),
        ("I am going to", False, "Incomplete preposition"),
        ("Because I think that", False, "Incomplete subordinate clause"),
        ("Okay thanks", True, "Polite closing"),
        ("I need to check with my", False, "Incomplete possessive"),
        ("The meeting is scheduled for next Tuesday", True, "Complete with specific time"),
        ("The meeting is scheduled for", False, "Incomplete preposition"),
        ("Can you help me with this project?", True, "Complete request question"),
        ("Can you help me with", False, "Incomplete request"),
        ("Perfect", True, "Single word completion"),
        ("So I was thinking we could", False, "Incomplete thought"),
        ("I'll call you back tomorrow morning", True, "Complete future action"),
        ("I'll call you back", True, "Complete future action (shorter)"),
        ("I'll call you", False, "Incomplete (missing object/time)"),
    ]

    vad_llm = ExcellenceVADv2(use_llm=True)
    vad_regex = ExcellenceVADv2(use_llm=False)

    llm_correct = 0
    regex_correct = 0

    print(f"{'Text':<50} {'Expected':<12} {'LLM':<12} {'Regex':<12}")
    print("-" * 86)

    for text, expected, description in test_cases:
        # LLM prediction
        llm_result = vad_llm.semantic_detector.predict_completion(text)
        llm_pred = llm_result['complete_prob'] > 0.6
        llm_match = llm_pred == expected

        # Regex prediction
        regex_result = vad_regex.semantic_detector.is_complete(text)
        regex_pred = regex_result['complete_prob'] > 0.6
        regex_match = regex_pred == expected

        if llm_match:
            llm_correct += 1
        if regex_match:
            regex_correct += 1

        expected_str = "COMPLETE" if expected else "INCOMPLETE"
        llm_str = f"{'COMPLETE' if llm_pred else 'INCOMPLETE'} ({llm_result['complete_prob']:.0%})"
        regex_str = f"{'COMPLETE' if regex_pred else 'INCOMPLETE'} ({regex_result['complete_prob']:.0%})"

        llm_mark = "OK" if llm_match else "XX"
        regex_mark = "OK" if regex_match else "XX"

        print(f"{text:<50} {expected_str:<12} {llm_str:<12} {llm_mark}  {regex_str:<12} {regex_mark}")

    print("-" * 86)
    print()
    print(f"LLM Accuracy: {llm_correct}/{len(test_cases)} ({llm_correct/len(test_cases):.1%})")
    print(f"Regex Accuracy: {regex_correct}/{len(test_cases)} ({regex_correct/len(test_cases):.1%})")
    print()

    improvement = llm_correct - regex_correct
    if improvement > 0:
        print(f"✓ LLM is {improvement} predictions better (+{improvement/len(test_cases)*100:.1f}%)")
    elif improvement < 0:
        print(f"✗ Regex is {-improvement} predictions better")
    else:
        print("= Both methods tied")

    print()
    return llm_correct / len(test_cases), regex_correct / len(test_cases)


def test_realistic_scenarios():
    """Test v2.0 on realistic conversation scenarios"""
    print("=" * 80)
    print(" REALISTIC SCENARIO TESTING (v2.0)")
    print("=" * 80)
    print()

    sr = 16000
    vad = ExcellenceVADv2(sample_rate=sr, use_llm=True)

    scenarios = []

    # Scenario 1: Complete statement with natural ending
    scenarios.append({
        'name': 'Complete Statement (Natural Ending)',
        'audio': generate_realistic_audio(2.0, 150, 130, 5.0, energy_decay=0.6),
        'text': "I think we should schedule the meeting for next Tuesday",
        'expected_turn_end': 'HIGH',  # Should be >70%
        'description': 'Complete declarative with time reference + falling pitch + energy decay'
    })

    # Scenario 2: Incomplete with conjunction
    scenarios.append({
        'name': 'Incomplete Statement (Conjunction)',
        'audio': generate_realistic_audio(1.5, 150, 155, 5.0, energy_decay=1.0),
        'text': "I think we should schedule the meeting and",
        'expected_turn_end': 'LOW',  # Should be <50%
        'description': 'Incomplete with conjunction + level pitch + no energy decay'
    })

    # Scenario 3: Question (complete)
    scenarios.append({
        'name': 'Complete Question',
        'audio': generate_realistic_audio(1.2, 150, 180, 5.5, energy_decay=0.9),
        'text': "What time would you like to meet?",
        'expected_turn_end': 'HIGH',  # Should be >80%
        'description': 'Question with rising pitch + slight energy decay'
    })

    # Scenario 4: Incomplete question
    scenarios.append({
        'name': 'Incomplete Question',
        'audio': generate_realistic_audio(1.0, 150, 160, 5.0, energy_decay=1.0),
        'text': "What time would you like to",
        'expected_turn_end': 'LOW',  # Should be <50%
        'description': 'Incomplete question + slight rise + no decay'
    })

    # Scenario 5: Short acknowledgment
    scenarios.append({
        'name': 'Short Acknowledgment',
        'audio': generate_realistic_audio(0.5, 150, 135, 5.0, energy_decay=0.5),
        'text': "Perfect",
        'expected_turn_end': 'HIGH',  # Should be >80%
        'description': 'Single word + falling pitch + strong energy decay'
    })

    # Scenario 6: Hesitation (incomplete)
    scenarios.append({
        'name': 'Hesitation Pattern',
        'audio': generate_realistic_audio(1.2, 155, 155, 4.5, energy_decay=0.9),
        'text': "I think the answer is um",
        'expected_turn_end': 'LOW',  # Should be <50%
        'description': 'Hesitation + level pitch + slight decay (uncertain)'
    })

    # Scenario 7: Complete with preposition ending
    scenarios.append({
        'name': 'Complete Prepositional Phrase',
        'audio': generate_realistic_audio(1.8, 150, 135, 5.0, energy_decay=0.6),
        'text': "I'll meet you at the coffee shop",
        'expected_turn_end': 'HIGH',  # Should be >70%
        'description': 'Complete statement ending with location + falling pitch'
    })

    total_tests = 0
    total_correct = 0

    for scenario in scenarios:
        print("-" * 80)
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Text: \"{scenario['text']}\"")
        print()

        vad.reset()

        audio = scenario['audio']
        text = scenario['text']
        frame_size = 160

        # Process frames
        results = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]

            # AI speaking
            result = vad.process_frame(
                np.zeros(frame_size),  # User silent
                frame,                  # AI speaking
                text
            )

            if result.get('ai_speaking'):
                results.append(result)

        if results:
            avg_turn_end = np.mean([r['turn_end_prob'] for r in results])
            avg_prosody = np.mean([r['prosody_prob'] for r in results])
            avg_semantic = np.mean([r['semantic_prob'] for r in results])

            # Get last result for details
            last = results[-1]

            print(f"Turn-End Probability: {avg_turn_end:.1%}")
            print(f"  Prosody: {avg_prosody:.1%}")
            print(f"  Semantic: {avg_semantic:.1%}")

            if 'llm_ending_prob' in last:
                print(f"  LLM Ending: {last['llm_ending_prob']:.1%}")
                print(f"  LLM Continuing: {last['llm_continuing_prob']:.1%}")

            # Check if passed
            if scenario['expected_turn_end'] == 'HIGH':
                threshold = 0.70
                passed = avg_turn_end >= threshold
                print(f"Expected: >{threshold:.0%} - {'PASS' if passed else 'FAIL'}")
            else:  # LOW
                threshold = 0.50
                passed = avg_turn_end < threshold
                print(f"Expected: <{threshold:.0%} - {'PASS' if passed else 'FAIL'}")

            if passed:
                total_correct += 1
            total_tests += 1

        print()

    # Results
    print("=" * 80)
    print(" FINAL RESULTS - REALISTIC SCENARIOS")
    print("=" * 80)
    print()

    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    print(f"Accuracy: {total_correct}/{total_tests} ({accuracy:.1f}%)")
    print()

    if accuracy >= 95:
        print(f"🎯 EXCELLENT: {accuracy:.1f}% - Matches human neuroscience target!")
    elif accuracy >= 85:
        print(f"✓ VERY GOOD: {accuracy:.1f}% - Strong performance")
    elif accuracy >= 70:
        print(f"○ GOOD: {accuracy:.1f}% - Acceptable performance")
    else:
        print(f"✗ NEEDS WORK: {accuracy:.1f}% - Further tuning required")

    print()

    # Performance stats
    stats = vad.get_stats()
    print("PERFORMANCE:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    if 'llm_avg_latency_ms' in stats:
        print(f"  LLM average: {stats['llm_avg_latency_ms']:.2f}ms")
        print(f"  LLM P95: {stats['llm_p95_latency_ms']:.2f}ms")
        print(f"  Cache hit rate: {stats['llm_cache_hit_rate']:.1%}")
    print()

    return accuracy


def main():
    """Run all tests"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  EXCELLENCE VAD v2.0 - COMPREHENSIVE TEST SUITE".center(78) + "*")
    print("*" + "  LLM-Based Human Brain-Matched Turn-Taking Detection".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    # Test 1: Semantic predictions
    llm_acc, regex_acc = test_semantic_predictions()

    # Test 2: Realistic scenarios
    scenario_acc = test_realistic_scenarios()

    # Summary
    print("=" * 80)
    print(" OVERALL SUMMARY")
    print("=" * 80)
    print()
    print(f"Semantic Prediction (LLM): {llm_acc:.1%}")
    print(f"Semantic Prediction (Regex): {regex_acc:.1%}")
    print(f"Realistic Scenarios: {scenario_acc:.1f}%")
    print()

    improvement = llm_acc - regex_acc
    print(f"LLM Improvement: +{improvement:.1%} over regex")
    print()

    if scenario_acc >= 95:
        print("🎯 v2.0 READY FOR PRODUCTION")
        print("   Matches human neuroscience target (95%+ accuracy)")
    elif scenario_acc >= 85:
        print("✓ v2.0 STRONG PERFORMANCE")
        print("   Exceeds v1.0, suitable for production with monitoring")
    else:
        print("○ v2.0 PROMISING")
        print("   Further tuning recommended before production")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
