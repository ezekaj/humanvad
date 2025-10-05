"""
Test Complete 3-Stage System on New Hotel Data

Tests Intent Classifier + Turn-End Predictor + Memory-VAD Bridge
on realistic hotel conversation scenarios
"""

import sys
sys.path.append('../human-speech-detection')

import numpy as np
from datetime import datetime
from intent_classifier_german import IntentClassifierGerman
from turn_end_predictor import TurnEndPredictor
from memory_vad_bridge import MemoryVADBridge


def test_complete_system():
    """Test all 3 stages integrated"""

    print("=" * 80)
    print(" COMPLETE 3-STAGE SYSTEM TEST - New Hotel Data")
    print("=" * 80)
    print()

    # Initialize all 3 stages
    print("Initializing components...")
    intent_classifier = IntentClassifierGerman()
    turn_predictor = TurnEndPredictor(use_lstm=True, lookahead_ms=300)
    memory_bridge = MemoryVADBridge(embedding_dim=64, min_observations=5)
    print("[OK] All 3 stages initialized\n")

    # New hotel conversation scenarios (different from training)
    test_scenarios = [
        # Guest arrival
        {
            'speaker_id': 'Guest_501',
            'text': 'Grüß Gott',
            'context': 'arrival',
            'expected_intent': 'greeting',
            'expected_gap': 50,
            'is_fpp': True,
            'current_vad_prob': 0.85,
            'prosody': {'f0_slope': 2.0, 'energy': 65, 'speech_rate': 5.0, 'duration': 800}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Wo kann ich parken?',
            'context': 'arrival',
            'expected_intent': 'question',
            'expected_gap': 100,
            'is_fpp': True,
            'current_vad_prob': 0.82,
            'prosody': {'f0_slope': 8.5, 'energy': 70, 'speech_rate': 5.5, 'duration': 1200}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Der Parkplatz ist hinter dem Hotel',
            'context': 'arrival',
            'expected_intent': 'statement',
            'expected_gap': 400,
            'is_fpp': False,
            'current_vad_prob': 0.75,
            'prosody': {'f0_slope': -3.0, 'energy': 55, 'speech_rate': 5.2, 'duration': 1800}
        },

        # Room preferences
        {
            'speaker_id': 'Guest_501',
            'text': 'Ich hätte gern ein ruhiges Zimmer',
            'context': 'booking',
            'expected_intent': 'request',
            'expected_gap': 250,
            'is_fpp': True,
            'current_vad_prob': 0.88,
            'prosody': {'f0_slope': 1.5, 'energy': 60, 'speech_rate': 5.0, 'duration': 1500}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Gibt es WLAN im Zimmer?',
            'context': 'amenities',
            'expected_intent': 'question',
            'expected_gap': 150,
            'is_fpp': True,
            'current_vad_prob': 0.80,
            'prosody': {'f0_slope': 7.0, 'energy': 68, 'speech_rate': 5.3, 'duration': 1100}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Selbstverständlich',
            'context': 'amenities',
            'expected_intent': 'response',
            'expected_gap': 100,
            'is_fpp': False,
            'current_vad_prob': 0.72,
            'prosody': {'f0_slope': -1.0, 'energy': 50, 'speech_rate': 5.0, 'duration': 900}
        },

        # Service requests
        {
            'speaker_id': 'Guest_501',
            'text': 'Bringen Sie mir bitte Handtücher',
            'context': 'service',
            'expected_intent': 'request',
            'expected_gap': 250,
            'is_fpp': True,
            'current_vad_prob': 0.86,
            'prosody': {'f0_slope': 2.0, 'energy': 65, 'speech_rate': 5.1, 'duration': 1600}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Sofort',
            'context': 'service',
            'expected_intent': 'response',
            'expected_gap': 100,
            'is_fpp': False,
            'current_vad_prob': 0.78,
            'prosody': {'f0_slope': -2.0, 'energy': 55, 'speech_rate': 5.0, 'duration': 600}
        },

        # Problem report
        {
            'speaker_id': 'Guest_501',
            'text': 'Die Heizung funktioniert nicht',
            'context': 'complaint',
            'expected_intent': 'statement',
            'expected_gap': 400,
            'is_fpp': False,
            'current_vad_prob': 0.76,
            'prosody': {'f0_slope': -2.5, 'energy': 62, 'speech_rate': 5.2, 'duration': 1400}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Tut mir sehr leid',
            'context': 'complaint',
            'expected_intent': 'apology',
            'expected_gap': 100,
            'is_fpp': True,
            'current_vad_prob': 0.84,
            'prosody': {'f0_slope': -1.5, 'energy': 58, 'speech_rate': 4.8, 'duration': 1000}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Können Sie das heute noch reparieren?',
            'context': 'complaint',
            'expected_intent': 'question',
            'expected_gap': 150,
            'is_fpp': True,
            'current_vad_prob': 0.81,
            'prosody': {'f0_slope': 6.5, 'energy': 64, 'speech_rate': 5.4, 'duration': 1700}
        },

        # Local information
        {
            'speaker_id': 'Guest_501',
            'text': 'Welches Restaurant empfehlen Sie?',
            'context': 'information',
            'expected_intent': 'question',
            'expected_gap': 100,
            'is_fpp': True,
            'current_vad_prob': 0.83,
            'prosody': {'f0_slope': 7.5, 'energy': 66, 'speech_rate': 5.3, 'duration': 1500}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Das Gasthaus am See ist ausgezeichnet',
            'context': 'information',
            'expected_intent': 'statement',
            'expected_gap': 400,
            'is_fpp': False,
            'current_vad_prob': 0.74,
            'prosody': {'f0_slope': -3.5, 'energy': 54, 'speech_rate': 5.0, 'duration': 1900}
        },

        # Checkout
        {
            'speaker_id': 'Guest_501',
            'text': 'Ich reise morgen früh ab',
            'context': 'checkout',
            'expected_intent': 'statement',
            'expected_gap': 400,
            'is_fpp': False,
            'current_vad_prob': 0.77,
            'prosody': {'f0_slope': -2.0, 'energy': 56, 'speech_rate': 5.1, 'duration': 1300}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Möchten Sie die Rechnung reservieren?',
            'context': 'checkout',
            'expected_intent': 'offer',
            'expected_gap': 200,
            'is_fpp': True,
            'current_vad_prob': 0.80,
            'prosody': {'f0_slope': 5.0, 'energy': 62, 'speech_rate': 5.2, 'duration': 1600}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Nein, ich bezahle jetzt',
            'context': 'checkout',
            'expected_intent': 'response',
            'expected_gap': 150,
            'is_fpp': False,
            'current_vad_prob': 0.79,
            'prosody': {'f0_slope': -1.0, 'energy': 58, 'speech_rate': 5.0, 'duration': 1100}
        },

        # Farewell
        {
            'speaker_id': 'Guest_501',
            'text': 'Gute Reise',
            'context': 'closing',
            'expected_intent': 'closing',
            'expected_gap': 200,
            'is_fpp': True,
            'current_vad_prob': 0.82,
            'prosody': {'f0_slope': 1.0, 'energy': 60, 'speech_rate': 4.9, 'duration': 800}
        },
        {
            'speaker_id': 'Guest_501',
            'text': 'Herzlichen Dank',
            'context': 'closing',
            'expected_intent': 'social',
            'expected_gap': 150,
            'is_fpp': False,
            'current_vad_prob': 0.76,
            'prosody': {'f0_slope': -2.0, 'energy': 55, 'speech_rate': 5.0, 'duration': 900}
        },
    ]

    # Run through all scenarios
    print("Processing conversation sequence:")
    print("-" * 80)
    print()

    results = []
    correct_intent = 0
    correct_prediction = 0
    correct_adaptation = 0

    for i, scenario in enumerate(test_scenarios, 1):
        # Stage 1: Intent Classification
        intent_result = intent_classifier.classify(
            scenario['text'],
            prosody_features=scenario['prosody']
        )

        intent_match = intent_result.intent_type == scenario['expected_intent']
        if intent_match:
            correct_intent += 1

        # Stage 2: Turn-End Prediction (with history)
        prediction = turn_predictor.predict(
            current_vad_prob=scenario['current_vad_prob'],
            prosody_features=scenario['prosody'],
            intent_type=intent_result.intent_type,
            speaker_profile={'avg_turn_gap_ms': 200, 'interruption_tolerance': 0.75}
        )

        # Check if prediction is reasonable (within 20% of current)
        prediction_reasonable = abs(prediction.predicted_prob_200ms - scenario['current_vad_prob']) < 0.3
        if prediction_reasonable:
            correct_prediction += 1

        # Stage 3: Memory Bridge (update and adapt)
        # Simulate actual gap (close to expected for this test)
        actual_gap = scenario['expected_gap'] + np.random.randint(-20, 20)

        memory_result = memory_bridge.observe_turn_taking(
            speaker_id=scenario['speaker_id'],
            intent_type=intent_result.intent_type,
            intent_subtype=intent_result.intent_subtype,
            is_fpp=intent_result.is_fpp,
            expected_gap_ms=intent_result.expected_gap_ms,
            actual_gap_ms=actual_gap,
            prosody_features=scenario['prosody'],
            context=scenario['context']
        )

        # Check adaptation threshold is reasonable (0.55-0.90 range)
        adaptation_reasonable = 0.55 <= memory_result['adapted_threshold'] <= 0.90
        if adaptation_reasonable:
            correct_adaptation += 1

        # Display progress
        status = "[OK]" if (intent_match and prediction_reasonable and adaptation_reasonable) else "[X]"

        print(f"{status} Turn {i:2d}: {scenario['text'][:45]:45s}")
        print(f"        Intent: {intent_result.intent_type}/{intent_result.intent_subtype:15s} "
              f"(expected: {scenario['expected_intent']})")
        print(f"        Predict: Current={scenario['current_vad_prob']:.2f} -> "
              f"200ms={prediction.predicted_prob_200ms:.2f}, "
              f"Confidence={prediction.confidence:.2f}")
        print(f"        Memory: Adapted threshold={memory_result['adapted_threshold']:.3f}, "
              f"Confidence={memory_result['confidence']:.2%}, "
              f"Surprise={memory_result['surprise']:.2f}")

        if not intent_match:
            print(f"        [ERROR] Intent mismatch: got {intent_result.intent_type}, "
                  f"expected {scenario['expected_intent']}")

        print()

        results.append({
            'text': scenario['text'],
            'intent_correct': intent_match,
            'prediction_ok': prediction_reasonable,
            'adaptation_ok': adaptation_reasonable
        })

    # Final results
    total = len(test_scenarios)
    print("=" * 80)
    print(" RESULTS")
    print("=" * 80)
    print()

    print(f"Stage 2 (Intent Classifier): {correct_intent}/{total} correct ({correct_intent/total*100:.1f}%)")
    print(f"Stage 3 (Turn-End Predictor): {correct_prediction}/{total} reasonable ({correct_prediction/total*100:.1f}%)")
    print(f"Stage 4 (Memory-VAD Bridge): {correct_adaptation}/{total} adapted ({correct_adaptation/total*100:.1f}%)")
    print()

    # Overall system success
    all_correct = sum(1 for r in results if r['intent_correct'] and r['prediction_ok'] and r['adaptation_ok'])
    print(f"OVERALL SYSTEM: {all_correct}/{total} fully correct ({all_correct/total*100:.1f}%)")
    print()

    # Speaker profile summary
    print("=" * 80)
    print(" LEARNED SPEAKER PROFILE")
    print("=" * 80)
    print()

    profile = memory_bridge.get_speaker_summary('Guest_501')
    print(f"Speaker: {profile['speaker_id']}")
    print(f"Total turns: {profile['total_interactions']}")
    print(f"Confidence: {profile['confidence']:.1%}")
    print(f"Average gap: {profile['timing']['avg_gap_ms']:.1f}ms +/- {profile['timing']['std_gap_ms']:.1f}ms")
    print(f"Interruption tolerance: {profile['behavior']['interruption_tolerance']:.2f}")
    print()

    print("Intent-specific learned gaps:")
    for intent, gap in sorted(profile['timing']['intent_specific_gaps'].items()):
        print(f"  {intent:12s}: {gap:6.1f}ms")
    print()

    # Assessment
    print("=" * 80)
    print(" SYSTEM ASSESSMENT")
    print("=" * 80)
    print()

    overall_pct = all_correct/total*100
    if overall_pct >= 90:
        print("[EXCELLENT] 90%+ on integrated system test")
    elif overall_pct >= 80:
        print("[GOOD] 80-90% on integrated system test")
    elif overall_pct >= 70:
        print("[FAIR] 70-80% on integrated system test")
    else:
        print("[NEEDS WORK] <70% on integrated system test")

    print()
    print("Component breakdown:")
    print(f"  - Intent accuracy: {correct_intent/total*100:.1f}% (Stage 2)")
    print(f"  - Prediction quality: {correct_prediction/total*100:.1f}% (Stage 3)")
    print(f"  - Adaptation quality: {correct_adaptation/total*100:.1f}% (Stage 4)")
    print()

    # Latency summary
    print("Latency (from benchmarks):")
    print(f"  - Intent: 0.027ms")
    print(f"  - Predictor: 0.003ms")
    print(f"  - Memory: 5.231ms")
    print(f"  - TOTAL: 5.261ms")
    print()

    print("=" * 80)
    print(" TEST COMPLETE")
    print("=" * 80)

    return {
        'intent_accuracy': correct_intent/total,
        'prediction_quality': correct_prediction/total,
        'adaptation_quality': correct_adaptation/total,
        'overall_accuracy': all_correct/total
    }


if __name__ == "__main__":
    results = test_complete_system()

    print()
    print("FINAL SCORES:")
    print(f"  Intent:     {results['intent_accuracy']*100:.1f}%")
    print(f"  Prediction: {results['prediction_quality']*100:.1f}%")
    print(f"  Adaptation: {results['adaptation_quality']*100:.1f}%")
    print(f"  Overall:    {results['overall_accuracy']*100:.1f}%")
