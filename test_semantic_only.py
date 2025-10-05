"""
Test SEMANTIC DETECTOR ONLY with real German speech
Bypasses ProductionVAD to focus on text-based turn-end detection
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman


def test_semantic_detector():
    """
    Test semantic completion detector with real German sentences
    """

    print("=" * 80)
    print(" SEMANTIC DETECTOR TEST (Text-Only)")
    print("=" * 80)
    print()

    scenarios = [
        {
            'text': 'Das Hotel hat fünfzig Zimmer',
            'expected': 'interrupt',
            'description': 'Complete sentence (hotel rooms)'
        },
        {
            'text': 'Vielen Dank für Ihren Anruf',
            'expected': 'interrupt',
            'description': 'Complete sentence (thank you)'
        },
        {
            'text': 'Guten Tag, wie kann ich Ihnen helfen',
            'expected': 'interrupt',
            'description': 'Complete greeting'
        },
        {
            'text': 'Ich möchte Ihnen sagen dass',
            'expected': 'wait',
            'description': 'Incomplete (ends with "dass")'
        },
        {
            'text': 'Ich gehe zur',
            'expected': 'wait',
            'description': 'Incomplete (ends with preposition)'
        },
        {
            'text': 'Der Preis beträgt zweihundert Euro',
            'expected': 'interrupt',
            'description': 'Complete with number'
        },
        {
            'text': 'Ja, das ist korrekt',
            'expected': 'interrupt',
            'description': 'Complete confirmation'
        },
        {
            'text': 'Das Zimmer ist verfügbar und',
            'expected': 'wait',
            'description': 'Incomplete (ends with "und")'
        },
        {
            'text': 'Haben Sie noch weitere Fragen',
            'expected': 'interrupt',
            'description': 'Complete question'
        },
        {
            'text': 'Sehr gerne, ich helfe Ihnen',
            'expected': 'interrupt',
            'description': 'Complete polite response'
        },
    ]

    # Initialize VAD
    vad = ExcellenceVADGerman(turn_end_threshold=0.60)

    # Access semantic detector directly
    semantic_detector = vad.semantic_detector

    print("Testing semantic completion patterns...")
    print("-" * 80)

    correct = 0
    results = []

    for scenario in scenarios:
        text = scenario['text']

        # Test semantic detector directly
        semantic_result = semantic_detector.is_complete(text)
        semantic_prob = semantic_result['complete_prob']

        # Compute final turn-end probability (45% prosody + 55% semantic)
        # For this test, assume prosody=0.30 (neutral)
        prosody_prob = 0.30
        final_prob = 0.45 * prosody_prob + 0.55 * semantic_prob

        # Determine action based on threshold
        if final_prob >= 0.60:
            action = 'interrupt'
        else:
            action = 'wait'

        is_correct = action == scenario['expected']
        correct += is_correct

        status = "OK  " if is_correct else "FAIL"

        print(f"  {status} {scenario['description']:40} sem={semantic_prob:.3f} final={final_prob:.3f} -> {action:10}")

        results.append({
            'scenario': scenario,
            'semantic_prob': semantic_prob,
            'final_prob': final_prob,
            'action': action,
            'correct': is_correct
        })

    print()
    print("-" * 80)

    accuracy = (correct / len(scenarios)) * 100
    print(f"Semantic Detector Accuracy: {accuracy:.0f}% ({correct}/{len(scenarios)})")
    print()

    # Show detailed analysis
    print("DETAILED ANALYSIS:")
    print("-" * 80)
    print()
    print("Complete sentences (should have HIGH semantic_prob > 0.7):")
    for r in results:
        if r['scenario']['expected'] == 'interrupt':
            mark = "OK" if r['correct'] else "FAIL"
            print(f"  {mark} \"{r['scenario']['text'][:50]:50}\" sem={r['semantic_prob']:.3f}")

    print()
    print("Incomplete sentences (should have LOW semantic_prob < 0.3):")
    for r in results:
        if r['scenario']['expected'] == 'wait':
            mark = "OK" if r['correct'] else "FAIL"
            print(f"  {mark} \"{r['scenario']['text'][:50]:50}\" sem={r['semantic_prob']:.3f}")

    print()
    print("=" * 80)

    return accuracy, results


if __name__ == "__main__":
    print()
    print("SEMANTIC DETECTOR TEST (Bypassing ProductionVAD)")
    print()

    accuracy, results = test_semantic_detector()

    print()
    print("NOTE: This test focuses on semantic patterns only.")
    print("ProductionVAD (audio) is not tested here.")
    print()
