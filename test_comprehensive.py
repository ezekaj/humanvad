"""
Comprehensive test of semantic detector with 30+ scenarios
Tests edge cases, various sentence types, and real-world hotel conversations
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman


def test_comprehensive():
    """
    Comprehensive test with 30+ German sentences
    """

    print("=" * 80)
    print(" COMPREHENSIVE SEMANTIC DETECTOR TEST")
    print("=" * 80)
    print()

    scenarios = [
        # === COMPLETE SENTENCES (interrupt) ===

        # Hotel-specific complete sentences
        {'text': 'Das Hotel hat fünfzig Zimmer', 'expected': 'interrupt', 'category': 'Hotel Info'},
        {'text': 'Der Preis beträgt zweihundert Euro', 'expected': 'interrupt', 'category': 'Price'},
        {'text': 'Das Zimmer ist verfügbar', 'expected': 'interrupt', 'category': 'Availability'},
        {'text': 'Frühstück ist inklusive', 'expected': 'interrupt', 'category': 'Amenities'},
        {'text': 'Check-in ist um vierzehn Uhr', 'expected': 'interrupt', 'category': 'Time'},

        # Polite phrases
        {'text': 'Vielen Dank für Ihren Anruf', 'expected': 'interrupt', 'category': 'Polite'},
        {'text': 'Guten Tag, wie kann ich Ihnen helfen', 'expected': 'interrupt', 'category': 'Greeting'},
        {'text': 'Sehr gerne, ich helfe Ihnen', 'expected': 'interrupt', 'category': 'Polite'},
        {'text': 'Auf Wiedersehen', 'expected': 'interrupt', 'category': 'Closing'},

        # Confirmations
        {'text': 'Ja, das ist korrekt', 'expected': 'interrupt', 'category': 'Confirmation'},
        {'text': 'Genau', 'expected': 'interrupt', 'category': 'Confirmation'},
        {'text': 'Perfekt', 'expected': 'interrupt', 'category': 'Confirmation'},
        {'text': 'Das ist richtig', 'expected': 'interrupt', 'category': 'Confirmation'},

        # Questions (complete)
        {'text': 'Haben Sie noch weitere Fragen', 'expected': 'interrupt', 'category': 'Question'},
        {'text': 'Kann ich Ihnen noch helfen', 'expected': 'interrupt', 'category': 'Question'},
        {'text': 'Möchten Sie buchen', 'expected': 'interrupt', 'category': 'Question'},

        # Statements with objects
        {'text': 'Ich helfe Ihnen gerne', 'expected': 'interrupt', 'category': 'Statement'},
        {'text': 'Wir haben drei Zimmer frei', 'expected': 'interrupt', 'category': 'Statement'},
        {'text': 'Das kostet hundert Euro pro Nacht', 'expected': 'interrupt', 'category': 'Statement'},

        # === INCOMPLETE SENTENCES (wait) ===

        # Ends with conjunction
        {'text': 'Ich möchte Ihnen sagen dass', 'expected': 'wait', 'category': 'Conjunction'},
        {'text': 'Das Zimmer ist verfügbar und', 'expected': 'wait', 'category': 'Conjunction'},
        {'text': 'Wir haben Zimmer aber', 'expected': 'wait', 'category': 'Conjunction'},
        {'text': 'Der Preis ist niedrig weil', 'expected': 'wait', 'category': 'Conjunction'},

        # Ends with preposition
        {'text': 'Ich gehe zur', 'expected': 'wait', 'category': 'Preposition'},
        {'text': 'Das Hotel ist in', 'expected': 'wait', 'category': 'Preposition'},
        {'text': 'Wir fahren mit', 'expected': 'wait', 'category': 'Preposition'},

        # Ends with article
        {'text': 'Ich nehme das', 'expected': 'wait', 'category': 'Article'},
        {'text': 'Haben Sie einen', 'expected': 'wait', 'category': 'Article'},

        # Ends with auxiliary verb
        {'text': 'Ich kann', 'expected': 'wait', 'category': 'Auxiliary'},
        {'text': 'Sie müssen', 'expected': 'wait', 'category': 'Auxiliary'},
        {'text': 'Wir haben', 'expected': 'wait', 'category': 'Auxiliary'},

        # Hesitation/fillers
        {'text': 'Das ist äh verfügbar', 'expected': 'wait', 'category': 'Filler'},
        {'text': 'Ich möchte... sagen', 'expected': 'wait', 'category': 'Ellipsis'},
        {'text': 'Hmm, das kostet', 'expected': 'wait', 'category': 'Filler'},

        # === EDGE CASES ===

        # Very short complete
        {'text': 'Ja', 'expected': 'interrupt', 'category': 'Short Complete'},
        {'text': 'Nein', 'expected': 'interrupt', 'category': 'Short Complete'},
        {'text': 'Danke', 'expected': 'interrupt', 'category': 'Short Complete'},

        # Numbers
        {'text': 'Zweihundert Euro', 'expected': 'interrupt', 'category': 'Number'},
        {'text': 'Fünfzig Zimmer', 'expected': 'interrupt', 'category': 'Number'},
    ]

    # Initialize VAD
    vad = ExcellenceVADGerman(turn_end_threshold=0.60)
    semantic_detector = vad.semantic_detector

    print(f"Testing {len(scenarios)} scenarios...")
    print("-" * 80)

    correct = 0
    results = []
    errors_by_category = {}

    for scenario in scenarios:
        text = scenario['text']
        expected = scenario['expected']
        category = scenario['category']

        # Test semantic detector directly
        semantic_result = semantic_detector.is_complete(text)
        semantic_prob = semantic_result['complete_prob']

        # Compute final turn-end probability (45% prosody + 55% semantic)
        prosody_prob = 0.30  # Neutral
        final_prob = 0.45 * prosody_prob + 0.55 * semantic_prob

        # Determine action based on threshold
        if final_prob >= 0.60:
            action = 'interrupt'
        else:
            action = 'wait'

        is_correct = action == expected
        correct += is_correct

        if not is_correct:
            if category not in errors_by_category:
                errors_by_category[category] = []
            errors_by_category[category].append({
                'text': text,
                'expected': expected,
                'got': action,
                'semantic_prob': semantic_prob
            })

        status = "OK  " if is_correct else "FAIL"

        print(f"  {status} [{category:15}] \"{text[:45]:45}\" sem={semantic_prob:.2f} -> {action:9}")

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
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{len(scenarios)})")
    print()

    # Breakdown by expected action
    complete_scenarios = [r for r in results if r['scenario']['expected'] == 'interrupt']
    incomplete_scenarios = [r for r in results if r['scenario']['expected'] == 'wait']

    complete_correct = sum(1 for r in complete_scenarios if r['correct'])
    incomplete_correct = sum(1 for r in incomplete_scenarios if r['correct'])

    complete_acc = (complete_correct / len(complete_scenarios)) * 100 if complete_scenarios else 0
    incomplete_acc = (incomplete_correct / len(incomplete_scenarios)) * 100 if incomplete_scenarios else 0

    print("BREAKDOWN BY TYPE:")
    print("-" * 80)
    print(f"Complete sentences:   {complete_acc:.1f}% ({complete_correct}/{len(complete_scenarios)})")
    print(f"Incomplete sentences: {incomplete_acc:.1f}% ({incomplete_correct}/{len(incomplete_scenarios)})")
    print()

    # Show errors by category
    if errors_by_category:
        print("ERRORS BY CATEGORY:")
        print("-" * 80)
        for category, errors in errors_by_category.items():
            print(f"\n{category}:")
            for error in errors:
                print(f"  FAIL: \"{error['text']}\"")
                print(f"        Expected: {error['expected']}, Got: {error['got']}, sem={error['semantic_prob']:.2f}")
        print()

    # Category breakdown
    categories = {}
    for r in results:
        cat = r['scenario']['category']
        if cat not in categories:
            categories[cat] = {'correct': 0, 'total': 0}
        categories[cat]['total'] += 1
        if r['correct']:
            categories[cat]['correct'] += 1

    print("ACCURACY BY CATEGORY:")
    print("-" * 80)
    for cat in sorted(categories.keys()):
        stats = categories[cat]
        cat_acc = (stats['correct'] / stats['total']) * 100
        print(f"  {cat:20} {cat_acc:5.1f}% ({stats['correct']}/{stats['total']})")

    print()
    print("=" * 80)

    return accuracy, results


if __name__ == "__main__":
    print()
    print("COMPREHENSIVE SEMANTIC DETECTOR TEST")
    print("Testing 30+ real-world German hotel conversation scenarios")
    print()

    accuracy, results = test_comprehensive()

    print()
    if accuracy >= 95:
        print(f"[EXCELLENT] {accuracy:.1f}% accuracy - Production ready!")
    elif accuracy >= 90:
        print(f"[VERY GOOD] {accuracy:.1f}% accuracy - Minor tuning recommended")
    elif accuracy >= 80:
        print(f"[GOOD] {accuracy:.1f}% accuracy - Some improvements needed")
    else:
        print(f"[NEEDS WORK] {accuracy:.1f}% accuracy - Significant improvements needed")
    print()
