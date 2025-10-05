"""
Test Intent Classifier with Real German Hotel Conversation Data

Honest evaluation of accuracy on realistic hotel scenarios
"""

import sys
sys.path.append('../human-speech-detection')

from intent_classifier_german import IntentClassifierGerman


def test_real_hotel_conversations():
    """Test with real hotel conversation scenarios"""

    classifier = IntentClassifierGerman()

    # Real hotel conversation examples (typical Sofia interactions)
    test_data = [
        # === HOTEL RECEPTION SCENARIOS ===
        {
            'text': "Guten Tag, Hotel Alpenblick",
            'expected_intent': 'greeting',
            'expected_gap': 50,
            'expected_fpp': True,
            'scenario': 'Reception greeting'
        },
        {
            'text': "Haben Sie ein Zimmer frei?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Availability question'
        },
        {
            'text': "Wann möchten Sie anreisen?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Check-in date question'
        },
        {
            'text': "Am Montag",
            'expected_intent': 'statement',  # Short answer
            'expected_gap': None,  # Variable
            'expected_fpp': False,
            'scenario': 'Date answer'
        },
        {
            'text': "Wie viele Personen?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Guest count question'
        },
        {
            'text': "Zwei Personen",
            'expected_intent': 'statement',
            'expected_gap': None,
            'expected_fpp': False,
            'scenario': 'Guest count answer'
        },

        # === BOOKING PROCESS ===
        {
            'text': "Möchten Sie das Zimmer buchen?",
            'expected_intent': 'offer',  # or question
            'expected_gap': 200,
            'expected_fpp': True,
            'scenario': 'Booking offer'
        },
        {
            'text': "Ja, bitte",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Acceptance'
        },
        {
            'text': "Das macht 180 Euro pro Nacht",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Price information'
        },
        {
            'text': "Ist Frühstück inklusive?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Breakfast question'
        },

        # === SERVICE REQUESTS ===
        {
            'text': "Können Sie mir helfen?",
            'expected_intent': 'question',  # or request
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Help request'
        },
        {
            'text': "Natürlich, was benötigen Sie?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Service response'
        },
        {
            'text': "Ich brauche ein Taxi zum Flughafen",
            'expected_intent': 'request',
            'expected_gap': 250,
            'expected_fpp': True,
            'scenario': 'Taxi request'
        },

        # === EDGE CASES ===
        {
            'text': "Äh, ich denke... vielleicht morgen?",
            'expected_intent': 'discourse',  # Filler detected
            'expected_gap': 500,
            'expected_fpp': False,
            'scenario': 'Hesitation'
        },
        {
            'text': "Das Zimmer ist verfügbar",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Statement'
        },
        {
            'text': "Vielen Dank für Ihren Anruf",
            'expected_intent': 'social',
            'expected_gap': 150,
            'expected_fpp': False,
            'scenario': 'Thanks'
        },
        {
            'text': "Auf Wiederhören",
            'expected_intent': 'closing',
            'expected_gap': 200,
            'expected_fpp': True,
            'scenario': 'Call closing'
        },

        # === COMPLEX CASES ===
        {
            'text': "Haben Sie noch weitere Fragen?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Closing question'
        },
        {
            'text': "Nein, danke",
            'expected_intent': 'response',
            'expected_gap': 150,
            'expected_fpp': False,
            'scenario': 'Negative response'
        },
        {
            'text': "Das Zimmer kostet zweihundert Euro",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Price statement'
        },
    ]

    print("=" * 80)
    print(" REAL DATA TEST: German Hotel Conversations")
    print("=" * 80)
    print()

    correct = 0
    total = len(test_data)

    errors = []

    for i, test in enumerate(test_data, 1):
        result = classifier.classify(test['text'])

        # Check intent type
        intent_match = result.intent_type == test['expected_intent']

        # Check FPP status
        fpp_match = result.is_fpp == test['expected_fpp']

        # Check gap (allow 50ms tolerance)
        gap_match = True
        if test['expected_gap']:
            gap_diff = abs(result.expected_gap_ms - test['expected_gap'])
            gap_match = gap_diff <= 50

        # Overall match
        is_correct = intent_match and fpp_match and gap_match

        if is_correct:
            correct += 1
            status = "[OK]"
        else:
            status = "[X]"
            errors.append({
                'text': test['text'],
                'expected': test['expected_intent'],
                'got': result.intent_type,
                'scenario': test['scenario']
            })

        print(f"{status} Test {i}/{total}: {test['scenario']}")
        print(f"   Text: \"{test['text']}\"")
        print(f"   Expected: {test['expected_intent']}, Gap: {test['expected_gap']}ms, FPP: {test['expected_fpp']}")
        print(f"   Got:      {result.intent_type}/{result.intent_subtype}, Gap: {result.expected_gap_ms}ms, FPP: {result.is_fpp}")

        if not is_correct:
            if not intent_match:
                print(f"   [ERROR] INTENT MISMATCH")
            if not fpp_match:
                print(f"   [ERROR] FPP STATUS MISMATCH")
            if not gap_match:
                print(f"   [ERROR] GAP TIMING MISMATCH")

        print()

    # Results
    accuracy = (correct / total) * 100
    print("=" * 80)
    print(f" RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)
    print()

    if errors:
        print("ERRORS TO FIX:")
        print("-" * 80)
        for error in errors:
            print(f"Scenario: {error['scenario']}")
            print(f"  Text: \"{error['text']}\"")
            print(f"  Expected: {error['expected']}, Got: {error['got']}")
            print()

    # Honest assessment
    print("=" * 80)
    print(" HONEST ASSESSMENT")
    print("=" * 80)
    print()

    if accuracy >= 95:
        print("[EXCELLENT] 95%+ accuracy on real data")
        print("   Ready for production use")
    elif accuracy >= 85:
        print("[GOOD] 85-95% accuracy")
        print("   Minor pattern adjustments needed")
    elif accuracy >= 75:
        print("[FAIR] 75-85% accuracy")
        print("   Significant pattern improvements required")
    else:
        print("[NEEDS WORK] <75% accuracy")
        print("   Major redesign or ML approach recommended")

    print()
    print("Next steps:")
    print("  1. Fix identified pattern mismatches")
    print("  2. Add missing intent categories")
    print("  3. Test with actual Sofia conversation transcripts")
    print("  4. Integrate with prosody features for higher accuracy")
    print()

    return accuracy, errors


if __name__ == "__main__":
    accuracy, errors = test_real_hotel_conversations()

    print()
    print("=" * 80)
    print(f"FINAL SCORE: {accuracy:.1f}%")
    print("=" * 80)
