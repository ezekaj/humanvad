"""
Test Intent Classifier with NEW German Hotel Conversation Data

Fresh test cases to validate generalization
"""

import sys
sys.path.append('../human-speech-detection')

from intent_classifier_german import IntentClassifierGerman


def test_new_hotel_conversations():
    """Test with completely new hotel scenarios"""

    classifier = IntentClassifierGerman()

    # NEW test cases - different from training
    test_data = [
        # === ARRIVAL & CHECK-IN ===
        {
            'text': "Grüß Gott",
            'expected_intent': 'greeting',
            'expected_gap': 50,
            'expected_fpp': True,
            'scenario': 'Regional greeting'
        },
        {
            'text': "Wo kann ich parken?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Parking question'
        },
        {
            'text': "Der Parkplatz ist hinter dem Hotel",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Parking directions'
        },
        {
            'text': "Ich hätte gern ein ruhiges Zimmer",
            'expected_intent': 'request',
            'expected_gap': 250,
            'expected_fpp': True,
            'scenario': 'Room preference request'
        },

        # === ROOM SERVICE & AMENITIES ===
        {
            'text': "Gibt es WLAN im Zimmer?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'WiFi availability'
        },
        {
            'text': "Selbstverständlich",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Affirmative response'
        },
        {
            'text': "Bringen Sie mir bitte Handtücher",
            'expected_intent': 'request',
            'expected_gap': 250,
            'expected_fpp': True,
            'scenario': 'Housekeeping request'
        },
        {
            'text': "Sofort",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Quick acknowledgment'
        },

        # === COMPLAINTS & ISSUES ===
        {
            'text': "Die Heizung funktioniert nicht",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Problem report'
        },
        {
            'text': "Tut mir sehr leid",
            'expected_intent': 'apology',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Apology for issue'
        },
        {
            'text': "Können Sie das heute noch reparieren?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Repair timeline question'
        },

        # === LOCAL INFORMATION ===
        {
            'text': "Welches Restaurant empfehlen Sie?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Restaurant recommendation'
        },
        {
            'text': "Das Gasthaus am See ist ausgezeichnet",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Recommendation statement'
        },
        {
            'text': "Wie komme ich zum Bahnhof?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Direction question'
        },

        # === CHECKOUT & DEPARTURE ===
        {
            'text': "Ich reise morgen früh ab",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Departure announcement'
        },
        {
            'text': "Möchten Sie die Rechnung reservieren?",
            'expected_intent': 'offer',
            'expected_gap': 200,
            'expected_fpp': True,
            'scenario': 'Bill reservation offer'
        },
        {
            'text': "Nein, ich bezahle jetzt",
            'expected_intent': 'response',
            'expected_gap': 150,
            'expected_fpp': False,
            'scenario': 'Payment preference'
        },

        # === EDGE CASES ===
        {
            'text': "Also... hmm... vielleicht später?",
            'expected_intent': 'discourse',
            'expected_gap': 500,
            'expected_fpp': False,
            'scenario': 'Multiple fillers'
        },
        {
            'text': "Gute Reise",
            'expected_intent': 'closing',
            'expected_gap': 200,
            'expected_fpp': True,
            'scenario': 'Travel farewell'
        },
        {
            'text': "Herzlichen Dank",
            'expected_intent': 'social',
            'expected_gap': 150,
            'expected_fpp': False,
            'scenario': 'Warm thanks'
        },
    ]

    print("=" * 80)
    print(" NEW DATA TEST: Different German Hotel Scenarios")
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

    # Assessment
    print("=" * 80)
    print(" GENERALIZATION ASSESSMENT")
    print("=" * 80)
    print()

    if accuracy >= 95:
        print("[EXCELLENT] 95%+ accuracy - Great generalization!")
    elif accuracy >= 85:
        print("[GOOD] 85-95% accuracy - Good generalization with minor gaps")
    elif accuracy >= 75:
        print("[FAIR] 75-85% accuracy - Some overfitting to training data")
    else:
        print("[POOR] <75% accuracy - Significant overfitting detected")

    print()
    return accuracy, errors


if __name__ == "__main__":
    accuracy, errors = test_new_hotel_conversations()

    print()
    print("=" * 80)
    print(f"GENERALIZATION SCORE: {accuracy:.1f}%")
    print("=" * 80)
