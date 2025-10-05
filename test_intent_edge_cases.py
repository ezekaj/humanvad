"""
Test Intent Classifier with EDGE CASES & DIFFICULT Scenarios

Stress test with ambiguous, complex, and tricky German utterances
"""

import sys
sys.path.append('../human-speech-detection')

from intent_classifier_german import IntentClassifierGerman


def test_edge_cases():
    """Test with challenging edge cases"""

    classifier = IntentClassifierGerman()

    # DIFFICULT edge cases - ambiguous, complex, unusual
    test_data = [
        # === AMBIGUOUS CASES ===
        {
            'text': "Vielleicht",
            'expected_intent': 'statement',
            'expected_gap': 350,
            'expected_fpp': False,
            'scenario': 'Single word uncertainty'
        },
        {
            'text': "Mal sehen",
            'expected_intent': 'statement',
            'expected_gap': 350,
            'expected_fpp': False,
            'scenario': 'Informal deliberation'
        },
        {
            'text': "Könnte sein",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Uncertain confirmation'
        },
        {
            'text': "Eigentlich schon",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Hedged agreement'
        },

        # === COMPLEX MULTI-CLAUSE ===
        {
            'text': "Wenn Sie Zeit haben, können Sie mir helfen?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Conditional question'
        },
        {
            'text': "Ich weiß nicht, ob das möglich ist",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Embedded clause uncertainty'
        },
        {
            'text': "Das hängt davon ab",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Conditional statement'
        },

        # === INTERRUPTED/INCOMPLETE ===
        {
            'text': "Ich möchte...",
            'expected_intent': 'request',
            'expected_gap': 250,
            'expected_fpp': True,
            'scenario': 'Incomplete request'
        },
        {
            'text': "Können Sie... äh...",
            'expected_intent': 'discourse',
            'expected_gap': 500,
            'expected_fpp': False,
            'scenario': 'Interrupted with filler'
        },
        {
            'text': "Wann... also... wann genau?",
            'expected_intent': 'question',
            'expected_gap': 100,
            'expected_fpp': True,
            'scenario': 'Self-repair question'
        },

        # === SARCASM/RHETORICAL ===
        {
            'text': "Ach wirklich?",
            'expected_intent': 'discourse',
            'expected_gap': 500,
            'expected_fpp': False,
            'scenario': 'Sarcastic response'
        },
        {
            'text': "Na toll",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Ironic acknowledgment'
        },

        # === INDIRECT SPEECH ACTS ===
        {
            'text': "Es ist ziemlich kalt hier",
            'expected_intent': 'statement',
            'expected_gap': 400,
            'expected_fpp': False,
            'scenario': 'Indirect complaint'
        },
        {
            'text': "Ich würde gerne zahlen",
            'expected_intent': 'request',
            'expected_gap': 250,
            'expected_fpp': True,
            'scenario': 'Indirect request (polite)'
        },
        {
            'text': "Haben Sie vielleicht Zeit?",
            'expected_intent': 'question',
            'expected_gap': 150,
            'expected_fpp': True,
            'scenario': 'Hedged question'
        },

        # === MINIMAL UTTERANCES ===
        {
            'text': "Hmm",
            'expected_intent': 'discourse',
            'expected_gap': 500,
            'expected_fpp': False,
            'scenario': 'Pure filler'
        },
        {
            'text': "Ach so",
            'expected_intent': 'discourse',
            'expected_gap': 500,
            'expected_fpp': False,
            'scenario': 'Realization marker'
        },
        {
            'text': "Genau so",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Emphatic confirmation'
        },

        # === EMOTIONAL/EXCLAMATION ===
        {
            'text': "Toll!",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Exclamation'
        },
        {
            'text': "Endlich!",
            'expected_intent': 'response',
            'expected_gap': 100,
            'expected_fpp': False,
            'scenario': 'Relief exclamation'
        },
    ]

    print("=" * 80)
    print(" EDGE CASE TEST: Difficult & Ambiguous German Utterances")
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
    print(" EDGE CASE HANDLING")
    print("=" * 80)
    print()

    if accuracy >= 90:
        print("[EXCELLENT] 90%+ on edge cases - Robust classifier")
    elif accuracy >= 75:
        print("[GOOD] 75-90% on edge cases - Handles most ambiguity")
    elif accuracy >= 60:
        print("[FAIR] 60-75% on edge cases - Struggles with ambiguity")
    else:
        print("[POOR] <60% on edge cases - Needs better patterns")

    print()
    return accuracy, errors


if __name__ == "__main__":
    accuracy, errors = test_edge_cases()

    print()
    print("=" * 80)
    print(f"EDGE CASE SCORE: {accuracy:.1f}%")
    print("=" * 80)
