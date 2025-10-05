"""
Generate diverse German test scenarios from various domains
Not just hotel - includes customer service, medical, retail, etc.
"""

import random
from excellence_vad_german import ExcellenceVADGerman


def generate_diverse_german_scenarios():
    """
    Generate 50+ diverse German conversation scenarios
    From different domains: customer service, medical, retail, restaurant, travel
    """

    scenarios = [
        # === CUSTOMER SERVICE (Complete) ===
        {'text': 'Ihr Paket wird morgen zugestellt', 'expected': 'interrupt', 'domain': 'Customer Service'},
        {'text': 'Ich habe Ihre Bestellung gefunden', 'expected': 'interrupt', 'domain': 'Customer Service'},
        {'text': 'Die Lieferung kostet fünf Euro', 'expected': 'interrupt', 'domain': 'Customer Service'},
        {'text': 'Wir bearbeiten Ihre Anfrage sofort', 'expected': 'interrupt', 'domain': 'Customer Service'},
        {'text': 'Ihre Reklamation wurde angenommen', 'expected': 'interrupt', 'domain': 'Customer Service'},

        # === MEDICAL (Complete) ===
        {'text': 'Der Termin ist am Montag', 'expected': 'interrupt', 'domain': 'Medical'},
        {'text': 'Nehmen Sie die Tabletten täglich', 'expected': 'interrupt', 'domain': 'Medical'},
        {'text': 'Die Untersuchung dauert zwanzig Minuten', 'expected': 'interrupt', 'domain': 'Medical'},
        {'text': 'Ihr Befund ist unauffällig', 'expected': 'interrupt', 'domain': 'Medical'},
        {'text': 'Kommen Sie bitte um zehn Uhr', 'expected': 'interrupt', 'domain': 'Medical'},

        # === RETAIL (Complete) ===
        {'text': 'Das Produkt ist auf Lager', 'expected': 'interrupt', 'domain': 'Retail'},
        {'text': 'Wir haben drei Farben vorrätig', 'expected': 'interrupt', 'domain': 'Retail'},
        {'text': 'Die Größe passt perfekt', 'expected': 'interrupt', 'domain': 'Retail'},
        {'text': 'Zahlung per Karte ist möglich', 'expected': 'interrupt', 'domain': 'Retail'},
        {'text': 'Ihre Rechnung beträgt dreißig Euro', 'expected': 'interrupt', 'domain': 'Retail'},

        # === RESTAURANT (Complete) ===
        {'text': 'Ihr Tisch ist bereit', 'expected': 'interrupt', 'domain': 'Restaurant'},
        {'text': 'Die Speisekarte bringe ich sofort', 'expected': 'interrupt', 'domain': 'Restaurant'},
        {'text': 'Das Gericht ist vegetarisch', 'expected': 'interrupt', 'domain': 'Restaurant'},
        {'text': 'Möchten Sie ein Getränk bestellen', 'expected': 'interrupt', 'domain': 'Restaurant'},
        {'text': 'Guten Appetit', 'expected': 'interrupt', 'domain': 'Restaurant'},

        # === TRAVEL (Complete) ===
        {'text': 'Ihr Flug startet pünktlich', 'expected': 'interrupt', 'domain': 'Travel'},
        {'text': 'Das Gate ist B zwölf', 'expected': 'interrupt', 'domain': 'Travel'},
        {'text': 'Ihr Koffer wurde gefunden', 'expected': 'interrupt', 'domain': 'Travel'},
        {'text': 'Die Verbindung fährt stündlich', 'expected': 'interrupt', 'domain': 'Travel'},
        {'text': 'Steigen Sie am Hauptbahnhof um', 'expected': 'interrupt', 'domain': 'Travel'},

        # === BANKING (Complete) ===
        {'text': 'Ihr Konto ist gedeckt', 'expected': 'interrupt', 'domain': 'Banking'},
        {'text': 'Die Überweisung wurde ausgeführt', 'expected': 'interrupt', 'domain': 'Banking'},
        {'text': 'Ihre PIN ist gesperrt', 'expected': 'interrupt', 'domain': 'Banking'},
        {'text': 'Wir benötigen Ihren Ausweis', 'expected': 'interrupt', 'domain': 'Banking'},
        {'text': 'Der Zinssatz beträgt zwei Prozent', 'expected': 'interrupt', 'domain': 'Banking'},

        # === INCOMPLETE - CUSTOMER SERVICE ===
        {'text': 'Ihr Paket wird', 'expected': 'wait', 'domain': 'Customer Service'},
        {'text': 'Wir bearbeiten Ihre Anfrage und', 'expected': 'wait', 'domain': 'Customer Service'},
        {'text': 'Die Lieferung kostet fünf Euro aber', 'expected': 'wait', 'domain': 'Customer Service'},
        {'text': 'Ich möchte Ihnen mitteilen dass', 'expected': 'wait', 'domain': 'Customer Service'},

        # === INCOMPLETE - MEDICAL ===
        {'text': 'Der Termin ist am', 'expected': 'wait', 'domain': 'Medical'},
        {'text': 'Nehmen Sie die Tabletten', 'expected': 'wait', 'domain': 'Medical'},
        {'text': 'Die Untersuchung dauert zwanzig Minuten und', 'expected': 'wait', 'domain': 'Medical'},
        {'text': 'Ihr Befund ist', 'expected': 'wait', 'domain': 'Medical'},

        # === INCOMPLETE - RETAIL ===
        {'text': 'Das Produkt ist auf', 'expected': 'wait', 'domain': 'Retail'},
        {'text': 'Wir haben drei Farben aber', 'expected': 'wait', 'domain': 'Retail'},
        {'text': 'Zahlung per Karte ist möglich wenn', 'expected': 'wait', 'domain': 'Retail'},

        # === INCOMPLETE - RESTAURANT ===
        {'text': 'Ihr Tisch ist', 'expected': 'wait', 'domain': 'Restaurant'},
        {'text': 'Die Speisekarte bringe ich', 'expected': 'wait', 'domain': 'Restaurant'},
        {'text': 'Das Gericht ist vegetarisch oder', 'expected': 'wait', 'domain': 'Restaurant'},

        # === INCOMPLETE - TRAVEL ===
        {'text': 'Ihr Flug startet um', 'expected': 'wait', 'domain': 'Travel'},
        {'text': 'Das Gate ist', 'expected': 'wait', 'domain': 'Travel'},
        {'text': 'Die Verbindung fährt stündlich aber', 'expected': 'wait', 'domain': 'Travel'},

        # === INCOMPLETE - BANKING ===
        {'text': 'Ihr Konto ist', 'expected': 'wait', 'domain': 'Banking'},
        {'text': 'Die Überweisung wurde ausgeführt und', 'expected': 'wait', 'domain': 'Banking'},
        {'text': 'Wir benötigen Ihren', 'expected': 'wait', 'domain': 'Banking'},

        # === EDGE CASES - COMPLEX SENTENCES ===
        {'text': 'Wenn Sie möchten, kann ich Ihnen helfen', 'expected': 'interrupt', 'domain': 'General'},
        {'text': 'Falls Sie Fragen haben, rufen Sie an', 'expected': 'interrupt', 'domain': 'General'},
        {'text': 'Obwohl es teuer ist, lohnt es sich', 'expected': 'interrupt', 'domain': 'General'},
        {'text': 'Nachdem ich prüfe, sage ich Bescheid', 'expected': 'interrupt', 'domain': 'General'},

        # === EDGE CASES - WITH HESITATION ===
        {'text': 'Das ist äh korrekt', 'expected': 'wait', 'domain': 'General'},
        {'text': 'Wir haben... drei Optionen', 'expected': 'wait', 'domain': 'General'},
        {'text': 'Hmm, lassen Sie mich nachsehen', 'expected': 'wait', 'domain': 'General'},

        # === POLITE CLOSING PHRASES ===
        {'text': 'Vielen Dank für Ihr Verständnis', 'expected': 'interrupt', 'domain': 'General'},
        {'text': 'Schönen Tag noch', 'expected': 'interrupt', 'domain': 'General'},
        {'text': 'Auf Wiederhören', 'expected': 'interrupt', 'domain': 'General'},
        {'text': 'Bis bald', 'expected': 'interrupt', 'domain': 'General'},
    ]

    return scenarios


def test_diverse_german_data():
    """
    Test VAD with diverse German conversation data
    """

    print("=" * 80)
    print(" DIVERSE GERMAN DATA TEST - 60+ SCENARIOS")
    print("=" * 80)
    print()
    print("Testing scenarios from:")
    print("  - Customer Service")
    print("  - Medical appointments")
    print("  - Retail/Shopping")
    print("  - Restaurant/Food")
    print("  - Travel/Transportation")
    print("  - Banking/Finance")
    print("  - General conversation")
    print()
    print("-" * 80)
    print()

    scenarios = generate_diverse_german_scenarios()

    # Initialize VAD
    vad = ExcellenceVADGerman(turn_end_threshold=0.60)
    semantic_detector = vad.semantic_detector

    correct = 0
    results = []
    errors_by_domain = {}
    stats_by_domain = {}

    for scenario in scenarios:
        text = scenario['text']
        expected = scenario['expected']
        domain = scenario['domain']

        # Test semantic detector
        semantic_result = semantic_detector.is_complete(text)
        semantic_prob = semantic_result['complete_prob']

        # Compute final probability
        prosody_prob = 0.30
        final_prob = 0.45 * prosody_prob + 0.55 * semantic_prob

        # Determine action
        if final_prob >= 0.60:
            action = 'interrupt'
        else:
            action = 'wait'

        is_correct = action == expected
        correct += is_correct

        # Track by domain
        if domain not in stats_by_domain:
            stats_by_domain[domain] = {'correct': 0, 'total': 0}
        stats_by_domain[domain]['total'] += 1
        if is_correct:
            stats_by_domain[domain]['correct'] += 1

        if not is_correct:
            if domain not in errors_by_domain:
                errors_by_domain[domain] = []
            errors_by_domain[domain].append({
                'text': text,
                'expected': expected,
                'got': action,
                'semantic_prob': semantic_prob
            })

        status = "OK  " if is_correct else "FAIL"
        print(f"  {status} [{domain:18}] \"{text[:45]:45}\" sem={semantic_prob:.2f} -> {action:9}")

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

    # Breakdown by domain
    print("ACCURACY BY DOMAIN:")
    print("-" * 80)
    for domain in sorted(stats_by_domain.keys()):
        stats = stats_by_domain[domain]
        domain_acc = (stats['correct'] / stats['total']) * 100
        print(f"  {domain:20} {domain_acc:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")

    print()

    # Show errors if any
    if errors_by_domain:
        print("ERRORS BY DOMAIN:")
        print("-" * 80)
        for domain, errors in errors_by_domain.items():
            print(f"\n{domain}:")
            for error in errors:
                print(f"  FAIL: \"{error['text']}\"")
                print(f"        Expected: {error['expected']}, Got: {error['got']}, sem={error['semantic_prob']:.2f}")
        print()

    # Breakdown by type
    complete_scenarios = [r for r in results if r['scenario']['expected'] == 'interrupt']
    incomplete_scenarios = [r for r in results if r['scenario']['expected'] == 'wait']

    complete_correct = sum(1 for r in complete_scenarios if r['correct'])
    incomplete_correct = sum(1 for r in incomplete_scenarios if r['correct'])

    complete_acc = (complete_correct / len(complete_scenarios)) * 100 if complete_scenarios else 0
    incomplete_acc = (incomplete_correct / len(incomplete_scenarios)) * 100 if incomplete_scenarios else 0

    print("BREAKDOWN BY SENTENCE TYPE:")
    print("-" * 80)
    print(f"Complete sentences:   {complete_acc:.1f}% ({complete_correct}/{len(complete_scenarios)})")
    print(f"Incomplete sentences: {incomplete_acc:.1f}% ({incomplete_correct}/{len(incomplete_scenarios)})")
    print()

    print("=" * 80)

    if accuracy >= 95:
        print(f"[EXCELLENT] {accuracy:.1f}% accuracy - Production ready!")
    elif accuracy >= 90:
        print(f"[VERY GOOD] {accuracy:.1f}% accuracy - Minor tuning recommended")
    elif accuracy >= 80:
        print(f"[GOOD] {accuracy:.1f}% accuracy - Some improvements needed")
    else:
        print(f"[NEEDS WORK] {accuracy:.1f}% accuracy - Significant improvements needed")

    print("=" * 80)

    return accuracy, results


if __name__ == "__main__":
    print()
    print("TESTING WITH DIVERSE GERMAN CONVERSATION DATA")
    print("From multiple domains: customer service, medical, retail, etc.")
    print()

    accuracy, results = test_diverse_german_data()

    print()
