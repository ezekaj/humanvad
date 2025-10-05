"""
Test Excellence VAD - Deutsche Version
======================================

Comprehensive test suite for German language turn-taking detection
"""

import numpy as np
from excellence_vad_german import ExcellenceVADGerman, SemanticCompletionDetectorGerman
import time


def test_german_semantic_patterns():
    """Test German-specific semantic completion patterns"""
    print("=" * 80)
    print(" DEUTSCHE SEMANTIK-MUSTER TEST")
    print("=" * 80)
    print()

    detector = SemanticCompletionDetectorGerman()

    # Test cases: (text, expected_complete, description)
    test_cases = [
        # Vollständige Sätze
        ("Ich gehe zum Laden", True, "Complete action with location"),
        ("Das Hotel ist sehr schön", True, "Complete declarative"),
        ("Wie spät ist es?", True, "Complete question"),
        ("Ja natürlich", True, "Confirmation"),
        ("Okay danke", True, "Polite closing"),
        ("Perfekt", True, "Single word acknowledgment"),
        ("Ich treffe Sie morgen am Bahnhof", True, "Complete with time + location"),
        ("Wir haben drei Zimmertypen verfügbar", True, "Complete statement"),
        ("Das Frühstück wird von 7 bis 10 Uhr serviert", True, "Complete with time range"),

        # Unvollständige Sätze
        ("Ich gehe zum", False, "Incomplete preposition"),
        ("Weil ich denke dass", False, "Incomplete conjunction"),
        ("Lass mich", False, "Incomplete auxiliary"),
        ("Ich möchte", False, "Incomplete desire"),
        ("Der Preis ist", False, "Incomplete statement"),
        ("Können Sie", False, "Incomplete question"),
        ("Ich bin in der", False, "Incomplete location"),
        ("Das Zimmer hat und", False, "Incomplete conjunction"),
        ("Wenn Sie möchten", False, "Incomplete conditional"),

        # Grenzfälle
        ("Ich gehe", True, "Very short but complete"),
        ("Verstanden", True, "Single word complete"),
        ("Moment bitte", True, "Short polite phrase"),
        ("Das ist gut und", False, "Complete + conjunction"),
        ("Am Montag", True, "Time expression alone"),
        ("In Berlin", True, "Location alone can be complete"),
    ]

    correct = 0
    total = len(test_cases)

    print(f"{'Text':<50} {'Expected':<12} {'Result':<12} {'Score':<8} {'Status'}")
    print("-" * 90)

    for text, expected, description in test_cases:
        result = detector.is_complete(text)
        predicted = result['complete_prob'] > 0.6
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        expected_str = "VOLLSTÄNDIG" if expected else "UNVOLLSTÄNDIG"
        predicted_str = "VOLLSTÄNDIG" if predicted else "UNVOLLSTÄNDIG"
        status = "OK" if is_correct else "XX"

        print(f"{text:<50} {expected_str:<12} {predicted_str:<12} {result['complete_prob']:.0%}      {status}")

    print("-" * 90)
    accuracy = (correct / total) * 100
    print(f"\nGenauigkeit: {correct}/{total} ({accuracy:.1f}%)")
    print()

    if accuracy >= 90:
        print(f"AUSGEZEICHNET: {accuracy:.1f}% - Produktionsbereit!")
    elif accuracy >= 80:
        print(f"SEHR GUT: {accuracy:.1f}% - Starke Leistung")
    elif accuracy >= 70:
        print(f"GUT: {accuracy:.1f}% - Akzeptabel")
    else:
        print(f"VERBESSERUNG NÖTIG: {accuracy:.1f}%")

    print()
    return accuracy


def test_german_full_system():
    """Test complete German VAD system with audio"""
    print("=" * 80)
    print(" VOLLSTÄNDIGER SYSTEM-TEST (Audio + Text)")
    print("=" * 80)
    print()

    sr = 16000
    vad = ExcellenceVADGerman(sample_rate=sr)

    # Test scenarios
    scenarios = [
        {
            'name': 'Höfliche Anfrage (Vollständig)',
            'text': 'Ich hätte gerne ein Zimmer für zwei Nächte',
            'duration': 1.5,
            'energy_decay': 0.6,
            'expected': 'HIGH'
        },
        {
            'name': 'Unvollständige Frage',
            'text': 'Können Sie mir sagen ob',
            'duration': 1.0,
            'energy_decay': 1.0,
            'expected': 'LOW'
        },
        {
            'name': 'Bestätigung',
            'text': 'Ja perfekt danke',
            'duration': 0.8,
            'energy_decay': 0.5,
            'expected': 'HIGH'
        },
        {
            'name': 'Zeitangabe am Ende',
            'text': 'Ich komme morgen um 15 Uhr an',
            'duration': 1.2,
            'energy_decay': 0.6,
            'expected': 'HIGH'
        },
        {
            'name': 'Konjunktion am Ende',
            'text': 'Ich brauche ein Zimmer und',
            'duration': 1.0,
            'energy_decay': 1.0,
            'expected': 'LOW'
        },
    ]

    correct = 0
    total = len(scenarios)

    for scenario in scenarios:
        print("-" * 80)
        print(f"Szenario: {scenario['name']}")
        print(f"Text: \"{scenario['text']}\"")
        print()

        vad.reset()

        # Generate simple audio
        duration = scenario['duration']
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * 150 * t) * 0.3

        # Apply energy decay
        if scenario['energy_decay'] < 1.0:
            decay = np.linspace(1.0, scenario['energy_decay'], samples)
            audio *= decay

        # Process frames
        frame_size = 160
        results = []

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]
            result = vad.process_frame(
                np.zeros(frame_size),  # User silent
                frame,                  # AI speaking
                scenario['text']
            )
            if result.get('ai_speaking'):
                results.append(result)

        if results:
            avg_turn_end = np.mean([r['turn_end_prob'] for r in results])
            avg_prosody = np.mean([r['prosody_prob'] for r in results])
            avg_semantic = np.mean([r['semantic_prob'] for r in results])

            print(f"Turn-End-Wahrscheinlichkeit: {avg_turn_end:.1%}")
            print(f"  Prosodie: {avg_prosody:.1%}")
            print(f"  Semantik: {avg_semantic:.1%}")

            if scenario['expected'] == 'HIGH':
                threshold = 0.70
                passed = avg_turn_end >= threshold
                print(f"Erwartet: >{threshold:.0%} - {'BESTANDEN' if passed else 'FEHLER'}")
            else:
                threshold = 0.50
                passed = avg_turn_end < threshold
                print(f"Erwartet: <{threshold:.0%} - {'BESTANDEN' if passed else 'FEHLER'}")

            if passed:
                correct += 1

        print()

    print("=" * 80)
    print(" ENDERGEBNIS")
    print("=" * 80)
    accuracy = (correct / total) * 100
    print(f"Genauigkeit: {correct}/{total} ({accuracy:.1f}%)")
    print()

    stats = vad.get_stats()
    print("LEISTUNG:")
    print(f"  Durchschnittliche Latenz: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 Latenz: {stats['p95_latency_ms']:.2f}ms")
    print()

    return accuracy


def test_hotel_conversation():
    """Test realistic hotel conversation scenarios in German"""
    print("=" * 80)
    print(" HOTEL-GESPRÄCH SZENARIEN")
    print("=" * 80)
    print()

    detector = SemanticCompletionDetectorGerman()

    conversations = [
        # Rezeption -> Gast -> Rezeption
        ("Guten Tag, wie kann ich Ihnen helfen?", True, "Rezeption Begrüßung"),
        ("Ich möchte ein Zimmer buchen", True, "Gast Anfrage"),
        ("Für wie viele Nächte?", True, "Rezeption Rückfrage"),
        ("Für drei Nächte ab", False, "Gast unvollständig"),
        ("Für drei Nächte ab Montag", True, "Gast vollständig"),
        ("Perfekt, ich prüfe die Verfügbarkeit", True, "Rezeption Bestätigung"),
        ("Wir haben noch", False, "Rezeption unvollständig"),
        ("Wir haben noch Zimmer verfügbar", True, "Rezeption vollständig"),
        ("Was kostet das?", True, "Gast Preisfrage"),
        ("Der Preis beträgt 120 Euro pro Nacht", True, "Rezeption Antwort"),
    ]

    print(f"{'Utterance':<50} {'Expected':<12} {'Result':<12} {'Status'}")
    print("-" * 80)

    correct = 0
    for text, expected, speaker in conversations:
        result = detector.is_complete(text)
        predicted = result['complete_prob'] > 0.6
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        expected_str = "VOLLSTÄNDIG" if expected else "UNVOLLSTÄNDIG"
        predicted_str = "VOLLSTÄNDIG" if predicted else "UNVOLLSTÄNDIG"
        status = "OK" if is_correct else "XX"

        print(f"{text:<50} {expected_str:<12} {predicted_str:<12} {status}")

    print("-" * 80)
    accuracy = (correct / len(conversations)) * 100
    print(f"\nHotel-Gespräch Genauigkeit: {correct}/{len(conversations)} ({accuracy:.1f}%)")
    print()

    return accuracy


def main():
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  EXCELLENCE VAD - DEUTSCHE VERSION TEST SUITE".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    # Test 1: Semantic patterns
    semantic_acc = test_german_semantic_patterns()

    # Test 2: Full system
    system_acc = test_german_full_system()

    # Test 3: Hotel conversation
    hotel_acc = test_hotel_conversation()

    # Summary
    print("=" * 80)
    print(" GESAMTZUSAMMENFASSUNG")
    print("=" * 80)
    print()
    print(f"Semantik-Muster: {semantic_acc:.1f}%")
    print(f"Vollständiges System: {system_acc:.1f}%")
    print(f"Hotel-Gespräch: {hotel_acc:.1f}%")
    print()

    overall = (semantic_acc + system_acc + hotel_acc) / 3
    print(f"GESAMTGENAUIGKEIT: {overall:.1f}%")
    print()

    if overall >= 90:
        print("OK PRODUKTIONSBEREIT für deutsche Telefongespräche!")
    elif overall >= 80:
        print("OK SEHR GUT - Bereit für Tests mit echten Nutzern")
    elif overall >= 70:
        print("OK GUT - Weitere Optimierung empfohlen")
    else:
        print("XX VERBESSERUNG NÖTIG")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
