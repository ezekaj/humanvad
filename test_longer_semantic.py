"""
Test semantic detector on longer German sentences (text-only, no audio)
"""
from excellence_vad_german import ExcellenceVADGerman

# Longer test sentences
longer_tests = [
    # Complete (expected > 0.60)
    ("Das Hotel verfügt über fünfzig modern ausgestattete Zimmer mit kostenlosem WLAN und Klimaanlage für Ihren komfortablen Aufenthalt", True, "17w statement"),
    ("Ich habe Ihre Reservierung für drei Nächte vom fünfzehnten bis zum achtzehnten Oktober im Doppelzimmer mit Meerblick bestätigt", True, "18w confirmation"),
    ("Das Frühstücksbuffet wird täglich von sieben bis elf Uhr im Restaurantbereich im Erdgeschoss serviert und ist im Zimmerpreis bereits enthalten", True, "20w statement"),
    ("Der Gesamtpreis für Ihren Aufenthalt beträgt dreihundertfünfzig Euro inklusive Mehrwertsteuer und Frühstück für zwei Personen", True, "15w price"),
    ("Selbstverständlich können Sie das Zimmer auch früher beziehen wenn es bereits ab vierzehn Uhr zur Verfügung steht und gereinigt wurde", True, "20w confirmation"),

    # Questions (expected > 0.60)
    ("Haben Sie spezielle Wünsche bezüglich der Zimmerausstattung oder möchten Sie vielleicht ein Zimmer in einer bestimmten Etage reservieren", True, "18w question"),
    ("Darf ich Sie fragen ob Sie bereits bei uns übernachtet haben und ob Sie vielleicht Teil unseres Treueprogramms sind", True, "19w question"),
    ("Möchten Sie zusätzlich zu Ihrem Zimmer auch einen Parkplatz in unserer Tiefgarage buchen der fünfzehn Euro pro Nacht kostet", True, "19w question"),

    # Confirmations (expected > 0.60)
    ("Sehr gerne habe ich das für Sie notiert und werde sicherstellen dass alles zu Ihrer vollsten Zufriedenheit vorbereitet wird", True, "19w confirmation"),
    ("Perfekt, dann ist alles geklärt und ich freue mich sehr darauf Sie am fünfzehnten Oktober bei uns begrüßen zu dürfen", True, "20w confirmation"),

    # Incomplete (expected < 0.60, should be 0.2)
    ("Ich möchte Ihnen gerne noch mitteilen dass wir auch einen Wellnessbereich haben mit", False, "13w incomplete+mit"),
    ("Das Zimmer befindet sich im dritten Stock und Sie erreichen es über den Fahrstuhl oder", False, "15w incomplete+oder"),
    ("Für die Zahlung akzeptieren wir Kreditkarten, EC-Karten und natürlich auch", False, "10w incomplete+auch"),
    ("Die Stornierungsbedingungen sehen vor dass Sie bis zu vierundzwanzig Stunden vor", False, "11w incomplete+vor"),
    ("Unser Concierge-Service steht Ihnen jederzeit zur Verfügung wenn Sie Fragen haben oder", False, "12w incomplete+oder"),
]

def main():
    vad = ExcellenceVADGerman(turn_end_threshold=0.60)
    semantic = vad.semantic_detector

    print("Testing Semantic Detector on Longer German Sentences (Text-Only)\n")
    print("=" * 100)

    correct = 0
    total = 0

    for text, expected_complete, category in longer_tests:
        result = semantic.is_complete(text)

        actual_complete = (result['complete_prob'] >= 0.60)
        is_correct = (actual_complete == expected_complete)

        status = "OK" if is_correct else "FAIL"
        correct += is_correct
        total += 1

        # Truncate text for display
        display_text = text[:50] + "..." if len(text) > 50 else text

        print(f"{status:4} | {result['complete_prob']:.3f} | {display_text:55} | {category}")

    print("=" * 100)
    print(f"\nACCURACY: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Threshold: 0.60")
    print(f"Average sentence length: {sum(len(t[0].split()) for t in longer_tests) // len(longer_tests)} words")
    print(f"\nTesting LONGER German sentences (15-20 words avg)")

if __name__ == "__main__":
    main()
