"""
Generate longer, more realistic German hotel conversation sentences
for comprehensive VAD testing
"""
import asyncio
import edge_tts
import os

# Longer, more complex German sentences for hotel receptionist scenarios
longer_test_data = [
    # Complete sentences (15-30 words)
    ("Das Hotel verfügt über fünfzig modern ausgestattete Zimmer mit kostenlosem WLAN und Klimaanlage für Ihren komfortablen Aufenthalt", "long_complete_1.wav"),
    ("Ich habe Ihre Reservierung für drei Nächte vom fünfzehnten bis zum achtzehnten Oktober im Doppelzimmer mit Meerblick bestätigt", "long_complete_2.wav"),
    ("Das Frühstücksbuffet wird täglich von sieben bis elf Uhr im Restaurantbereich im Erdgeschoss serviert und ist im Zimmerpreis bereits enthalten", "long_complete_3.wav"),
    ("Der Gesamtpreis für Ihren Aufenthalt beträgt dreihundertfünfzig Euro inklusive Mehrwertsteuer und Frühstück für zwei Personen", "long_complete_4.wav"),
    ("Selbstverständlich können Sie das Zimmer auch früher beziehen wenn es bereits ab vierzehn Uhr zur Verfügung steht und gereinigt wurde", "long_complete_5.wav"),

    # Incomplete sentences (cut off mid-thought)
    ("Ich möchte Ihnen gerne noch mitteilen dass wir auch einen Wellnessbereich haben mit", "long_incomplete_1.wav"),
    ("Das Zimmer befindet sich im dritten Stock und Sie erreichen es über den Fahrstuhl oder", "long_incomplete_2.wav"),
    ("Für die Zahlung akzeptieren wir Kreditkarten, EC-Karten und natürlich auch", "long_incomplete_3.wav"),
    ("Die Stornierungsbedingungen sehen vor dass Sie bis zu vierundzwanzig Stunden vor", "long_incomplete_4.wav"),
    ("Unser Concierge-Service steht Ihnen jederzeit zur Verfügung wenn Sie Fragen haben oder", "long_incomplete_5.wav"),

    # Complete questions
    ("Haben Sie spezielle Wünsche bezüglich der Zimmerausstattung oder möchten Sie vielleicht ein Zimmer in einer bestimmten Etage reservieren", "long_question_1.wav"),
    ("Darf ich Sie fragen ob Sie bereits bei uns übernachtet haben und ob Sie vielleicht Teil unseres Treueprogramms sind", "long_question_2.wav"),
    ("Möchten Sie zusätzlich zu Ihrem Zimmer auch einen Parkplatz in unserer Tiefgarage buchen der fünfzehn Euro pro Nacht kostet", "long_question_3.wav"),

    # Complete confirmations
    ("Sehr gerne habe ich das für Sie notiert und werde sicherstellen dass alles zu Ihrer vollsten Zufriedenheit vorbereitet wird", "long_confirm_1.wav"),
    ("Perfekt, dann ist alles geklärt und ich freue mich sehr darauf Sie am fünfzehnten Oktober bei uns begrüßen zu dürfen", "long_confirm_2.wav"),
]

async def generate_longer_sentences():
    output_dir = "longer_german_audio"
    os.makedirs(output_dir, exist_ok=True)

    # Use German female voice (de-DE-KatjaNeural - natural sounding)
    voice = "de-DE-KatjaNeural"

    print(f"Generating longer German sentences with {voice}...")
    print(f"Output: {output_dir}/")
    print()

    for text, filename in longer_test_data:
        filepath = f"{output_dir}/{filename}"

        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filepath)

        word_count = len(text.split())
        print(f"OK {filename:30} ({word_count:2} words) - \"{text[:60]}...\"")

    print()
    print(f"Generated {len(longer_test_data)} longer German speech files")
    print(f"Location: {output_dir}/")
    print(f"Average length: {sum(len(t[0].split()) for t in longer_test_data) // len(longer_test_data)} words")

    # Save metadata
    with open(f"{output_dir}/metadata.txt", "w", encoding="utf-8") as f:
        f.write("Generated with edge-tts (Microsoft Azure Neural Voice)\n")
        f.write(f"Voice: {voice}\n")
        f.write("Content: Longer German hotel conversation sentences (15-30 words)\n\n")
        for text, filename in longer_test_data:
            expected = "interrupt" if "complete" in filename or "question" in filename or "confirm" in filename else "wait"
            f.write(f"{filename}: [{expected}] {text}\n")

if __name__ == "__main__":
    asyncio.run(generate_longer_sentences())
