"""
Generate high-quality German speech using Microsoft Edge TTS
(More natural than gTTS, better for VAD testing)
"""

import asyncio
import edge_tts
import os

# Test sentences - same as gTTS test
test_data = [
    ("Das Hotel hat funfzig Zimmer", "complete_sentence_1.wav"),
    ("Vielen Dank fur Ihren Anruf", "complete_sentence_2.wav"),
    ("Guten Tag, wie kann ich Ihnen helfen", "complete_sentence_3.wav"),
    ("Ich mochte Ihnen sagen dass", "incomplete_hesitation.wav"),
    ("Ich gehe zur", "incomplete_preposition.wav"),
    ("Der Preis betragt zweihundert Euro", "complete_with_number.wav"),
    ("Ja, das ist korrekt", "complete_confirmation.wav"),
    ("Das Zimmer ist verfugbar und", "incomplete_conjunction.wav"),
    ("Haben Sie noch weitere Fragen", "complete_question.wav"),
    ("Sehr gerne, ich helfe Ihnen", "complete_polite.wav"),
]

async def generate_speech():
    os.makedirs("real_audio", exist_ok=True)
    
    # Use German female voice (de-DE-KatjaNeural - natural sounding)
    voice = "de-DE-KatjaNeural"
    
    print(f"Generating speech with {voice}...")
    print(f"Output: real_audio/")
    print()
    
    for text, filename in test_data:
        filepath = f"real_audio/{filename}"
        
        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filepath)
        
        print(f"OK {filename:30} - \"{text}\"")
    
    print()
    print(f"Generated {len(test_data)} natural German speech files")
    print("Location: real_audio/")
    
    # Save metadata
    with open("real_audio/metadata.txt", "w", encoding="utf-8") as f:
        f.write("Generated with edge-tts (Microsoft Azure Neural Voice)\n")
        f.write(f"Voice: {voice}\n\n")
        for text, filename in test_data:
            f.write(f"{filename}: {text}\n")

if __name__ == "__main__":
    asyncio.run(generate_speech())
