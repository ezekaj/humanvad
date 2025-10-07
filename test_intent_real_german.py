"""
Test Intent Integration with REAL German Audio
===============================================
"""

import glob
import wave
import numpy as np
from flexduo_silero_vad import FlexDuoSileroVAD, DuplexFrame


def load_wav(file_path):
    """Load WAV and resample to 8kHz"""
    import scipy.signal as signal
    with wave.open(file_path, 'rb') as wf:
        sr = wf.getframerate()
        audio_bytes = wf.readframes(wf.getnframes())
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != 8000:
            audio = signal.resample(audio, int(len(audio) * 8000 / sr))
        return audio


def test_with_intent():
    """Test with intent classification enabled"""
    print("="*80)
    print(" INTENT CLASSIFICATION TEST (REAL GERMAN AUDIO)")
    print("="*80)
    print()

    # Load German audio files
    german_files = glob.glob("C:/Users/User/Desktop/human-like-speech-ai/german_offline_speech/*.wav")
    print(f"Found {len(german_files)} German audio files")
    print()

    # Scenario mapping (audio file to expected text)
    scenarios = {
        'guten_tag.wav': ('guten_tag', 'greeting', False),  # Greeting - don't stop
        'ja.wav': ('ja', 'response:confirm', False),  # Acknowledgment - don't stop
        'entschuldigung.wav': ('entschuldigung', 'apology', True),  # Apology - stop
        'danke.wav': ('danke', 'social:thanks', False),  # Thanks - don't stop (expect "bitte")
        'wie_gehts.wav': ('wie gehts', 'question', True),  # Question - stop
        'bitte.wav': ('bitte', 'request', True),  # Request - stop
    }

    results_without_intent = []
    results_with_intent = []

    for filename, (text, expected_intent, should_stop) in scenarios.items():
        filepath = f"C:/Users/User/Desktop/human-like-speech-ai/german_offline_speech/{filename}"

        try:
            audio = load_wav(filepath)
        except:
            continue

        mid = len(audio) // 2
        user_audio = audio[:mid]
        ai_audio = audio[mid:]

        frame_size = 240
        timestamp_ms = 0

        print(f"Testing: {filename} ('{text}') - Expected: {expected_intent}")

        # Test 1: WITHOUT intent classification (LOWER thresholds to detect real speech)
        vad_no_intent = FlexDuoSileroVAD(sample_rate=8000, user_threshold=0.10, interrupt_threshold=0.30, enable_intent=False)

        # AI speaking
        for i in range(0, min(len(ai_audio), 8000), frame_size):
            ai_chunk = ai_audio[i:i+frame_size]
            if len(ai_chunk) < frame_size:
                ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))
            user_chunk = np.random.randn(frame_size) * 0.01
            vad_no_intent.process_frame(DuplexFrame(timestamp_ms, user_chunk, ai_chunk))
            timestamp_ms += 30

        # User interrupts
        stopped_no_intent = False
        for i in range(0, min(len(user_audio), 2400), frame_size):
            user_chunk = user_audio[i:i+frame_size]
            ai_chunk = ai_audio[min(i + 6000, len(ai_audio)-frame_size):min(i + 6000 + frame_size, len(ai_audio))]

            if len(user_chunk) < frame_size:
                user_chunk = np.pad(user_chunk, (0, frame_size - len(user_chunk)))
            if len(ai_chunk) < frame_size:
                ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))

            decision = vad_no_intent.process_frame(DuplexFrame(timestamp_ms, user_chunk, ai_chunk))
            if decision.should_stop_ai:
                stopped_no_intent = True
            timestamp_ms += 30

        # Test 2: WITH intent classification (LOWER thresholds to detect real speech)
        vad_with_intent = FlexDuoSileroVAD(sample_rate=8000, user_threshold=0.10, interrupt_threshold=0.30, enable_intent=True)

        timestamp_ms = 0

        # AI speaking
        for i in range(0, min(len(ai_audio), 8000), frame_size):
            ai_chunk = ai_audio[i:i+frame_size]
            if len(ai_chunk) < frame_size:
                ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))
            user_chunk = np.random.randn(frame_size) * 0.01
            vad_with_intent.process_frame(DuplexFrame(timestamp_ms, user_chunk, ai_chunk))
            timestamp_ms += 30

        # User interrupts WITH TEXT for intent
        stopped_with_intent = False
        detected_intent = None
        for i in range(0, min(len(user_audio), 2400), frame_size):
            user_chunk = user_audio[i:i+frame_size]
            ai_chunk = ai_audio[min(i + 6000, len(ai_audio)-frame_size):min(i + 6000 + frame_size, len(ai_audio))]

            if len(user_chunk) < frame_size:
                user_chunk = np.pad(user_chunk, (0, frame_size - len(user_chunk)))
            if len(ai_chunk) < frame_size:
                ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))

            decision = vad_with_intent.process_frame(DuplexFrame(timestamp_ms, user_chunk, ai_chunk, user_text=text))
            if decision.should_stop_ai:
                stopped_with_intent = True
            if decision.intent:
                detected_intent = decision.intent
            timestamp_ms += 30

        stats_no = vad_no_intent.get_statistics()
        stats_with = vad_with_intent.get_statistics()

        print(f"  Without intent: stopped={stopped_no_intent}, barge_ins={stats_no['total_barge_ins']}")
        print(f"  With intent: stopped={stopped_with_intent}, barge_ins={stats_with['total_barge_ins']}, intent={detected_intent}")
        print(f"  Prevented: {stats_with['false_barge_ins_prevented']}")

        # Evaluate
        correct_without = stopped_no_intent == should_stop
        correct_with = stopped_with_intent == should_stop

        results_without_intent.append(correct_without)
        results_with_intent.append(correct_with)

        if correct_with and not correct_without:
            print(f"  [IMPROVED] Intent classification fixed the decision!")
        elif correct_with == correct_without:
            print(f"  [SAME] Both got it {'correct' if correct_with else 'wrong'}")
        else:
            print(f"  [WORSE] Intent made it worse")

        print()

    # Summary
    print("="*80)
    print(" RESULTS")
    print("="*80)
    print()

    acc_without = sum(results_without_intent) / len(results_without_intent) * 100 if results_without_intent else 0
    acc_with = sum(results_with_intent) / len(results_with_intent) * 100 if results_with_intent else 0

    print(f"WITHOUT Intent: {sum(results_without_intent)}/{len(results_without_intent)} ({acc_without:.1f}%)")
    print(f"WITH Intent:    {sum(results_with_intent)}/{len(results_with_intent)} ({acc_with:.1f}%)")
    print()
    print(f"Improvement: {acc_with - acc_without:+.1f}%")
    print()

    if acc_with > acc_without:
        print(f"[SUCCESS] Intent classification improved accuracy by {acc_with - acc_without:.1f}%!")
    elif acc_with == acc_without:
        print("[NEUTRAL] Intent classification had no effect")
    else:
        print("[ISSUE] Intent classification reduced accuracy")

    print()


if __name__ == "__main__":
    import scipy.signal
    test_with_intent()
