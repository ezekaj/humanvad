"""
Live Excellence VAD Test
=========================

Real-time testing with your voice
Simulates AI speaking with text transcription
"""

import numpy as np
import pyaudio
import time
from excellence_vad import ExcellenceVAD


def test_live():
    """Live microphone test with simulated AI scenarios"""

    print("=" * 80)
    print(" EXCELLENCE VAD - LIVE TEST")
    print("=" * 80)
    print()
    print("This will simulate AI speaking different phrases while detecting your voice.")
    print()
    print("Test scenarios:")
    print("  1. AI says complete sentence - you respond (natural turn-taking)")
    print("  2. AI says incomplete phrase - you interrupt (real interruption)")
    print()
    print("Press Ctrl+C to stop")
    print()
    print("-" * 80)

    sr = 16000
    chunk = 160  # 10ms

    vad = ExcellenceVAD(sample_rate=sr, turn_end_threshold=0.75)
    p = pyaudio.PyAudio()

    # Simulated AI scenarios (cycling through)
    ai_scenarios = [
        {
            'text': "I think we should meet tomorrow at noon",
            'complete': True,
            'duration': 3.0
        },
        {
            'text': "Let me think about",
            'complete': False,
            'duration': 2.0
        },
        {
            'text': "What time works best for you?",
            'complete': True,
            'duration': 2.5
        },
        {
            'text': "I am going to",
            'complete': False,
            'duration': 1.5
        },
        {
            'text': "Okay sounds good thanks",
            'complete': True,
            'duration': 2.0
        },
    ]

    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sr,
            input=True,
            frames_per_buffer=chunk
        )

        print(f"{'Time':<8} {'AI Text':<40} {'User':<8} {'Turn-End':<10} {'Action':<30}")
        print("-" * 80)

        start = time.time()
        scenario_idx = 0
        scenario_start = start
        current_scenario = ai_scenarios[scenario_idx]

        while True:
            elapsed = time.time() - start
            scenario_elapsed = time.time() - scenario_start

            # Read user microphone
            data = stream.read(chunk, exception_on_overflow=False)
            user_audio = np.frombuffer(data, dtype=np.float32)

            if len(user_audio) < chunk:
                user_audio = np.pad(user_audio, (0, chunk - len(user_audio)))

            # Simulate AI speaking (silence for now, in production would be real AI audio)
            ai_audio = np.zeros(chunk)

            # Process with VAD (include AI text for semantic analysis)
            ai_text = current_scenario['text'] if scenario_elapsed < current_scenario['duration'] else None

            result = vad.process_frame(user_audio, ai_audio, ai_text)

            # Display when user speaking or significant event
            if result['user_speaking']:
                t = time.time() - start

                user_str = "SPEAKING"
                turn_prob = f"{result['turn_end_prob']:.1%}"
                action = result['action']
                ai_display = current_scenario['text'][:35] if ai_text else "(silence)"

                # Highlight important events
                if result.get('overlap', False):
                    indicator = "<!>"
                else:
                    indicator = ""

                print(f"{t:>6.1f}s  {ai_display:<40} {user_str:<8} {turn_prob:<10} {action:<30} {indicator}")

            # Switch scenario after duration
            if scenario_elapsed > current_scenario['duration'] + 1.0:
                scenario_idx = (scenario_idx + 1) % len(ai_scenarios)
                current_scenario = ai_scenarios[scenario_idx]
                scenario_start = time.time()
                vad.reset()

                print()
                print(f"--- New AI utterance: \"{current_scenario['text']}\" ({'complete' if current_scenario['complete'] else 'incomplete'}) ---")
                print()

    except KeyboardInterrupt:
        print()
        print("-" * 80)

        stats = vad.get_stats()
        print(f"\nPerformance:")
        print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
        print()
        print("Test complete!")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    test_live()
