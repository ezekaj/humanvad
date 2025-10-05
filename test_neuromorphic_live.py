"""
Live Neuromorphic VAD Test
===========================
Real-time turn-taking detection with brain-inspired mechanisms
"""

import numpy as np
import pyaudio
import time
from neuromorphic_vad import NeuromorphicVAD


def test_live():
    print("=" * 80)
    print(" NEUROMORPHIC LIVE TEST - Brain-Inspired Turn-Taking")
    print("=" * 80)
    print()
    print("Neural mechanisms active:")
    print("  - Theta oscillations (4-8 Hz)")
    print("  - STG onset detection (200ms boundaries)")
    print("  - Cortical entrainment")
    print("  - Hierarchical prediction")
    print()
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print()
    print("-" * 80)
    print(f"{'Time':<8} {'Speaking':<10} {'Turn-End':<10} {'Action':<30} {'Latency'}")
    print("-" * 80)

    sr = 16000
    chunk = 160  # 10ms

    vad = NeuromorphicVAD(sample_rate=sr)
    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sr,
            input=True,
            frames_per_buffer=chunk
        )

        start = time.time()

        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)

            if len(audio) < chunk:
                audio = np.pad(audio, (0, chunk - len(audio)))

            # User frame = microphone, AI frame = silence (single speaker test)
            result = vad.process_frame(audio, np.zeros(chunk))

            # Only print when speaking or significant event
            if result['user_speaking'] or result.get('onset_boundary', False):
                t = time.time() - start
                speaking = "SPEAKING" if result['user_speaking'] else "silence"
                turn_prob = f"{result.get('turn_end_prob', 0):.1%}"
                action = result.get('action', 'continue')
                latency = f"{result['latency_ms']:.2f}ms"

                print(f"{t:>6.1f}s  {speaking:<10} {turn_prob:<10} {action:<30} {latency}")

    except KeyboardInterrupt:
        print()
        print("-" * 80)
        stats = vad.get_stats()
        print(f"\nAvg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
        print("\nTest complete!")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    test_live()
