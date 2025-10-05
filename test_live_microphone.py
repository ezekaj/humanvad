"""
Live microphone test - speak into your mic to test VAD
Press Ctrl+C to stop
"""

import numpy as np
import pyaudio
from production_vad import ProductionVAD
import time

def test_live():
    print("=" * 80)
    print(" LIVE MICROPHONE TEST - Voice Activity Detection")
    print("=" * 80)
    print()
    print("Starting microphone... Speak into your microphone!")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 80)
    print(f"{'Time':<12} {'Status':<15} {'Confidence':<12} {'Latency':<10}")
    print("-" * 80)

    # Initialize VAD
    sr = 16000
    vad = ProductionVAD(sample_rate=sr)

    # Audio settings
    chunk_size = 480  # 30ms at 16kHz

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    try:
        # Open microphone stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            frames_per_buffer=chunk_size
        )

        print("Microphone active!")
        print()

        start_time = time.time()
        speech_count = 0
        silence_count = 0

        while True:
            # Read audio
            audio_data = stream.read(chunk_size, exception_on_overflow=False)

            # Convert to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # Normalize to [-1, 1]

            # Detect speech
            result = vad.detect_frame(audio_np)

            # Track statistics
            if result['is_speech']:
                speech_count += 1
            else:
                silence_count += 1

            # Print result
            elapsed = time.time() - start_time
            status = "SPEECH" if result['is_speech'] else "silence"

            # Color coding (won't show in Windows but helps readability)
            status_display = f"[{status}]"

            print(f"{elapsed:>6.1f}s     {status_display:<15} {result['confidence']:>6.1%}      {result['latency_ms']:>5.2f}ms")

    except KeyboardInterrupt:
        print()
        print("-" * 80)
        print()
        print("STATISTICS:")
        print(f"  Total frames:   {speech_count + silence_count}")
        print(f"  Speech frames:  {speech_count} ({speech_count/(speech_count+silence_count)*100:.1f}%)")
        print(f"  Silence frames: {silence_count} ({silence_count/(speech_count+silence_count)*100:.1f}%)")
        print()

        stats = vad.get_stats()
        if stats:
            print("PERFORMANCE:")
            print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"  P50 latency: {stats['p50_latency_ms']:.2f}ms")
            print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
            print(f"  Max latency: {stats['max_latency_ms']:.2f}ms")
            print()

        print("Test stopped.")

    finally:
        # Cleanup
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    test_live()
