"""
Silero VAD Wrapper - Production ML-based VAD
=============================================

Pre-trained deep learning model with 95%+ accuracy
Ultra-fast inference: ~1-5ms per frame
"""

import torch
import numpy as np
from collections import deque
from typing import Dict
import time


class SileroVAD:
    """
    Silero VAD wrapper for ultra-fast, accurate speech detection

    Pre-trained on 4000+ hours of speech data
    Works on: English, Spanish, German, Russian, and more
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ):
        self.sr = sample_rate
        self.threshold = threshold
        self.min_speech_frames = int(min_speech_duration_ms / 30)  # 30ms per frame
        self.min_silence_frames = int(min_silence_duration_ms / 30)

        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False  # Use PyTorch model for faster CPU inference
        )

        # State
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0

        # Performance tracking
        self.processing_times = deque(maxlen=100)

    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Detect speech in single audio frame

        Args:
            frame: Audio samples (16kHz, 512 samples = 32ms recommended)

        Returns:
            Dict with is_speech, confidence, latency_ms
        """
        start_time = time.perf_counter()

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(frame).float()

        # Run model inference
        with torch.no_grad():
            confidence = self.model(audio_tensor, self.sr).item()

        # Threshold decision
        is_speech = confidence >= self.threshold

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        return {
            'is_speech': is_speech,
            'confidence': confidence,
            'latency_ms': latency_ms
        }

    def process_stream(self, frame: np.ndarray) -> Dict:
        """Process streaming audio with state tracking"""
        result = self.detect_frame(frame)

        if result['is_speech']:
            self.speech_frame_count += 1
            self.silence_frame_count = 0

            if not self.is_speaking and self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
        else:
            self.silence_frame_count += 1

            if self.is_speaking and self.silence_frame_count >= self.min_silence_frames:
                self.is_speaking = False
                self.speech_frame_count = 0

        return {
            'is_speaking': self.is_speaking,
            'confidence': result['confidence'],
            'latency_ms': result['latency_ms'],
            'speech_frames': self.speech_frame_count,
            'silence_frames': self.silence_frame_count
        }

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if len(self.processing_times) > 0:
            return {
                'avg_latency_ms': np.mean(self.processing_times),
                'p50_latency_ms': np.percentile(self.processing_times, 50),
                'p95_latency_ms': np.percentile(self.processing_times, 95),
                'max_latency_ms': np.max(self.processing_times)
            }
        return {}

    def reset(self):
        """Reset state"""
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" SILERO VAD - Production ML Model")
    print("=" * 80)
    print()
    print("Loading model...")

    vad = SileroVAD(sample_rate=16000)

    print("Model loaded!")
    print()
    print("Speed Test:")
    print("-" * 80)

    # Test with 512 samples (32ms at 16kHz - Silero recommended)
    test_frame = np.random.randn(512).astype(np.float32)
    n_iterations = 100

    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.detect_frame(test_frame)
    avg_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Average latency: {avg_time:.2f}ms per frame (32ms audio)")
    print(f"Real-time factor: {avg_time / 32:.2f}x")
    print(f"Target: <10ms {'PASS' if avg_time < 10 else 'FAIL'}")
    print()

    stats = vad.get_stats()
    print(f"P50: {stats['p50_latency_ms']:.2f}ms")
    print(f"P95: {stats['p95_latency_ms']:.2f}ms")
    print()

    print("=" * 80)
    print()
    print("Run test_realistic_noise.py with SileroVAD for full evaluation")


if __name__ == "__main__":
    demo()
