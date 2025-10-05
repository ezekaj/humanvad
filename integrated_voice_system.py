"""
Integrated Voice AI System
===========================

Complete pipeline:
Audio -> STT (Speech-to-Text) -> Excellence VAD -> Turn-Taking Decision

Production-ready system combining:
1. Real-time speech recognition (Whisper/Faster-Whisper)
2. Excellence VAD (prosody + semantic turn-taking)
3. Smart interruption detection

Target: Production voice AI (Sofia, phone bots, etc.)
"""

import numpy as np
import pyaudio
import time
from collections import deque
from typing import Dict, Optional
from excellence_vad import ExcellenceVAD


class IntegratedVoiceSystem:
    """
    Complete voice AI system with STT + turn-taking detection

    Pipeline:
    1. Audio input (user + AI channels)
    2. STT transcription (streaming)
    3. Excellence VAD analysis (prosody + semantics)
    4. Turn-taking decision (interrupt vs wait)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        use_whisper: bool = False  # Set True if whisper installed
    ):
        self.sr = sample_rate

        # Turn-taking detection
        self.vad = ExcellenceVAD(
            sample_rate=sample_rate,
            turn_end_threshold=0.75
        )

        # Speech recognition
        self.use_whisper = use_whisper
        if use_whisper:
            try:
                import faster_whisper
                self.stt_model = faster_whisper.WhisperModel(
                    "base",  # or "tiny" for faster, "small" for better
                    device="cpu",
                    compute_type="int8"
                )
                print("[OK] Faster-Whisper STT loaded")
            except:
                print("⚠ Faster-Whisper not available, using simulated STT")
                self.use_whisper = False

        # Transcription buffers
        self.user_text_buffer = deque(maxlen=10)
        self.ai_text_buffer = deque(maxlen=10)

        # Audio buffers for STT (need longer chunks)
        self.user_audio_buffer = deque(maxlen=sample_rate * 3)  # 3s
        self.ai_audio_buffer = deque(maxlen=sample_rate * 3)

        # State
        self.ai_currently_saying = ""
        self.user_currently_saying = ""

        # Statistics
        self.stats = {
            'interruptions_detected': 0,
            'natural_turn_takings': 0,
            'total_decisions': 0
        }

    def transcribe_audio(self, audio: np.ndarray, is_user: bool) -> str:
        """
        Transcribe audio to text using STT

        In production: Use faster-whisper for real-time streaming
        For demo: Simulate based on audio characteristics
        """

        if self.use_whisper and len(audio) >= self.sr:  # At least 1s
            # Real STT with Whisper
            segments, info = self.stt_model.transcribe(
                audio,
                beam_size=1,  # Fast, greedy decoding
                language="en"
            )
            text = " ".join([seg.text for seg in segments])
            return text.strip()
        else:
            # Simulated STT for demo (in production, replace with real STT)
            # Detect if speaking based on energy
            energy = np.sqrt(np.mean(audio ** 2))

            if energy > 0.01:
                # Simulate transcription based on who's speaking
                if is_user:
                    return self.user_currently_saying or "(user speaking)"
                else:
                    return self.ai_currently_saying or "(AI speaking)"
            return ""

    def process_frame(
        self,
        user_frame: np.ndarray,
        ai_frame: np.ndarray,
        ai_text_override: Optional[str] = None  # For testing/simulation
    ) -> Dict:
        """
        Process audio frame through complete pipeline

        Args:
            user_frame: User microphone audio (10ms, 160 samples)
            ai_frame: AI speech output audio (10ms, 160 samples)
            ai_text_override: Optional AI text (for testing without real TTS)

        Returns:
            {
                'action': 'interrupt_ai_immediately' | 'wait_for_ai_completion' | 'continue',
                'user_speaking': bool,
                'ai_speaking': bool,
                'user_text': str,
                'ai_text': str,
                'turn_end_prob': float,
                'reasoning': str
            }
        """

        # Add to buffers
        self.user_audio_buffer.extend(user_frame)
        self.ai_audio_buffer.extend(ai_frame)

        # Run Excellence VAD first (fast prosody check)
        vad_result = self.vad.process_frame(user_frame, ai_frame, ai_text_override or self.ai_currently_saying)

        user_speaking = vad_result['user_speaking']
        ai_speaking = vad_result['ai_speaking']

        # If significant speech event, run STT
        user_text = ""
        ai_text = ai_text_override or self.ai_currently_saying

        if user_speaking and len(self.user_audio_buffer) >= self.sr * 0.5:  # 500ms minimum
            # Transcribe user speech
            user_audio = np.array(list(self.user_audio_buffer))
            user_text = self.transcribe_audio(user_audio, is_user=True)
            if user_text:
                self.user_text_buffer.append(user_text)
                self.user_currently_saying = " ".join(list(self.user_text_buffer)[-3:])

        if ai_speaking and not ai_text_override:
            # In production: Get from TTS system
            # For demo: Use simulated text
            ai_text = self.ai_currently_saying

        # Update AI text from override or TTS
        if ai_text_override:
            self.ai_currently_saying = ai_text_override

        # Track statistics
        if vad_result['action'] == 'interrupt_ai_immediately' and vad_result.get('overlap'):
            if vad_result['turn_end_prob'] > 0.75:
                self.stats['natural_turn_takings'] += 1
            else:
                self.stats['interruptions_detected'] += 1
            self.stats['total_decisions'] += 1

        return {
            'action': vad_result['action'],
            'user_speaking': user_speaking,
            'ai_speaking': ai_speaking,
            'overlap': vad_result.get('overlap', False),
            'user_text': user_text or self.user_currently_saying,
            'ai_text': ai_text,
            'turn_end_prob': vad_result.get('turn_end_prob', 0.0),
            'reasoning': vad_result.get('reasoning', 'unknown'),
            'prosody_prob': vad_result.get('prosody_prob', 0.0),
            'semantic_prob': vad_result.get('semantic_prob', 0.0),
            'latency_ms': vad_result.get('latency_ms', 0.0)
        }

    def set_ai_text(self, text: str):
        """Update what AI is currently saying (from TTS)"""
        self.ai_currently_saying = text

    def set_user_text(self, text: str):
        """Update user text (for testing)"""
        self.user_currently_saying = text

    def get_stats(self) -> Dict:
        """Get system statistics"""
        vad_stats = self.vad.get_stats()
        return {
            **vad_stats,
            **self.stats,
            'interruption_rate': self.stats['interruptions_detected'] / max(self.stats['total_decisions'], 1)
        }

    def reset(self):
        """Reset system state"""
        self.vad.reset()
        self.user_audio_buffer.clear()
        self.ai_audio_buffer.clear()
        self.user_text_buffer.clear()
        self.ai_text_buffer.clear()
        self.ai_currently_saying = ""
        self.user_currently_saying = ""


def demo():
    """Demo of integrated system"""
    print("=" * 80)
    print(" INTEGRATED VOICE AI SYSTEM")
    print("=" * 80)
    print()
    print("Complete Pipeline:")
    print("  Audio -> STT (Speech Recognition) -> Excellence VAD -> Turn-Taking Decision")
    print()
    print("Components:")
    print("  [OK] Real-time audio processing")
    print("  [OK] Speech-to-text transcription")
    print("  [OK] Prosodic analysis (pitch, energy, timing)")
    print("  [OK] Semantic completion detection")
    print("  [OK] Intelligent interruption handling")
    print()

    # Initialize system
    system = IntegratedVoiceSystem(sample_rate=16000, use_whisper=False)

    print("System initialized")
    print()

    # Simulate conversation scenario
    print("=" * 80)
    print(" SIMULATED CONVERSATION TEST")
    print("=" * 80)
    print()

    # Generate test audio
    def make_speech(duration, f0, rate):
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        f0_arr = np.full(len(t), f0)
        signal = np.sin(2 * np.pi * f0_arr * t)
        signal += 0.5 * np.sin(2 * np.pi * 2 * f0_arr * t)
        mod = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
        return (signal * mod) / np.max(np.abs(signal * mod))

    # Scenario: AI says complete sentence, user responds
    print("Scenario 1: Natural turn-taking")
    print("-" * 80)

    ai_speech = make_speech(1.0, 160, 5.0)
    ai_text = "What time should we meet tomorrow?"

    system.reset()
    system.set_ai_text(ai_text)

    frame_size = 160
    results = []

    # Process AI speech
    for i in range(0, len(ai_speech) - frame_size, frame_size * 5):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)

        result = system.process_frame(user_frame, ai_frame, ai_text)
        if result['ai_speaking']:
            results.append(result)

    # Pause
    for _ in range(15):
        result = system.process_frame(np.zeros(frame_size), np.zeros(frame_size), "")

    # User responds
    user_speech = make_speech(0.5, 150, 5.0)
    system.set_user_text("How about 3 PM")

    for i in range(0, len(user_speech) - frame_size, frame_size):
        user_frame = user_speech[i:i+frame_size]
        ai_frame = np.zeros(frame_size)

        result = system.process_frame(user_frame, ai_frame, "")
        if result['user_speaking']:
            results.append(result)
            break

    final = results[-1]
    print(f"AI said: \"{ai_text}\"")
    print(f"User responds: \"{final['user_text']}\"")
    print(f"Turn-end probability: {final['turn_end_prob']:.1%}")
    print(f"Action: {final['action']}")
    print(f"Result: {'[OK] Natural turn-taking' if final['turn_end_prob'] > 0.7 else '[X] Unexpected'}")
    print()

    # Scenario 2: User interrupts mid-sentence
    print("Scenario 2: User interruption")
    print("-" * 80)

    ai_speech = make_speech(0.8, 160, 5.5)
    ai_text = "I think we should go to"  # Incomplete

    system.reset()
    system.set_ai_text(ai_text)

    results = []

    # Process AI speech
    for i in range(0, len(ai_speech) // 2 - frame_size, frame_size * 3):
        ai_frame = ai_speech[i:i+frame_size]
        user_frame = np.zeros(frame_size)
        result = system.process_frame(user_frame, ai_frame, ai_text)

    # User interrupts (overlap)
    user_speech = make_speech(0.4, 155, 5.0)
    system.set_user_text("Wait, let me check")

    for i in range(0, len(user_speech) - frame_size, frame_size):
        user_frame = user_speech[i:i+frame_size]
        ai_idx = min(len(ai_speech) // 2 + i, len(ai_speech) - frame_size)
        ai_frame = ai_speech[ai_idx:ai_idx+frame_size]

        result = system.process_frame(user_frame, ai_frame, ai_text)
        if result['overlap']:
            results.append(result)
            break

    if results:
        final = results[-1]
        print(f"AI saying: \"{ai_text}\" (incomplete)")
        print(f"User interrupts: \"{final['user_text']}\"")
        print(f"Turn-end probability: {final['turn_end_prob']:.1%}")
        print(f"Action: {final['action']}")
        print(f"Result: {'[OK] Interruption detected' if final['turn_end_prob'] < 0.5 else '[X] Unexpected'}")

    print()
    print("=" * 80)

    # Statistics
    stats = system.get_stats()
    print()
    print("SYSTEM STATISTICS:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Natural turn-takings: {stats['natural_turn_takings']}")
    print(f"  Interruptions detected: {stats['interruptions_detected']}")
    print()

    print("Production Integration:")
    print("  1. Replace simulated STT with faster-whisper (pip install faster-whisper)")
    print("  2. Connect to your TTS system (get current AI text)")
    print("  3. Feed real microphone audio")
    print("  4. Use action decisions to control conversation flow")
    print()
    print("Ready for Sofia, phone bots, or any voice AI system!")
    print()


if __name__ == "__main__":
    demo()
