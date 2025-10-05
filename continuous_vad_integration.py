"""
Continuous VAD Integration
===========================

Real-time audio streaming pipeline connecting:
- Excellence VAD German (Stage 1)
- Intent Classifier (Stage 2)
- Turn-End Predictor (Stage 3)
- Memory-VAD Bridge (Stage 4)

Handles:
- Frame buffering (10ms frames at 16kHz)
- State management across frames
- Continuous prediction
- Speaker profile updates
"""

import sys
import numpy as np
from collections import deque
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to import Excellence VAD
sys.path.append('../human-speech-detection')

try:
    from excellence_vad_german import ExcellenceVADGerman
except ImportError:
    print("WARNING: Excellence VAD not found. Using stub.")
    ExcellenceVADGerman = None

from intent_classifier_german import IntentClassifierGerman
from turn_end_predictor import TurnEndPredictor
from memory_vad_bridge import MemoryVADBridge


@dataclass
class AudioFrame:
    """Single audio frame"""
    timestamp_ms: int
    user_audio: np.ndarray  # 160 samples (10ms at 16kHz)
    ai_audio: np.ndarray
    user_text: str = ""  # Transcription if available
    ai_text: str = ""


@dataclass
class TurnTakingDecision:
    """Real-time turn-taking decision"""
    action: str  # 'interrupt' | 'wait' | 'continue'
    confidence: float
    advance_warning_ms: float
    turn_end_prob: float
    predicted_prob: float
    adapted_threshold: float
    intent_type: Optional[str]
    should_interrupt_early: bool
    timestamp_ms: int


class ContinuousVADIntegration:
    """
    Continuous real-time VAD integration with all 4 stages

    Frame-by-frame processing pipeline:
    Audio (10ms) -> VAD -> Intent -> Predictor -> Memory -> Decision
    """

    def __init__(
        self,
        speaker_id: str,
        use_lstm: bool = True,
        lookahead_ms: int = 300,
        buffer_size: int = 50  # 500ms buffer
    ):
        # Stage 1: Excellence VAD (if available)
        if ExcellenceVADGerman:
            self.vad = ExcellenceVADGerman()
            print("[OK] Excellence VAD loaded")
        else:
            self.vad = None
            print("[STUB] Excellence VAD not available - using stub")

        # Stage 2: Intent Classifier
        self.intent_classifier = IntentClassifierGerman()

        # Stage 3: Turn-End Predictor
        self.predictor = TurnEndPredictor(
            use_lstm=use_lstm,
            lookahead_ms=lookahead_ms
        )

        # Stage 4: Memory-VAD Bridge
        self.memory = MemoryVADBridge(
            embedding_dim=64,
            min_observations=5
        )

        self.speaker_id = speaker_id
        self.current_threshold = 0.75

        # Frame buffer for state management
        self.frame_buffer = deque(maxlen=buffer_size)
        self.vad_history = deque(maxlen=30)  # 300ms history

        # Turn-taking state
        self.current_turn_start_ms = None
        self.last_user_text = ""
        self.last_intent = None

        # Statistics
        self.total_frames_processed = 0
        self.total_interruptions = 0
        self.total_early_interruptions = 0

    def process_frame(self, frame: AudioFrame) -> TurnTakingDecision:
        """
        Process single 10ms audio frame through full pipeline

        Args:
            frame: AudioFrame with user/AI audio + optional text

        Returns:
            TurnTakingDecision with action and metadata
        """
        self.total_frames_processed += 1
        self.frame_buffer.append(frame)

        # Stage 1: VAD Detection
        if self.vad:
            vad_result = self.vad.process_frame(
                frame.user_audio,
                frame.ai_audio,
                frame.ai_text
            )
        else:
            # Stub VAD (use energy-based simple detection)
            vad_result = self._stub_vad(frame)

        current_prob = vad_result.get('turn_end_prob', 0.5)
        prosody_prob = vad_result.get('prosody_prob', 0.5)
        user_speaking = vad_result.get('user_speaking', False)

        # Extract prosody features from result
        prosody = {
            'f0_slope': 0.0,  # Excellence VAD doesn't expose this
            'energy': prosody_prob * 100,  # Approximate from prosody_prob
            'speech_rate': 5.0,
            'duration': 10
        }

        # Track VAD history
        self.vad_history.append(current_prob)

        # Stage 2: Intent Classification (if user speaking and text available)
        intent_result = None
        if user_speaking and frame.user_text:
            intent_result = self.intent_classifier.classify(
                frame.user_text,
                prosody_features=prosody
            )
            self.last_intent = intent_result
            self.last_user_text = frame.user_text

            # Detect turn start
            if self.current_turn_start_ms is None:
                self.current_turn_start_ms = frame.timestamp_ms

        # Stage 3: Predictive Turn-End
        speaker_profile = self.memory.get_speaker_summary(self.speaker_id)

        prediction = self.predictor.predict(
            current_vad_prob=current_prob,
            prosody_features=prosody,
            intent_type=intent_result.intent_type if intent_result else None,
            speaker_profile=speaker_profile
        )

        # Stage 4: Memory Update (when turn ends)
        if self._is_turn_end(current_prob) and intent_result:
            gap_ms = frame.timestamp_ms - self.current_turn_start_ms

            memory_result = self.memory.observe_turn_taking(
                speaker_id=self.speaker_id,
                intent_type=intent_result.intent_type,
                intent_subtype=intent_result.intent_subtype,
                is_fpp=intent_result.is_fpp,
                expected_gap_ms=intent_result.expected_gap_ms,
                actual_gap_ms=gap_ms,
                prosody_features=prosody,
                context="conversation"
            )

            # Update threshold if confident
            if memory_result['should_adapt']:
                self.current_threshold = memory_result['adapted_threshold']

            # Reset turn state
            self.current_turn_start_ms = None

        # Decision Logic
        decision = self._make_decision(
            frame=frame,
            current_prob=current_prob,
            predicted_prob=prediction.predicted_prob_200ms,
            should_interrupt_early=prediction.should_interrupt_early,
            threshold=self.current_threshold,
            confidence=prediction.confidence,
            intent_result=intent_result
        )

        # Track interruptions
        if decision.action == 'interrupt':
            self.total_interruptions += 1
            if decision.should_interrupt_early:
                self.total_early_interruptions += 1

        return decision

    def _stub_vad(self, frame: AudioFrame) -> Dict:
        """Simple energy-based VAD stub when Excellence VAD not available"""
        user_energy = np.mean(np.abs(frame.user_audio))
        ai_energy = np.mean(np.abs(frame.ai_audio))

        # Simple heuristic: turn-end likely when user energy drops
        turn_end_prob = 1.0 - min(user_energy / 0.1, 1.0)  # Normalize

        return {
            'turn_end_prob': turn_end_prob,
            'user_speaking': user_energy > 0.01,
            'prosody_features': {
                'f0_slope': 0.0,
                'energy': user_energy * 100,
                'speech_rate': 5.0,
                'duration': 10
            }
        }

    def _is_turn_end(self, vad_prob: float) -> bool:
        """Detect if turn actually ended"""
        # Simple: if probability drops below threshold
        return vad_prob < 0.3 and len(self.vad_history) > 5

    def _make_decision(
        self,
        frame: AudioFrame,
        current_prob: float,
        predicted_prob: float,
        should_interrupt_early: bool,
        threshold: float,
        confidence: float,
        intent_result
    ) -> TurnTakingDecision:
        """
        Turn-taking decision logic with priority:
        1. Early interruption (based on prediction)
        2. Current threshold met
        3. Predicted turn-end approaching
        4. Continue listening
        """

        action = 'continue'
        advance_warning = 0.0

        # Priority 1: Early interruption (predictive)
        if should_interrupt_early:
            action = 'interrupt'
            advance_warning = self.predictor.lookahead_ms

        # Priority 2: Current turn-end detected
        elif current_prob > threshold:
            action = 'interrupt'
            advance_warning = 0.0

        # Priority 3: Predicted turn-end approaching
        elif predicted_prob > threshold + 0.1:
            action = 'wait'  # Prepare to interrupt
            advance_warning = self.predictor.lookahead_ms * 0.5

        return TurnTakingDecision(
            action=action,
            confidence=confidence,
            advance_warning_ms=advance_warning,
            turn_end_prob=current_prob,
            predicted_prob=predicted_prob,
            adapted_threshold=threshold,
            intent_type=intent_result.intent_type if intent_result else None,
            should_interrupt_early=should_interrupt_early,
            timestamp_ms=frame.timestamp_ms
        )

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            'total_frames': self.total_frames_processed,
            'total_interruptions': self.total_interruptions,
            'early_interruptions': self.total_early_interruptions,
            'early_rate': (self.total_early_interruptions / max(self.total_interruptions, 1)) * 100,
            'speaker_profile': self.memory.get_speaker_summary(self.speaker_id),
            'current_threshold': self.current_threshold
        }

    def reset_turn_state(self):
        """Reset turn state (e.g., when conversation ends)"""
        self.current_turn_start_ms = None
        self.last_user_text = ""
        self.last_intent = None
        self.frame_buffer.clear()
        self.vad_history.clear()


def demo_continuous_processing():
    """Demo: Simulate continuous audio processing"""
    print("=" * 80)
    print(" CONTINUOUS VAD INTEGRATION - Real-time Pipeline Demo")
    print("=" * 80)
    print()

    # Initialize system
    system = ContinuousVADIntegration(
        speaker_id="Guest_Demo",
        use_lstm=True,
        lookahead_ms=300
    )

    print("System initialized with:")
    print("  - Excellence VAD (Stage 1)")
    print("  - Intent Classifier (Stage 2)")
    print("  - Turn-End Predictor (Stage 3)")
    print("  - Memory-VAD Bridge (Stage 4)")
    print()
    print("Simulating conversation...")
    print("-" * 80)
    print()

    # Simulate conversation with frames
    conversation = [
        # Turn 1: Guest greeting
        {'text': 'Guten Tag', 'duration_frames': 80, 'energy_pattern': 'rising'},
        # Turn 2: Question
        {'text': 'Haben Sie ein Zimmer frei?', 'duration_frames': 120, 'energy_pattern': 'question'},
        # Turn 3: Confirmation
        {'text': 'Ja, ich nehme es', 'duration_frames': 90, 'energy_pattern': 'falling'},
    ]

    timestamp_ms = 0

    for turn_idx, turn in enumerate(conversation, 1):
        print(f"Turn {turn_idx}: {turn['text']}")

        # Simulate frames for this turn
        frames = turn['duration_frames']

        for frame_idx in range(frames):
            # Simulate audio energy pattern
            if turn['energy_pattern'] == 'rising':
                energy = 0.05 + (frame_idx / frames) * 0.05
            elif turn['energy_pattern'] == 'question':
                energy = 0.08 + 0.02 * np.sin(frame_idx / 10)
            else:  # falling
                energy = 0.08 - (frame_idx / frames) * 0.06

            # Create frame
            user_audio = np.random.randn(160) * energy
            ai_audio = np.random.randn(160) * 0.01

            frame = AudioFrame(
                timestamp_ms=timestamp_ms,
                user_audio=user_audio,
                ai_audio=ai_audio,
                user_text=turn['text'] if frame_idx == frames // 2 else "",
                ai_text=""
            )

            # Process frame
            decision = system.process_frame(frame)

            # Log key events
            if decision.action == 'interrupt':
                print(f"  [{timestamp_ms}ms] INTERRUPT - Warning: {decision.advance_warning_ms:.0f}ms, "
                      f"Intent: {decision.intent_type}, Threshold: {decision.adapted_threshold:.3f}")
            elif decision.action == 'wait':
                print(f"  [{timestamp_ms}ms] WAIT - Preparing to interrupt soon")

            timestamp_ms += 10

        print()

    # Show statistics
    print("=" * 80)
    print(" STATISTICS")
    print("=" * 80)
    print()

    stats = system.get_statistics()
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total interruptions: {stats['total_interruptions']}")
    print(f"Early interruptions: {stats['early_interruptions']} ({stats['early_rate']:.1f}%)")
    print(f"Current threshold: {stats['current_threshold']:.3f}")
    print()

    profile = stats['speaker_profile']
    print(f"Speaker Profile:")
    print(f"  Total turns: {profile['total_interactions']}")
    print(f"  Confidence: {profile['confidence']:.1%}")
    print(f"  Avg gap: {profile['timing']['avg_gap_ms']:.1f}ms")
    print()

    print("=" * 80)
    print(" Demo Complete - System ready for real-time audio")
    print("=" * 80)


if __name__ == "__main__":
    demo_continuous_processing()
