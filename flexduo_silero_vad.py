"""
FlexDuo + Silero VAD - Production-Grade Duplex VAD
===================================================

Combines:
1. Silero VAD (neural network, trained on 6000+ languages)
2. FlexDuo architecture (7-state FSM + semantic buffering)

Based on arXiv:2502.13472v1 (FlexDuo, Feb 2025)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
from collections import deque
import sys
import os

# Add path for intent classifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'human-speech-detection'))
try:
    from intent_classifier_german import IntentClassifierGerman
    INTENT_AVAILABLE = True
except ImportError:
    INTENT_AVAILABLE = False
    print("Warning: Intent classifier not available")


# FlexDuo 7-State Model (from paper)
class FlexDuoState(Enum):
    """7 dialogue states from FlexDuo paper"""
    IDLE = "idle"                          # No activity, filter noise
    AI_SPEAKING = "ai_speaking"            # AI is speaking
    USER_SPEAKING = "user_speaking"        # User is speaking
    USER_INTERRUPT = "user_interrupt"      # User interrupts AI (barge-in)
    AI_BACKCHANNELING = "ai_backchanneling"  # AI gives feedback during user speech
    USER_BACKCHANNELING = "user_backchanneling"  # User gives feedback during AI speech
    SIMULTANEOUS = "simultaneous"          # Both speaking (resolve to interrupt/backchannel)


@dataclass
class DuplexFrame:
    """Single frame for duplex processing"""
    timestamp_ms: int
    user_audio: np.ndarray
    ai_audio: np.ndarray
    user_text: str = ""
    ai_text: str = ""


@dataclass
class FlexDuoDecision:
    """FlexDuo VAD decision"""
    state: FlexDuoState
    user_speaking: bool
    ai_speaking: bool
    barge_in_detected: bool
    should_stop_ai: bool
    user_vad_prob: float
    ai_vad_prob: float
    confidence: float
    timestamp_ms: int
    intent: Optional[str] = None  # For future intent classification


class SlidingWindowBuffer:
    """FlexDuo sliding window for semantic buffering"""
    def __init__(self, window_ms: int = 200, frame_ms: int = 30):
        self.window_frames = window_ms // frame_ms
        self.buffer = deque(maxlen=self.window_frames)

    def add(self, user_prob: float, ai_prob: float):
        self.buffer.append({'user': user_prob, 'ai': ai_prob})

    def get_buffer(self):
        return list(self.buffer)

    def user_sustained(self, threshold: float = 0.5) -> bool:
        """Check if user speech is sustained across window"""
        if len(self.buffer) < self.window_frames:
            return False
        return sum(1 for f in self.buffer if f['user'] > threshold) >= self.window_frames * 0.8

    def ai_sustained(self, threshold: float = 0.5) -> bool:
        """Check if AI speech is sustained across window"""
        if len(self.buffer) < self.window_frames:
            return False
        return sum(1 for f in self.buffer if f['ai'] > threshold) >= self.window_frames * 0.8


class ContextManager:
    """FlexDuo context manager - tracks conversation history"""
    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)
        self.last_speaker = None
        self.conversation_turns = 0

    def update(self, state: FlexDuoState, user_text: str = "", ai_text: str = ""):
        self.history.append({
            'state': state,
            'user_text': user_text,
            'ai_text': ai_text
        })

        if state == FlexDuoState.USER_SPEAKING:
            self.last_speaker = "user"
            self.conversation_turns += 1
        elif state == FlexDuoState.AI_SPEAKING:
            self.last_speaker = "ai"

    def get_context(self) -> Dict:
        return {
            'history': list(self.history),
            'last_speaker': self.last_speaker,
            'turns': self.conversation_turns
        }


class FlexDuoStateFSM:
    """FlexDuo Finite State Machine (7 states)"""
    def __init__(self,
                 user_threshold: float = 0.5,
                 ai_threshold: float = 0.5,
                 interrupt_threshold: float = 0.7,
                 semantic_buffer_ms: int = 200):
        self.current_state = FlexDuoState.IDLE
        self.user_threshold = user_threshold
        self.ai_threshold = ai_threshold
        self.interrupt_threshold = interrupt_threshold
        self.semantic_buffer_ms = semantic_buffer_ms

        # Track potential interrupt start
        self.potential_interrupt_start_ms = None

        # FIX: Track AI recent activity (for barge-in from IDLE)
        self.ai_last_active_ms = None
        self.ai_recent_window_ms = 500  # Consider AI "recently speaking" within 500ms

    def transition(self,
                   user_prob: float,
                   ai_prob: float,
                   window: SlidingWindowBuffer,
                   timestamp_ms: int) -> tuple[FlexDuoState, bool, bool]:
        """
        FlexDuo state transition logic

        Returns:
            (new_state, barge_in_detected, should_stop_ai)
        """
        barge_in_detected = False
        should_stop_ai = False

        user_active = user_prob > self.user_threshold
        ai_active = ai_prob > self.ai_threshold

        # Track AI activity
        if ai_active:
            self.ai_last_active_ms = timestamp_ms

        # Check if AI recently spoke (within 500ms)
        ai_recently_active = (self.ai_last_active_ms is not None and
                              timestamp_ms - self.ai_last_active_ms < self.ai_recent_window_ms)

        # State: IDLE
        if self.current_state == FlexDuoState.IDLE:
            if ai_active and not user_active:
                return FlexDuoState.AI_SPEAKING, False, False
            elif user_active and not ai_active:
                # Check if this is a late barge-in (user interrupting right after AI finished)
                if ai_recently_active and user_prob > self.interrupt_threshold:
                    # User interrupting within 500ms of AI finishing - treat as barge-in
                    if self.potential_interrupt_start_ms is None:
                        self.potential_interrupt_start_ms = timestamp_ms

                    buffer_duration = timestamp_ms - self.potential_interrupt_start_ms
                    if buffer_duration >= self.semantic_buffer_ms:
                        # Sustained for 200ms - confirm barge-in
                        return FlexDuoState.USER_INTERRUPT, True, True
                    else:
                        # Still buffering
                        return FlexDuoState.IDLE, False, False
                else:
                    # Normal user speech start
                    return FlexDuoState.USER_SPEAKING, False, False
            elif user_active and ai_active:
                # Both active from IDLE - check if late barge-in
                if ai_recently_active and user_prob > self.interrupt_threshold:
                    # User barging in while AI also speaking
                    if self.potential_interrupt_start_ms is None:
                        self.potential_interrupt_start_ms = timestamp_ms

                    buffer_duration = timestamp_ms - self.potential_interrupt_start_ms
                    if buffer_duration >= self.semantic_buffer_ms:
                        # Sustained for 200ms - confirm barge-in
                        return FlexDuoState.USER_INTERRUPT, True, True
                    else:
                        # Still buffering - stay in simultaneous
                        return FlexDuoState.SIMULTANEOUS, False, False
                else:
                    # Simultaneous start, default to user
                    return FlexDuoState.USER_SPEAKING, False, False
            else:
                # Stay in IDLE, filter noise
                self.potential_interrupt_start_ms = None  # Reset buffer
                return FlexDuoState.IDLE, False, False

        # State: AI_SPEAKING (critical - detect barge-in)
        elif self.current_state == FlexDuoState.AI_SPEAKING:
            if not ai_active and not user_active:
                # AI finished
                return FlexDuoState.IDLE, False, False
            elif not ai_active and user_active:
                # AI finished, user starts (normal turn-taking)
                return FlexDuoState.USER_SPEAKING, False, False
            elif ai_active and user_active:
                # Potential barge-in or backchannel
                if user_prob > self.interrupt_threshold:
                    # Strong user signal - check semantic buffer
                    if self.potential_interrupt_start_ms is None:
                        self.potential_interrupt_start_ms = timestamp_ms

                    buffer_duration = timestamp_ms - self.potential_interrupt_start_ms

                    if buffer_duration >= self.semantic_buffer_ms:
                        # Confirmed barge-in (sustained for 200ms)
                        barge_in_detected = True
                        should_stop_ai = True
                        self.potential_interrupt_start_ms = None
                        return FlexDuoState.USER_INTERRUPT, barge_in_detected, should_stop_ai
                    else:
                        # Still buffering
                        return FlexDuoState.AI_SPEAKING, False, False
                else:
                    # Weak user signal - backchannel (e.g., "mhm", "yeah")
                    if self.potential_interrupt_start_ms is not None:
                        self.potential_interrupt_start_ms = None  # Reset
                    return FlexDuoState.USER_BACKCHANNELING, False, False
            else:
                # AI continues speaking
                if self.potential_interrupt_start_ms is not None:
                    self.potential_interrupt_start_ms = None  # Reset
                return FlexDuoState.AI_SPEAKING, False, False

        # State: USER_SPEAKING
        elif self.current_state == FlexDuoState.USER_SPEAKING:
            if user_active and not ai_active:
                # User continues
                return FlexDuoState.USER_SPEAKING, False, False
            elif not user_active and ai_active:
                # User finished, AI responds
                return FlexDuoState.AI_SPEAKING, False, False
            elif not user_active and not ai_active:
                # User finished
                return FlexDuoState.IDLE, False, False
            else:
                # AI backchannels during user speech
                return FlexDuoState.AI_BACKCHANNELING, False, False

        # State: USER_INTERRUPT
        elif self.current_state == FlexDuoState.USER_INTERRUPT:
            if user_active:
                # User continues after interrupt
                return FlexDuoState.USER_SPEAKING, False, False
            else:
                return FlexDuoState.IDLE, False, False

        # State: USER_BACKCHANNELING
        elif self.current_state == FlexDuoState.USER_BACKCHANNELING:
            if not user_active and ai_active:
                # Back to AI speaking
                return FlexDuoState.AI_SPEAKING, False, False
            elif user_active and user_prob > self.interrupt_threshold:
                # Backchannel becomes interrupt
                return FlexDuoState.USER_INTERRUPT, True, True
            else:
                return FlexDuoState.USER_BACKCHANNELING, False, False

        # State: AI_BACKCHANNELING
        elif self.current_state == FlexDuoState.AI_BACKCHANNELING:
            if user_active and not ai_active:
                # Back to user speaking
                return FlexDuoState.USER_SPEAKING, False, False
            else:
                return FlexDuoState.AI_BACKCHANNELING, False, False

        # State: SIMULTANEOUS
        elif self.current_state == FlexDuoState.SIMULTANEOUS:
            # Check if this is a barge-in scenario (user interrupting while AI recently spoke)
            if ai_recently_active and user_prob > self.interrupt_threshold:
                # Continue buffering for barge-in
                if self.potential_interrupt_start_ms is None:
                    self.potential_interrupt_start_ms = timestamp_ms

                buffer_duration = timestamp_ms - self.potential_interrupt_start_ms
                if buffer_duration >= self.semantic_buffer_ms:
                    # Sustained for 200ms - confirm barge-in
                    return FlexDuoState.USER_INTERRUPT, True, True
                else:
                    # Still buffering
                    return FlexDuoState.SIMULTANEOUS, False, False
            else:
                # Resolve simultaneous to dominant speaker
                if user_prob > ai_prob:
                    return FlexDuoState.USER_SPEAKING, False, False
                else:
                    return FlexDuoState.AI_SPEAKING, False, False

        # Default: stay in current state
        return self.current_state, False, False


class FlexDuoSileroVAD:
    """
    FlexDuo + Silero VAD - Production Duplex VAD

    Combines:
    - Silero VAD (neural network, 6000+ languages)
    - FlexDuo 7-state FSM
    - Semantic buffering (sliding window)
    - Context tracking
    """

    def __init__(self,
                 sample_rate: int = 8000,
                 frame_duration_ms: int = 30,
                 user_threshold: float = 0.35,  # Lowered from 0.5 (Silero is conservative)
                 ai_threshold: float = 0.35,     # Lowered from 0.5
                 interrupt_threshold: float = 0.55,  # Lowered from 0.7
                 semantic_buffer_ms: int = 200,
                 enable_intent: bool = True):

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.enable_intent = enable_intent and INTENT_AVAILABLE

        # Load Silero VAD model
        print("Loading Silero VAD model...")
        self.silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        print("Silero VAD loaded successfully!")

        # Intent classifier (optional)
        self.intent_classifier = None
        if self.enable_intent:
            try:
                self.intent_classifier = IntentClassifierGerman()
                print("Intent classifier loaded!")
            except Exception as e:
                print(f"Intent classifier failed to load: {e}")
                self.enable_intent = False

        # FlexDuo components
        self.state_fsm = FlexDuoStateFSM(
            user_threshold=user_threshold,
            ai_threshold=ai_threshold,
            interrupt_threshold=interrupt_threshold,
            semantic_buffer_ms=semantic_buffer_ms
        )
        self.sliding_window = SlidingWindowBuffer(
            window_ms=semantic_buffer_ms,
            frame_ms=frame_duration_ms
        )
        self.context_manager = ContextManager()

        # Statistics
        self.total_frames = 0
        self.total_barge_ins = 0
        self.false_barge_ins_prevented = 0

    def _silero_vad(self, audio: np.ndarray) -> float:
        """
        Get VAD probability from Silero

        Args:
            audio: numpy array (frame_samples,)

        Returns:
            VAD probability (0.0-1.0)
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Silero VAD expects 16kHz, resample if needed
        target_sr = 16000
        if self.sample_rate != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=target_sr
            )
            audio_tensor = resampler(audio_tensor)

        # Silero requires EXACTLY 512 samples @ 16kHz (32ms)
        required_samples = 512
        if len(audio_tensor) < required_samples:
            # Pad
            padding = required_samples - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        elif len(audio_tensor) > required_samples:
            # Truncate
            audio_tensor = audio_tensor[:required_samples]

        # Get VAD probability
        with torch.no_grad():
            vad_prob = self.silero_model(audio_tensor, target_sr).item()

        return vad_prob

    def process_frame(self, frame: DuplexFrame) -> FlexDuoDecision:
        """
        Process single duplex frame

        Args:
            frame: DuplexFrame with user + AI audio

        Returns:
            FlexDuoDecision with state and actions
        """
        self.total_frames += 1

        # Get VAD probabilities from Silero
        user_vad_prob = self._silero_vad(frame.user_audio)
        ai_vad_prob = self._silero_vad(frame.ai_audio)

        # Update sliding window
        self.sliding_window.add(user_vad_prob, ai_vad_prob)

        # FlexDuo state transition
        new_state, barge_in_detected, should_stop_ai = self.state_fsm.transition(
            user_prob=user_vad_prob,
            ai_prob=ai_vad_prob,
            window=self.sliding_window,
            timestamp_ms=frame.timestamp_ms
        )

        # Update context
        self.context_manager.update(new_state, frame.user_text, frame.ai_text)

        # Intent classification (if enabled and barge-in detected)
        intent_type = None
        if barge_in_detected and self.enable_intent and frame.user_text:
            try:
                intent_result = self.intent_classifier.classify(frame.user_text)
                intent_type = f"{intent_result.intent_type}:{intent_result.intent_subtype}"

                # BACKCHANNEL: Don't stop AI for minimal responses
                if intent_result.intent_type == 'discourse' and intent_result.intent_subtype == 'backchannel':
                    should_stop_ai = False
                    barge_in_detected = False  # Reclassify as not a barge-in
                    self.false_barge_ins_prevented += 1

                # ACKNOWLEDGMENT: Don't stop AI for "ja", "okay", etc.
                elif intent_result.intent_type == 'response' and intent_result.intent_subtype in ['acknowledge', 'confirm']:
                    # Only stop if it's a strong confirmation with more content
                    if len(frame.user_text.split()) <= 2:  # Short response
                        should_stop_ai = False
                        barge_in_detected = False
                        self.false_barge_ins_prevented += 1

                # QUESTION/REQUEST: Always stop AI to let user speak
                elif intent_result.intent_type in ['question', 'request']:
                    should_stop_ai = True

            except Exception as e:
                print(f"Intent classification error: {e}")

        # Update statistics
        if barge_in_detected:
            self.total_barge_ins += 1

        # Compute confidence
        confidence = abs(user_vad_prob - ai_vad_prob)

        # Update FSM state
        self.state_fsm.current_state = new_state

        return FlexDuoDecision(
            state=new_state,
            user_speaking=user_vad_prob > self.state_fsm.user_threshold,
            ai_speaking=ai_vad_prob > self.state_fsm.ai_threshold,
            barge_in_detected=barge_in_detected,
            should_stop_ai=should_stop_ai,
            user_vad_prob=user_vad_prob,
            ai_vad_prob=ai_vad_prob,
            confidence=confidence,
            timestamp_ms=frame.timestamp_ms,
            intent=intent_type
        )

    def get_statistics(self) -> Dict:
        """Get FlexDuo VAD statistics"""
        context = self.context_manager.get_context()

        return {
            'total_frames': self.total_frames,
            'total_barge_ins': self.total_barge_ins,
            'false_barge_ins_prevented': self.false_barge_ins_prevented,
            'current_state': self.state_fsm.current_state.value,
            'conversation_turns': context['turns'],
            'last_speaker': context['last_speaker']
        }

    def reset(self):
        """Reset state for new conversation"""
        self.state_fsm.current_state = FlexDuoState.IDLE
        self.state_fsm.potential_interrupt_start_ms = None
        self.sliding_window = SlidingWindowBuffer(
            window_ms=self.state_fsm.semantic_buffer_ms,
            frame_ms=self.frame_duration_ms
        )
        self.context_manager = ContextManager()


def demo():
    """Demo: FlexDuo + Silero on simple scenario"""
    print("="*80)
    print(" FLEXDUO + SILERO VAD DEMO")
    print("="*80)
    print()

    # Initialize
    vad = FlexDuoSileroVAD(
        sample_rate=8000,
        frame_duration_ms=30,
        user_threshold=0.5,
        ai_threshold=0.5,
        interrupt_threshold=0.7,
        semantic_buffer_ms=200
    )

    print("System initialized:")
    print("  - Silero VAD (neural network)")
    print("  - FlexDuo 7-state FSM")
    print("  - Semantic buffer: 200ms")
    print()

    # Simulate: AI speaks, user interrupts
    print("Scenario: AI speaking, user interrupts")
    print("-"*80)
    print()

    timestamp_ms = 0
    frame_samples = int(8000 * 0.03)  # 30ms at 8kHz

    # AI speaking (500ms)
    print("AI: Speaking (500ms)")
    for _ in range(16):  # 16 frames * 30ms = 480ms
        ai_audio = np.random.randn(frame_samples) * 0.3
        user_audio = np.random.randn(frame_samples) * 0.01

        frame = DuplexFrame(timestamp_ms, user_audio, ai_audio)
        decision = vad.process_frame(frame)

        timestamp_ms += 30

    print(f"  State: {decision.state.value}")
    print()

    # User interrupts (300ms)
    print("User: Interrupts (300ms)")
    for i in range(10):  # 10 frames * 30ms = 300ms
        ai_audio = np.random.randn(frame_samples) * 0.3
        user_audio = np.random.randn(frame_samples) * 0.8  # Strong signal

        frame = DuplexFrame(timestamp_ms, user_audio, ai_audio)
        decision = vad.process_frame(frame)

        if decision.barge_in_detected:
            print(f"  [{timestamp_ms}ms] BARGE-IN DETECTED!")
            print(f"  State: {decision.state.value}")
            print(f"  Should stop AI: {decision.should_stop_ai}")

        timestamp_ms += 30

    print()
    stats = vad.get_statistics()
    print("Statistics:")
    print(f"  Total barge-ins: {stats['total_barge_ins']}")
    print(f"  Conversation turns: {stats['conversation_turns']}")
    print(f"  Final state: {stats['current_state']}")
    print()
    print("="*80)


if __name__ == "__main__":
    demo()
