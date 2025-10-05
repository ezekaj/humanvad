"""
Excellence VAD - Human Telephone Model
=======================================

Hybrid prosody + semantic turn-taking detection matching human performance.

Based on neuroscience research:
- Humans achieve 95% accuracy on telephone using BOTH:
  1. Prosodic cues (pitch, timing, energy) - 40-50% contribution
  2. Semantic/syntactic completion - 50-60% contribution

Target: 90-95% accuracy matching human telephone performance
"""

import numpy as np
from collections import deque
from typing import Dict, Optional
import time
from production_vad import ProductionVAD
import re


class SemanticCompletionDetector:
    """
    Detects if utterance is semantically/syntactically complete

    Simulates human "prediction-by-simulation" mechanism
    Uses linguistic rules + patterns (in production would use LLM)
    """

    def __init__(self):
        # Common sentence-final patterns
        self.completion_patterns = [
            # Statements ending with period-like prosody
            r'\b(yes|no|okay|sure|thanks|thank you|alright|fine|perfect|great)\s*$',
            r'\b(done|finished|complete|ready|good)\s*$',

            # Complete grammatical structures
            r'\bi (am|was|will be|would be|have been|can|could|should|will|would)\s+\w+\s*$',
            r'\byou (are|were|will be|can|could|should|have|had)\s+\w+\s*$',
            r'\b(it is|that is|this is|there is|here is)\s+\w+\s*$',

            # Questions (complete)
            r'\b(what|where|when|why|who|how)\s+.*\?$',
            r'\b(is|are|was|were|will|would|can|could|should)\s+\w+.*\?$',

            # Sentence-final structures (time/place)
            r'\s+(today|tomorrow|yesterday|now|then|there|here)\s*$',
            r'\s+(tonight|morning|afternoon|evening|night)\s*$',
            r'\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*$',
            r'\s+(minute|hour|day|week|month|year)s?\s*$',

            # Complete sentences ending with action verbs + object
            r'\b(check in|check out|book|arrive|leave|meet|schedule)\s+.*$',

            # Complete statements (verb + prepositional phrase ending)
            r'\s+(at the|in the|on the|from the|with the|to the)\s+\w+(\s+\w+)*\s*$',

            # Default: Long sentences (>8 words) without incomplete markers tend to be complete
        ]

        # Incomplete patterns (turn-holding)
        self.incomplete_patterns = [
            # Conjunctions (expecting continuation)
            r'\b(and|but|or|because|since|although|if|unless|while|when)\s*$',
            r'\b(so|therefore|however|moreover|furthermore)\s*$',

            # Prepositions (incomplete phrases)
            r'\b(in|on|at|to|from|with|by|for|of|about)\s*$',

            # Auxiliary verbs (expecting main verb)
            r'\b(am|is|are|was|were|will|would|can|could|should|have|has|had)\s*$',

            # Articles (expecting noun)
            r'\b(a|an|the)\s*$',

            # Possessives
            r'\b(my|your|his|her|its|our|their)\s*$',

            # Incomplete phrases
            r'\bi (want to|need to|have to|going to)\s*$',
            r'\blet me\s*$',
        ]

        # Recent text buffer
        self.text_buffer = deque(maxlen=50)  # Last ~10 seconds of text

    def is_complete(self, text: str) -> Dict:
        """
        Check if text represents semantically/syntactically complete utterance

        Returns:
            complete_prob: 0.0-1.0 probability of completion
            reason: Why it's complete/incomplete
        """

        if not text or len(text.strip()) == 0:
            return {'complete_prob': 0.0, 'reason': 'empty'}

        text = text.strip().lower()

        # Store in buffer
        self.text_buffer.append(text)

        # Get last few words (most recent utterance)
        recent_text = ' '.join(list(self.text_buffer)[-10:])

        score = 0.5  # Default: uncertain
        reason = "unknown"

        # 1. Check for INCOMPLETE patterns (turn-holding)
        for pattern in self.incomplete_patterns:
            if re.search(pattern, recent_text, re.IGNORECASE):
                score = max(score - 0.3, 0.0)
                reason = f"incomplete_pattern: {pattern[:30]}"
                break

        # 2. Check for COMPLETE patterns (turn-ending)
        for pattern in self.completion_patterns:
            if re.search(pattern, recent_text, re.IGNORECASE):
                score = min(score + 0.4, 1.0)
                reason = f"complete_pattern: {pattern[:30]}"
                break

        # 3. Question mark = complete
        if '?' in text:
            score = min(score + 0.3, 1.0)
            reason = "question_mark"

        # 4. Trailing off (um, uh, er) = incomplete
        if re.search(r'\b(um|uh|er|ah|hmm)\s*$', text):
            score = max(score - 0.4, 0.0)
            reason = "trailing_filler"

        # 5. Word count heuristics
        word_count = len(text.split())

        # Very short utterances (3-8 words) often complete if no incomplete markers
        if 3 <= word_count <= 8 and score >= 0.5:
            score = min(score + 0.1, 1.0)
            reason = "short_complete"

        # Longer sentences (>8 words) default to complete if no incomplete markers found
        elif word_count > 8 and score == 0.5:
            score = 0.7  # Likely complete
            reason = "long_sentence"

        return {
            'complete_prob': score,
            'reason': reason
        }


class ExcellenceVAD:
    """
    Excellence-level turn-taking VAD matching human telephone performance

    Combines:
    1. Prosody (ProductionVAD: pitch, timing, energy) - 40-50%
    2. Semantics (completion detection) - 50-60%

    Target: 90-95% accuracy
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        turn_end_threshold: float = 0.75
    ):
        self.sr = sample_rate
        self.turn_end_threshold = turn_end_threshold

        # Prosodic detection
        self.prosody_detector = ProductionVAD(sample_rate=sample_rate)

        # Semantic detection
        self.semantic_detector = SemanticCompletionDetector()

        # Audio buffers
        self.user_buffer = deque(maxlen=sample_rate * 3)
        self.ai_buffer = deque(maxlen=sample_rate * 3)

        # Prosodic features history
        self.energy_history = deque(maxlen=150)  # 3s at 10ms frames
        self.confidence_history = deque(maxlen=150)

        # Performance tracking
        self.processing_times = deque(maxlen=100)

    def _analyze_prosody(self, audio: np.ndarray, is_speech: bool) -> Dict:
        """
        Analyze prosodic turn-end cues

        Returns:
            prosody_turn_end_prob: 0.0-1.0
        """

        if len(self.energy_history) < 20:
            return {'prosody_turn_end_prob': 0.3}

        recent_energy = list(self.energy_history)[-30:]

        score = 0.0

        # 1. Energy trend (falling = ending)
        if len(recent_energy) >= 20:
            first_half = np.mean(recent_energy[:10])
            second_half = np.mean(recent_energy[-10:])

            if second_half < first_half * 0.7:
                score += 0.4  # Strong energy drop
            elif second_half < first_half * 0.85:
                score += 0.2  # Moderate drop

        # 2. Absolute energy level (very low = ending)
        if len(recent_energy) > 0:
            current = recent_energy[-1]
            if current < 0.01:
                score += 0.3
            elif current < 0.05:
                score += 0.15

        # 3. Pause detection (silence after speech = boundary)
        if not is_speech and len(self.energy_history) >= 10:
            # Check if we had speech recently
            recent_had_speech = any(e > 0.05 for e in list(self.energy_history)[-20:-5])
            if recent_had_speech:
                score += 0.3  # Pause after speech

        return {
            'prosody_turn_end_prob': min(score, 1.0)
        }

    def process_frame(
        self,
        user_frame: np.ndarray,
        ai_frame: np.ndarray,
        ai_text: Optional[str] = None
    ) -> Dict:
        """
        Process stereo audio + AI text

        Args:
            user_frame: User audio (10ms, 160 samples at 16kHz)
            ai_frame: AI audio (10ms, 160 samples at 16kHz)
            ai_text: Current AI utterance text (optional, for semantic analysis)
        """

        start_time = time.perf_counter()

        # Update buffers
        self.user_buffer.extend(user_frame)
        self.ai_buffer.extend(ai_frame)

        # 1. Fast prosodic speech detection
        user_result = self.prosody_detector.detect_frame(user_frame)
        ai_result = self.prosody_detector.detect_frame(ai_frame)

        user_speaking = user_result['is_speech']
        ai_speaking = ai_result['is_speech']

        # Track prosodic features
        energy = np.sqrt(np.mean(ai_frame ** 2))
        self.energy_history.append(energy)
        self.confidence_history.append(ai_result['confidence'])

        # ALWAYS calculate turn-end probability (continuous monitoring)
        # 2. Prosodic turn-end analysis (40-50% weight)
        prosody_result = self._analyze_prosody(ai_frame, ai_speaking)
        prosody_prob = prosody_result['prosody_turn_end_prob']

        # 3. Semantic turn-end analysis (50-60% weight)
        if ai_text and len(ai_text.strip()) > 0:
            semantic_result = self.semantic_detector.is_complete(ai_text)
            semantic_prob = semantic_result['complete_prob']
            semantic_reason = semantic_result['reason']
        else:
            # No text available - rely on prosody only
            semantic_prob = 0.5
            semantic_reason = "no_text"

        # 4. FUSION (matching human weighting)
        # Research shows: semantics 50-60%, prosody 40-50%
        final_turn_end_prob = (
            0.45 * prosody_prob +      # Prosodic cues
            0.55 * semantic_prob        # Semantic/syntactic completion
        )

        # If user not speaking, return monitoring info
        if not user_speaking:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(latency_ms)
            return {
                'action': 'continue',
                'user_speaking': False,
                'ai_speaking': ai_speaking,
                'turn_end_prob': final_turn_end_prob,
                'prosody_prob': prosody_prob,
                'semantic_prob': semantic_prob,
                'latency_ms': latency_ms
            }

        # 5. Decision logic
        if user_speaking and ai_speaking:
            # OVERLAP: User speaking while AI speaking
            if final_turn_end_prob >= self.turn_end_threshold:
                # AI utterance is complete - user doing natural turn-taking
                action = "wait_for_ai_completion"
                reasoning = "natural_turn_taking"
            else:
                # AI mid-sentence - user interrupting
                action = "interrupt_ai_immediately"
                reasoning = "interruption"
        elif user_speaking:
            # User speaking, AI silent - clear to proceed
            action = "interrupt_ai_immediately"
            reasoning = "ai_silent"
        else:
            action = "continue"
            reasoning = "user_silent"

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        return {
            'action': action,
            'reasoning': reasoning,
            'turn_end_prob': final_turn_end_prob,
            'user_speaking': user_speaking,
            'ai_speaking': ai_speaking,
            'overlap': user_speaking and ai_speaking,

            # Debug info
            'prosody_prob': prosody_prob,
            'semantic_prob': semantic_prob,
            'semantic_reason': semantic_reason if ai_text else None,
            'ai_text': ai_text,

            # Performance
            'latency_ms': latency_ms
        }

    def get_stats(self) -> Dict:
        """Performance statistics"""
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
        self.user_buffer.clear()
        self.ai_buffer.clear()
        self.energy_history.clear()
        self.confidence_history.clear()
        self.semantic_detector.text_buffer.clear()


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" EXCELLENCE VAD - Human Telephone Model")
    print("=" * 80)
    print()
    print("Matching human audio-only turn-taking performance:")
    print()
    print("Components:")
    print("  1. Prosodic analysis (pitch, timing, energy) - 45% weight")
    print("  2. Semantic completion detection - 55% weight")
    print()
    print("Target: 90-95% accuracy (human telephone level)")
    print()

    sr = 16000
    vad = ExcellenceVAD(sample_rate=sr)

    # Speed test
    print("Speed Test:")
    print("-" * 80)

    test_frame = np.random.randn(160)
    test_text = "I think we should go there tomorrow"

    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.process_frame(test_frame, test_frame, test_text)
    avg_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Average latency: {avg_time:.2f}ms per frame")
    print(f"Target: <10ms {'PASS' if avg_time < 10 else 'FAIL'}")
    print()

    stats = vad.get_stats()
    print(f"P50: {stats['p50_latency_ms']:.2f}ms")
    print(f"P95: {stats['p95_latency_ms']:.2f}ms")
    print()

    # Example semantic detection
    print("Semantic Completion Examples:")
    print("-" * 80)

    examples = [
        ("I am going to the store", "Should be complete"),
        ("I am going to", "Incomplete (preposition)"),
        ("What time is it?", "Complete (question)"),
        ("Because I think that", "Incomplete (conjunction)"),
        ("Okay thanks", "Complete (acknowledgment)"),
        ("Let me", "Incomplete (auxiliary)"),
    ]

    detector = SemanticCompletionDetector()
    for text, expected in examples:
        result = detector.is_complete(text)
        status = "COMPLETE" if result['complete_prob'] > 0.6 else "INCOMPLETE"
        print(f"{text:<30} -> {status:<12} ({result['complete_prob']:.1%}) | {result['reason']}")

    print()
    print("Run test_excellence.py for full evaluation")
    print()


if __name__ == "__main__":
    demo()
