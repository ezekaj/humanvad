"""
Memory-VAD Bridge: Speaker-Specific Turn-Taking Profiles
=========================================================

Connects neuro-memory system with VAD/Intent classifier for personalized
turn-taking adaptation based on learned speaker patterns.

Research Foundation:
- Voice Activity Projection (VAP) - Cross-attention transformers
- Speaker-aware timing simulation (arXiv 2509.15808)
- Prosody adaptation via speaker profiles
- Bayesian surprise detection for novelty in timing patterns
- EM-LLM episodic memory architecture

Architecture:
1. Speaker Profile Learning - Track individual timing patterns
2. Pattern Recognition - Detect deviation from expected behavior
3. Adaptive Thresholds - Personalize turn-end detection per speaker
4. Memory Integration - Store and retrieve speaker-specific patterns
"""

import sys
sys.path.append('../neuro-memory-agent/src')

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Neuro-memory components
from surprise import BayesianSurpriseEngine, SurpriseConfig
from memory import EpisodicMemoryStore, EpisodicMemoryConfig


@dataclass
class SpeakerProfile:
    """Individual speaker's turn-taking characteristics"""
    speaker_id: str

    # Timing patterns (learned)
    avg_turn_gap_ms: float = 200.0  # Average gap after speaker finishes
    std_turn_gap_ms: float = 100.0  # Variation in gaps

    # Intent-specific gaps (learned per intent type)
    intent_gaps: Dict[str, float] = None  # e.g., {'question': 150, 'statement': 400}

    # Prosodic patterns
    avg_speech_rate: float = 5.0  # syllables/sec
    typical_f0_range: Tuple[float, float] = (100.0, 300.0)  # Hz

    # Interruption behavior
    interruption_tolerance: float = 0.75  # Threshold for accepting interruptions
    polite_waiting_time_ms: float = 300.0  # How long they wait before speaking

    # Conversation style
    turn_length_avg_ms: float = 3000.0  # Average utterance length
    backchannel_frequency: float = 0.2  # How often they use "uh-huh", "ja"

    # Learning metadata
    total_turns: int = 0
    last_updated: datetime = None
    confidence_score: float = 0.0  # 0-1, based on data quantity

    def __post_init__(self):
        if self.intent_gaps is None:
            self.intent_gaps = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class TurnTakingEvent:
    """Turn-taking observation for memory storage"""
    speaker_id: str
    timestamp: datetime

    # Turn characteristics
    intent_type: str
    intent_subtype: str
    is_fpp: bool  # First pair part
    expected_gap_ms: int
    actual_gap_ms: int  # Measured actual gap

    # Prosody features
    final_f0_slope: float  # Rising/falling intonation
    speech_rate: float  # syllables/sec
    utterance_duration_ms: int

    # Context
    was_interrupted: bool
    interruption_timing_ms: Optional[int]  # When interruption occurred
    conversation_context: str  # "hotel_booking", "complaint", etc.

    # Surprise/novelty
    timing_surprise: float  # How unexpected was this timing?


class MemoryVADBridge:
    """
    Bridge between neuro-memory system and VAD/Intent classifier

    Learns speaker-specific turn-taking patterns and adapts VAD thresholds
    based on accumulated experience with each speaker.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        surprise_threshold: float = 0.7,
        min_observations: int = 10  # Min turns before trusting profile
    ):
        # Speaker profiles (in-memory, fast access)
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}

        # Episodic memory (for consolidation and retrieval)
        self.memory = EpisodicMemoryStore(
            config=EpisodicMemoryConfig(
                max_episodes=1000,
                embedding_dim=embedding_dim
            )
        )

        # Surprise detection (for novelty in timing patterns)
        self.surprise_engine = BayesianSurpriseEngine(
            input_dim=embedding_dim,
            config=SurpriseConfig(
                window_size=20,
                surprise_threshold=surprise_threshold
            )
        )

        self.min_observations = min_observations

    def get_or_create_profile(self, speaker_id: str) -> SpeakerProfile:
        """Get existing profile or create new one"""
        if speaker_id not in self.speaker_profiles:
            self.speaker_profiles[speaker_id] = SpeakerProfile(speaker_id=speaker_id)
        return self.speaker_profiles[speaker_id]

    def observe_turn_taking(
        self,
        speaker_id: str,
        intent_type: str,
        intent_subtype: str,
        is_fpp: bool,
        expected_gap_ms: int,
        actual_gap_ms: int,
        prosody_features: Dict[str, float],
        context: str = "general"
    ) -> Dict[str, any]:
        """
        Observe a turn-taking event and update speaker profile

        Returns:
            - updated_profile: Current speaker profile
            - surprise: How unexpected this timing was
            - should_adapt: Whether to adjust VAD threshold
            - adapted_threshold: Suggested new threshold
        """

        profile = self.get_or_create_profile(speaker_id)

        # Create event
        event = TurnTakingEvent(
            speaker_id=speaker_id,
            timestamp=datetime.now(),
            intent_type=intent_type,
            intent_subtype=intent_subtype,
            is_fpp=is_fpp,
            expected_gap_ms=expected_gap_ms,
            actual_gap_ms=actual_gap_ms,
            final_f0_slope=prosody_features.get('final_f0_slope', 0.0),
            speech_rate=prosody_features.get('speech_rate', 5.0),
            utterance_duration_ms=prosody_features.get('duration', 1000),
            was_interrupted=prosody_features.get('interrupted', False),
            interruption_timing_ms=prosody_features.get('interruption_at_ms'),
            conversation_context=context,
            timing_surprise=0.0  # Will compute below
        )

        # Create embedding from event features
        event_embedding = self._create_event_embedding(event)

        # Compute surprise (how unexpected is this timing?)
        surprise_result = self.surprise_engine.compute_surprise(event_embedding)
        event.timing_surprise = surprise_result['surprise']

        # Update profile with new observation
        self._update_profile(profile, event)

        # Store in episodic memory (for long-term learning)
        self.memory.store_episode(
            content=event_embedding,
            embedding=event_embedding,
            surprise=event.timing_surprise,
            timestamp=event.timestamp,
            location=context,
            entities=[speaker_id],
            metadata={
                'intent_type': intent_type,
                'actual_gap_ms': actual_gap_ms,
                'expected_gap_ms': expected_gap_ms
            }
        )

        # Decide if we should adapt VAD threshold
        should_adapt = profile.total_turns >= self.min_observations
        adapted_threshold = self._compute_adapted_threshold(profile, intent_type)

        return {
            'updated_profile': profile,
            'surprise': event.timing_surprise,
            'is_novel': surprise_result['is_novel'],
            'should_adapt': should_adapt,
            'adapted_threshold': adapted_threshold,
            'confidence': profile.confidence_score
        }

    def _create_event_embedding(self, event: TurnTakingEvent) -> np.ndarray:
        """Create feature vector from turn-taking event"""
        features = [
            # Timing features (normalized)
            event.actual_gap_ms / 1000.0,  # Scale to seconds
            event.expected_gap_ms / 1000.0,
            (event.actual_gap_ms - event.expected_gap_ms) / 1000.0,  # Gap error

            # Prosody features
            event.final_f0_slope / 10.0,  # Normalize slope
            event.speech_rate / 10.0,
            event.utterance_duration_ms / 1000.0,

            # Categorical features (one-hot style)
            1.0 if event.is_fpp else 0.0,
            1.0 if event.was_interrupted else 0.0,
            event.interruption_timing_ms / 1000.0 if event.interruption_timing_ms else 0.0,

            # Intent encoding (simple hash-based)
            hash(event.intent_type) % 10 / 10.0,
            hash(event.intent_subtype) % 10 / 10.0,
        ]

        # Pad to embedding_dim
        embedding = np.array(features, dtype=np.float32)
        if len(embedding) < 64:
            embedding = np.pad(embedding, (0, 64 - len(embedding)), 'constant')
        else:
            embedding = embedding[:64]

        return embedding

    def _update_profile(self, profile: SpeakerProfile, event: TurnTakingEvent):
        """Update speaker profile with new observation (online learning)"""

        n = profile.total_turns

        # Incremental mean update (Welford's algorithm for stability)
        if n == 0:
            profile.avg_turn_gap_ms = event.actual_gap_ms
            profile.std_turn_gap_ms = 0.0
        else:
            # Update mean
            old_mean = profile.avg_turn_gap_ms
            profile.avg_turn_gap_ms = old_mean + (event.actual_gap_ms - old_mean) / (n + 1)

            # Update variance (online)
            profile.std_turn_gap_ms = np.sqrt(
                (n * profile.std_turn_gap_ms**2 + (event.actual_gap_ms - old_mean) * (event.actual_gap_ms - profile.avg_turn_gap_ms)) / (n + 1)
            )

        # Update intent-specific gaps (exponential moving average)
        alpha = 0.2  # Learning rate
        if event.intent_type in profile.intent_gaps:
            profile.intent_gaps[event.intent_type] = (
                (1 - alpha) * profile.intent_gaps[event.intent_type] +
                alpha * event.actual_gap_ms
            )
        else:
            profile.intent_gaps[event.intent_type] = event.actual_gap_ms

        # Update prosody patterns
        if n > 0:
            profile.avg_speech_rate = (
                (profile.avg_speech_rate * n + event.speech_rate) / (n + 1)
            )

        # Update interruption tolerance (if interrupted, they're more tolerant)
        if event.was_interrupted:
            # Lower threshold means more tolerant
            profile.interruption_tolerance = max(0.5, profile.interruption_tolerance - 0.05)

        # Metadata updates
        profile.total_turns += 1
        profile.last_updated = datetime.now()

        # Confidence score (based on data quantity, caps at 0.95)
        profile.confidence_score = min(0.95, profile.total_turns / 50.0)

    def _compute_adapted_threshold(self, profile: SpeakerProfile, intent_type: str) -> float:
        """
        Compute personalized turn-end threshold for this speaker

        Based on:
        - Speaker's learned timing patterns
        - Intent-specific gaps
        - Confidence in the profile
        """

        # Default threshold (from research: 0.75 works well)
        base_threshold = 0.75

        if profile.total_turns < self.min_observations:
            return base_threshold  # Not enough data

        # Get speaker-specific expected gap for this intent
        expected_gap = profile.intent_gaps.get(intent_type, profile.avg_turn_gap_ms)

        # Adjust threshold based on speaker's typical gap
        # Faster speakers (shorter gaps) → lower threshold (more responsive)
        # Slower speakers (longer gaps) → higher threshold (wait longer)
        gap_factor = expected_gap / 200.0  # Normalize around 200ms baseline

        # Adjust threshold (bounded)
        adapted = base_threshold * gap_factor
        adapted = np.clip(adapted, 0.55, 0.90)  # Keep in reasonable range

        # Weight by confidence
        final_threshold = (
            profile.confidence_score * adapted +
            (1 - profile.confidence_score) * base_threshold
        )

        return final_threshold

    def predict_next_gap(
        self,
        speaker_id: str,
        intent_type: str,
        prosody_features: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Predict expected gap and uncertainty for this speaker/intent

        Returns:
            (expected_gap_ms, uncertainty_std)
        """

        profile = self.get_or_create_profile(speaker_id)

        if profile.total_turns < self.min_observations:
            # Not enough data, use intent classifier defaults
            # (This is where intent_classifier_german.py gaps are used)
            return (200.0, 100.0)  # Default with high uncertainty

        # Get speaker-specific gap for this intent
        if intent_type in profile.intent_gaps:
            expected_gap = profile.intent_gaps[intent_type]
        else:
            expected_gap = profile.avg_turn_gap_ms

        # Uncertainty based on speaker's variability and confidence
        uncertainty = profile.std_turn_gap_ms / (1.0 + profile.confidence_score)

        # Adjust for prosody if available
        if prosody_features and 'final_f0_slope' in prosody_features:
            # Rising intonation (question) → expect tighter gap
            if prosody_features['final_f0_slope'] > 5:
                expected_gap *= 0.85
            # Falling intonation (statement) → expect looser gap
            elif prosody_features['final_f0_slope'] < -5:
                expected_gap *= 1.15

        return (expected_gap, uncertainty)

    def retrieve_similar_situations(
        self,
        speaker_id: str,
        intent_type: str,
        context: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve past similar turn-taking situations from memory

        Uses episodic memory to find similar conversations
        """

        # Create query embedding
        query_event = TurnTakingEvent(
            speaker_id=speaker_id,
            timestamp=datetime.now(),
            intent_type=intent_type,
            intent_subtype="",
            is_fpp=True,
            expected_gap_ms=200,
            actual_gap_ms=200,
            final_f0_slope=0.0,
            speech_rate=5.0,
            utterance_duration_ms=1000,
            was_interrupted=False,
            interruption_timing_ms=None,
            conversation_context=context,
            timing_surprise=0.0
        )

        query_embedding = self._create_event_embedding(query_event)

        # Search memory
        # Note: TwoStageRetriever from neuro-memory would be used here in production
        # For now, simplified direct search
        similar_episodes = []

        for episode in self.memory.episodes:
            if episode.entities and speaker_id in episode.entities:
                # Check if same intent type
                if episode.metadata.get('intent_type') == intent_type:
                    similar_episodes.append({
                        'timestamp': episode.timestamp,
                        'actual_gap': episode.metadata.get('actual_gap_ms'),
                        'expected_gap': episode.metadata.get('expected_gap_ms'),
                        'surprise': episode.surprise,
                        'context': episode.location
                    })

        # Return most recent k
        return sorted(similar_episodes, key=lambda x: x['timestamp'], reverse=True)[:k]

    def get_speaker_summary(self, speaker_id: str) -> Dict:
        """Get comprehensive summary of speaker's turn-taking profile"""

        profile = self.get_or_create_profile(speaker_id)

        return {
            'speaker_id': speaker_id,
            'total_interactions': profile.total_turns,
            'confidence': profile.confidence_score,
            'timing': {
                'avg_gap_ms': profile.avg_turn_gap_ms,
                'std_gap_ms': profile.std_turn_gap_ms,
                'intent_specific_gaps': profile.intent_gaps
            },
            'prosody': {
                'speech_rate': profile.avg_speech_rate,
                'f0_range': profile.typical_f0_range
            },
            'behavior': {
                'interruption_tolerance': profile.interruption_tolerance,
                'polite_waiting_ms': profile.polite_waiting_time_ms,
                'avg_turn_length_ms': profile.turn_length_avg_ms
            },
            'last_updated': profile.last_updated.isoformat(),
            'ready_for_adaptation': profile.total_turns >= self.min_observations
        }


# Demo usage
if __name__ == "__main__":
    print("=" * 80)
    print(" MEMORY-VAD BRIDGE - Speaker Profile Learning")
    print("=" * 80)
    print()

    bridge = MemoryVADBridge(embedding_dim=64, min_observations=5)

    # Simulate learning from multiple turns with a speaker
    print("Learning from Speaker 'Guest_001' (German hotel guest):")
    print("-" * 80)

    # Simulate 15 turns
    scenarios = [
        ("question", "wh_question", True, 100, 120, "booking"),
        ("response", "confirm", False, 100, 95, "booking"),
        ("question", "yn_modal", True, 150, 180, "booking"),
        ("statement", "assertion", False, 400, 420, "booking"),
        ("request", "polite_request", True, 250, 270, "service"),
        ("response", "acknowledge", False, 100, 85, "service"),
        ("question", "wh_question", True, 100, 110, "information"),
        ("statement", "assertion", False, 400, 450, "information"),
        ("social", "thanks", False, 150, 140, "closing"),
        ("closing", "farewell", True, 200, 190, "closing"),
        ("question", "wh_question", True, 100, 95, "booking"),
        ("question", "yn_modal", True, 150, 165, "booking"),
        ("request", "need_request", True, 250, 240, "service"),
        ("response", "deny", False, 150, 155, "service"),
        ("statement", "assertion", False, 400, 380, "information"),
    ]

    for i, (intent, subtype, is_fpp, expected, actual, context) in enumerate(scenarios, 1):
        result = bridge.observe_turn_taking(
            speaker_id="Guest_001",
            intent_type=intent,
            intent_subtype=subtype,
            is_fpp=is_fpp,
            expected_gap_ms=expected,
            actual_gap_ms=actual,
            prosody_features={'final_f0_slope': np.random.randn() * 3, 'speech_rate': 5.0, 'duration': 1000},
            context=context
        )

        if i % 5 == 0:
            print(f"Turn {i}: Surprise={result['surprise']:.3f}, "
                  f"Confidence={result['confidence']:.2f}, "
                  f"Adapted threshold={result['adapted_threshold']:.3f}")

    print()

    # Get profile summary
    print("Learned Speaker Profile:")
    print("-" * 80)
    summary = bridge.get_speaker_summary("Guest_001")
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Confidence: {summary['confidence']:.2%}")
    print(f"Average gap: {summary['timing']['avg_gap_ms']:.1f}ms ± {summary['timing']['std_gap_ms']:.1f}ms")
    print(f"Interruption tolerance: {summary['behavior']['interruption_tolerance']:.2f}")
    print(f"Ready for adaptation: {summary['ready_for_adaptation']}")
    print()

    print("Intent-specific learned gaps:")
    for intent, gap in summary['timing']['intent_specific_gaps'].items():
        print(f"  {intent}: {gap:.1f}ms")
    print()

    # Predict next gap
    expected, uncertainty = bridge.predict_next_gap("Guest_001", "question")
    print(f"Prediction for next question: {expected:.1f}ms ± {uncertainty:.1f}ms")
    print()

    print("=" * 80)
    print(" Memory-VAD Bridge ready for integration!")
    print("=" * 80)
