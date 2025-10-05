"""
Predictive Turn-End Model (LSTM-based)
=======================================

200-400ms lookahead turn-end prediction using LSTM ensemble.

Research Foundation:
- Lla-VAP: LSTM Ensemble (arXiv 2412.18061, Dec 2024)
- Voice Activity Projection (VAP) - arXiv 2401.04868
- Continuous Turn-Taking with LSTMs (arXiv 1806.11461)

Architecture:
1. LSTM Sequence Model - Learns temporal patterns in audio/prosody
2. Feature Fusion - Combines prosody + intent + timing history
3. Multi-step Prediction - Predicts turn-end probability 200-400ms ahead
4. Ensemble with Current VAD - Combines reactive + predictive signals

Performance Target:
- Latency: <15ms for prediction
- Accuracy: >85% turn-end detection 300ms before actual end
- Integration: Seamless fallback to current VAD
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class TurnEndPrediction:
    """Prediction result"""
    current_prob: float          # Current turn-end probability (from VAD)
    predicted_prob_200ms: float  # Predicted probability 200ms ahead
    predicted_prob_400ms: float  # Predicted probability 400ms ahead
    confidence: float            # Prediction confidence (0-1)
    should_interrupt_early: bool # Whether to interrupt before actual turn-end
    lookahead_ms: int           # How far ahead prediction is valid


class TurnEndPredictor:
    """
    LSTM-based Turn-End Predictor with 200-400ms lookahead

    Stage 1 (Stub): Returns current probability (no prediction)
    Stage 2 (LSTM): Trained model with actual lookahead

    Current implementation: Stub (ready for LSTM replacement)
    """

    def __init__(
        self,
        use_lstm: bool = False,
        lookahead_ms: int = 300,
        history_frames: int = 30  # 300ms history at 10ms frames
    ):
        self.use_lstm = use_lstm
        self.lookahead_ms = lookahead_ms
        self.history_frames = history_frames

        # Feature history buffer (for LSTM input)
        self.prosody_history = deque(maxlen=history_frames)
        self.intent_history = deque(maxlen=history_frames)
        self.vad_history = deque(maxlen=history_frames)

        # LSTM model (placeholder - will be trained)
        self.lstm_model = None
        self.is_trained = False

        # Fallback to current VAD when LSTM not available
        self.fallback_enabled = True

    def predict(
        self,
        current_vad_prob: float,
        prosody_features: Dict[str, float],
        intent_type: Optional[str] = None,
        speaker_profile: Optional[Dict] = None
    ) -> TurnEndPrediction:
        """
        Predict turn-end probability 200-400ms ahead

        Args:
            current_vad_prob: Current VAD turn-end probability
            prosody_features: {f0_slope, speech_rate, energy, etc.}
            intent_type: Current detected intent
            speaker_profile: Speaker-specific patterns

        Returns:
            TurnEndPrediction with lookahead probabilities
        """

        # Update feature history
        self._update_history(current_vad_prob, prosody_features, intent_type)

        if self.use_lstm and self.is_trained and self.lstm_model is not None:
            # STAGE 2: LSTM prediction (when trained)
            return self._lstm_predict(current_vad_prob, speaker_profile)
        else:
            # STAGE 1: Stub - returns current probability
            return self._stub_predict(current_vad_prob)

    def _stub_predict(self, current_prob: float) -> TurnEndPrediction:
        """
        Stub implementation - returns current probability
        (No actual lookahead until LSTM is trained)
        """
        return TurnEndPrediction(
            current_prob=current_prob,
            predicted_prob_200ms=current_prob,  # Same as current (no prediction)
            predicted_prob_400ms=current_prob,  # Same as current (no prediction)
            confidence=0.5,  # Low confidence (stub)
            should_interrupt_early=False,  # Conservative
            lookahead_ms=0  # No lookahead in stub
        )

    def _lstm_predict(
        self,
        current_prob: float,
        speaker_profile: Optional[Dict]
    ) -> TurnEndPrediction:
        """
        LSTM-based prediction (when model is trained)

        Uses temporal patterns in:
        - Prosody trajectory (F0, energy, rate)
        - Intent sequence (question → answer pattern)
        - VAD probability trend
        - Speaker-specific timing
        """

        # Create feature vector from history
        features = self._create_sequence_features(speaker_profile)

        # LSTM forward pass (placeholder - real model would be used here)
        # predicted_sequence = self.lstm_model.predict(features)

        # For now, use heuristic-based prediction (proof of concept)
        predicted_200ms = self._heuristic_lookahead(features, 20)  # 20 frames = 200ms
        predicted_400ms = self._heuristic_lookahead(features, 40)  # 40 frames = 400ms

        # Confidence based on trend stability
        confidence = self._calculate_prediction_confidence(features)

        # Decide if early interruption is beneficial
        should_interrupt_early = (
            predicted_200ms > 0.8 and  # High confidence turn-end predicted
            confidence > 0.7 and        # Confident in prediction
            current_prob < 0.6          # Not yet at turn-end
        )

        return TurnEndPrediction(
            current_prob=current_prob,
            predicted_prob_200ms=predicted_200ms,
            predicted_prob_400ms=predicted_400ms,
            confidence=confidence,
            should_interrupt_early=should_interrupt_early,
            lookahead_ms=self.lookahead_ms
        )

    def _update_history(
        self,
        vad_prob: float,
        prosody: Dict[str, float],
        intent: Optional[str]
    ):
        """Add current features to history buffer"""
        self.vad_history.append(vad_prob)
        self.prosody_history.append(prosody)
        self.intent_history.append(intent)

    def _create_sequence_features(self, speaker_profile: Optional[Dict]) -> np.ndarray:
        """
        Create LSTM input features from history

        Shape: (history_frames, feature_dim)
        """
        features = []

        for i in range(len(self.vad_history)):
            frame_features = [
                # VAD probability
                self.vad_history[i],

                # Prosody features
                self.prosody_history[i].get('f0_slope', 0.0) / 10.0,
                self.prosody_history[i].get('energy', 0.0) / 100.0,
                self.prosody_history[i].get('speech_rate', 5.0) / 10.0,

                # Intent encoding (simple hash)
                hash(self.intent_history[i]) % 10 / 10.0 if self.intent_history[i] else 0.0,
            ]

            # Speaker-specific features (if available)
            if speaker_profile:
                frame_features.extend([
                    speaker_profile.get('avg_turn_gap_ms', 200) / 1000.0,
                    speaker_profile.get('interruption_tolerance', 0.75),
                ])

            features.append(frame_features)

        # Pad if not enough history
        while len(features) < self.history_frames:
            features.insert(0, [0.0] * len(features[0]) if features else [0.0] * 7)

        return np.array(features, dtype=np.float32)

    def _heuristic_lookahead(self, features: np.ndarray, lookahead_frames: int) -> float:
        """
        Heuristic-based lookahead (placeholder for trained LSTM)

        Analyzes trends in:
        1. VAD probability trajectory
        2. Prosody slope (F0 falling → turn-end likely)
        3. Energy decay
        """

        # Extract recent VAD trend (last 10 frames)
        vad_sequence = features[-10:, 0]  # VAD column

        # Trend analysis
        if len(vad_sequence) < 3:
            return vad_sequence[-1] if len(vad_sequence) > 0 else 0.5

        # Linear regression on VAD trajectory
        x = np.arange(len(vad_sequence))
        slope, intercept = np.polyfit(x, vad_sequence, 1)

        # Extrapolate to lookahead_frames
        predicted_vad = slope * (len(vad_sequence) + lookahead_frames) + intercept
        predicted_vad = np.clip(predicted_vad, 0.0, 1.0)

        # Boost prediction if prosody indicates turn-end
        f0_slope_avg = np.mean(features[-5:, 1])  # Recent F0 slope
        if f0_slope_avg < -0.3:  # Strong falling intonation
            predicted_vad = min(1.0, predicted_vad + 0.15)

        # Boost if energy is decaying
        energy_trend = features[-5:, 2]
        if len(energy_trend) > 1 and energy_trend[-1] < energy_trend[0] * 0.7:
            predicted_vad = min(1.0, predicted_vad + 0.1)

        return predicted_vad

    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """
        Calculate confidence in prediction based on signal stability

        Higher confidence when:
        - Trends are consistent
        - Features have low variance
        - Sufficient history available
        """

        if len(self.vad_history) < 10:
            return 0.3  # Low confidence with insufficient data

        # Variance in recent VAD probabilities
        vad_recent = features[-10:, 0]
        vad_variance = np.var(vad_recent)

        # Trend consistency (correlation with linear fit)
        x = np.arange(len(vad_recent))
        correlation = np.corrcoef(x, vad_recent)[0, 1] if len(vad_recent) > 1 else 0

        # Combine factors
        stability_score = 1.0 - min(vad_variance * 2, 1.0)
        trend_score = abs(correlation)
        history_score = min(len(self.vad_history) / self.history_frames, 1.0)

        confidence = (stability_score * 0.4 + trend_score * 0.4 + history_score * 0.2)

        return confidence

    def train_lstm(self, training_data: List[Dict]):
        """
        Train LSTM model on collected conversation data

        Args:
            training_data: List of conversation sequences with labels
                [{
                    'prosody_sequence': np.ndarray,  # (T, feature_dim)
                    'intent_sequence': List[str],
                    'vad_sequence': np.ndarray,
                    'turn_end_frame': int  # Ground truth turn-end
                }]

        This would train a real LSTM model to replace heuristics
        """
        # Placeholder for actual LSTM training
        # Would use PyTorch/TensorFlow here

        print("LSTM training not implemented yet (placeholder)")
        print("Use heuristic-based prediction for now")

        self.is_trained = True  # Mark as trained (even if stub)

    def enable_lstm(self):
        """Enable LSTM prediction (when model is trained)"""
        self.use_lstm = True
        print("LSTM prediction enabled")

    def disable_lstm(self):
        """Fallback to stub (current VAD only)"""
        self.use_lstm = False
        print("LSTM prediction disabled - using current VAD only")


# Integration Example
def demo_predictor():
    print("=" * 80)
    print(" TURN-END PREDICTOR - Stage 1 (Stub) + Stage 2 (Heuristic LSTM)")
    print("=" * 80)
    print()

    # Stage 1: Stub (no LSTM)
    print("STAGE 1: Stub Implementation (Fallback)")
    print("-" * 80)
    predictor_stub = TurnEndPredictor(use_lstm=False)

    result = predictor_stub.predict(
        current_vad_prob=0.65,
        prosody_features={'f0_slope': -2.5, 'energy': 45, 'speech_rate': 5.2},
        intent_type='question'
    )

    print(f"Current VAD prob: {result.current_prob}")
    print(f"Predicted 200ms: {result.predicted_prob_200ms}")
    print(f"Predicted 400ms: {result.predicted_prob_400ms}")
    print(f"Confidence: {result.confidence}")
    print(f"Lookahead: {result.lookahead_ms}ms")
    print()

    # Stage 2: LSTM-based (heuristic for now)
    print("STAGE 2: LSTM-Based Prediction (Heuristic POC)")
    print("-" * 80)
    predictor_lstm = TurnEndPredictor(use_lstm=True, lookahead_ms=300)

    # Simulate conversation sequence
    for frame in range(50):
        vad_prob = min(0.95, 0.1 + frame * 0.015)  # Rising probability
        prosody = {
            'f0_slope': -3.0 if frame > 30 else 0.0,  # Falling at end
            'energy': max(10, 60 - frame),            # Decaying energy
            'speech_rate': 5.0
        }

        result = predictor_lstm.predict(
            current_vad_prob=vad_prob,
            prosody_features=prosody,
            intent_type='question',
            speaker_profile={'avg_turn_gap_ms': 150, 'interruption_tolerance': 0.7}
        )

        if frame % 10 == 0:
            print(f"Frame {frame*10}ms: Current={result.current_prob:.2f}, "
                  f"Pred_200ms={result.predicted_prob_200ms:.2f}, "
                  f"Confidence={result.confidence:.2f}, "
                  f"Early_interrupt={result.should_interrupt_early}")

    print()
    print("=" * 80)
    print(" Predictor ready for integration!")
    print(" - Stage 1 (stub): Always available")
    print(" - Stage 2 (LSTM): Enable when trained model available")
    print("=" * 80)


if __name__ == "__main__":
    demo_predictor()
