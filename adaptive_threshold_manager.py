"""
Adaptive Threshold Manager - Feature #7
Dynamic turn-end threshold based on conversation context
"""

import numpy as np
from collections import deque


class AdaptiveThresholdManager:
    """
    Lightweight adaptive threshold based on conversation context

    Adjusts threshold dynamically based on:
    - User interruption patterns (eager vs polite)
    - Signal confidence (trust high confidence signals)
    - Semantic clarity (lower threshold for clear completions)
    """

    def __init__(self, base_threshold: float = 0.60):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold

        # User behavior tracking
        self.user_interruptions = 0
        self.total_turns = 0

        # Confidence tracking
        self.recent_confidences = deque(maxlen=10)

        # Adjustment parameters
        self.min_threshold = 0.50
        self.max_threshold = 0.80

    def update_from_user_behavior(self, user_interrupted: bool):
        """Learn from user interruption patterns"""
        self.total_turns += 1

        if user_interrupted:
            self.user_interruptions += 1

        # Compute interruption rate
        if self.total_turns >= 5:
            interrupt_rate = self.user_interruptions / self.total_turns

            # High interrupt rate → Raise threshold (user is eager)
            if interrupt_rate > 0.6:
                adjustment = +0.10
            elif interrupt_rate > 0.4:
                adjustment = +0.05
            # Low interrupt rate → Lower threshold (user is polite)
            elif interrupt_rate < 0.2:
                adjustment = -0.05
            else:
                adjustment = 0.0

            self.current_threshold = np.clip(
                self.base_threshold + adjustment,
                self.min_threshold,
                self.max_threshold
            )

    def get_threshold(
        self,
        prosody_confidence: float,
        semantic_prob: float
    ) -> float:
        """
        Get adaptive threshold for current frame

        Args:
            prosody_confidence: 0.0-1.0 confidence in prosody signal
            semantic_prob: 0.0-1.0 semantic completion probability

        Returns:
            Adjusted threshold for turn-end detection
        """

        threshold = self.current_threshold

        # Adjust based on confidence
        self.recent_confidences.append(prosody_confidence)
        avg_confidence = np.mean(self.recent_confidences)

        if avg_confidence > 0.8:
            # High confidence → Lower threshold (trust the signal)
            threshold -= 0.05
        elif avg_confidence < 0.5:
            # Low confidence → Raise threshold (be conservative)
            threshold += 0.05

        # Adjust based on semantic clarity
        if semantic_prob > 0.9:
            # Very clear completion → Lower threshold
            threshold -= 0.10
        elif semantic_prob < 0.3:
            # Very unclear → Raise threshold
            threshold += 0.10

        # Clip to bounds
        threshold = float(np.clip(threshold, self.min_threshold, self.max_threshold))

        return threshold

    def reset(self):
        """Reset to base threshold"""
        self.current_threshold = self.base_threshold
        self.user_interruptions = 0
        self.total_turns = 0
        self.recent_confidences.clear()
