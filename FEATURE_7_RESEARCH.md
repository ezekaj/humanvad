# Feature #7: Adaptive Thresholds for Turn-Taking

**Research Date:** 2025-10-05
**Status:** Research Complete

---

## Overview

Adaptive thresholds automatically adjust the turn-end detection threshold based on conversation context, user behavior, and confidence levels. Instead of fixed 0.75 threshold for all situations, the system learns and adapts.

---

## Key Research Findings

### 1. Industry Practice (Retell AI, Deepgram, Tavus - 2024)

**Current Problem with Fixed Thresholds:**
- Fixed silence threshold (e.g., 700ms) causes issues:
  - **Too short** → Interrupts user mid-sentence
  - **Too long** → System seems unresponsive
- Cannot adapt to different conversation styles (formal vs casual)
- Doesn't account for speaker patterns

**Adaptive Solutions:**
- Adjust thresholds based on ambient noise levels
- Adapt to conversation dynamics in real-time
- Learn from user interruption patterns

**Quote (Retell AI 2024):**
> "Turn-taking systems with fixed silence thresholds face challenges such as limited response speed and the inability to adapt to different turn-taking dynamics in varying situations."

---

### 2. Feedback-Based Adaptive VAD

**Approach:**
- Use VAD decision to improve noise estimate
- Adaptively vary thresholds based on recent history
- Improves performance in non-stationary noise

**Implementation:**
- Track recent false positives/negatives
- Adjust threshold up (fewer interrupts) or down (more responsive)
- Continuous feedback loop

---

## How This Could Help Turn-Taking Detection

### Current Problem:
Excellence VAD uses **fixed 0.75 threshold** for all situations:
- User interrupts often → Need higher threshold (0.80-0.85)
- User waits politely → Need lower threshold (0.65-0.70)
- High confidence predictions → Can use lower threshold
- Low confidence predictions → Need higher threshold

### Adaptive Threshold Solution:

**Adjust threshold based on:**

1. **User Interruption History**
   - If user interrupts 5+ times in conversation → Raise threshold to 0.85
   - If user waits patiently → Lower threshold to 0.70
   - Learn user's interruption style

2. **Confidence Level**
   - High prosody confidence (>0.8) → Lower threshold to 0.70
   - Low prosody confidence (<0.5) → Raise threshold to 0.80
   - Accounts for signal quality

3. **Semantic Completeness**
   - Strong completion signal (>0.9) → Lower threshold to 0.65
   - Weak completion signal (<0.4) → Raise threshold to 0.85
   - More lenient when semantics are clear

4. **Recent Turn History**
   - Just started conversation → Higher threshold (0.80) - be conservative
   - Mid-conversation flow → Lower threshold (0.70) - be responsive
   - Adapts to conversation stage

---

## Simplified Implementation for VAD

### Concept: Dynamic Threshold with Feedback

```python
class AdaptiveThresholdManager:
    """
    Lightweight adaptive threshold based on conversation context
    """

    def __init__(self, base_threshold: float = 0.75):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold

        # User behavior tracking
        self.user_interruptions = 0
        self.total_turns = 0

        # Confidence tracking
        self.recent_confidences = deque(maxlen=10)

        # Adjustment parameters
        self.min_threshold = 0.60
        self.max_threshold = 0.90

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
```

---

## Integration with Excellence VAD

**Minimal changes needed:**

```python
# In ExcellenceVADGerman.__init__()
self.threshold_manager = AdaptiveThresholdManager(base_threshold=0.75)

# In process_frame()
# Replace fixed threshold with adaptive
adaptive_threshold = self.threshold_manager.get_threshold(
    prosody_confidence=ai_result['confidence'],
    semantic_prob=semantic_prob
)

# Use adaptive threshold instead of self.turn_end_threshold
if final_turn_end_prob >= adaptive_threshold:
    action = "interrupt_ai_immediately"
else:
    action = "wait_for_ai_completion"
```

---

## Expected Accuracy Impact

### Baseline: 40%
- Fixed 0.75 threshold for all situations

### With Adaptive Thresholds:
- **Expected improvement:** +10-20%
- **Accuracy target:** 50-60%

**Why this could work:**

1. **Accounts for signal quality** - High confidence → More responsive
2. **Learns user style** - Eager interrupter vs polite waiter
3. **Semantic clarity** - Clear completion → Lower barrier
4. **Zero latency** - Just threshold adjustment (<0.01ms)
5. **Simple** - No complex models, just math

**Conservative estimate:** +10% improvement (40% → 50%)

---

## Advantages Over Previous Features

| Feature | Complexity | Latency | Accuracy | Status |
|---------|-----------|---------|----------|--------|
| #1: Prediction Error | Medium | +0.11ms | 0% | ❌ Failed |
| #2: Rhythm Tracking | Medium | <2ms | -75% | ❌ Failed |
| #3: Temporal Context | High | <0.1ms | -40% | ❌ Failed |
| #4: N400 Prediction | Very High | +50-100ms | +10-15%? | ❌ Skipped |
| #7: Adaptive Thresholds | **Low** | **<0.01ms** | **+10-20%** | ✅ **Best candidate** |

**Why Feature #7 is different:**
- ✅ **Simple** - Just threshold math
- ✅ **Fast** - No additional computation
- ✅ **Practical** - Addresses real problem (one-size-fits-all threshold)
- ✅ **Testable** - Clear success criteria
- ✅ **Production-ready** - No dependencies

---

## Implementation Plan

### Step 1: Design ✅
- AdaptiveThresholdManager class (above)
- Tracks: user behavior, confidence, semantic clarity
- Adjusts: +/- 0.05 to 0.10 from base

### Step 2: Implement
- Create `adaptive_threshold_manager.py`
- Integrate into `excellence_vad_german.py`
- Add threshold tracking to debug output

### Step 3: Test
- Same 5 scenarios as Feature #3 benchmark
- Compare: Fixed 0.75 vs Adaptive
- Target: >10% improvement to integrate

---

## Risks

### 1. **Overfitting to Test Scenarios**
- Adaptive logic might overfit to specific tests
- **Mitigation:** Test on diverse scenarios

### 2. **Learning from Wrong Signals**
- If user interrupts due to AI error, system learns wrong pattern
- **Mitigation:** Require 5+ turns before adapting

### 3. **Threshold Oscillation**
- Threshold might bounce up/down unstably
- **Mitigation:** Use moving average, minimum adjustment intervals

### 4. **Still Doesn't Fix Root Problem**
- If baseline is broken, adaptive threshold won't help
- **Mitigation:** Track if improvements are real or just threshold tuning

---

## Success Criteria

**Minimum to integrate:**
- +10% accuracy improvement vs fixed threshold
- <0.1ms latency overhead
- Stable threshold (doesn't oscillate)
- Works across all 5 test scenarios

**Ideal outcome:**
- +20% accuracy improvement
- Learns user patterns within 5 turns
- Adapts to semantic clarity automatically

---

## Recommendation

✅ **IMPLEMENT AND TEST Feature #7**

**Reasons:**
1. Simplest feature yet (just threshold math)
2. Addresses real problem (fixed threshold is naive)
3. Zero latency impact
4. High probability of success (practical, not theoretical)
5. If it fails, we know the problem is deeper than thresholds

**Next steps:**
1. Implement `AdaptiveThresholdManager`
2. Integrate into Excellence VAD
3. Run benchmark with same 5 scenarios
4. If >10% improvement → KEEP
5. If <10% → **STOP adding features, reassess baseline**

---

**This is the last feature attempt. If Feature #7 fails, we need real conversation data.**
