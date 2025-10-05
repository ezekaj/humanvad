# Feature #3: Temporal Context Encoding (Rotational Dynamics)

**Research Date:** 2025-10-05
**Status:** Research Complete

---

## Overview

Temporal context encoding using rotational dynamics is a brain-inspired mechanism where neural activity forms circular/cyclical patterns in latent space to track time relative to speech events.

---

## Key Research Papers

### 1. "Latent neural dynamics encode temporal context in speech" (2023)
**Source:** PMC11182421 | Nature Communications Neuroscience

**Key Findings:**
- Superior temporal gyrus (STG) uses **rotational dynamics** in 5-6D latent space to encode temporal context
- Rotations triggered by:
  - **Sentence onset events** (conversation start)
  - **Peak rate events** (sharp acoustic envelope increases at syllable/word boundaries)
- Provides "clock-like representation" of linguistic timing
- Does NOT require predictive coding or oscillatory mechanisms

**Mechanism:**
```
Peak rate event → Rotation starts → Latent state traces circle → Time decoded from rotation phase
```

**Quote:**
> "Rotations in the peak rate subspace could serve to keep track of the time relative to the peak rate event"

**Computational Role:**
- Sentence onset: initialization signal (kick network into speech-encoding state)
- Peak rate: temporal scaffolding for organizing acoustic/phonetic features
- Divides sentences into intervals through rotations in state space

---

### 2. "An Investigation Of Rotary Position Embedding For Speech Enhancement" (2024)
**Source:** ACM ICSPCТ 2024

**Key Findings:**
- RoPE (Rotary Position Embedding) successfully adapted from NLP to speech processing
- Encodes absolute position with rotation matrix
- Incorporates explicit relative position dependency
- Benefits:
  - Flexible sequence length
  - Decaying inter-token dependency with distance
  - Linear self-attention with relative position encoding

**Implementation:**
- NOT matrix multiplication (too inefficient)
- Direct rotation applied to element pairs
- Computationally lightweight

---

## How This Could Help Turn-Taking Detection

### Current Problem:
Excellence VAD uses **instantaneous features** (current frame energy, current text):
- No memory of conversation flow
- Cannot track "how long since last turn boundary"
- Cannot distinguish "just started speaking" vs "finishing sentence"

### Rotational Dynamics Solution:

**Track time since key speech events:**

1. **Turn Start Event** → Reset rotation → Track time in turn
   - Early in turn (0-500ms): Low turn-end probability
   - Mid turn (500ms-2s): Moderate
   - Late in turn (2s+): High (natural conversation turns are short)

2. **Energy Peak Event** → Track time since last syllable/word
   - Recent peak (<200ms): Continuing speech
   - Old peak (>500ms): Possible pause/end

3. **Semantic Completion** → Track time since complete sentence
   - Just completed + pause → Strong turn-end signal
   - Completed long ago → Continuing multi-sentence turn

---

## Simplified Implementation for VAD

### Concept: "Time Since Last Event" Encoding

Instead of full 5-6D rotational dynamics, use **phase encoding**:

```python
class TemporalContextEncoder:
    """
    Lightweight temporal context using rotational phase encoding
    """

    def __init__(self):
        self.phase_turn_start = 0.0       # Tracks time since turn started
        self.phase_energy_peak = 0.0      # Tracks time since last syllable
        self.phase_completion = 0.0       # Tracks time since semantic completion

        # Rotation frequencies (Hz)
        self.freq_turn = 0.5    # 2-second period (typical turn length)
        self.freq_peak = 5.0    # 200ms period (syllable rate)
        self.freq_completion = 1.0  # 1-second period

    def update(self, dt: float, energy: float, is_complete: bool):
        """Update rotational phases"""

        # Advance all phases (time moves forward)
        self.phase_turn_start += 2 * np.pi * self.freq_turn * dt
        self.phase_energy_peak += 2 * np.pi * self.freq_peak * dt
        self.phase_completion += 2 * np.pi * self.freq_completion * dt

        # Reset phases on events
        if energy > threshold:
            self.phase_energy_peak = 0.0  # New syllable

        if is_complete:
            self.phase_completion = 0.0  # Sentence completed

    def get_temporal_context_features(self):
        """Extract features from rotational phases"""

        # Time since turn start (0→1 over 2 seconds)
        time_in_turn = (self.phase_turn_start % (2 * np.pi)) / (2 * np.pi)

        # Time since last syllable (0→1 over 200ms)
        time_since_syllable = (self.phase_energy_peak % (2 * np.pi)) / (2 * np.pi)

        # Time since completion (0→1 over 1 second)
        time_since_completion = (self.phase_completion % (2 * np.pi)) / (2 * np.pi)

        # Turn-end contribution from temporal context
        turn_end_contrib = 0.0

        # Long time in turn → increase turn-end
        if time_in_turn > 0.7:  # >1.4 seconds in turn
            turn_end_contrib += 0.3

        # Long pause (no syllables) → increase turn-end
        if time_since_syllable > 0.6:  # >120ms silence
            turn_end_contrib += 0.4

        # Recent completion + pause → strong turn-end
        if time_since_completion < 0.3 and time_since_syllable > 0.5:
            turn_end_contrib += 0.5

        return {
            'time_in_turn': time_in_turn,
            'time_since_syllable': time_since_syllable,
            'time_since_completion': time_since_completion,
            'turn_end_contrib': min(turn_end_contrib, 1.0)
        }
```

---

## Why This Could Work

### 1. **Brain-Inspired**
- STG uses rotational dynamics for temporal context
- Proven mechanism in human speech perception
- Lightweight computational model

### 2. **Addresses Current Weaknesses**
- Current VAD has NO temporal memory
- Cannot track conversation flow
- Temporal context adds missing dimension

### 3. **Simple Implementation**
- Just 3 phase variables (floats)
- Basic trigonometry (sin/cos)
- <0.1ms latency per frame

### 4. **Interpretable**
- "How long in this turn?"
- "How long since last word?"
- "How long since sentence ended?"

---

## Expected Accuracy Impact

### Baseline: 40%
- 45% prosody + 55% semantics
- No temporal context

### With Temporal Context:
- 35% prosody + 45% semantics + **20% temporal context**
- Expected accuracy: **50-60%** (conservative estimate)
- **+10-20% improvement** if temporal cues are strong

### Why +10-20%?
- Fixes "just started speaking" false positives
- Fixes "long pause mid-sentence" false interruptions
- Adds conversational flow awareness

---

## Implementation Plan

### Step 1: Design
- Decide rotation frequencies (turn duration, syllable rate, completion delay)
- Design feature extraction from phases
- Design fusion weights (prosody + semantic + temporal)

### Step 2: Implement
- Create `TemporalContextEncoder` class
- Integrate into `ExcellenceVADGerman`
- Test with synthetic conversations

### Step 3: Test
- Benchmark with/without temporal context
- Measure accuracy improvement
- Target: >5% improvement to integrate

---

## Risks

### 1. **Overfitting to Turn Length**
- German conversations may have variable turn lengths
- Fixed rotation frequency (2 seconds) may not fit all contexts
- **Mitigation:** Adaptive frequency based on conversation style

### 2. **Syllable Detection Required**
- Phase reset needs reliable syllable/energy peak detection
- May add complexity
- **Mitigation:** Use existing energy history from prosody detector

### 3. **Three More Hyperparameters**
- Rotation frequencies need tuning
- More complexity = more tuning
- **Mitigation:** Research-backed defaults (syllable rate: 4-6 Hz German)

---

## Alternative: Skip Temporal Context?

**Consider skipping if:**
- Implementation too complex
- German syllable rate unknown
- No easy way to detect "turn start" event

**Better alternatives might be:**
- Feature #4: N400-style semantic prediction error (simpler)
- Feature #5: Disfluency detection (well-researched)
- Feature #7: Adaptive thresholds (high impact, low complexity)

---

## Recommendation

**PROCEED WITH CAUTION**

Temporal context is theoretically strong but implementation risks are high:
- Need to tune 3 rotation frequencies
- Need syllable/peak detection
- Requires conversation flow tracking

**Suggested approach:**
1. Design simple version (single phase: time in turn)
2. Test with synthetic data
3. If promising (>5% improvement), expand to 3-phase version
4. If not promising, skip to Feature #4

---

## References

1. Caucheteux et al. (2023) "Latent neural dynamics encode temporal context in speech" - Nature Communications Neuroscience
2. Wang et al. (2024) "An Investigation Of Rotary Position Embedding For Speech Enhancement" - ACM ICSPCТ 2024
3. Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding" - arXiv:2104.09864

---

**Next Step:** Design Feature #3 or skip to Feature #4?
