# Feature #4: N400-Style Semantic Prediction Error

**Research Date:** 2025-10-05
**Status:** Research Complete

---

## Overview

N400 is a brain signal (ERP component) that occurs ~400ms after encountering semantically unexpected words. It reflects "semantic surprise" - the difference between predicted and actual word meaning.

---

## Key Research Findings

### 1. "Tracking Lexical and Semantic Prediction Error Underlying the N400" (Lopopolo & Rabovsky, 2024)
**Source:** Neurobiology of Language, MIT Press

**Key Findings:**
- N400 reflects **change in implicit predictive representation of meaning**
- NOT just next-word prediction, but **full sentence-level semantic prediction**
- Tracks how each word updates the overall meaning representation
- Measured as "change in probability distribution over semantic features"

**Computational Model:**
- Sentence Gestalt (SG) model: LSTM recurrent neural network
- Trained on large-scale text corpus
- Predicts sentence meaning through role-filler pairs
- Computes prediction error word-by-word

**Quote:**
> "Surprise at the level of meaning" by tracking how each word changes the implicit semantic representation of a sentence

---

### 2. Sentence-BERT for Semantic Similarity (2024)
**Source:** Hugging Face, Sentence Transformers

**Key Findings:**
- **DO NOT** use standard BERT for sentence similarity (poor results)
- **Sentence-BERT (SBERT)** designed specifically for semantic similarity
- 50 million computations (65 hours) with BERT → **5 seconds** with SBERT
- Uses siamese/triplet network structures
- 15,000+ pre-trained models available on Hugging Face

**Performance:**
- Real-time semantic similarity computation
- Cosine similarity between sentence embeddings
- State-of-the-art on MTEB (Massive Text Embeddings Benchmark)

---

## How This Could Help Turn-Taking Detection

### Current Problem:
Excellence VAD uses **pattern matching** for semantic completion:
- "Ich gehe zum" → Incomplete (preposition pattern)
- "Ich gehe zum Laden" → Complete (complete phrase pattern)
- **Problem:** Brittle, misses context

### N400-Style Solution:

**Track semantic coherence/prediction:**

1. **Expected Continuation** (Low N400 = coherent)
   - AI: "Ich möchte Ihnen sagen dass..."
   - Predicted next: "ich", "wir", "das", "es"
   - User interrupts → Check if semantically coherent to continue

2. **Unexpected Continuation** (High N400 = incoherent)
   - AI: "Das Hotel hat..."
   - User: "Ja!" (interrupts)
   - Semantic prediction: AI likely continuing with hotel info
   - High prediction error → Incomplete thought → DON'T interrupt

3. **Completion Prediction**
   - AI: "Das ist alles"
   - Semantic prediction: Low probability of continuation
   - Low prediction error → Complete thought → ALLOW interrupt

---

## Simplified Implementation for VAD

### Concept: Semantic Embedding Similarity

Instead of full LSTM + prediction error, use **lightweight semantic similarity**:

```python
from sentence_transformers import SentenceTransformer

class SemanticPredictionDetector:
    """
    Lightweight N400-style semantic prediction using SBERT
    """

    def __init__(self):
        # Use tiny model for speed (<100ms inference)
        self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        # German: 'paraphrase-multilingual-MiniLM-L12-v2'

        self.recent_sentences = []  # Last 3 sentences

    def compute_semantic_coherence(self, current_text: str) -> float:
        """
        Compute semantic coherence of current text with recent context

        Returns:
            coherence: 0.0-1.0 (high = coherent continuation)
        """

        if len(self.recent_sentences) == 0:
            return 0.5  # No context

        # Embed current text
        current_emb = self.model.encode(current_text)

        # Embed recent context
        context = ' '.join(self.recent_sentences)
        context_emb = self.model.encode(context)

        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        coherence = cosine_similarity([current_emb], [context_emb])[0][0]

        return float(coherence)

    def predict_continuation_probability(self, current_text: str) -> float:
        """
        Predict probability that speaker will continue

        Low coherence → High continuation probability (incomplete thought)
        High coherence → Low continuation probability (complete thought)
        """

        coherence = self.compute_semantic_coherence(current_text)

        # Invert: low coherence = incomplete = will continue
        continuation_prob = 1.0 - coherence

        return continuation_prob
```

---

## PROBLEMS WITH THIS APPROACH

### 1. **Latency**
- Sentence-BERT inference: 50-100ms per sentence
- **Too slow for real-time VAD** (target: <10ms per frame)
- Would need GPU acceleration

### 2. **Model Size**
- Even "tiny" SBERT models: 50-100MB
- Requires PyTorch/TensorFlow
- Memory overhead: 200-500MB
- **Too heavy for production VAD**

### 3. **Semantic Similarity ≠ Prediction Error**
- SBERT measures how similar two sentences are
- N400 measures how UNEXPECTED the current word is
- **Different concepts:**
  - "Ich gehe zum Laden" vs "Ich gehe zur Schule" → HIGH similarity, LOW prediction error
  - "Ich gehe zum" → LOW similarity with previous context, but EXPECTED continuation

### 4. **German Language Support**
- Need multilingual model: `paraphrase-multilingual-MiniLM-L12-v2`
- Larger model → More latency
- German-specific models may not exist for all features

### 5. **Integration Complexity**
- Requires sentence-level processing
- Current VAD processes 10ms frames
- Need to buffer and segment sentences
- Adds state management complexity

---

## Alternative: Simpler Semantic Features

Instead of full N400 prediction error, use **lightweight semantic cues**:

### Option A: Word Embeddings Distance
```python
# Use pre-computed word embeddings (FastText, GloVe)
# Compare last word to expected next words
# Fast: <1ms per lookup
# Problem: Misses sentence-level context
```

### Option B: Completion Keywords
```python
# Detect completion-signaling words in German
completion_words = ['fertig', 'erledigt', 'alles', 'ende', 'danke']

# If last word in completion_words → High completion probability
# Fast: <0.1ms
# Problem: Brittle, language-specific
```

### Option C: Sentence Length Heuristic
```python
# Short sentences (<5 words) → Likely complete
# Long sentences (>10 words) → Check for completion patterns
# Very fast: <0.01ms
# Problem: Inaccurate
```

---

## Expected Accuracy Impact

### Baseline: 40%
- 45% prosody + 55% semantics (pattern matching)

### With N400-Style Prediction (SBERT):
- Would need to reweight: 35% prosody + 40% semantics + 25% prediction error
- **Expected latency:** +50-100ms ❌ (too slow)
- **Expected accuracy:** 50-55% (optimistic, +10-15%)

### With Lightweight Semantic Features:
- 40% prosody + 50% semantics (improved) + 10% semantic features
- **Expected latency:** +1-2ms ✓ (acceptable)
- **Expected accuracy:** 45-50% (+5-10%)

---

## Recommendation

### ❌ **DO NOT** implement full N400-style prediction error

**Reasons:**
1. Too slow (50-100ms latency)
2. Too complex (SBERT, PyTorch)
3. Too heavy (100MB+ model)
4. Semantic similarity ≠ true prediction error
5. German language support limited

### ✅ **SKIP** to simpler features instead

**Better alternatives:**
- **Feature #5:** Disfluency detection (clear signal, lightweight)
- **Feature #7:** Adaptive thresholds (high impact, zero complexity)
- Improve existing semantic detector with better patterns

**Why skip N400?**
- Brain-inspired doesn't mean practical
- Trade-off: 10-15% accuracy gain vs 50-100ms latency = BAD
- Production VAD needs <10ms latency
- Already 3 failed features (#1, #2, #3) - pattern is clear

---

## Lessons from Features #1-4

| Feature | Concept Quality | Implementation Complexity | Latency Impact | Accuracy Gain | Integrated? |
|---------|----------------|---------------------------|----------------|---------------|-------------|
| #1: Prediction Error | ⭐⭐⭐⭐ | Medium | +0.11ms | 0% | ❌ No |
| #2: Rhythm Tracking | ⭐⭐⭐ | Medium | <2ms | -75% | ❌ No |
| #3: Temporal Context | ⭐⭐⭐⭐⭐ | High | <0.1ms | -40% | ❌ No |
| #4: N400 Prediction | ⭐⭐⭐⭐⭐ | Very High | +50-100ms | +10-15%? | ❌ No |

**Pattern:**
- Strong research ≠ production value
- Brain-inspired features are too complex
- Need to focus on **simple, fast, practical** features

---

## Next Steps

1. ✅ Document Feature #4 research (this file)
2. ⏭️ **SKIP** Feature #4 implementation
3. ⏭️ Move to Feature #5 (Disfluency Detection) or Feature #7 (Adaptive Thresholds)
4. 🎯 Focus on features with <5ms latency and >10% accuracy gain

---

## References

1. Lopopolo & Rabovsky (2024) "Tracking Lexical and Semantic Prediction Error Underlying the N400" - Neurobiology of Language
2. Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
3. Hugging Face Sentence Transformers: https://huggingface.co/tasks/sentence-similarity
4. MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard

---

**Status:** Feature #4 research complete, NOT recommended for implementation
**Next:** Skip to Feature #5 or Feature #7
