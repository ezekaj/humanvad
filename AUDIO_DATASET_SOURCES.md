# German Conversational Audio Datasets for VAD Testing

## Recommended Free Datasets

### 1. Mozilla Common Voice (German) - **BEST FOR TESTING**

**Why:** Free, CC0 license, 1000+ hours German speech, includes conversational/spontaneous speech

**Download:**
- Website: https://commonvoice.mozilla.org/
- Hugging Face: https://huggingface.co/datasets/mozilla-foundation/common_voice_20_0

**Quick Download (Python):**
```python
from datasets import load_dataset

# Download German subset
dataset = load_dataset("mozilla-foundation/common_voice_20_0", "de", split="train")

# Access audio files
for sample in dataset:
    audio = sample['audio']  # Audio array
    text = sample['sentence']  # Transcription
    # Use for testing
```

**Stats:**
- Language: German (de)
- Hours: 1040+ hours (as of 2022, likely more now)
- License: CC0 (Public Domain)
- Format: MP3
- Sample Rate: 48kHz (downsample to 16kHz for our VAD)
- Type: Mix of read speech and spontaneous conversation

**Pros:**
- ✅ Free and open
- ✅ Easy programmatic access
- ✅ Large German dataset
- ✅ Includes spontaneous speech (v20+)
- ✅ Diverse speakers

**Cons:**
- ⚠️ Not all samples are conversational dialogue (many are single-speaker read speech)
- ⚠️ Need to filter for multi-turn conversations

---

### 2. UniDataPro German Speech Dataset (Hugging Face)

**Why:** 431 hours of telephone dialogues, real conversations

**Download:**
```python
from datasets import load_dataset

dataset = load_dataset("UniDataPro/german-speech-recognition-dataset")
```

**Stats:**
- Type: Telephone dialogues
- Hours: 431 hours
- Speakers: 590+ native German speakers
- Accuracy: 95% sentence accuracy
- License: Check Hugging Face page

**Pros:**
- ✅ Real telephone conversations (perfect for hotel AI use case!)
- ✅ Native speakers
- ✅ Turn-taking dynamics

**Cons:**
- ⚠️ License unclear (check before commercial use)

---

### 3. CALLHOME German (TalkBank)

**Why:** Classic research corpus, real family phone calls

**Download:**
- TalkBank CABank: https://ca.talkbank.org/access/CallHome/deu.html

**Stats:**
- Type: Telephone conversations
- Duration: 100 conversations
- Context: Family/friends calling internationally
- Format: Audio + transcripts

**Pros:**
- ✅ Real conversational dynamics
- ✅ Academic research standard
- ✅ Natural turn-taking

**Cons:**
- ⚠️ Smaller dataset
- ⚠️ May require registration

---

## Quick Test Script

Create `test_with_real_audio.py`:

```python
from datasets import load_dataset
import librosa
import numpy as np
from excellence_vad_german import ExcellenceVADGerman

# Load Mozilla Common Voice German
print("Loading dataset...")
dataset = load_dataset(
    "mozilla-foundation/common_voice_20_0",
    "de",
    split="validation",
    streaming=True  # Stream to avoid downloading everything
)

vad = ExcellenceVADGerman(sample_rate=16000)

# Test on first 10 samples
for i, sample in enumerate(dataset):
    if i >= 10:
        break

    # Get audio
    audio = sample['audio']['array']
    sr = sample['audio']['sampling_rate']

    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Process in frames
    frame_size = 160  # 10ms at 16kHz
    text = sample['sentence']

    print(f"\nSample {i+1}: {text}")

    for j in range(0, len(audio) - frame_size, frame_size):
        frame = audio[j:j+frame_size]
        result = vad.process_frame(
            np.zeros(frame_size),  # User silent
            frame,  # AI speaking
            text
        )

        if result.get('ai_speaking'):
            print(f"  Frame {j//frame_size}: Turn-end prob = {result['turn_end_prob']:.2f}")
```

---

## Testing Checklist for Features #1 & #2

### Feature #1: Prediction Error Detection
- [ ] Test with real German speech (not sine waves)
- [ ] Verify prediction error varies (not stuck at 1.4%-2.7%)
- [ ] Check if high prediction error correlates with interruptions
- [ ] Benchmark accuracy improvement vs baseline

### Feature #2: Rhythm Tracking
- [ ] Test with rhythmic vs non-rhythmic speech patterns
- [ ] Verify rhythm disruption detection at turn boundaries
- [ ] Check tempo deceleration detection
- [ ] Adjust autocorrelation window based on real speech patterns

---

## Next Steps

1. **Download Mozilla Common Voice German**
   ```bash
   pip install datasets librosa
   python test_with_real_audio.py
   ```

2. **Test Features #1 & #2 with real audio**

3. **If accuracy improves >5%, integrate into production**

4. **If no improvement, document and move to next features**
