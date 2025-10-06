# German Voice Activity Detection (VAD) with Semantic Turn-End Detection

**Production-ready German conversational AI turn-taking system**

## Production Recommendation: ExcellenceVADGerman

**96.7% multi-domain accuracy | 0.077ms latency | Full German language support**

### Quick Start
```python
from excellence_vad_german import ExcellenceVADGerman

vad = ExcellenceVADGerman(turn_end_threshold=0.60)
result = vad.process_audio(audio_chunk, transcript="Das Hotel hat funfzig Zimmer")

if result['action'] == 'interrupt':
    agent.start_speaking()  # User finished speaking
else:
    agent.wait()  # User still speaking
```

### Performance Summary
- 96.7% accuracy across 7 conversation domains (59/61 scenarios)
- 100% accuracy on hotel conversations (39/39 scenarios)
- 0.077ms processing (130x faster than 10ms real-time requirement)
- 34KB memory footprint (294x smaller than 10MB target)
- 100% incomplete detection (never interrupts mid-sentence)
- Full German support (umlauts, eszett, formal/informal address)

## Main File

**`excellence_vad_german.py`** - Production German VAD system

## Test Results

### Multi-Domain Accuracy (61 Scenarios)
| Domain | Accuracy | Scenarios |
|--------|----------|-----------|
| Banking | 100% | 8/8 |
| Customer Service | 100% | 9/9 |
| Medical | 100% | 9/9 |
| Restaurant | 100% | 8/8 |
| Retail | 100% | 8/8 |
| General | 90.9% | 10/11 |
| Travel | 87.5% | 7/8 |
| OVERALL | 96.7% | 59/61 |

### Speed Performance
- Average: 0.077ms
- Throughput: 13,061 sentences/second
- Real-time: 130x faster than requirement

## Documentation
- `PERFORMANCE_SUMMARY.md` - Complete performance analysis
- `DIVERSE_DATA_RESULTS.md` - Multi-domain test results
- `FINAL_RESULTS_100_PERCENT.md` - Hotel conversation results

## Usage

### Basic Usage (Semantic-Only)
```python
from excellence_vad_german import ExcellenceVADGerman

vad = ExcellenceVADGerman(turn_end_threshold=0.60)
semantic = vad.semantic_detector

# Test German sentence
result = semantic.is_complete("Das Hotel hat funfzig Zimmer")
print(result['complete_prob'])  # 0.9 (complete)

# Determine action
prosody_prob = 0.30  # neutral
final_prob = 0.45 * prosody_prob + 0.55 * result['complete_prob']
action = 'interrupt' if final_prob >= 0.60 else 'wait'
```

### With Ollama (Local LLM)
```python
from excellence_vad_german import ExcellenceVADGerman
import requests

vad = ExcellenceVADGerman(turn_end_threshold=0.60)

# Generate response with Ollama
response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'llama3.2',
    'prompt': 'Guten Tag, wie kann ich helfen?',
    'stream': True
})

# Check turn-end as text streams
for line in response.iter_lines():
    data = json.loads(line)
    text = data.get('response', '')

    # Real-time turn-end detection
    result = vad.semantic_detector.is_complete(text)
    if result['complete_prob'] >= 0.70:
        print("Turn complete - user can speak")
        break
```

### With Audio + Text
```python
import numpy as np
from excellence_vad_german import ExcellenceVADGerman

vad = ExcellenceVADGerman(sample_rate=16000)

# Process 10ms audio frame (160 samples at 16kHz)
result = vad.process_frame(
    user_frame=user_audio,  # np.ndarray, 160 samples
    ai_frame=ai_audio,      # np.ndarray, 160 samples
    ai_text="Guten Tag"
)

print(f"Action: {result['action']}")           # 'interrupt' or 'wait'
print(f"Latency: {result['latency_ms']}ms")    # ~0.077ms
```

## Integration Examples

### Ollama + VAD Pipeline
```python
from excellence_vad_german import ExcellenceVADGerman
import ollama

vad = ExcellenceVADGerman(turn_end_threshold=0.60)

def chat_with_vad(user_input):
    response = ""
    for chunk in ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': user_input}],
        stream=True
    ):
        response += chunk['message']['content']

        # Check if turn is complete
        result = vad.semantic_detector.is_complete(response)
        if result['complete_prob'] >= 0.70:
            yield response
            response = ""

# Usage
for sentence in chat_with_vad("Wie geht es dir?"):
    print(sentence)
    # User can interrupt here
```

### Dependencies
```bash
pip install numpy requests ollama
```

## Running Tests
```bash
python test_comprehensive.py          # Hotel test (100%)
python generate_diverse_german_test.py  # Multi-domain (96.7%)
python test_performance.py            # Speed/memory benchmark
python evaluate_excellence_real_audio.py  # Compare Excellence vs Production on real WAV clips
# To export JSON metrics alongside the console summary:
# python evaluate_excellence_real_audio.py --report reports/real_audio_summary.json
```
