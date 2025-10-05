"""
Test Excellence VAD on Diverse Real-World Scenarios
====================================================

Tests various conversation types:
- Customer service calls
- Meeting discussions
- Casual conversations
- Different accents and speaking styles
"""

import numpy as np
from excellence_vad import ExcellenceVAD


def generate_diverse_scenarios(sr=16000):
    """Generate diverse conversation scenarios"""

    def make_utterance(duration, f0_start, f0_end, modulation_rate, energy_level=1.0):
        t = np.linspace(0, duration, int(sr * duration))
        f0 = np.linspace(f0_start, f0_end, len(t))

        # Rich harmonic signal
        signal = energy_level * np.sin(2 * np.pi * f0 * t)
        signal += 0.8 * energy_level * np.sin(2 * np.pi * 2 * f0 * t)
        signal += 0.6 * energy_level * np.sin(2 * np.pi * 3 * f0 * t)
        signal += 0.4 * energy_level * np.sin(2 * np.pi * 500 * t)
        signal += 0.3 * energy_level * np.sin(2 * np.pi * 1500 * t)

        # Amplitude modulation
        modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * modulation_rate * t))
        signal *= modulation

        return signal / np.max(np.abs(signal))

    scenarios = []

    # ============================================================================
    # SCENARIO 1: Customer Service - Polite Interruption
    # ============================================================================
    scenario1 = {
        'name': 'Customer Service - Polite Interruption',
        'description': 'Agent explaining, customer interjects politely',
        'segments': []
    }

    # Agent: "Your order will be shipped within 3 to 5 business days and"
    utt1 = make_utterance(2.0, 150, 155, 5.0)  # Level pitch (continuing)
    scenario1['segments'].append({
        'audio': utt1,
        'text': "Your order will be shipped within 3 to 5 business days and",
        'speaker': 'AI',
        'label': 'incomplete',
        'expected_turn_end': 'LOW'  # <50% (incomplete)
    })

    # Customer: "Actually, I need it sooner"
    utt2 = make_utterance(1.2, 145, 140, 5.2)  # Falling pitch
    scenario1['segments'].append({
        'audio': utt2,
        'text': "Actually, I need it sooner",
        'speaker': 'USER',
        'label': 'interruption',
        'expected_action': 'interrupt_ai_immediately'
    })

    scenarios.append(scenario1)

    # ============================================================================
    # SCENARIO 2: Meeting - Natural Turn-Taking
    # ============================================================================
    scenario2 = {
        'name': 'Meeting - Natural Turn-Taking',
        'description': 'Speaker finishes point, colleague responds',
        'segments': []
    }

    # Speaker A: "I think we should schedule the meeting for next Tuesday"
    utt1 = make_utterance(2.2, 145, 130, 4.8)  # Falling pitch (complete)
    scenario2['segments'].append({
        'audio': utt1,
        'text': "I think we should schedule the meeting for next Tuesday",
        'speaker': 'AI',
        'label': 'complete_statement',
        'expected_turn_end': 'HIGH'  # >70%
    })

    # Pause (200ms)
    pause = np.zeros(int(sr * 0.2))
    scenario2['segments'].append({
        'audio': pause,
        'text': "",
        'speaker': 'NONE',
        'label': 'pause',
        'expected_turn_end': 'N/A'
    })

    # Speaker B: "That works for me"
    utt2 = make_utterance(0.8, 150, 135, 5.0)
    scenario2['segments'].append({
        'audio': utt2,
        'text': "That works for me",
        'speaker': 'USER',
        'label': 'natural_response',
        'expected_action': 'wait_for_ai_completion'
    })

    scenarios.append(scenario2)

    # ============================================================================
    # SCENARIO 3: Fast-Paced Conversation - Quick Back-and-Forth
    # ============================================================================
    scenario3 = {
        'name': 'Fast-Paced Conversation',
        'description': 'Quick exchanges, minimal pauses',
        'segments': []
    }

    # AI: "What about Friday?"
    utt1 = make_utterance(0.8, 155, 180, 5.5)  # Rising (question)
    scenario3['segments'].append({
        'audio': utt1,
        'text': "What about Friday?",
        'speaker': 'AI',
        'label': 'complete_question',
        'expected_turn_end': 'HIGH'  # >70%
    })

    # Short pause (100ms - fast conversation)
    pause = np.zeros(int(sr * 0.1))
    scenario3['segments'].append({
        'audio': pause,
        'text': "",
        'speaker': 'NONE',
        'label': 'short_pause',
        'expected_turn_end': 'N/A'
    })

    # User: "Perfect"
    utt2 = make_utterance(0.5, 150, 135, 5.0)
    scenario3['segments'].append({
        'audio': utt2,
        'text': "Perfect",
        'speaker': 'USER',
        'label': 'quick_acknowledgment',
        'expected_action': 'wait_for_ai_completion'
    })

    scenarios.append(scenario3)

    # ============================================================================
    # SCENARIO 4: Hesitation - AI Uncertain
    # ============================================================================
    scenario4 = {
        'name': 'Hesitation Pattern',
        'description': 'AI hesitates, user waits then responds',
        'segments': []
    }

    # AI: "I think the answer is... um..."
    utt1 = make_utterance(1.5, 155, 155, 4.5, energy_level=0.7)  # Level, quieter
    scenario4['segments'].append({
        'audio': utt1,
        'text': "I think the answer is um",
        'speaker': 'AI',
        'label': 'hesitation_incomplete',
        'expected_turn_end': 'LOW'  # <50% (hesitation, continuing)
    })

    # User: "Take your time"
    utt2 = make_utterance(0.9, 148, 138, 5.0)
    scenario4['segments'].append({
        'audio': utt2,
        'text': "Take your time",
        'speaker': 'USER',
        'label': 'supportive_comment',
        'expected_action': 'interrupt_ai_immediately'
    })

    scenarios.append(scenario4)

    # ============================================================================
    # SCENARIO 5: Multi-Clause Sentence
    # ============================================================================
    scenario5 = {
        'name': 'Multi-Clause Sentence',
        'description': 'Long sentence with multiple clauses',
        'segments': []
    }

    # AI: "When you arrive at the hotel, please check in at the front desk"
    utt1 = make_utterance(2.5, 150, 135, 4.8)  # Falling (complete)
    scenario5['segments'].append({
        'audio': utt1,
        'text': "When you arrive at the hotel, please check in at the front desk",
        'speaker': 'AI',
        'label': 'complete_multi_clause',
        'expected_turn_end': 'HIGH'  # >70%
    })

    # Pause
    pause = np.zeros(int(sr * 0.25))
    scenario5['segments'].append({
        'audio': pause,
        'text': "",
        'speaker': 'NONE',
        'label': 'pause',
        'expected_turn_end': 'N/A'
    })

    # User: "Okay, got it"
    utt2 = make_utterance(0.7, 150, 140, 5.0)
    scenario5['segments'].append({
        'audio': utt2,
        'text': "Okay, got it",
        'speaker': 'USER',
        'label': 'acknowledgment',
        'expected_action': 'wait_for_ai_completion'
    })

    scenarios.append(scenario5)

    # ============================================================================
    # SCENARIO 6: List Enumeration
    # ============================================================================
    scenario6 = {
        'name': 'List Enumeration',
        'description': 'AI listing items, user interrupts',
        'segments': []
    }

    # AI: "We have three options available: economy, business, and"
    utt1 = make_utterance(2.0, 155, 155, 5.0)  # Level (continuing list)
    scenario6['segments'].append({
        'audio': utt1,
        'text': "We have three options available: economy, business, and",
        'speaker': 'AI',
        'label': 'incomplete_list',
        'expected_turn_end': 'LOW'  # <50%
    })

    # User: "Business class please"
    utt2 = make_utterance(1.0, 148, 138, 5.2)
    scenario6['segments'].append({
        'audio': utt2,
        'text': "Business class please",
        'speaker': 'USER',
        'label': 'decisive_interruption',
        'expected_action': 'interrupt_ai_immediately'
    })

    scenarios.append(scenario6)

    # ============================================================================
    # SCENARIO 7: Confirmation Question
    # ============================================================================
    scenario7 = {
        'name': 'Confirmation Question',
        'description': 'AI asks for confirmation',
        'segments': []
    }

    # AI: "So you want to book for 2 people, is that correct?"
    utt1 = make_utterance(2.0, 150, 175, 5.0)  # Rising (question)
    scenario7['segments'].append({
        'audio': utt1,
        'text': "So you want to book for 2 people, is that correct?",
        'speaker': 'AI',
        'label': 'confirmation_question',
        'expected_turn_end': 'HIGH'  # >80%
    })

    # Short pause
    pause = np.zeros(int(sr * 0.15))
    scenario7['segments'].append({
        'audio': pause,
        'text': "",
        'speaker': 'NONE',
        'label': 'pause',
        'expected_turn_end': 'N/A'
    })

    # User: "Yes"
    utt2 = make_utterance(0.4, 150, 140, 5.0)
    scenario7['segments'].append({
        'audio': utt2,
        'text': "Yes",
        'speaker': 'USER',
        'label': 'confirmation',
        'expected_action': 'wait_for_ai_completion'
    })

    scenarios.append(scenario7)

    return scenarios


def test_diverse_scenarios():
    """Test Excellence VAD on diverse scenarios"""

    print("=" * 80)
    print(" EXCELLENCE VAD - DIVERSE REAL-WORLD SCENARIOS")
    print("=" * 80)
    print()

    vad = ExcellenceVAD(sample_rate=16000, turn_end_threshold=0.75)

    scenarios = generate_diverse_scenarios()

    total_correct = 0
    total_tests = 0

    for scenario in scenarios:
        print("-" * 80)
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 80)
        print()

        vad.reset()
        frame_size = 160

        for seg_idx, segment in enumerate(scenario['segments']):
            audio = segment['audio']
            text = segment['text']
            speaker = segment['speaker']
            label = segment['label']

            print(f"Segment {seg_idx + 1}: [{speaker}] {label}")
            print(f"  Text: \"{text}\"" if text else "  Text: (silence)")

            # Process audio frames
            results = []
            for i in range(0, len(audio) - frame_size, frame_size):
                frame = audio[i:i+frame_size]

                # Simulate stereo channels
                if speaker == 'AI':
                    user_frame = np.zeros(frame_size)
                    ai_frame = frame
                elif speaker == 'USER':
                    user_frame = frame
                    ai_frame = np.zeros(frame_size)
                else:  # NONE (silence)
                    user_frame = np.zeros(frame_size)
                    ai_frame = np.zeros(frame_size)

                result = vad.process_frame(user_frame, ai_frame, text)

                if result.get('user_speaking') or result.get('ai_speaking'):
                    results.append(result)

            # Evaluate based on expected outcome
            if 'expected_turn_end' in segment and segment['expected_turn_end'] != 'N/A':
                if results:
                    avg_turn_end = np.mean([r.get('turn_end_prob', 0.0) for r in results])

                    if segment['expected_turn_end'] == 'HIGH':
                        threshold = 0.70
                        passed = avg_turn_end >= threshold
                        print(f"  Turn-end: {avg_turn_end:.1%} (expected: >{threshold:.0%}) - {'PASS' if passed else 'FAIL'}")
                    elif segment['expected_turn_end'] == 'LOW':
                        threshold = 0.50
                        passed = avg_turn_end < threshold
                        print(f"  Turn-end: {avg_turn_end:.1%} (expected: <{threshold:.0%}) - {'PASS' if passed else 'FAIL'}")

                    if passed:
                        total_correct += 1
                    total_tests += 1

            if 'expected_action' in segment:
                if results:
                    actions = [r['action'] for r in results if r.get('overlap') or r.get('user_speaking')]
                    if actions:
                        expected = segment['expected_action']
                        actual = actions[-1]  # Last action

                        if expected == 'interrupt_ai_immediately':
                            passed = 'interrupt' in actual
                        else:
                            passed = actual == expected

                        print(f"  Action: {actual} (expected: {expected}) - {'PASS' if passed else 'FAIL'}")

                        if passed:
                            total_correct += 1
                        total_tests += 1

            print()

        print()

    # Final results
    print("=" * 80)
    print(" FINAL RESULTS - DIVERSE SCENARIOS")
    print("=" * 80)
    print()

    if total_tests > 0:
        accuracy = (total_correct / total_tests) * 100
        print(f"Accuracy: {total_correct}/{total_tests} ({accuracy:.1f}%)")
        print()

        if accuracy >= 90:
            print(f"EXCELLENT: {accuracy:.1f}% - Production-ready performance")
        elif accuracy >= 80:
            print(f"GOOD: {accuracy:.1f}% - Strong real-world performance")
        elif accuracy >= 70:
            print(f"ACCEPTABLE: {accuracy:.1f}% - Usable with monitoring")
        else:
            print(f"NEEDS WORK: {accuracy:.1f}% - Further tuning required")

    print()

    # Performance stats
    stats = vad.get_stats()
    print("PERFORMANCE:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Target: <10ms {'PASS' if stats['avg_latency_ms'] < 10 else 'FAIL'}")
    print()

    return accuracy if total_tests > 0 else None


if __name__ == "__main__":
    accuracy = test_diverse_scenarios()
