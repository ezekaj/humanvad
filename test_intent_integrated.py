"""
Test FlexDuo + Silero VAD with Intent Classification
=====================================================

Test on scenarios with text to enable intent-based barge-in handling
"""

import numpy as np
from flexduo_silero_vad import FlexDuoSileroVAD, DuplexFrame


def test_backchannel_scenario():
    """Test: User says 'mhm' during AI speech (should NOT stop AI)"""
    print("\n" + "="*80)
    print(" TEST 1: Backchannel (mhm) - Should NOT Stop AI")
    print("="*80)

    vad = FlexDuoSileroVAD(
        sample_rate=8000,
        user_threshold=0.20,
        ai_threshold=0.20,
        interrupt_threshold=0.50,
        enable_intent=True
    )

    timestamp_ms = 0
    frame_size = 240  # 30ms @ 8kHz

    # AI speaking
    print("\nAI: 'Das Zimmer kostet 120 Euro pro Nacht...' (speaking)")
    ai_speech = np.random.randn(8000) * 0.6  # 1s of AI speech

    for i in range(0, len(ai_speech), frame_size):
        ai_chunk = ai_speech[i:i+frame_size]
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))
        user_chunk = np.random.randn(frame_size) * 0.01

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk)
        decision = vad.process_frame(frame)
        timestamp_ms += 30

    # User says "mhm" (backchannel) - should NOT stop AI
    print("User: 'Mhm' (backchannel during AI speech)")
    user_backchannel = np.random.randn(600) * 0.7  # 75ms "mhm"

    for i in range(0, len(user_backchannel), frame_size):
        user_chunk = user_backchannel[i:i+frame_size]
        ai_chunk = ai_speech[min(i + 6000, len(ai_speech)-frame_size):min(i + 6000 + frame_size, len(ai_speech))]

        if len(user_chunk) < frame_size:
            user_chunk = np.pad(user_chunk, (0, frame_size - len(user_chunk)))
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))

        # Add text for intent classification
        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk, user_text="mhm")
        decision = vad.process_frame(frame)

        if decision.should_stop_ai:
            print(f"  ERROR: AI stopped for backchannel at {timestamp_ms}ms!")
            return False

        if decision.intent:
            print(f"  Intent detected: {decision.intent}")

        timestamp_ms += 30

    stats = vad.get_statistics()
    print(f"\nResult: {stats['total_barge_ins']} barge-ins, {stats['false_barge_ins_prevented']} prevented")

    if stats['total_barge_ins'] == 0 and stats['false_barge_ins_prevented'] > 0:
        print("[PASS] Backchannel correctly ignored")
        return True
    else:
        print("[FAIL] Backchannel incorrectly triggered barge-in")
        return False


def test_question_interruption():
    """Test: User asks question during AI speech (should STOP AI)"""
    print("\n" + "="*80)
    print(" TEST 2: Question Interruption - Should STOP AI")
    print("="*80)

    vad = FlexDuoSileroVAD(
        sample_rate=8000,
        user_threshold=0.20,
        interrupt_threshold=0.50,
        enable_intent=True
    )

    timestamp_ms = 0
    frame_size = 240

    # AI speaking
    print("\nAI: 'Die Buchung ist bestätigt...' (speaking)")
    ai_speech = np.random.randn(8000) * 0.6

    for i in range(0, len(ai_speech), frame_size):
        ai_chunk = ai_speech[i:i+frame_size]
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))
        user_chunk = np.random.randn(frame_size) * 0.01

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk)
        vad.process_frame(frame)
        timestamp_ms += 30

    # User asks question - should STOP AI
    print("User: 'Wie viel kostet das?' (question interruption)")
    user_question = np.random.randn(1600) * 0.7  # 200ms question

    stopped = False
    for i in range(0, len(user_question), frame_size):
        user_chunk = user_question[i:i+frame_size]
        ai_chunk = ai_speech[min(i + 6000, len(ai_speech)-frame_size):min(i + 6000 + frame_size, len(ai_speech))]

        if len(user_chunk) < frame_size:
            user_chunk = np.pad(user_chunk, (0, frame_size - len(user_chunk)))
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk, user_text="Wie viel kostet das?")
        decision = vad.process_frame(frame)

        if decision.should_stop_ai:
            stopped = True
            print(f"  AI stopped at {timestamp_ms}ms for question")
            if decision.intent:
                print(f"  Intent: {decision.intent}")

        timestamp_ms += 30

    stats = vad.get_statistics()
    print(f"\nResult: {stats['total_barge_ins']} barge-ins, stopped={stopped}")

    if stopped and stats['total_barge_ins'] >= 1:
        print("[PASS] ✓ Question correctly triggered barge-in")
        return True
    else:
        print("[FAIL] Question did not trigger barge-in")
        return False


def test_acknowledgment_short():
    """Test: User says 'Ja' (short acknowledgment - should NOT stop AI)"""
    print("\n" + "="*80)
    print(" TEST 3: Short Acknowledgment (Ja) - Should NOT Stop AI")
    print("="*80)

    vad = FlexDuoSileroVAD(
        sample_rate=8000,
        user_threshold=0.20,
        interrupt_threshold=0.50,
        enable_intent=True
    )

    timestamp_ms = 0
    frame_size = 240

    # AI speaking
    print("\nAI: 'Das Zimmer hat Meerblick...' (speaking)")
    ai_speech = np.random.randn(8000) * 0.6

    for i in range(0, len(ai_speech), frame_size):
        ai_chunk = ai_speech[i:i+frame_size]
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))
        user_chunk = np.random.randn(frame_size) * 0.01

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk)
        vad.process_frame(frame)
        timestamp_ms += 30

    # User says "Ja" (short acknowledgment)
    print("User: 'Ja' (short acknowledgment)")
    user_ack = np.random.randn(400) * 0.7  # 50ms "ja"

    for i in range(0, len(user_ack), frame_size):
        user_chunk = user_ack[i:i+frame_size]
        ai_chunk = ai_speech[min(i + 6000, len(ai_speech)-frame_size):min(i + 6000 + frame_size, len(ai_speech))]

        if len(user_chunk) < frame_size:
            user_chunk = np.pad(user_chunk, (0, frame_size - len(user_chunk)))
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk, user_text="Ja")
        decision = vad.process_frame(frame)

        if decision.should_stop_ai:
            print(f"  ERROR: AI stopped for short acknowledgment at {timestamp_ms}ms!")
            return False

        timestamp_ms += 30

    stats = vad.get_statistics()
    print(f"\nResult: {stats['total_barge_ins']} barge-ins, {stats['false_barge_ins_prevented']} prevented")

    if stats['total_barge_ins'] == 0 and stats['false_barge_ins_prevented'] > 0:
        print("[PASS] ✓ Short acknowledgment correctly ignored")
        return True
    else:
        print("[FAIL] Short acknowledgment incorrectly triggered barge-in")
        return False


def test_request_interruption():
    """Test: User makes request during AI speech (should STOP AI)"""
    print("\n" + "="*80)
    print(" TEST 4: Request Interruption - Should STOP AI")
    print("="*80)

    vad = FlexDuoSileroVAD(
        sample_rate=8000,
        user_threshold=0.20,
        interrupt_threshold=0.50,
        enable_intent=True
    )

    timestamp_ms = 0
    frame_size = 240

    # AI speaking
    print("\nAI: 'Wir haben verschiedene Zimmertypen...' (speaking)")
    ai_speech = np.random.randn(8000) * 0.6

    for i in range(0, len(ai_speech), frame_size):
        ai_chunk = ai_speech[i:i+frame_size]
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))
        user_chunk = np.random.randn(frame_size) * 0.01

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk)
        vad.process_frame(frame)
        timestamp_ms += 30

    # User makes request
    print("User: 'Ich möchte ein Doppelzimmer' (request)")
    user_request = np.random.randn(2000) * 0.7  # 250ms request

    stopped = False
    for i in range(0, len(user_request), frame_size):
        user_chunk = user_request[i:i+frame_size]
        ai_chunk = ai_speech[min(i + 6000, len(ai_speech)-frame_size):min(i + 6000 + frame_size, len(ai_speech))]

        if len(user_chunk) < frame_size:
            user_chunk = np.pad(user_chunk, (0, frame_size - len(user_chunk)))
        if len(ai_chunk) < frame_size:
            ai_chunk = np.pad(ai_chunk, (0, frame_size - len(ai_chunk)))

        frame = DuplexFrame(timestamp_ms, user_chunk, ai_chunk, user_text="Ich möchte ein Doppelzimmer")
        decision = vad.process_frame(frame)

        if decision.should_stop_ai:
            stopped = True
            print(f"  AI stopped at {timestamp_ms}ms for request")
            if decision.intent:
                print(f"  Intent: {decision.intent}")

        timestamp_ms += 30

    stats = vad.get_statistics()
    print(f"\nResult: {stats['total_barge_ins']} barge-ins, stopped={stopped}")

    if stopped and stats['total_barge_ins'] >= 1:
        print("[PASS] ✓ Request correctly triggered barge-in")
        return True
    else:
        print("[FAIL] Request did not trigger barge-in")
        return False


def run_all_tests():
    """Run all intent-integrated tests"""
    print("="*80)
    print(" FLEXDUO + SILERO VAD + INTENT CLASSIFICATION TESTS")
    print("="*80)
    print("\nTesting intent-based barge-in handling:")
    print("  - Backchannel (mhm, ja) -> Don't stop AI")
    print("  - Question -> Stop AI")
    print("  - Request -> Stop AI")
    print()

    results = []
    results.append(("Backchannel ignored", test_backchannel_scenario()))
    results.append(("Question stops AI", test_question_interruption()))
    results.append(("Short acknowledgment ignored", test_acknowledgment_short()))
    results.append(("Request stops AI", test_request_interruption()))

    print("\n" + "="*80)
    print(" INTENT INTEGRATION RESULTS")
    print("="*80)
    print()

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for scenario, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {scenario}")

    print()
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print()

    if passed == total:
        print("[SUCCESS] 🎉 Intent classification fully integrated!")
        print("\nExpected accuracy improvement: +7.6% (from FlexDuo paper)")
    elif passed >= 3:
        print("[GOOD] Intent classification working well")
    else:
        print("[NEEDS WORK] Intent integration issues")

    print()


if __name__ == "__main__":
    run_all_tests()
