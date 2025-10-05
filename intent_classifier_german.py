"""
Intent Classifier for German Conversational AI
===============================================

Research-based intent detection with adjacency pair awareness and response timing.

Based on:
- Levinson (2016) - Turn-taking timing (0-200ms gaps)
- Sacks et al. (1974) - Adjacency pairs
- Dialogue Act classification (DAMSL/ISO 24617-2)
- German pragmatics and prosody patterns

"""

import re
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class IntentResult:
    """Intent classification result"""
    intent_type: str           # Main intent category
    intent_subtype: str        # Specific subtype
    expected_gap_ms: int       # Expected response gap (milliseconds)
    is_fpp: bool               # Is First Pair Part (expects immediate response)
    expected_spp: Optional[str] # Expected Second Pair Part type
    confidence: float          # Classification confidence 0-1


class IntentClassifierGerman:
    """
    German Intent Classifier with Adjacency Pair Awareness

    Classifies speaker intent and predicts expected response timing
    based on dialogue act theory and German pragmatics.

    Intent Categories (based on DAMSL + German extensions):
    1. Questions (WH, Yes/No, Tag)
    2. Statements (Assert, Report, Inform)
    3. Requests (Command, Polite Request, Suggestion)
    4. Responses (Answer, Confirm, Deny)
    5. Social (Greeting, Closing, Thanks, Apology)
    6. Discourse Markers (Backchannel, Filler, Repair)
    """

    def __init__(self):
        # ADJACENCY PAIR FIRST PARTS (expect immediate response)
        # Format: (pattern, intent_type, subtype, gap_ms, expected_spp)

        self.fpp_patterns = [
            # === GREETINGS === (50-100ms gap)
            (r'^(hallo|guten tag|guten morgen|guten abend|moin)\b',
             'greeting', 'initial', 50, 'greeting'),

            # === WH-QUESTIONS === (100-150ms gap - expect specific answer)
            (r'^\s*(wann|wie|wo|wohin|woher)\s+',
             'question', 'wh_question', 100, 'answer_specific'),
            (r'^\s*(was|welche[rs]?|wer|wessen|wem|wen)\s+',
             'question', 'wh_question', 100, 'answer_specific'),
            (r'^\s*warum\s+',
             'question', 'wh_question', 120, 'answer_reason'),
            (r'^\s*wie\s+viele?\s+',
             'question', 'wh_quantity', 100, 'answer_number'),

            # === YES/NO QUESTIONS === (150-200ms gap - expect ja/nein)
            (r'^\s*(können|möchten|wollen|dürfen|sollen|müssen)\s+sie\b',
             'question', 'yn_modal', 150, 'answer_yn'),
            (r'^\s*(haben|sind|ist|war|werden|wird)\s+sie\b',
             'question', 'yn_be_have', 150, 'answer_yn'),
            (r'\?\s*$',  # Any sentence ending with ?
             'question', 'yn_general', 150, 'answer_yn'),

            # === TAG QUESTIONS === (100ms gap - strong expectation)
            (r'\b(oder|nicht wahr|gell|ne|wa)\s*\?\s*$',
             'question', 'tag_question', 100, 'confirmation'),

            # === OFFERS === (200ms gap - expect accept/reject)
            (r'^\s*(kann|könnte|darf|möchte)\s+ich\s+(ihnen\s+)?helfen\b',
             'offer', 'help_offer', 200, 'accept_reject'),
            (r'^\s*möchten\s+sie\s+(noch\s+)?(etwas|was)\b',
             'offer', 'service_offer', 200, 'accept_reject'),

            # === REQUESTS === (250ms gap - expect compliance)
            (r'^\s*könnten\s+sie\b',
             'request', 'polite_request', 250, 'comply_refuse'),
            (r'^\s*(bitte|würden\s+sie)\b',
             'request', 'polite_request', 250, 'comply_refuse'),
            (r'^\s*(geben|zeigen|sagen|erklären)\s+sie\s+mir\b',
             'request', 'direct_request', 200, 'comply_refuse'),

            # === APOLOGY EXPECTATION === (100ms - expect acceptance)
            (r'\b(entschuldigung|tut mir leid|verzeihung)\b',
             'apology', 'apologize', 100, 'accept_apology'),

            # === CLOSING === (200ms - expect reciprocal)
            (r'\b(auf wiedersehen|auf wiederhören|tschüss|bis bald|schönes wochenende)\b',
             'closing', 'farewell', 200, 'farewell'),
        ]

        # NON-PAIR-INITIATING (looser timing)
        self.non_fpp_patterns = [
            # === STATEMENTS === (300-500ms gap - optional response)
            (r'^\s*ich\s+(denke|glaube|meine|finde)\b',
             'statement', 'opinion', 400, None),
            (r'^\s*(es\s+ist|das\s+ist|hier\s+ist)\b',
             'statement', 'assertion', 400, None),
            (r'\.\s*$',  # Sentence ending with period
             'statement', 'declarative', 400, None),

            # === CONFIRMATIONS === (100-200ms - often followed by elaboration)
            (r'^\s*(ja|jawohl|genau|richtig|stimmt|korrekt)\b',
             'response', 'confirm', 100, 'elaboration_optional'),

            # === DENIALS === (150ms - may expect clarification)
            (r'^\s*(nein|nicht|niemals|keineswegs)\b',
             'response', 'deny', 150, 'clarification_optional'),

            # === ACKNOWLEDGMENTS === (200ms - conversation continues)
            (r'^\s*(okay|gut|in ordnung|alles klar|perfekt|super)\b',
             'response', 'acknowledge', 200, None),

            # === THANKS === (150ms - expect "bitte"/"gerne")
            (r'\b(danke|vielen dank|dankeschön)\b',
             'social', 'thanks', 150, 'accept_thanks'),

            # === BACKCHANNEL INVITATION === (200ms - expects minimal response)
            (r'\b(verstehen\s+sie|wissen\s+sie)\s*\?\s*$',
             'discourse', 'check_understanding', 200, 'backchannel'),

            # === REPAIR INITIATION === (100ms - expect repetition/clarification)
            (r'^\s*(was|wie|bitte)\s*\?\s*$',
             'discourse', 'repair_request', 100, 'repetition'),
        ]

        # SPECIAL: Discourse markers (no specific timing)
        self.discourse_patterns = [
            (r'\b(äh|ähm|ehm|hm|hmm|ach|nun|also)\b',
             'discourse', 'filler', 500, None),
            (r'^\s*(übrigens|überhaupt|eigentlich|sowieso)\b',
             'discourse', 'digression', 300, None),
        ]

    def classify(self, text: str, prosody_features: Optional[Dict] = None) -> IntentResult:
        """
        Classify intent from German text and optional prosody

        Args:
            text: German utterance text
            prosody_features: Optional dict with:
                - final_f0_slope: float (rising > 0, falling < 0)
                - mean_f0: float (pitch in Hz)
                - duration: float (utterance length in ms)

        Returns:
            IntentResult with intent type and expected gap timing
        """

        if not text or len(text.strip()) == 0:
            return IntentResult(
                intent_type='unknown',
                intent_subtype='empty',
                expected_gap_ms=300,
                is_fpp=False,
                expected_spp=None,
                confidence=0.0
            )

        text_clean = text.strip().lower()

        # PRIORITY 1: Check FPP patterns (adjacency pair starters)
        for pattern, intent_type, subtype, gap_ms, expected_spp in self.fpp_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                confidence = self._calculate_confidence(text_clean, pattern, prosody_features)

                # Adjust gap based on prosody if available
                adjusted_gap = self._adjust_gap_for_prosody(gap_ms, prosody_features, intent_type)

                return IntentResult(
                    intent_type=intent_type,
                    intent_subtype=subtype,
                    expected_gap_ms=adjusted_gap,
                    is_fpp=True,
                    expected_spp=expected_spp,
                    confidence=confidence
                )

        # PRIORITY 2: Check non-FPP patterns
        for pattern, intent_type, subtype, gap_ms, expected_spp in self.non_fpp_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                confidence = self._calculate_confidence(text_clean, pattern, prosody_features)
                adjusted_gap = self._adjust_gap_for_prosody(gap_ms, prosody_features, intent_type)

                return IntentResult(
                    intent_type=intent_type,
                    intent_subtype=subtype,
                    expected_gap_ms=adjusted_gap,
                    is_fpp=False,
                    expected_spp=expected_spp,
                    confidence=confidence
                )

        # PRIORITY 3: Check discourse markers
        for pattern, intent_type, subtype, gap_ms, expected_spp in self.discourse_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return IntentResult(
                    intent_type=intent_type,
                    intent_subtype=subtype,
                    expected_gap_ms=gap_ms,
                    is_fpp=False,
                    expected_spp=expected_spp,
                    confidence=0.7
                )

        # DEFAULT: Assume statement with moderate gap
        return IntentResult(
            intent_type='statement',
            intent_subtype='unknown',
            expected_gap_ms=350,
            is_fpp=False,
            expected_spp=None,
            confidence=0.3
        )

    def _calculate_confidence(self, text: str, pattern: str, prosody: Optional[Dict]) -> float:
        """Calculate classification confidence"""
        base_confidence = 0.8

        # Boost confidence if prosody supports intent
        if prosody:
            # Rising intonation + question pattern = high confidence
            if '?' in text and prosody.get('final_f0_slope', 0) > 0:
                base_confidence = 0.95
            # Falling intonation + statement pattern = high confidence
            elif '.' in text and prosody.get('final_f0_slope', 0) < 0:
                base_confidence = 0.9

        return min(base_confidence, 1.0)

    def _adjust_gap_for_prosody(self, base_gap: int, prosody: Optional[Dict], intent_type: str) -> int:
        """Adjust expected gap based on prosodic cues"""
        if not prosody:
            return base_gap

        adjusted_gap = base_gap

        # Strong rising intonation (question) → tighter gap
        if intent_type == 'question' and prosody.get('final_f0_slope', 0) > 5:
            adjusted_gap = int(base_gap * 0.8)  # 20% tighter

        # Falling intonation (statement) → looser gap acceptable
        elif intent_type == 'statement' and prosody.get('final_f0_slope', 0) < -5:
            adjusted_gap = int(base_gap * 1.2)  # 20% looser

        # Very short utterance → tighter gap (backchannel expected)
        if prosody.get('duration', 1000) < 300:  # < 300ms utterance
            adjusted_gap = int(adjusted_gap * 0.7)

        return max(50, min(adjusted_gap, 600))  # Clamp to 50-600ms

    def get_adjacency_pair_type(self, result: IntentResult) -> Optional[str]:
        """Get adjacency pair type for dialogue state tracking"""
        if not result.is_fpp:
            return None

        pair_map = {
            ('question', 'wh_question'): 'Q-A',
            ('question', 'yn_modal'): 'Q-YN',
            ('greeting', 'initial'): 'Greeting-Greeting',
            ('offer', 'help_offer'): 'Offer-Accept/Reject',
            ('request', 'polite_request'): 'Request-Comply/Refuse',
            ('apology', 'apologize'): 'Apology-Accept',
            ('closing', 'farewell'): 'Closing-Closing',
        }

        return pair_map.get((result.intent_type, result.intent_subtype))


def demo():
    """Demonstrate intent classifier"""
    print("=" * 80)
    print(" GERMAN INTENT CLASSIFIER - Research-Based")
    print("=" * 80)
    print()
    print("Based on:")
    print("  - Levinson (2016): Turn-taking timing (0-200ms)")
    print("  - Sacks et al. (1974): Adjacency pairs")
    print("  - DAMSL/ISO 24617-2: Dialogue act taxonomy")
    print()

    classifier = IntentClassifierGerman()

    test_cases = [
        # Adjacency Pair First Parts (tight timing)
        ("Wann möchten Sie anreisen?", "WH-Question"),
        ("Können Sie mir helfen?", "Y/N Question"),
        ("Guten Tag", "Greeting"),
        ("Möchten Sie noch etwas?", "Offer"),
        ("Könnten Sie das wiederholen?", "Polite Request"),
        ("Tut mir leid", "Apology"),
        ("Auf Wiedersehen", "Closing"),

        # Non-FPP (looser timing)
        ("Das Zimmer ist verfügbar.", "Statement"),
        ("Ja, genau", "Confirmation"),
        ("Nein, danke", "Denial"),
        ("Okay, verstanden", "Acknowledgment"),
        ("Vielen Dank", "Thanks"),

        # Discourse markers
        ("Äh, ich denke...", "Filler"),
        ("Übrigens, wissen Sie...", "Digression"),

        # Edge cases
        ("Haben Sie weitere Fragen?", "Tag-like question"),
        ("...oder?", "Tag question"),
    ]

    print("=" * 80)
    print("Test Cases:")
    print("=" * 80)
    print()

    for text, description in test_cases:
        result = classifier.classify(text)
        pair_type = classifier.get_adjacency_pair_type(result)

        print(f"Text: {text:<40}")
        print(f"  Expected: {description}")
        print(f"  Intent: {result.intent_type}/{result.intent_subtype}")
        print(f"  Expected Gap: {result.expected_gap_ms}ms")
        print(f"  FPP: {'Yes' if result.is_fpp else 'No'}")
        if pair_type:
            print(f"  Adjacency Pair: {pair_type}")
        print(f"  Confidence: {result.confidence:.1%}")
        print()

    # Prosody influence demo
    print("=" * 80)
    print("Prosody Influence on Timing:")
    print("=" * 80)
    print()

    text = "Können Sie mir helfen"

    # Without prosody
    result1 = classifier.classify(text)
    print(f"Without prosody: {result1.expected_gap_ms}ms")

    # With strong rising intonation (clear question)
    result2 = classifier.classify(text, {'final_f0_slope': 10, 'duration': 800})
    print(f"With rising intonation: {result2.expected_gap_ms}ms (tighter)")

    # With very short duration (backchannel)
    result3 = classifier.classify(text, {'duration': 200})
    print(f"Very short utterance: {result3.expected_gap_ms}ms (much tighter)")

    print()
    print("Intent Classifier ready for integration!")
    print()


if __name__ == "__main__":
    demo()
