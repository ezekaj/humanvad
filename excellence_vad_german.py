"""
Excellence VAD - Deutsche Version (German Language Support)
============================================================

Hybrid Prosody + Semantic Turn-Taking Detection für Deutsch.

Angepasst für deutsche Sprachmuster und Satzstrukturen.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional
import time
from production_vad import ProductionVAD
from disfluency_detector import DisfluencyDetector
import re


class SemanticCompletionDetectorGerman:
    """
    Erkennt semantische/syntaktische Vollständigkeit für DEUTSCHE Sprache

    German-specific patterns for sentence completion detection
    """

    def __init__(self):
        # Vollständige Sätze (Complete sentences) - DEUTSCH
        self.completion_patterns = [
            # Bestätigungen und kurze Antworten (highest priority - check first)
            r'^(ja|nein|okay|gut|sicher|danke|bitte|gerne|perfekt|super)[\s\.,!?]*$',
            r'^(fertig|erledigt|klar|verstanden|genau)[\s\.,!?]*$',

            # Greetings and closings
            r'\bvielen dank\b',
            r'\b(hallo|guten tag|guten morgen|guten abend|auf wiedersehen|auf wiederhören|tschüss)\b',
            r'\b(schönen|guten)\s+(tag|abend|morgen)\s+(noch|wünsche)\b',  # "Schönen Tag noch"
            r'\bbis\s+(bald|später|morgen|gleich)\b',  # "Bis bald"
            r'\bguten appetit\b',  # Restaurant specific

            # Complete sentences with period (BUT NOT with ellipsis "...")
            r'^[^.]{10,}\.$',  # At least 10 chars before period, NO ellipsis

            # === NEW: Complete sentences WITHOUT period ===

            # Common complete sentence structures (Verb + Object/Complement at end)
            r'\b(ist|sind|war|waren|hat|haben|wird|wurde)\s+\w+\s*$',  # "ist korrekt", "hat Zimmer"
            r'\b(kann|könnte|soll|sollte|muss|musste)\s+\w+\s*$',  # "kann helfen"

            # Subject + Verb + Object (complete declarative sentences)
            r'^(ich|er|sie|es|wir|ihr|sie)\s+\w+\s+\w+(\s+\w+)*\s*$',  # "Ich helfe Ihnen"

            # Question patterns (W-questions that are complete)
            r'^(haben|können|möchten|wollen)\s+sie\s+\w+\w+(\s+\w+)+\s*$',  # "Haben Sie weitere Fragen" (at least 2 words after "Sie")
            r'^(möchten|wollen|können)\s+sie\s+\w+\s*$',  # "Möchten Sie buchen" (single verb after Sie is complete)
            r'^wie\s+kann\s+ich\s+\w+(\s+\w+)*\s*$',  # "Wie kann ich helfen"
            r'^was\s+(ist|sind|bedeutet)\s+\w+(\s+\w+)*\s*$',  # "Was ist das"

            # Polite phrases and helping expressions
            r'\bich\s+helfe\s+\w+\s*$',  # "ich helfe Ihnen"
            r'\bgerne\b.*\w+\s*$',  # "sehr gerne, ich helfe"
            r'\b(sehr|ganz)\s+gerne\s*$',  # "sehr gerne"

            # Numbers and quantities (complete info)
            r'\d+\s+(zimmer|euro|personen|tage|stunden|minuten)\s*$',
            r'\b(hat|haben|kosten|kostet)\s+\d+',  # "hat 50", "kostet 200"

            # Price/quantity expressions
            r'\d+\s+euro\s*$',  # "zweihundert Euro"
            r'\bbeträgt\s+\w+\s+euro\s*$',  # "beträgt zweihundert Euro"

            # Specific complete phrases
            r'\b(anruf|gespräch|nachricht)\s*\.?\s*$',  # "für Ihren Anruf"
            r'\bfür\s+(ihren|deinen|seinen)\s+(anruf|besuch)\s*$',

            # Präpositionale Phrasen am Ende (complete) - like "zur Schule"
            r'\s+(am|im|auf dem|vom|mit dem)\s+\w+(\s+\w+)*\s*$',
            r'\s+(zur|in der|auf der|von der|mit der)\s+\w+(\s+\w+)*\s*$',

            # Zeitangaben am Satzende
            r'\s+(heute|morgen|gestern|jetzt|dann|dort|hier)\s*$',
            r'\s+(abends|morgens|nachmittags|nachts)\s*$',
            r'\s+(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\s*$',

            # Nebensätze beendet (past participles at end)
            r'\s+(gemacht|gesagt|gedacht|gegangen|gekommen|gewesen)\s*$',

            # Adjectives/Nouns at end (complete descriptions)
            r'\s+(korrekt|richtig|falsch|gut|schlecht|wichtig|möglich)\s*$',  # "ist korrekt"
            r'\s+(fragen|zimmer|euro|personen|tage|wochen)\s*$',  # "weitere Fragen"
            r'\s+(lager|vorrat|stock|verfügung|bereit)\s*$',  # "auf Lager", "ist bereit"
            r'\s+(perfekt|unauffällig|pünktlich|täglich|stündlich)\s*$',  # Medical/time adverbs
            r'\s+(sofort|gleich|direkt|jetzt|bald)\s*$',  # Immediacy adverbs
            r'\s+(vegetarisch|vegan|glutenfrei|laktosefrei)\s*$',  # Food/dietary
            r'\s+(gesperrt|gedeckt|gefunden|zugestellt|angenommen)\s*$',  # Past participles (complete)

            # Time expressions (complete)
            r'\bum\s+\w+\s+uhr\s*$',  # "um vierzehn Uhr"
            r'\b(morgens|abends|mittags|nachts|täglich|stündlich|wöchentlich)\s*$',

            # Helping/service phrases
            r'\bhelfen\s*$',  # "kann ich helfen", "noch helfen"
            r'\bich\s+ihnen\s+(noch\s+)?helfen\s*$',

            # Price/cost with "pro"
            r'\bpro\s+(nacht|tag|woche|monat|person)\s*$',  # "Euro pro Nacht"
            r'\bprozent\s*$',  # "zwei Prozent"

            # Imperative verbs (commands - complete)
            r'^(nehmen|kommen|steigen|rufen|lassen)\s+sie\s+\w+\w+(\s+\w+){2,}\s*$',  # "Nehmen Sie die Tabletten täglich" (at least 3 words after "Sie")
            r'^steigen\s+sie\s+am\s+\w+(\s+\w+)*\s+um\s*$',  # "Steigen Sie am Hauptbahnhof um"
            r'^\w+\s+sie\s+(am|um|an|auf|in)\s+\w+\s+\w+(\s+\w+)*\s*$',  # "... Sie am Ort [action]"

            # Subordinate clauses that ARE complete (with comma + main clause)
            r'^(falls|wenn|obwohl|nachdem)\s+.+,\s+\w+\s+\w+\s+\w+\s*$',  # "Falls Sie Fragen haben, rufen Sie an" (at least 3 words after comma)
            r'^falls\s+sie\s+\w+\s+\w+,\s+\w+\s+sie\s+\w+\s*$',  # "Falls Sie Fragen haben, rufen Sie an" (specific pattern)

            # Location/Place complete patterns
            r'\b(ist|liegt|befindet sich)\s+[A-Z]\s+\w+\s*$',  # "ist B zwölf" (gate numbers)
            r'\bam\s+\w+\s+um\s*$',  # "am Hauptbahnhof um"

            # Duration patterns
            r'\bdauert\s+\w+\s+(minuten|stunden|tage)\s*$',  # "dauert zwanzig Minuten"

            # Action complete patterns
            r'\bbringe\s+(ich|wir)\s+\w+\s*$',  # "bringe ich sofort"
            r'\bpasst\s+\w+\s*$',  # "passt perfekt"
        ]

        # Unvollständige Muster (Incomplete patterns) - DEUTSCH
        self.incomplete_patterns = [
            # Ellipsis and hesitation markers (HIGHEST PRIORITY)
            r'\.\.\.',  # Any ellipsis indicates incomplete
            r'\b(äh|ähm|ehm|hm|hmm|ach|nun)\b',  # Filler words anywhere

            # Subordinating conjunctions at end (sentence incomplete)
            r'\b(dass|das)\s*$',  # "sagen dass" = incomplete
            r'\b(und|aber|oder|weil|da|obwohl|wenn|falls|während|als)\s*$',
            r'\b(also|deshalb|jedoch|zudem|außerdem)\s*$',

            # Präpositionen (unvollständige Phrasen)
            r'\b(zum|zur|ins|ans|aufs)\s*$',  # Contracted prepositions alone
            r'\b(in|auf|an|zu|von|mit|bei|für|über|um)\s*$',
            r'\b(durch|gegen|ohne|hinter|vor|neben|zwischen)\s*$',

            # Hilfsverben (erwarte Hauptverb)
            r'\b(bin|ist|sind|war|waren|wird|würde|kann|könnte|soll|sollte|muss|musste)\s*$',
            r'\b(habe|hat|haben|hatte|hatten)\s*$',

            # Artikel (erwarte Nomen)
            r'\b(der|die|das|den|dem|des|ein|eine|einem|einer)\s*$',

            # Possessivpronomen (incomplete - need noun)
            r'\b(mein|dein|sein|ihr|unser|euer|meine|deine|seine|ihre|ihren|seinen|unseren)\s*$',

            # "ist/sind am" without following word (incomplete - expect location/time)
            r'\b(ist|sind|war|waren)\s+am\s*$',

            # Unvollständige Phrasen
            r'\bich (möchte|will|muss|soll)\s*$',
            r'\blass mich\s*$',
            r'\bkönnen sie\s*$',
        ]

        # Text-Puffer
        self.text_buffer = deque(maxlen=50)

    def is_complete(self, text: str) -> Dict:
        """
        Prüft ob deutscher Text semantisch/syntaktisch vollständig ist

        Returns:
            complete_prob: 0.0-1.0 Wahrscheinlichkeit der Vollständigkeit
            reason: Grund für die Einschätzung
        """

        if not text or len(text.strip()) == 0:
            return {'complete_prob': 0.0, 'reason': 'leer'}

        text = text.strip().lower()

        # In Puffer speichern
        self.text_buffer.append(text)

        score = 0.5  # Standard: unsicher
        reason = "unbekannt"

        # 1. Prüfe UNVOLLSTÄNDIGE Muster ZUERST (höchste Priorität)
        # Check CURRENT text only (not buffer) for incomplete patterns
        for pattern in self.incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score = 0.2  # STARK unvollständig
                reason = f"unvollständiges_muster: {pattern[:30]}"
                # FRÜHER RETURN - keine weiteren Prüfungen
                return {'complete_prob': score, 'reason': reason}

        # 2. Prüfe VOLLSTÄNDIGE Muster
        # Check CURRENT text only (not buffer) for completion patterns
        for pattern in self.completion_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score = 0.9  # STARK vollständig
                reason = f"vollständiges_muster: {pattern[:30]}"
                break

        # 3. Fragezeichen = vollständig
        if '?' in text:
            score = 0.9
            reason = "fragezeichen"

        # 4. Füllwörter am Ende = unvollständig
        if re.search(r'\b(äh|ähm|hm|hmm|ach|nun)\s*$', text):
            score = 0.1
            reason = "füllwort_am_ende"

        # 5. Wortanzahl Heuristik (NUR wenn noch unsicher)
        if score == 0.5:
            word_count = len(text.split())

            # Sehr kurze Äußerungen (1-2 Wörter)
            if word_count <= 2:
                score = 0.8  # Wahrscheinlich vollständig (ja, nein, okay)
                reason = "sehr_kurz"

            # Kurze Äußerungen (3-8 Wörter)
            elif 3 <= word_count <= 8:
                score = 0.6
                reason = "kurz_neutral"

            # Längere Sätze (>8 Wörter)
            elif word_count > 8:
                score = 0.7
                reason = "langer_satz"

        return {
            'complete_prob': score,
            'reason': reason
        }


class ExcellenceVADGerman:
    """
    Excellence VAD für DEUTSCHE Sprache

    Kombiniert:
    1. Prosodie-Analyse (Tonhöhe, Timing, Energie) - 45%
    2. Deutsche Semantik-Erkennung - 55%

    Ziel: 90-95% Genauigkeit für deutsche Telefongespräche
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        turn_end_threshold: float = 0.60,  # Tuned for current semantic detector
        use_disfluency_detection: bool = False,  # Feature #5
        use_adaptive_threshold: bool = False,  # Feature #7
    ):
        self.sr = sample_rate
        self.turn_end_threshold = turn_end_threshold
        self.use_disfluency_detection = use_disfluency_detection
        self.use_adaptive_threshold = use_adaptive_threshold

        # Prosodie-Erkennung (sprachunabhängig)
        self.prosody_detector = ProductionVAD(sample_rate=sample_rate)

        # DEUTSCHE Semantik-Erkennung
        self.semantic_detector = SemanticCompletionDetectorGerman()

        # Feature #5: Disfluency Detection
        if self.use_disfluency_detection:
            from disfluency_detector import DisfluencyDetector
            self.disfluency_detector = DisfluencyDetector()

        # Feature #7: Adaptive Thresholds
        if self.use_adaptive_threshold:
            from adaptive_threshold_manager import AdaptiveThresholdManager
            self.threshold_manager = AdaptiveThresholdManager(base_threshold=turn_end_threshold)

        # Audio-Puffer
        self.user_buffer = deque(maxlen=sample_rate * 3)
        self.ai_buffer = deque(maxlen=sample_rate * 3)

        # Prosodische Features
        self.energy_history = deque(maxlen=150)
        self.confidence_history = deque(maxlen=150)

        # Performance-Tracking
        self.processing_times = deque(maxlen=100)

    def _analyze_prosody(self, audio: np.ndarray, is_speech: bool) -> Dict:
        """
        Prosodie-Analyse (sprachunabhängig)

        Returns:
            prosody_turn_end_prob: 0.0-1.0
        """

        if len(self.energy_history) < 20:
            return {'prosody_turn_end_prob': 0.3}

        recent_energy = list(self.energy_history)[-30:]

        score = 0.0

        # 1. Energie-Trend (fallend = Ende)
        if len(recent_energy) >= 20:
            first_half = np.mean(recent_energy[:10])
            second_half = np.mean(recent_energy[-10:])

            if second_half < first_half * 0.7:
                score += 0.4
            elif second_half < first_half * 0.85:
                score += 0.2

        # 2. Absolute Energie (sehr niedrig = Ende)
        if len(recent_energy) > 0:
            current = recent_energy[-1]
            if current < 0.01:
                score += 0.3
            elif current < 0.05:
                score += 0.15

        # 3. Pause nach Sprache
        if not is_speech and len(self.energy_history) >= 10:
            recent_had_speech = any(e > 0.05 for e in list(self.energy_history)[-20:-5])
            if recent_had_speech:
                score += 0.3

        return {
            'prosody_turn_end_prob': min(score, 1.0)
        }

    def process_frame(
        self,
        user_frame: np.ndarray,
        ai_frame: np.ndarray,
        ai_text: Optional[str] = None
    ) -> Dict:
        """
        Verarbeite Stereo-Audio + deutschen AI-Text

        Args:
            user_frame: Nutzer-Audio (10ms, 160 Samples bei 16kHz)
            ai_frame: AI-Audio (10ms, 160 Samples bei 16kHz)
            ai_text: Aktueller deutscher AI-Text (optional)
        """

        start_time = time.perf_counter()

        # Puffer aktualisieren
        self.user_buffer.extend(user_frame)
        self.ai_buffer.extend(ai_frame)

        # 1. Schnelle Sprach-Erkennung
        user_result = self.prosody_detector.detect_frame(user_frame)
        ai_result = self.prosody_detector.detect_frame(ai_frame)

        user_speaking = user_result['is_speech']
        ai_speaking = ai_result['is_speech']

        # Prosodie-Features tracken
        energy = np.sqrt(np.mean(ai_frame ** 2))
        self.energy_history.append(energy)
        self.confidence_history.append(ai_result['confidence'])

        # 2. Prosodie-Analyse (45%)
        prosody_result = self._analyze_prosody(ai_frame, ai_speaking)
        prosody_prob = prosody_result['prosody_turn_end_prob']

        # 3. DEUTSCHE Semantik-Analyse (55%)
        if ai_text and len(ai_text.strip()) > 0:
            semantic_result = self.semantic_detector.is_complete(ai_text)
            semantic_prob = semantic_result['complete_prob']
            semantic_reason = semantic_result['reason']
        else:
            semantic_prob = 0.5
            semantic_reason = "kein_text"

        # 3.5. Feature #5: Disfluency Detection (Adjust semantic_prob)
        hesitation_result = None
        if self.use_disfluency_detection and ai_text:
            # Detect hesitations (fillers, pauses, repairs)
            # TODO: Track pause duration from silence frames
            pause_ms = 0  # Placeholder - would need silence frame tracking
            hesitation_result = self.disfluency_detector.detect_hesitation(
                text=ai_text,
                pause_ms=pause_ms
            )

            # Adjust semantic probability based on hesitation
            semantic_prob = self.disfluency_detector.adjust_completion_probability(
                semantic_prob=semantic_prob,
                hesitation_prob=hesitation_result['hesitation_prob']
            )

        # 4. FUSION: 45% prosody + 55% semantic (adjusted)
        final_turn_end_prob = (
            0.45 * prosody_prob +
            0.55 * semantic_prob
        )

        # 5. Feature #7: Get adaptive threshold if enabled
        if self.use_adaptive_threshold:
            current_threshold = self.threshold_manager.get_threshold(
                prosody_confidence=ai_result['confidence'],
                semantic_prob=semantic_prob
            )
        else:
            current_threshold = self.turn_end_threshold

        # Wenn Nutzer nicht spricht, Monitoring-Info zurückgeben
        if not user_speaking:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(latency_ms)
            return {
                'action': 'continue',
                'user_speaking': False,
                'ai_speaking': ai_speaking,
                'turn_end_prob': final_turn_end_prob,
                'prosody_prob': prosody_prob,
                'semantic_prob': semantic_prob,
                'semantic_reason': semantic_reason if ai_text else None,
                'latency_ms': latency_ms
            }

        # 6. Entscheidungs-Logik
        if user_speaking and ai_speaking:
            # ÜBERLAPPUNG: Nutzer spricht während AI spricht
            if final_turn_end_prob >= current_threshold:
                # AI-Äußerung ist vollständig - Turn endet - ALLOW interrupt
                action = "interrupt_ai_immediately"
                reasoning = "natürliche_übernahme"
            else:
                # AI mitten im Satz - Turn continues - DON'T interrupt
                action = "wait_for_ai_completion"
                reasoning = "unterbrechung"
        elif user_speaking:
            # Nutzer spricht, AI schweigt
            action = "interrupt_ai_immediately"
            reasoning = "ai_schweigt"
        else:
            action = "continue"
            reasoning = "nutzer_schweigt"

        # Latenz tracken
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        result = {
            'action': action,
            'reasoning': reasoning,
            'turn_end_prob': final_turn_end_prob,
            'user_speaking': user_speaking,
            'ai_speaking': ai_speaking,
            'overlap': user_speaking and ai_speaking,

            # Debug-Info
            'prosody_prob': prosody_prob,
            'semantic_prob': semantic_prob,
            'semantic_reason': semantic_reason if ai_text else None,
            'ai_text': ai_text,

            # Performance
            'latency_ms': latency_ms
        }

        # Add disfluency detection info if enabled
        if self.use_disfluency_detection and hesitation_result:
            result['hesitation_prob'] = hesitation_result['hesitation_prob']
            result['detected_fillers'] = hesitation_result['detected_fillers']
            result['pause_detected'] = hesitation_result['pause_detected']

        # Add adaptive threshold info if enabled
        if self.use_adaptive_threshold:
            result['adaptive_threshold'] = current_threshold
            result['threshold_adjustment'] = current_threshold - self.turn_end_threshold

        return result

    def get_stats(self) -> Dict:
        """Performance-Statistiken"""
        if len(self.processing_times) > 0:
            return {
                'avg_latency_ms': np.mean(self.processing_times),
                'p50_latency_ms': np.percentile(self.processing_times, 50),
                'p95_latency_ms': np.percentile(self.processing_times, 95),
                'max_latency_ms': np.max(self.processing_times)
            }
        return {}

    def reset(self):
        """Zustand zurücksetzen"""
        self.user_buffer.clear()
        self.ai_buffer.clear()
        self.energy_history.clear()
        self.confidence_history.clear()
        self.semantic_detector.text_buffer.clear()


def demo():
    """Demo für deutsche Sprache"""
    print("=" * 80)
    print(" EXCELLENCE VAD - DEUTSCHE VERSION")
    print("=" * 80)
    print()
    print("Turn-Taking-Erkennung für deutsche Telefongespräche:")
    print()
    print("Komponenten:")
    print("  1. Prosodie-Analyse (Energie, Timing, Tonhöhe) - 45%")
    print("  2. Deutsche Semantik-Erkennung - 55%")
    print()
    print("Ziel: 90-95% Genauigkeit (menschliches Niveau)")
    print()

    sr = 16000
    vad = ExcellenceVADGerman(sample_rate=sr)

    # Geschwindigkeitstest
    print("Geschwindigkeitstest:")
    print("-" * 80)

    test_frame = np.random.randn(160) * 0.1
    test_text = "Ich denke wir sollten das morgen besprechen"

    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.process_frame(test_frame, test_frame, test_text)
    avg_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Durchschnittliche Latenz: {avg_time:.2f}ms pro Frame")
    print(f"Ziel: <10ms {'BESTANDEN' if avg_time < 10 else 'FEHLER'}")
    print()

    stats = vad.get_stats()
    print(f"P50: {stats['p50_latency_ms']:.2f}ms")
    print(f"P95: {stats['p95_latency_ms']:.2f}ms")
    print()

    # Beispiel deutsche Sätze
    print("Beispiele deutsche Semantik-Erkennung:")
    print("-" * 80)

    examples = [
        ("Ich gehe zum Laden", "Vollständig"),
        ("Ich gehe zum", "Unvollständig (Präposition)"),
        ("Wie spät ist es?", "Vollständig (Frage)"),
        ("Weil ich denke dass", "Unvollständig (Konjunktion)"),
        ("Okay danke", "Vollständig (Bestätigung)"),
        ("Lass mich", "Unvollständig (Hilfsverb)"),
        ("Ich treffe Sie morgen am Bahnhof", "Vollständig (mit Zeitangabe)"),
    ]

    detector = SemanticCompletionDetectorGerman()
    for text, expected in examples:
        result = detector.is_complete(text)
        status = "VOLLSTÄNDIG" if result['complete_prob'] > 0.6 else "UNVOLLSTÄNDIG"
        print(f"{text:<45} -> {status:<15} ({result['complete_prob']:.1%}) | {result['reason']}")

    print()
    print("Führen Sie test_excellence_german.py für vollständige Tests aus")
    print()


if __name__ == "__main__":
    demo()
