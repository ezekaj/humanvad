"""Improve semantic completion detector with better patterns"""

with open('excellence_vad.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the completion_patterns section
old_patterns = """        # Common sentence-final patterns
        self.completion_patterns = [
            # Statements ending with period-like prosody
            r'\\b(yes|no|okay|sure|thanks|thank you|alright|fine)\\s*$',
            r'\\b(done|finished|complete|ready|good)\\s*$',

            # Complete grammatical structures
            r'\\bi (am|was|will be|would be|have been|can|could|should|will|would)\\s+\\w+\\s*$',
            r'\\byou (are|were|will be|can|could|should|have|had)\\s+\\w+\\s*$',
            r'\\b(it is|that is|this is|there is|here is)\\s+\\w+\\s*$',

            # Questions (complete)
            r'\\b(what|where|when|why|who|how)\\s+.*\\?$',
            r'\\b(is|are|was|were|will|would|can|could|should)\\s+\\w+.*\\?$',

            # Sentence-final structures
            r'\\s+(today|tomorrow|yesterday|now|then|there|here)\\s*$',
            r'\\s+(minute|hour|day|week|month|year)s?\\s*$',
        ]"""

new_patterns = """        # Common sentence-final patterns
        self.completion_patterns = [
            # Statements ending with period-like prosody
            r'\\b(yes|no|okay|sure|thanks|thank you|alright|fine|perfect|great)\\s*$',
            r'\\b(done|finished|complete|ready|good)\\s*$',

            # Complete grammatical structures
            r'\\bi (am|was|will be|would be|have been|can|could|should|will|would)\\s+\\w+\\s*$',
            r'\\byou (are|were|will be|can|could|should|have|had)\\s+\\w+\\s*$',
            r'\\b(it is|that is|this is|there is|here is)\\s+\\w+\\s*$',

            # Questions (complete)
            r'\\b(what|where|when|why|who|how)\\s+.*\\?$',
            r'\\b(is|are|was|were|will|would|can|could|should)\\s+\\w+.*\\?$',

            # Sentence-final structures (time/place)
            r'\\s+(today|tomorrow|yesterday|now|then|there|here)\\s*$',
            r'\\s+(tonight|morning|afternoon|evening|night)\\s*$',
            r'\\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\s*$',
            r'\\s+(minute|hour|day|week|month|year)s?\\s*$',

            # Complete sentences ending with action verbs + object
            r'\\b(check in|check out|book|arrive|leave|meet|schedule)\\s+.*$',

            # Complete statements (verb + prepositional phrase ending)
            r'\\s+(at the|in the|on the|from the|with the|to the)\\s+\\w+(\\s+\\w+)*\\s*$',

            # Default: Long sentences (>8 words) without incomplete markers tend to be complete
        ]"""

if old_patterns in content:
    content = content.replace(old_patterns, new_patterns)

    # Also improve the scoring logic
    old_scoring = """        # 5. Word count (very short utterances often complete)
        word_count = len(text.split())
        if word_count >= 3 and word_count <= 8:
            score = min(score + 0.1, 1.0)
            reason = "short_complete"

        return {
            'complete_prob': score,
            'reason': reason
        }"""

    new_scoring = """        # 5. Word count heuristics
        word_count = len(text.split())

        # Very short utterances (3-8 words) often complete if no incomplete markers
        if 3 <= word_count <= 8 and score >= 0.5:
            score = min(score + 0.1, 1.0)
            reason = "short_complete"

        # Longer sentences (>8 words) default to complete if no incomplete markers found
        elif word_count > 8 and score == 0.5:
            score = 0.7  # Likely complete
            reason = "long_sentence"

        return {
            'complete_prob': score,
            'reason': reason
        }"""

    content = content.replace(old_scoring, new_scoring)

    with open('excellence_vad.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("[OK] Improved semantic completion patterns")
    print("  - Added time/day patterns")
    print("  - Added action verb patterns")
    print("  - Added prepositional phrase endings")
    print("  - Improved long sentence detection (>8 words)")
else:
    print("[ERROR] Could not find patterns to replace")
