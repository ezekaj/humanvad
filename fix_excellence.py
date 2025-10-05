"""Quick script to fix excellence_vad.py - move turn-end calculation before early return"""

with open('excellence_vad.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the section
old_section = """        # Track prosodic features
        energy = np.sqrt(np.mean(ai_frame ** 2))
        self.energy_history.append(energy)
        self.confidence_history.append(ai_result['confidence'])

        # If user not speaking, nothing to decide
        if not user_speaking:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(latency_ms)
            return {
                'action': 'continue',
                'user_speaking': False,
                'ai_speaking': ai_speaking,
                'latency_ms': latency_ms
            }

        # 2. Prosodic turn-end analysis (40-50% weight)
        prosody_result = self._analyze_prosody(ai_frame, ai_speaking)
        prosody_prob = prosody_result['prosody_turn_end_prob']

        # 3. Semantic turn-end analysis (50-60% weight)
        if ai_text and len(ai_text.strip()) > 0:
            semantic_result = self.semantic_detector.is_complete(ai_text)
            semantic_prob = semantic_result['complete_prob']
            semantic_reason = semantic_result['reason']
        else:
            # No text available - rely on prosody only
            semantic_prob = 0.5
            semantic_reason = "no_text"

        # 4. FUSION (matching human weighting)
        # Research shows: semantics 50-60%, prosody 40-50%
        final_turn_end_prob = (
            0.45 * prosody_prob +      # Prosodic cues
            0.55 * semantic_prob        # Semantic/syntactic completion
        )

        # 5. Decision logic"""

new_section = """        # Track prosodic features
        energy = np.sqrt(np.mean(ai_frame ** 2))
        self.energy_history.append(energy)
        self.confidence_history.append(ai_result['confidence'])

        # ALWAYS calculate turn-end probability (continuous monitoring)
        # 2. Prosodic turn-end analysis (40-50% weight)
        prosody_result = self._analyze_prosody(ai_frame, ai_speaking)
        prosody_prob = prosody_result['prosody_turn_end_prob']

        # 3. Semantic turn-end analysis (50-60% weight)
        if ai_text and len(ai_text.strip()) > 0:
            semantic_result = self.semantic_detector.is_complete(ai_text)
            semantic_prob = semantic_result['complete_prob']
            semantic_reason = semantic_result['reason']
        else:
            # No text available - rely on prosody only
            semantic_prob = 0.5
            semantic_reason = "no_text"

        # 4. FUSION (matching human weighting)
        # Research shows: semantics 50-60%, prosody 40-50%
        final_turn_end_prob = (
            0.45 * prosody_prob +      # Prosodic cues
            0.55 * semantic_prob        # Semantic/syntactic completion
        )

        # If user not speaking, return monitoring info
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
                'latency_ms': latency_ms
            }

        # 5. Decision logic"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('excellence_vad.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("[OK] Fixed excellence_vad.py - turn-end now calculated continuously")
else:
    print("[ERROR] Could not find section to replace")
