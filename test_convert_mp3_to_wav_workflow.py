"""Regression tests for the MP3 to WAV conversion workflow."""

import os
import unittest

from convert_mp3_to_wav import convert_mp3_to_wav


class ConvertMP3ToWavWorkflowTest(unittest.TestCase):
    """Ensure the conversion helpers integrate with the voice AI stack."""

    def test_validate_existing_wavs_runs_voice_ai_check(self) -> None:
        wav_files = convert_mp3_to_wav(
            input_dir="test_audio_mp3",
            output_dir="test_audio",
            validate_only=True,
            run_vad_check=True,
        )

        self.assertGreater(len(wav_files), 0, "Expected WAV files for validation")
        for wav_file in wav_files:
            self.assertTrue(
                wav_file.endswith(".wav"),
                f"Unexpected file returned from validation: {wav_file}",
            )
            self.assertTrue(
                os.path.exists(os.path.join("test_audio", wav_file)),
                f"Missing WAV file in output directory: {wav_file}",
            )


if __name__ == "__main__":
    unittest.main()
