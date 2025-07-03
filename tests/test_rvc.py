# Test for RVC pipeline
import unittest
from unittest.mock import patch
import audio_specialist.rvc_pipeline as rvc


class TestRVCPipeline(unittest.TestCase):

    @patch("subprocess.run")
    def test_convert_voice(self, mock_run):
        input_path = "tests/sample_input.wav"
        output_path = "tests/sample_output.wav"
        speaker_id = 1

        rvc.convert_voice(input_path, output_path, speaker_id)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("inference_main.py", args)
        self.assertIn(input_path, args)
        self.assertIn(output_path, args)
        self.assertIn(str(speaker_id), args)


if __name__ == "__main__":
    unittest.main()
