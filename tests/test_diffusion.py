# Test for diffusion model
import unittest
from unittest.mock import patch, MagicMock
import diffusion_model.generate as gen

class TestDiffusionGenerate(unittest.TestCase):

    @patch("diffusion_model.generate.AudioDiffusionPipeline")
    def test_generate_speech(self, mock_pipeline):
        # Mock the pipeline's return value and save method
        mock_instance = MagicMock()
        mock_audio = MagicMock()
        mock_instance.return_value = mock_audio
        mock_audio.save = MagicMock()

        mock_pipeline.from_pretrained.return_value = mock_instance

        prompt = "Hello world"
        output_path = "tests/generated.wav"

        gen.generate_speech(prompt, output_path)

        mock_pipeline.from_pretrained.assert_called_once()
        mock_instance.assert_called_once_with(prompt)
        mock_audio.save.assert_called_once_with(output_path)

if __name__ == "__main__":
    unittest.main()

