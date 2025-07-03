# tests/test_generate.py
import unittest
import os
from diffusion_model.generate import generate_audio

class TestDiffusionGenerate(unittest.TestCase):
    def test_generate_audio(self):
        prompt = "A soothing piano melody."
        output_path = "tests/test_outputs/generated_audio.wav"

        generate_audio(prompt, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith(".wav"))

        file_size = os.path.getsize(output_path)
        self.assertGreater(file_size, 1024, "Generated .wav file is unexpectedly small.")

if __name__ == "__main__":
    unittest.main()
