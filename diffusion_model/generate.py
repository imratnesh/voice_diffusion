# Generate audio/voice using model
import os

from diffusers import AudioDiffusionPipeline
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from diffusion_model.config import DiffusionConfig


def generate_speech(prompt: str, output_path: str):
    # Replace with your actual model path or HuggingFace model repo
    model_path = "facebook/audiogen-medium"  # Example: official AudioGen model
    pipe = AudioDiffusionPipeline.from_pretrained(model_path)
    audio = pipe(prompt)
    audio.save(output_path)

def generate_audio(prompt: str, output_path: str):
    cfg = DiffusionConfig()
    print(f"[INFO] Loading model: {cfg.pretrained_model}")

    processor = AutoProcessor.from_pretrained(cfg.pretrained_model)
    model = MusicgenForConditionalGeneration.from_pretrained(cfg.pretrained_model)

    inputs = processor(text=prompt, return_tensors="pt")
    audio_values = model.generate(**inputs, do_sample=True, max_new_tokens=256)

    audio_array = audio_values[0].cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure sampling rate is within valid range for WAV files
    sampling_rate = min(cfg.sampling_rate, 48000)
    if sampling_rate > 65535:
        sampling_rate = 48000  # fallback to safe value

    # Ensure audio_array is in correct format (int16) and shape
    print(f"[DEBUG] audio_array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
    if audio_array.ndim > 2:
        audio_array = audio_array.squeeze()
        print(f"[DEBUG] Squeezed audio_array shape: {audio_array.shape}")
    if audio_array.ndim == 1:
        pass  # mono
    elif audio_array.ndim == 2 and audio_array.shape[0] <= 2:
        audio_array = audio_array.T  # (channels, samples) -> (samples, channels)
    elif audio_array.ndim == 2 and audio_array.shape[1] <= 2:
        pass  # already (samples, channels)
    else:
        raise ValueError(f"Unexpected audio_array shape: {audio_array.shape}")

    if audio_array.dtype != "int16":
        # Normalize and convert to int16
        audio_array = (audio_array / max(abs(audio_array).max(), 1e-8) * 32767).astype("int16")

    write(output_path, sampling_rate, audio_array)
    print(f"[SUCCESS] Audio generated at {output_path}")