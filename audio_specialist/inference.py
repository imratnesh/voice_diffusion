# Inference using diffusion
import argparse

import soundfile as sf
import librosa

def load_audio(path, sr=22050):
    print(f"Loading audio from {path} with sample rate {sr}")
    audio, _ = librosa.load(path, sr=sr)
    print(f"Loaded audio shape: {audio.shape}")
    return audio


def run_inference(input_audio, output_audio, speaker_id=0):
    print(f"Running inference with input_audio={input_audio}, output_audio={output_audio}, speaker_id={speaker_id}")
    # Load input audio
    audio = load_audio(input_audio)
    # Placeholder: here you would run your actual model inference
    print("Performing (placeholder) inference...")
    # For now, just save the input as output
    sf.write(output_audio, audio, 22050)
    print(f"Inference complete. Output saved to {output_audio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for RVC pipeline.")
    parser.add_argument('--input_audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output_audio', type=str, required=True, help='Path to output audio file')
    parser.add_argument('--speaker_id', type=int, default=0, help='Speaker ID')
    args = parser.parse_args()
    run_inference(args.input_audio, args.output_audio, args.speaker_id)
