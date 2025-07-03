# Main voice conversion module
import subprocess
import sys
import os

def convert_voice(input_path: str, output_path: str, speaker_id: int = 0):
    """Convert input voice using RVC"""
    command = [
        "python3", os.path.join(os.path.dirname(__file__), "inference.py"),
        "--input_audio", input_path,
        "--output_audio", output_path,
        "--speaker_id", str(speaker_id),
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode == 0:
        print(f"Voice converted and saved to {output_path}")
    else:
        print(f"Voice conversion failed with return code {result.returncode}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python rvc_pipeline.py --input <input_path> --output <output_path> [--speaker_id <id>]")
        sys.exit(1)
    input_path = None
    output_path = None
    speaker_id = 0
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--input' and i + 1 < len(args):
            input_path = args[i + 1]
        elif arg == '--output' and i + 1 < len(args):
            output_path = args[i + 1]
        elif arg == '--speaker_id' and i + 1 < len(args):
            speaker_id = int(args[i + 1])
    print(f"Parsed arguments: input_path={input_path}, output_path={output_path}, speaker_id={speaker_id}")
    if not input_path or not output_path:
        print("Error: --input and --output arguments are required.")
        sys.exit(1)
    print("Starting voice conversion...")
    convert_voice(input_path, output_path, speaker_id)
    print("Voice conversion process finished.")

if __name__ == "__main__":
    main()
