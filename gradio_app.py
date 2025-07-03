import gradio as gr

from diffusion_model.generate import generate_audio


def inference(prompt_text):
    output_path = "tests/test_outputs/gradio_generated_audio.wav"
    generate_audio(prompt_text, output_path)
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("# Audio Diffusion Generator\nEnter a prompt to generate audio.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="A soothing piano melody.")
    with gr.Row():
        generate_btn = gr.Button("Generate Audio")
    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio", type="filepath")
    generate_btn.click(fn=inference, inputs=prompt, outputs=audio_output)

demo.launch()

