import gradio as gr
from faster_whisper import WhisperModel
import os
import tempfile

# Load model once at startup
print("Loading Whisper model (large-v3-turbo)...")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
print("Model loaded!")

def transcribe(audio_path, progress=gr.Progress()):
    if audio_path is None:
        return "No file uploaded.", None

    progress(0.1, desc="Starting transcription...")
    segments, info = model.transcribe(audio_path, beam_size=5, language="en")

    progress(0.3, desc=f"Detected: {info.language} ({info.language_probability:.0%}), Duration: {info.duration:.1f}s")

    lines = []
    for segment in segments:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        lines.append(line)
        progress(0.3 + 0.7 * (segment.end / info.duration), desc=f"Transcribing... {segment.end:.1f}s / {info.duration:.1f}s")

    transcript = "\n".join(lines)
    header = f"Language: {info.language} ({info.language_probability:.0%})\nDuration: {info.duration:.1f}s\nSegments: {len(lines)}\n{'='*50}\n\n"

    # Save to a temp file for download
    out_path = os.path.join(tempfile.gettempdir(), "transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + transcript)

    return header + transcript, out_path

# Build UI
with gr.Blocks(title="Whisper Transcription") as demo:
    gr.Markdown("# Whisper Transcription (large-v3-turbo)")
    gr.Markdown("Drag and drop an audio/video file, then click Transcribe.")

    with gr.Row():
        audio_input = gr.Audio(label="Upload Audio/Video", type="filepath", sources=["upload"])

    with gr.Row():
        transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

    with gr.Row():
        output_text = gr.Textbox(label="Transcript", lines=20)

    with gr.Row():
        download_btn = gr.File(label="Download Transcript")

    transcribe_btn.click(fn=transcribe, inputs=audio_input, outputs=[output_text, download_btn])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
