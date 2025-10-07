from faster_whisper import WhisperModel
import torch

# Choose model size
model_size = "small.en"

# Detect GPU availability
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"  # Recommended for GPU
    print("Running on GPU 🚀")
else:
    device = "cpu"
    compute_type = "int8"     # More efficient for CPU
    print("Running on CPU 🧠")

# Initialize model
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# Async function to transcribe audio
async def transcribe_audio(file_path: str) -> str:
    segments, info = model.transcribe(file_path, language="en", beam_size=5)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text
    return transcribed_text
