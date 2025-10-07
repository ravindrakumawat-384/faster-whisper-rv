from faster_whisper import WhisperModel

model_size = "small.en"

# Run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

async def transcribe_audio(file_path: str) -> str:
    segments, info = model.transcribe(file_path, language="en", beam_size=5)
    transcribed_text = ""
    for segment in segments:
        # transcribed_text += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcribed_text += segment.text
    return transcribed_text