import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import webrtcvad
from pyannote.audio import Pipeline
import re
import language_tool_python

# ---------------- SETTINGS ----------------
samplerate = 16000
block_duration = 1.0    # block size for audio capture
chunk_duration = 0.6    # chunk size for transcription
channels = 1            # mono audio
frame_per_block = int(samplerate * block_duration)
frame_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# ---------------- MODELS ----------------
model = WhisperModel("medium.en", device="cuda", compute_type="float16")
vad = webrtcvad.Vad(2)  # aggressiveness 0-3
tool = language_tool_python.LanguageTool('en')

# ---------------- BANNED / IGNORE ----------------
banned_words = {"ass", "fuck", "shit"}
IGNORE_PHRASES = [
    "thank you", "thanks for watching", "thank you very much",
    "i'm so sorry", "oh", "alright", "i love you",
    "happy birthday", "so let's see", "okay", "i'm going to",
    "next video", "see you"
]


def should_ignore(text: str) -> bool:
    text_lc = text.lower().strip()
    for phrase in IGNORE_PHRASES:
        if phrase in text_lc:
            return True
    return False

# ---------------- VAD ----------------
def is_speech(audio_chunk: np.ndarray) -> bool:
    if audio_chunk.size == 0:
        return False
    
    if audio_chunk.dtype != np.int16:
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
        int16_audio = (audio_chunk * 32767).astype(np.int16)
    else:
        int16_audio = audio_chunk

    frame_duration_ms = 30
    frame_size = int(samplerate * frame_duration_ms / 1000)

    for i in range(0, len(int16_audio) - frame_size + 1, frame_size):
        frame = int16_audio[i:i + frame_size]
        try:
            if vad.is_speech(frame.tobytes(), samplerate):
                return True
        except Exception:
            continue
    return False

# ---------------- AUDIO CALLBACK ----------------
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    print("Listening... Ctrl+C to stop")
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frame_per_block):
        while True:
            sd.sleep(100)

# ---------------- TEXT REFINEMENT ----------------
def refine_text_local(text: str) -> str:
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

# ---------------- SPEAKER DIARIZATION ----------------
# Fixing the revision issue by explicitly providing revision
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", # speaker-diarization@2.1
    revision="main"  # ensure the revision is passed
    )

# optional: ensure internal models also have revision
# diarization_pipeline.segmentation.model.revision = "main"
# diarization_pipeline.embedding.model.revision = "main"
# diarization_pipeline.clustering.model.revision = "main"

# ---------------- TRANSCRIBER ----------------
def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frame_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frame_per_chunk].flatten().astype(np.float32)
            audio_buffer = []

            if not is_speech(audio_data):
                continue

            # Whisper transcription
            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                condition_on_previous_text=False,
                suppress_blank=True
            )

            # Convert audio_data to temp WAV for pyannote
            import soundfile as sf
            sf.write("audio.mp3", audio_data, samplerate)

            # Speaker diarization
            diarization = diarization_pipeline({"uri": "chunk", "audio": "audio.mp3"})

            for segment in segments:
                raw_text = segment.text.strip()
                if not raw_text:
                    continue
                if any(re.search(rf"\b{bw}\b", raw_text.lower()) for bw in banned_words):
                    continue
                if should_ignore(raw_text):
                    continue
                if hasattr(segment, "avg_log_prob") and segment.avg_log_prob < -0.2:
                    continue

                refined_text = refine_text_local(raw_text)

                # Match diarization segment to ASR segment
                speaker_label = "Unknown"
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment.start <= turn.end:
                        speaker_label = speaker
                        break

                print(f"[{speaker_label}] {refined_text}")

# ---------------- START THREADS ----------------
threading.Thread(target=recorder, daemon=True).start()
transcriber()
