import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import webrtcvad
import re
import language_tool_python

# ---------------- SETTINGS ----------------
samplerate = 16000
block_duration = 2.5       # longer block to reduce repeated tokens
chunk_duration = 1.0       # bigger chunk for better context
channels = 1               # mono
frame_per_block = int(samplerate * block_duration)
frame_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

model = WhisperModel("medium.en", device="cuda", compute_type="float16")
vad = webrtcvad.Vad(2)  # 0-3, higher = more aggressive

tool = language_tool_python.LanguageTool('en')

# ---------------- FILTERS ----------------
banned_words = {"ass", "fuck", "shit"}
IGNORE_PHRASES = [
    "thank you",
    "thanks for watching",
    "thank you very much",
    "i'm so sorry",
    "oh",
    "alright",
    "i love you",
    "happy birthday",
    "so let's see",
    "okay",
    "i'm going to",
    "next video",
    "see you",
]

def should_ignore(text: str) -> bool:
    text_lc = text.lower().strip()
    for phrase in IGNORE_PHRASES:
        if phrase in text_lc:
            return True
    return False

# ---------------- VAD CHECK ----------------
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

    # Slide window with small overlap
    for i in range(0, len(int16_audio) - frame_size + 1, frame_size // 2):
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
    print("Listening... ctrl + C to stop")
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frame_per_block):
        while True:
            sd.sleep(100)

# ---------------- TEXT REFINEMENT ----------------
def refine_text_local(text: str) -> str:
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

# ---------------- TRANSCRIBER ----------------
def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frame_per_chunk:
            # concatenate chunk and flatten
            audio_data = np.concatenate(audio_buffer)[:frame_per_chunk].flatten().astype(np.float32)
            audio_buffer = []  # clear buffer

            # Only process if VAD detects speech
            if not is_speech(audio_data):
                continue

            # Whisper transcription
            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=5,                      # better accuracy for Indian English
                condition_on_previous_text=True,  # maintain context 
                suppress_blank=True
            )

            for segment in segments:
                raw_text = segment.text.strip()
                if not raw_text:
                    continue
                # Skip banned words
                if any(re.search(rf"\b{bw}\b", raw_text.lower()) for bw in banned_words):
                    continue
                # Skip ignored phrases
                if should_ignore(raw_text):
                    continue
                # Skip low confidence segments
                if hasattr(segment, "avg_log_prob") and segment.avg_log_prob < -0.3:
                    continue

                refined_text = refine_text_local(raw_text)
                print(refined_text)

# ---------------- START THREADS ----------------
threading.Thread(target=recorder, daemon=True).start()
transcriber()
