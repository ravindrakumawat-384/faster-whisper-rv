from vosk import Model, KaldiRecognizer
import sounddevice as sd
# import soundfile as sf

import queue
import json

# Settings
samplerate = 16000

# Load model
model = Model("vosk-model-small-en-in-0.4")
rec = KaldiRecognizer(model, samplerate)

# Thread-safe audio queue
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("Audio Status:", status)
    q.put(bytes(indata))

# Start microphone stream
with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                       dtype="int16", channels=1,
                       device=1,
                       callback=callback):

    print("Listening... Speak into the microphone (Ctrl+C to stop)")
    while True:
        print("While = true")
        data = q.get()
        # print("data",data)
        if rec.AcceptWaveform(data):
            print("if rec")
            result = json.loads(rec.Result())
            print("result--> ", result)
            print("result[text]--> ", result["text"])
            if result["text"].strip():  # skip empty results
                print("if result")
                print("Recognized:", result["text"])
        else:
            print("else..")
            partial = json.loads(rec.PartialResult())
            print("partial--> ",partial)
            if partial["partial"].strip():  # optional: show partials
                print("partial..")
                print("Partial:", partial["partial"])












import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import webrtcvad
import re
# from openai import OpenAI
# client = OpenAI()

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

#settings
samplerate = 16000
block_duration = 0.5 # seconds
chunk_duration = 2 
channels = 1

frame_per_block = int(samplerate * block_duration)
frame_per_chunk = int(samplerate * chunk_duration)
audio_queue = queue.Queue()
audio_buffer = []

model = WhisperModel("small.en", device="cuda", compute_type="float16")

vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3 (higher = more strict)


def audio_callback(indate, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indate.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback, blocksize=frame_per_block):
        print("listening... ctrl + C to stop.. ")
        while True:
            sd.sleep(100)


threading.Thread(target=recorder, daemon=True).start()
transcriber()