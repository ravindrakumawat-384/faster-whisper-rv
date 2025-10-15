import whisperx
from whisperx.diarize import DiarizationPipeline
import librosa
import soundfile as sf
import numpy as np
import warnings
from database import insert_with_uuid, diarization_collection
import os
hf_token = os.getenv("HF_TOKEN")
async def get_speaker_diarization_json(
    audio_file: str,
    device: str = "cpu",
    compute_type: str = "int8",
    min_speakers: int = 2,
    max_speakers: int = 6,
    # clustering_threshold: float = 0.65,
) -> list[dict]:
    """
    Perform speaker diarization and return segments with speaker, timing, and text in JSON format.
    """
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 1. Check if already processed
    existing_doc = diarization_collection.find_one({"file": audio_file})
    if existing_doc:
        return existing_doc
    
    # 2. Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    audio = librosa.util.normalize(librosa.effects.preemphasis(audio))
    audio = audio.astype("float32")
    duration = librosa.get_duration(y=audio, sr=sr)

    # 3. Transcribe
    model = whisperx.load_model("large-v2", device=device, compute_type=compute_type, language="en")
    result = model.transcribe(audio)
    
    # 4. Align transcript
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    t_result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # 5. Diarization
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=device
    )
    diarize_segments = diarize_model(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    # 6. Assign speaker labels
    finalResult = whisperx.assign_word_speakers(diarize_segments, t_result)

    speaker_order = []
    for seg in finalResult["segments"]:
        if seg["speaker"] not in speaker_order:
            speaker_order.append(seg["speaker"])

    # 2. Create a mapping from original speaker IDs to standardized IDs
    speaker_map = {orig: f"SPEAKER_{i:02d}" for i, orig in enumerate(speaker_order)}

    # 3. Apply mapping to all segments
    for seg in finalResult["segments"]:
        seg["speaker"] = speaker_map[seg["speaker"]]


    # 7. Prepare JSON output
    output = []
    speakers = []
    for segment in finalResult["segments"]:
        output.append({
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment['speaker'],
            "text": segment["text"]
        })
        if segment['speaker'] not in speakers:
            speakers.append(segment['speaker'])
    
    # 8. Store in MongoDB
    savedDoc = insert_with_uuid(diarization_collection, {
        "file": audio_file,
        "duration": duration,
        "speakers": len(speakers),
        "transcription": output
    })
    foundDoc = diarization_collection.find_one({"_id": savedDoc.inserted_id })
    if 'speakerMapping' in foundDoc:
        for segment in foundDoc["transcription"]:
            segment["speaker"] = foundDoc["speakerMapping"].get(segment["speaker"], segment["speaker"])

    return foundDoc