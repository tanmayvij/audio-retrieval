from faster_whisper import WhisperModel

from app.config import WHISPER_COMPUTE, WHISPER_MODEL, WHISPER_DEVICE


def generate_segments(audio_file):
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE
    )
    
    segments, info = model.transcribe(audio_file, vad_filter=True, beam_size=5)
    
    return segments
