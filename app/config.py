import os

DATABASE_URL = os.getenv("DATABASE_URL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
DIARIZATION_MODEL = os.getenv(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)
DIARIZATION_DEVICE = os.getenv("DIARIZATION_DEVICE", "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHUNK_TARGET_WORDS = int(os.getenv("CHUNK_TARGET_WORDS", "200"))
