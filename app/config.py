import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHUNK_TARGET_WORDS = int(os.getenv("CHUNK_TARGET_WORDS", "200"))
