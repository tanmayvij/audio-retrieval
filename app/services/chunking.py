from collections.abc import Iterable

from app.config import CHUNK_TARGET_WORDS


_OVERLAP_WORDS = 30
_PAUSE_BREAK_SECONDS = 2.5


def _word_count(text: str) -> int:
    return len(text.split())


def _overlap(segments: list[dict]) -> list[dict]:
    words = 0
    result: list[dict] = []

    for segment in reversed(segments):
        result.append(segment)
        words += segment["word_count"]
        if words >= _OVERLAP_WORDS:
            break

    return list(reversed(result))


def _make_chunk(segments: list[dict]) -> dict:
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(segment["text"] for segment in segments),
    }


def chunk_segments(segments: Iterable[object]) -> list[dict]:
    """Combine a Whisper segment iterable into timestamped text chunks.

    ``faster_whisper`` returns a generator, so this function consumes it once.
    Each output chunk contains its inclusive start time, end time, and text.
    Chunks end at Whisper's existing segment boundaries after reaching
    ``CHUNK_TARGET_WORDS``; a long pause also starts a new chunk. The final
    Whisper segment(s) of a chunk are repeated as a small contextual overlap.
    """
    chunks: list[dict] = []
    current: list[dict] = []
    current_words = 0
    previous_end: float | None = None

    for whisper_segment in segments:
        text = whisper_segment.text.strip()
        if not text:
            continue

        segment = {
            "start": whisper_segment.start,
            "end": whisper_segment.end,
            "text": text,
            "word_count": _word_count(text),
        }

        has_pause = (
            previous_end is not None
            and segment["start"] - previous_end >= _PAUSE_BREAK_SECONDS
        )
        if current and (current_words >= CHUNK_TARGET_WORDS or has_pause):
            chunks.append(_make_chunk(current))
            if has_pause:
                current = []
                current_words = 0
            else:
                current = _overlap(current)
                current_words = sum(item["word_count"] for item in current)

        current.append(segment)
        current_words += segment["word_count"]
        previous_end = segment["end"]

    if current:
        chunks.append(_make_chunk(current))

    return chunks
