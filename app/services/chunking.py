from collections.abc import Iterable

from app.config import CHUNK_TARGET_WORDS
from app.services.diarization import SpeakerTurn


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
        "speaker": segments[0]["speaker"],
    }


def _overlap_duration(
    start: float, end: float, speaker_turn: SpeakerTurn
) -> float:
    return max(0.0, min(end, speaker_turn.end) - max(start, speaker_turn.start))


def _distance(start: float, end: float, speaker_turn: SpeakerTurn) -> float:
    if end < speaker_turn.start:
        return speaker_turn.start - end
    if start > speaker_turn.end:
        return start - speaker_turn.end
    return 0.0


def _speaker_for_word(
    start: float, end: float, speaker_turns: list[SpeakerTurn]
) -> str:
    overlaps = [
        _overlap_duration(start, end, speaker_turn)
        for speaker_turn in speaker_turns
    ]
    largest_overlap = max(overlaps)
    if largest_overlap > 0:
        return speaker_turns[overlaps.index(largest_overlap)].speaker

    nearest_turn = min(
        speaker_turns,
        key=lambda speaker_turn: (
            _distance(start, end, speaker_turn),
            speaker_turn.start,
        ),
    )
    return nearest_turn.speaker


def _speaker_words(
    segments: Iterable[object], speaker_turns: list[SpeakerTurn]
) -> Iterable[dict]:
    for whisper_segment in segments:
        words = whisper_segment.words
        if words is None:
            raise ValueError("Whisper segments must include word timestamps.")

        for whisper_word in words:
            text = whisper_word.word.strip()
            if not text:
                continue

            start = float(whisper_word.start)
            end = float(whisper_word.end)
            yield {
                "start": start,
                "end": end,
                "text": text,
                "word_count": _word_count(text),
                "speaker": _speaker_for_word(start, end, speaker_turns),
            }


def chunk_segments(
    segments: Iterable[object], speaker_turns: Iterable[SpeakerTurn]
) -> list[dict]:
    """Combine timestamped Whisper words into speaker-aware text chunks.

    ``faster_whisper`` returns a generator, so this function consumes it once.
    Each output chunk contains one speaker, its start and end time, and text.
    Speaker changes and long pauses always start a new chunk. Chunks also end
    after reaching ``CHUNK_TARGET_WORDS``; only same-speaker words are repeated
    as a small contextual overlap after a size-based split.
    """
    normalized_turns = sorted(
        speaker_turns, key=lambda turn: (turn.start, turn.end)
    )
    if not normalized_turns:
        raise ValueError("At least one speaker turn is required.")

    chunks: list[dict] = []
    current: list[dict] = []
    current_words = 0
    previous_end: float | None = None

    for segment in _speaker_words(segments, normalized_turns):
        has_pause = (
            previous_end is not None
            and segment["start"] - previous_end >= _PAUSE_BREAK_SECONDS
        )
        has_speaker_change = (
            bool(current) and segment["speaker"] != current[-1]["speaker"]
        )
        reached_target = current_words >= CHUNK_TARGET_WORDS
        if current and (reached_target or has_pause or has_speaker_change):
            chunks.append(_make_chunk(current))
            if has_pause or has_speaker_change:
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
