"""Speaker diarization for audio recordings."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import DIARIZATION_DEVICE, DIARIZATION_MODEL, HF_TOKEN


_NUM_SPEAKERS = 2


@dataclass(frozen=True)
class SpeakerTurn:
    """A continuous interval attributed to one recording-local speaker."""

    start: float
    end: float
    speaker: str


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN must be configured for the protected pyannote model."
        )

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
    if pipeline is None:
        raise RuntimeError(f"Unable to load diarization model: {DIARIZATION_MODEL}")

    pipeline.to(torch.device(DIARIZATION_DEVICE))
    return pipeline


def diarize(audio_file: str) -> list[SpeakerTurn]:
    """Return a normalized, exclusive two-speaker timeline for an audio file."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_file)
    output = _get_pipeline()(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=_NUM_SPEAKERS,
    )
    annotation = output.exclusive_speaker_diarization

    raw_turns = sorted(
        (
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not raw_turns:
        raise RuntimeError(f"Diarization produced no speaker turns for {audio_file}")

    detected_speakers = {speaker for _, _, speaker in raw_turns}
    if len(detected_speakers) != _NUM_SPEAKERS:
        raise RuntimeError(
            f"Expected {_NUM_SPEAKERS} speakers in {audio_file}, "
            f"but diarization produced {len(detected_speakers)}"
        )

    normalized_speakers: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for start, end, speaker in raw_turns:
        normalized_speaker = normalized_speakers.setdefault(
            speaker, f"SPEAKER_{len(normalized_speakers) + 1:02d}"
        )
        turns.append(SpeakerTurn(start, end, normalized_speaker))

    return turns
