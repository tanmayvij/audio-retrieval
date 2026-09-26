"""Audio-file metadata helpers."""

from pathlib import Path

import av


def get_duration_seconds(path: Path) -> float | None:
    """Return an audio file's duration from container metadata, if available."""
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)

        for stream in container.streams.audio:
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)

    return None
