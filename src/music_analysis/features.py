"""
Dataclass representing per-track audio features with JSON serialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Features:
    track_name: str
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    key_strength: float | None = None
    loudness: float | None = None
    danceability: float | None = None
    effnet_embedding: list[float] = field(default_factory=list)
    vggish_embedding: list[float] = field(default_factory=list)
    clap_matches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, directory: Path) -> Path:
        output_path = directory / f"{self.track_name}.json"
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path

    @classmethod
    def from_dict(cls, data: dict) -> Features:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> Features:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
