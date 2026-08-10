from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleasePaths:
    root: Path

    @property
    def sources(self) -> Path:
        return self.root / "runtime-sources"

    @property
    def bridge_venvs(self) -> Path:
        return self.root / "bridge-venvs"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def activation(self) -> Path:
        return self.root / "activation"

    @property
    def run(self) -> Path:
        return self.root / "run"

    @property
    def secrets(self) -> Path:
        return self.root / "secrets"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    def source(self, commit: str) -> Path:
        return self.sources / commit

    def bridge_venv(self, commit: str) -> Path:
        return self.bridge_venvs / commit

    def release(self, commit: str) -> Path:
        return self.releases / f"{commit}.json"

    def create_layout(self) -> None:
        for path in (
            self.sources,
            self.bridge_venvs,
            self.releases,
            self.activation,
            self.run,
            self.secrets,
            self.state,
            self.backups,
        ):
            path.mkdir(parents=True, exist_ok=True)
