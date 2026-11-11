"""Load `config.yaml` into typed settings. Relative paths resolve against the repo root."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_FILE = "config.yaml"

class ConfigError(ValueError):
    """Raised when config.yaml is missing or invalid. Messages name the field to fix."""


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0
    auto_install: bool = True
    input_cost_per_million: float  # $ per million input tokens, for the cost in LLM call logs
    output_cost_per_million: float  # $ per million output tokens


class PathsConfig(BaseModel):
    cache_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")


class Config(BaseModel):
    llm: LLMConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    root: Path = Field(default_factory=Path.cwd, exclude=True)

    def resolve(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.paths.cache_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)


def find_repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml and config.yaml.

    Falls back to the package's own repo, so notebooks in any subfolder work.
    """
    for candidate in [Path.cwd(), Path(__file__).resolve()]:
        for parent in [candidate, *candidate.parents]:
            if (parent / "pyproject.toml").is_file() and (parent / "config.yaml").is_file():
                return parent
    return Path.cwd()


def load_config() -> Config:
    """Load `config.yaml` from the repo root. Its location is fixed."""
    config_path = find_repo_root() / CONFIG_FILE
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    try:
        return Config(**raw, root=config_path.parent)
    except Exception as exc:  # pydantic ValidationError -> friendlier message
        raise ConfigError(f"Invalid {config_path.name}: {exc}") from exc
