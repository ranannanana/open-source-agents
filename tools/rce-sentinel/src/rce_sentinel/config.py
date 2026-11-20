"""Config loading. Lookup order: $RCE_SENTINEL_CONFIG_DIR, <repo>/.rce-sentinel/config,
packaged defaults."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROMPT_VERSION = "p3"


def _default_config_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [here / "_config", here.parents[1] / "config"]


def config_dirs(repo_root: Path | None = None) -> list[Path]:
    dirs: list[Path] = []
    if env := os.environ.get("RCE_SENTINEL_CONFIG_DIR"):
        dirs.append(Path(env))
    if repo_root is not None:
        dirs.append(repo_root / ".rce-sentinel" / "config")
    dirs.extend(_default_config_dirs())
    return [d for d in dirs if d.is_dir()]


def find_config(name: str, repo_root: Path | None = None) -> Path:
    for d in config_dirs(repo_root):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"config file not found: {name}")


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "on", "true", "yes"}


@dataclass
class Settings:
    repo_root: Path
    state_dir: Path
    models: dict[str, Any]
    policy: dict[str, Any]
    sources: dict[str, Any]
    symbols: dict[str, Any]
    rule_files: list[Path]
    caveman: bool
    provider_mode: str  # "auto" | "mock" | "off"
    rules_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.state_dir / "sentinel.db"


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def load_env_file(path: Path | str) -> list[str]:
    """Load KEY=VALUE lines from an explicit file; never overrides set vars.

    Returns the names loaded, never the values.
    """
    loaded = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "=" if "=" in line else ":"  # also accept KEY: value
        if sep not in line:
            continue
        key, _, val = line.removeprefix("export ").partition(sep)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key.isidentifier() and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


TOOL_ROOT = Path(__file__).resolve().parents[2]
_autoloaded: list[str] = []


def env_file_candidates() -> list[Path]:
    """Explicit RCE_SENTINEL_ENV_FILE, then the user config dir, then the tool's .env."""
    explicit = os.environ.get("RCE_SENTINEL_ENV_FILE", "")
    return [*([Path(explicit).expanduser()] if explicit else []),
            Path.home() / ".config" / "rce-sentinel" / ".env", TOOL_ROOT / ".env"]


def autoload_env() -> list[str]:
    """Load API keys from every env file that exists, once per process."""
    if env_flag("RCE_SENTINEL_NO_ENV_AUTOLOAD", False):
        return []
    if not _autoloaded:
        _autoloaded.append("")
        for path in env_file_candidates():
            if path.is_file():
                _autoloaded.extend(load_env_file(path))
    return [n for n in _autoloaded if n]


def load_settings(repo_root: Path) -> Settings:
    autoload_env()
    repo_root = repo_root.resolve()
    state_dir = Path(os.environ.get("RCE_SENTINEL_HOME", repo_root / ".rce-sentinel"))
    state_dir.mkdir(parents=True, exist_ok=True)
    policy = load_yaml(find_config("policy.yaml", repo_root))
    rule_files = sorted(find_config("rules", repo_root).glob("*.yaml"))
    symbols_path = find_config("symbols.yaml", repo_root)
    caveman_default = bool(policy.get("caveman", True))
    return Settings(
        repo_root=repo_root,
        state_dir=state_dir,
        models=load_yaml(find_config("models.yaml", repo_root)),
        policy=policy,
        sources=load_yaml(find_config("sources.yaml", repo_root)),
        symbols=load_yaml(symbols_path),
        rule_files=rule_files,
        caveman=env_flag("CAVEMAN", caveman_default),
        provider_mode=os.environ.get("RCE_SENTINEL_PROVIDER", "auto").lower(),
        rules_version=_hash_files([*rule_files, symbols_path]),
    )
