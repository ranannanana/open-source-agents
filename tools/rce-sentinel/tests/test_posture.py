"""Security posture: key and database modes, git exposure, token strength, bad signatures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from rce_sentinel.config import Settings, load_settings
from rce_sentinel.memory.answers import AnswerCache
from rce_sentinel.memory.store import Store
from rce_sentinel.posture import posture


def by_name(settings: Settings) -> dict[str, str]:
    return {c.name: c.level for c in posture(settings)}


def test_fresh_setup_is_private(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RCE_SENTINEL_PROXY_TOKEN", raising=False)
    store = Store(settings.db_path)                    # creates key and database
    store.close()
    assert os.stat(settings.db_path).st_mode & 0o777 == 0o600
    got = by_name(settings)
    assert got["signing key"] == got["database"] == got["signatures"] == "ok"
    assert got["proxy token"] == "ok"


def test_readable_key_or_database_fails(settings: Settings) -> None:
    Store(settings.db_path).close()
    os.chmod(settings.db_path, 0o644)
    os.chmod(os.environ["RCE_SENTINEL_KEY_FILE"], 0o644)
    got = by_name(settings)
    assert got["database"] == "fail" and got["signing key"] == "fail"


def test_short_token_and_forged_rows_warn(settings: Settings,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCE_SENTINEL_PROXY_TOKEN", "short")
    store = Store(settings.db_path)
    cache = AnswerCache(store)
    key = cache.key({"messages": [{"role": "user", "content": "q"}]}, None, "")
    assert key
    cache.put(key, "m", {"role": "assistant", "content": "a"}, "stop", {}, 0.0)
    store.execute("UPDATE answer_cache SET model='other'")
    store.close()
    got = by_name(settings)
    assert got["proxy token"] == "warn" and got["signatures"] == "warn"


def test_state_dir_that_git_would_commit_warns(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setenv("RCE_SENTINEL_HOME", str(repo / ".rce-sentinel"))
    settings = load_settings(repo)
    assert by_name(settings)["git"] == "warn"
    (repo / ".gitignore").write_text(".rce-sentinel/\n")
    assert by_name(settings)["git"] == "ok"
