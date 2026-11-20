"""Security posture: refuse to trust a setup that would leak or accept forged memory.

Idea from adk-memshield's verify_rls_posture(): check the deployment, not just the code,
at startup. Here that means the signing key, the state directory and database, whether
git could pick the database up, the proxy token, and rows whose signature fails.

    fail   memory could be forged or read by other users: fix before relying on it
    warn   weaker than it should be
    ok     as intended
"""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rce_sentinel.config import Settings
from rce_sentinel.memory.trust import default_key_path

MIN_TOKEN_CHARS = 16


@dataclass
class Check:
    level: str   # ok | warn | fail
    name: str
    detail: str


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _owner_only(path: Path, name: str, *, missing: str) -> Check:
    if not path.exists():
        return Check("warn", name, missing)
    mode = _mode(path)
    if path.stat().st_uid != os.getuid():
        return Check("fail", name, f"{path} is owned by another user")
    if mode & 0o077:
        return Check("fail", name, f"{path} is mode {mode:o}; run chmod 600 {path}")
    return Check("ok", name, f"{path} mode {mode:o}")


def _git_ignored(path: Path) -> bool | None:
    """True/False inside a git work tree, None when not in one or git is missing."""
    try:
        inside = subprocess.run(["git", "-C", str(path.parent), "rev-parse",
                                 "--is-inside-work-tree"], capture_output=True, text=True,
                                timeout=10)
        if inside.returncode != 0:
            return None
        res = subprocess.run(["git", "-C", str(path.parent), "check-ignore", "-q", str(path)],
                             capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.returncode == 0


def _bad_signatures(settings: Settings) -> Check:
    """Count cached rows whose signature fails, without touching hit counters."""
    from rce_sentinel.memory.answers import AnswerCache
    from rce_sentinel.memory.store import Store

    if not settings.db_path.exists():
        return Check("ok", "signatures", "no database yet")
    store = Store(settings.db_path)
    try:
        bad = sum(1 for r in store.execute("SELECT * FROM result_cache")
                  if not store.signer.verify(
                      r["sig"], "cache", r["key"], r["verdict_json"], r["trust"] or 0,
                      r["rules_version"], r["cve_db_version"], r["prompt_version"]))
        bad += AnswerCache(store).count_unsigned()
    finally:
        store.close()
    if bad:
        return Check("warn", "signatures", f"{bad} cached row(s) have a wrong signature and "
                     "are ignored; the database may have been copied in or edited")
    return Check("ok", "signatures", "every cached row verifies")


def posture(settings: Settings) -> list[Check]:
    out = [_owner_only(default_key_path(), "signing key",
                       missing="no key yet; one is created on first use")]
    state = settings.state_dir
    if state.exists() and _mode(state) & 0o002:
        out.append(Check("fail", "state dir", f"{state} is world-writable"))
    else:
        out.append(Check("ok", "state dir", f"{state}"))
    out.append(_owner_only(settings.db_path, "database",
                           missing="no database yet; it is created owner-only"))
    ignored = _git_ignored(state)
    if ignored is False:
        out.append(Check("warn", "git", f"{state} is not git-ignored; call logs and cached "
                         "answers could be committed. Add .rce-sentinel/ to .gitignore"))
    elif ignored:
        out.append(Check("ok", "git", "state dir is git-ignored"))
    token = os.environ.get("RCE_SENTINEL_PROXY_TOKEN", "")
    if token and len(token) < MIN_TOKEN_CHARS:
        out.append(Check("warn", "proxy token", f"shorter than {MIN_TOKEN_CHARS} characters"))
    else:
        out.append(Check("ok", "proxy token", "set" if token else
                         "not set; serve stays on loopback only"))
    out.append(_bad_signatures(settings))
    return out
