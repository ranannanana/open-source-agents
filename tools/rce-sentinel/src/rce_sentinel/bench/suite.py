"""Benchmark suite: prompts with deterministic checks and an optional judge rubric.

A suite is a YAML file:

    name: dev
    version: 1
    items:
      - id: chore-commit-msg
        prompt: "..."
        must: ["regex", ...]        # every pattern must match the reply (case-insensitive)
        must_not: ["regex", ...]    # none may match
        rubric: "what a passing answer does"   # read by the judge model, if one is set

Each item lands in the train or test split by a hash of its id, so the split never
changes when items are added and the router is always scored on prompts it did not learn
from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rce_sentinel.config import TOOL_ROOT

SUITES = TOOL_ROOT / "demo" / "bench"
TEST_SHARE = 0.25


@dataclass
class Item:
    id: str
    prompt: str
    must: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    rubric: str = ""
    max_tokens: int = 8000

    @property
    def split(self) -> str:
        h = int(hashlib.sha256(self.id.encode()).hexdigest()[:8], 16)
        return "test" if h % 1000 < TEST_SHARE * 1000 else "train"

    def check(self, reply: str) -> dict[str, bool]:
        """Deterministic checks by pattern; True means the check passed."""
        out = {f"must:{p}": bool(re.search(p, reply, re.I | re.S)) for p in self.must}
        out |= {f"must_not:{p}": not re.search(p, reply, re.I | re.S) for p in self.must_not}
        return out


@dataclass
class Suite:
    name: str
    version: int
    items: list[Item]

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def load(cls, name_or_path: str) -> Suite:
        path = Path(name_or_path)
        if not path.suffix:
            path = SUITES / f"{name_or_path}.yaml"
        raw = yaml.safe_load(path.read_text())
        items = [Item(id=str(i["id"]), prompt=" ".join(str(i["prompt"]).split()),
                      must=list(i.get("must", [])), must_not=list(i.get("must_not", [])),
                      rubric=str(i.get("rubric", "")),
                      max_tokens=int(i.get("max_tokens", 8000)))
                 for i in raw["items"]]
        ids = [i.id for i in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate item ids in {path}")
        for i in items:
            for pat in i.must + i.must_not:
                re.compile(pat)
        return cls(name=str(raw["name"]), version=int(raw.get("version", 1)), items=items)
