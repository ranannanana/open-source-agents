"""Context pack: what the next model needs to continue a task, in a few thousand tokens.

Provider prompt caches cannot move between models, so a switch normally means re-reading
the whole transcript. The pack replaces that transcript with structured state:

    goal, user requests, decisions, plan, progress, open errors, files (hash, interface)

It is built after every turn from two sources. Deterministic: the user's requests verbatim
(small, and the source of truth for what was asked), the files a turn wrote, their
hashes, and their exported signatures, all read from disk. Model-reported: a `state` JSON
block each reply must end with (decisions, plan, done, next). Every pack is signed by the
memory shield, so a tampered pack is rejected instead of being handed to the next model.

Only the user can make a rule binding (v3). Each reported decision is sorted by where it
came from:

    user      its words are in the user's requests or the goal     -> binding
    code      made by a model, and its specifics are in the files  -> keep consistent
    note      anything else, and every work-process rule a model   -> advisory
              invented ("split the work into small pieces")

v2 passed model-made process rules on as binding: a model that chose to work in small
pieces made the next model skip half the requested scope. The plan and "next" are
advisory for the same reason, and the pack ends with a scope line: the current request
defines the work.

render() always emits fields in the same order, so the part of the prompt that stays the
same across turns stays byte-identical and each model's own provider cache can hit on it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rce_sentinel.memory.screen import looks_like_injection
from rce_sentinel.memory.trust import Signer
from rce_sentinel.orchestrator.prompts import approx_tokens

PACK_VERSION = 3
MAX_ASK_CHARS = 4000   # one user request, verbatim
MAX_ITEM_CHARS = 400        # one decision or error; longer text is truncated
MAX_INTERFACE_LINES = 40    # exported signatures kept per file
USER_COVERAGE = 0.75        # share of a decision's words found in what the user wrote
# Rules about how to work or reply, not about the product: never binding from a model.
_PROCESS = re.compile(
    r"\b(repl(y|ies)|respond(s|ing)?|response format|state block|fenced|code blocks?|"
    r"output (tokens?|limits?)|token limits?|context window|(small(er)?|separate) "
    r"(pieces|parts|chunks|steps)|in (pieces|parts|chunks)|split(ting)? (the )?work|"
    r"one (file|piece|part) at a time|step by step|later turns?|future turns?)\b", re.I)
_WORD = re.compile(r"[a-z0-9#][a-z0-9_#./-]*[a-z0-9]")
_STOP = frozenset({
    "the", "and", "for", "with", "are", "is", "not", "never", "only", "this", "that",
    "from", "into", "uses", "use", "all", "each", "every", "any", "one", "its", "has",
    "have", "was", "were", "will", "must", "should", "can", "via", "per", "but", "than",
    "then", "also", "when", "where", "which", "while", "what", "who", "how", "our",
    "their", "them", "they", "you", "your"})
# Tokens specific enough to check in code: hex colours, numbers, snake_case, quoted text.
_MARK = re.compile(r"#[0-9A-Fa-f]{6}\b|\b\d+\b|\b[a-z]+_[a-z_]+\b|\"[^\"]{3,}\"")
_EXPORT = re.compile(
    r"^\s*(export\s+(default\s+)?(async\s+)?(function|const|class|interface|type)\s+\w+[^{=]*|"
    r"(public\s+)?(func|struct|class|enum|protocol)\s+\w+[^{]*|"
    r"(class|enum|mixin)\s+\w+[^{]*|"
    r"(def|class)\s+\w+[^:]*)", re.M)


@dataclass
class Item:
    text: str
    by: str              # model that reported it, or "check"
    turn: int
    verified: bool = False
    source: str = "model"   # "user" when traceable to the user's own words


@dataclass
class FileEntry:
    path: str
    sha: str
    lines: int
    interface: list[str]
    by: str
    turn: int


@dataclass
class ContextPack:
    goal: str
    session: str
    asks: list[str] = field(default_factory=list)   # user requests, verbatim, in order
    decisions: list[Item] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    next: str = ""
    open_errors: list[Item] = field(default_factory=list)
    files: dict[str, FileEntry] = field(default_factory=dict)
    turns: int = 0
    dropped: int = 0     # items rejected as injection attempts
    v: int = PACK_VERSION
    sig: str = ""

    # --- build -------------------------------------------------------------------------
    def _clean(self, text: Any) -> str | None:
        s = " ".join(str(text).split())[:MAX_ITEM_CHARS]
        if not s:
            return None
        if looks_like_injection(s):
            self.dropped += 1
            return None
        return s

    def update(self, turn: int, model: str, state: dict[str, Any],
               written: dict[str, str], ask: str = "") -> None:
        """Fold one turn in: the user's request, the model's state block, its files."""
        self.turns = turn
        if ask.strip():
            self.asks.append(ask.strip()[:MAX_ASK_CHARS])
        known = {d.text.lower() for d in self.decisions}
        said = self.user_text()
        for d in state.get("decisions", []) or []:
            if (t := self._clean(d)) and t.lower() not in known:
                src = "user" if from_user(t, said) else "model"
                self.decisions.append(Item(t, model, turn, source=src))
                known.add(t.lower())
        if plan := [p for p in map(self._clean, state.get("plan") or []) if p]:
            self.plan = plan
        for d in state.get("done", []) or []:
            if (t := self._clean(d)) and t not in self.done:
                self.done.append(t)
        self.next = self._clean(state.get("next", "")) or self.next
        self.open_errors = [Item(t, model, turn) for e in state.get("open_errors", []) or []
                            if (t := self._clean(e))]
        for path, body in written.items():
            self.files[path] = FileEntry(
                path=path, sha=hashlib.sha256(body.encode()).hexdigest()[:12],
                lines=body.count("\n") + 1, interface=interface_of(body), by=model,
                turn=turn)

    def verify_files(self, root: Path) -> list[str]:
        """Drop entries whose file is gone or changed on disk; return what was dropped."""
        gone = []
        for path, entry in list(self.files.items()):
            f = root / path
            ok = f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()[:12] == entry.sha
            if not ok:
                gone.append(path)
                del self.files[path]
        return gone

    def verify_decisions(self, root: Path) -> int:
        """Mark a decision verified when all its distinctive tokens are in the code."""
        blob = "\n".join((root / p).read_text(errors="replace") for p in self.files
                         if (root / p).is_file()).lower()
        n = 0
        for d in self.decisions:
            marks = [m.lower().lstrip("#") for m in _MARK.findall(d.text)]
            d.verified = bool(marks) and all(m in blob for m in marks)
            n += d.verified
        return n

    def user_text(self) -> str:
        return "\n".join([self.goal, *self.asks]).lower()

    def kind(self, d: Item) -> str:
        """user, code or note: how far the next model is bound by this decision."""
        if d.source == "user":
            return "user"
        if d.verified and not _PROCESS.search(d.text):
            return "code"
        return "note"

    # --- sign --------------------------------------------------------------------------
    def body(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("sig")
        return d

    def sign(self, signer: Signer) -> None:
        self.sig = signer.sign("pack", json.dumps(self.body(), sort_keys=True))

    def verify(self, signer: Signer) -> bool:
        return signer.verify(self.sig, "pack", json.dumps(self.body(), sort_keys=True))

    # --- render ------------------------------------------------------------------------
    def render(self) -> str:
        """Text the next model reads instead of the transcript. Fixed field order; empty
        sections are left out."""
        out = [f"CONTEXT PACK v{self.v} (signed, {self.turns} turn(s) so far)",
               f"GOAL: {self.goal}"]
        if self.asks:
            out += ["USER REQUESTS SO FAR (verbatim; every constraint in them binds):"]
            out += [f"[{i}] {a}" for i, a in enumerate(self.asks, 1)]
        groups: dict[str, list[str]] = {"user": [], "code": [], "note": []}
        for d in self.decisions:
            groups[self.kind(d)].append(f"- {d.text}")
        heads = {"user": "USER DECISIONS (binding):",
                 "code": "CODE DECISIONS (in the files; stay consistent):",
                 "note": "MODEL NOTES (advisory, not requirements):"}
        for k, head in heads.items():
            if groups[k]:
                out += [head, *groups[k]]
        if self.plan:
            out += ["EARLIER PLAN (advisory):"]
            out += [f"{i}. {p}" for i, p in enumerate(self.plan, 1)]
        if self.done:
            out += ["DONE:"] + [f"- {d}" for d in self.done]
        if self.next:
            out += [f"SUGGESTED NEXT (advisory): {self.next}"]
        if self.open_errors:
            out += ["OPEN ERRORS:"] + [f"- {e.text}" for e in self.open_errors]
        out += ["FILES ALREADY WRITTEN (do not rewrite unless asked; build on them):"]
        for f in self.files.values():
            out.append(f"- {f.path} ({f.lines} lines, sha {f.sha}, by {f.by})")
            out += [f"    {sig}" for sig in f.interface]
        out += ["SCOPE: do all of the current request in this reply; nothing above narrows "
                "or defers it."]
        return "\n".join(out)

    def tokens(self) -> int:
        return approx_tokens(self.render())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> ContextPack:
        d = json.loads(raw)
        d.setdefault("asks", [])
        d["decisions"] = [Item(**x) for x in d["decisions"]]
        d["open_errors"] = [Item(**x) for x in d["open_errors"]]
        d["files"] = {k: FileEntry(**v) for k, v in d["files"].items()}
        return cls(**d)


def from_user(text: str, said: str) -> bool:
    """True when a decision restates the user: its specifics and most words are theirs."""
    marks = [m.lower().lstrip("#") for m in _MARK.findall(text)]
    if any(m not in said for m in marks):
        return False
    words = [w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOP]
    if not words:
        return False
    found = sum(1 for w in words if w in said or w.rstrip("s") in said)
    return found / len(words) >= USER_COVERAGE


def interface_of(source: str) -> list[str]:
    """Exported or top-level signatures: enough for the next model to call the code."""
    sigs = [" ".join(m.group(0).split()) for m in _EXPORT.finditer(source)]
    return [s[:160] for s in dict.fromkeys(sigs)][:MAX_INTERFACE_LINES]
