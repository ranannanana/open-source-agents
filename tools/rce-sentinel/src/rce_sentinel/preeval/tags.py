"""Prompt tags and the free rule-based tagger.

A tag is a set of probabilities, not labels, so the scorer can weigh them:
task type (one of TASK_TYPES), difficulty (a distribution over DIFFICULTY levels), and a few
yes/no predicates. Rules run first and cost nothing; when they are unsure, a decision model
(jev.py) or a local model answers the same questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TASK_TYPES = ("build", "edit", "debug", "refactor", "test", "explain", "design", "chore")
DIFFICULTY = ("trivial", "routine", "hard", "frontier")
PREDICATES = ("multi_file", "multi_target", "long_output")

TASK_HELP = {
    "build": "create a new app, feature, or project from scratch",
    "edit": "change or extend existing code",
    "debug": "find and fix a bug, error, or failing test",
    "refactor": "restructure code without changing behaviour",
    "test": "write or fix tests",
    "explain": "answer a question or explain code, no code change",
    "design": "architecture, API, or UI/UX design decisions",
    "chore": "commit messages, renames, config, docs, formatting",
}
DIFFICULTY_HELP = {
    "trivial": "one-liner, no reasoning",
    "routine": "a normal change a mid-size model handles",
    "hard": "multi-step reasoning, several files, subtle bugs",
    "frontier": "large build or deep design needing the strongest model",
}

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "build": (r"\bbuild\b", r"\bcreate\b", r"\bscaffold", r"\bfrom scratch\b", r"\bnew app\b",
              r"\bimplement\b", r"\bmake (an?|the) \w+ app\b",
              r"^(build|create|scaffold|implement)\b",
              r"\bwrite an? (\w+ )*(package|service|library|module|cli|server|app)\b"),
    "edit": (r"\badd\b", r"\bchange\b", r"\bupdate\b", r"\bextend\b", r"\bmodify\b",
             r"\bsupport for\b", r"\breplace\b", r"\bconvert\b", r"\bmake this\b",
             r"\bread the \w+ from\b"),
    "debug": (r"\bfix\b", r"\bbug\b", r"error\b", r"exception\b", r"\btraceback\b",
              r"\bfail(s|ing|ed)?\b", r"\bcrash", r"\bbroken\b", r"\bdoesn'?t work\b",
              r"\binstead of\b", r"\bhangs?\b", r"\bdeadlocks?\b",
              r"\bleak", r"\boom\b", r"\bslow", r"\bstops\b", r"\bflaky\b", r"\bforever\b",
              r"\bspikes?\b", r"\bcannot\b", r"\bduplicate", r"\bdiagnose\b", r"\bis off\b",
              r"\bnever deallocated\b", r"\bundefined\b", r"\breturns? negative\b"),
    "refactor": (r"\brefactor", r"\bclean ?up\b", r"\bsimplif", r"\bextract\b",
                 r"\brestructure", r"\bdeduplicate", r"\bnull safety\b"),
    "test": (r"\btests?\b", r"\bunit test", r"\bpytest\b", r"\bjest\b", r"\bcoverage\b",
             r"\bxctest\b", r"\buntested\b"),
    "explain": (r"^(what|why|how|where|when|which)\b", r"\bexplain\b", r"\bwhat does\b",
                r"\bhow does\b", r"\?\s*$"),
    "design": (r"\barchitect", r"\bdesign\b", r"\bui\b", r"\bux\b", r"\blayout\b",
               r"\bschema\b", r"\bapi design\b", r"\btrade-?offs?\b", r"\bdata model\b",
               r"\bhow (should|would) (we|you)\b", r"\bstrateg", r"\bdefen[cs]es?\b",
               r"\bconflict resolution\b", r"\bconsensus\b", r"\bno downtime\b",
               r"\bzero[- ]downtime\b", r"\bstep[- ]by[- ]step plan\b"),
    "chore": (r"\bcommit message\b", r"\brename\b", r"\btypos?\b", r"\bformat", r"\blint\b",
              r"\breadme\b", r"\bbump\b", r"\bconfig\b", r"\.gitignore\b", r"\bchangelog\b",
              r"\bpr title\b", r"\bpretty[- ]print", r"\bsort (these )?\w* ?imports\b",
              r"\.env\b", r"\blicense\b", r"\bdocstring\b", r"\bmakefile\b", r"\bsemver\b"),
}
_COMPILED = {k: [re.compile(p, re.I | re.M) for p in v] for k, v in _KEYWORDS.items()}
# Specific phrases outweigh generic verbs like "add" or "change".
_STRONG = {r"\bcommit message\b", r"\bfrom scratch\b", r"\btraceback\b", r"\brefactor",
           r"\bunit test", r"\bexplain\b", r"\barchitect", r"\btypos?\b", r"\bnew app\b",
           r"\binstead of\b", r"\bdiagnose\b", r"\bdeadlocks?\b", r"\bleak",
           r"\.gitignore\b", r"\bchangelog\b", r"\bpr title\b", r"\bpretty[- ]print",
           r"\bdocstring\b", r"\bsemver\b", r"\blicense\b", r"\bxctest\b", r"\buntested\b",
           r"\bnull safety\b", r"\bdata model\b", r"\btrade-?offs?\b",
           r"\bsort (these )?\w* ?imports\b", r"\bdesign\b",
           r"^(build|create|scaffold|implement)\b", r"\bconvert\b", r"\breplace\b",
           r"\bwrite an? (\w+ )*(package|service|library|module|cli|server|app)\b"}
# Explaining is secondary when another kind of work is asked for too.
_ACTION = ("build", "edit", "debug", "refactor", "test", "design", "chore")

# Frameworks and platforms; two or more in one prompt means several targets.
TARGETS = {
    "react native": r"\breact[- ]native\b", "flutter": r"\bflutter\b",
    "swift": r"\bswift(ui)?\b", "kotlin": r"\bkotlin\b", "react": r"\breact(?![- ]native)\b",
    "vue": r"\bvue\b", "angular": r"\bangular\b", "python": r"\bpython\b",
    "go": r"\bgolang\b|\bin go\b", "rust": r"\brust\b", "node": r"\bnode(\.js)?\b",
}
PLATFORMS = {"ios": r"\bios\b", "android": r"\bandroid\b", "web": r"\bweb\b",
             "desktop": r"\bdesktop\b|\bmacos\b|\bwindows\b"}

_HARD = (r"\bfrom scratch\b", r"\bbest possible\b", r"\bproduction\b", r"\bscalab",
         r"\barchitect", r"\bend[- ]to[- ]end\b", r"\bfull\b", r"\bcomplete\b",
         r"\bcleanest\b", r"\bconcurren", r"\brace condition\b", r"\bmemory leak\b",
         r"\bdeadlock", r"\bleaks?\b", r"\bthreads?\b", r"\bgoroutine", r"\bcoroutine",
         r"\bevent loop\b", r"\basync\b", r"\bunder load\b", r"\bp99\b", r"\bdistributed\b",
         r"\bexactly[- ]once\b", r"\bidempoten", r"\bconsisten", r"\bmulti-tenant\b",
         r"\btime ?zone\b", r"\bunicode\b", r"\boverflow\b", r"\bmigrations?\b",
         r"\bno downtime\b", r"\bzero[- ]downtime\b", r"\bbackoff\b", r"\bretries\b",
         r"\bstampede\b", r"\bretain cycle\b", r"\btransactions?\b", r"\bisolation\b",
         r"\bpartitions?\b", r"\bgraceful shutdown\b", r"\boffline\b", r"\bconflict",
         r"\b(all|every|each) (the )?(problems|leaks?|issues|bugs)\b", r"\bfind every\b",
         r"\btrade-?offs?\b", r"\bdefen[cs]es? in depth\b", r"\bsometimes\b",
         r"\bintermittent", r"\bmemory\b", r"\bcache[ds]?\b", r"\btwo (\w+ )?fixes\b",
         r"\bprecisely\b", r"\bfor both\b", r"\boverfit")
_TRIVIAL = (r"\btypo\b", r"\brename\b", r"\bcommit message\b", r"\bone[- ]liner\b",
            r"\bformat\b")
_MULTI_FILE = (r"\bacross\b", r"\bmodules?\b", r"\bfiles\b", r"\bcodebase\b", r"\bapp\b",
               r"\bproject\b", r"\brepo\b")
# Named source files, e.g. worker.go or 001_init.sql.
_FILE = re.compile(r"\b[\w/-]+\.(py|go|ts|tsx|js|jsx|swift|kt|dart|sql|ya?ml|rs|java|rb)\b",
                   re.I)


@dataclass
class Tags:
    task: dict[str, float]           # P(task type), sums to 1
    difficulty: list[float]          # P(level), index into DIFFICULTY
    predicates: dict[str, float]     # P(true) per PREDICATES entry
    source: str                      # rules | jev | local
    confidence: float                # how sure the tagger is overall
    targets: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    cost_usd: float = 0.0            # what tagging itself cost

    @property
    def top_task(self) -> str:
        return max(self.task, key=lambda k: self.task[k])

    @property
    def level(self) -> float:
        """Expected difficulty index, 0 (trivial) to 3 (frontier)."""
        return sum(i * p for i, p in enumerate(self.difficulty))

    def to_dict(self) -> dict[str, Any]:
        diff = dict(zip(DIFFICULTY, self.difficulty, strict=True))
        return {"task": self.task, "difficulty": diff,
                "predicates": self.predicates, "source": self.source,
                "confidence": round(self.confidence, 3), "targets": self.targets,
                "platforms": self.platforms, "subtasks": self.subtasks,
                "cost_usd": self.cost_usd}

    def summary(self) -> str:
        diff = DIFFICULTY[round(self.level)]
        return (f"{self.top_task} ({self.task[self.top_task]:.0%}), {diff} "
                f"(level {self.level:.2f}), via {self.source}, conf {self.confidence:.2f}")


def _hits(patterns: tuple[str, ...] | list[re.Pattern[str]], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.I | re.M))


def _found(table: dict[str, str], text: str) -> list[str]:
    return [name for name, pat in table.items() if re.search(pat, text, re.I)]


def _spread(center: float, sharp: float = 1.5) -> list[float]:
    """A distribution over difficulty levels peaked at `center`."""
    w = [1.0 / (1.0 + sharp * abs(i - center)) ** 2 for i in range(len(DIFFICULTY))]
    s = sum(w)
    return [x / s for x in w]


def rule_tags(prompt: str) -> Tags:
    """Free deterministic tags from keywords, length, and named targets."""
    text = prompt.strip()
    counts = {k: float(sum(2 if p.pattern in _STRONG else 1 for p in pats if p.search(text)))
              for k, pats in _COMPILED.items()}
    if any(counts[k] for k in _ACTION):
        counts["explain"] *= 0.25 if counts["debug"] or counts["design"] else 0.5
    total = sum(counts.values())
    # Laplace smoothing keeps every type possible.
    task = {k: (counts[k] + 0.25) / (total + 0.25 * len(TASK_TYPES)) for k in TASK_TYPES}
    targets = _found(TARGETS, text)
    platforms = _found(PLATFORMS, text)
    n_targets = max(len(targets), 1)
    n_files = len({m.group(0).lower() for m in _FILE.finditer(text)})
    # Clauses roughly count the separate requirements asked for.
    clauses = len(re.findall(r"[,;:]|\band\b|\(\d\)", text))

    level = 1.0
    level += 0.5 * min(_hits(_HARD, text), 4)
    level -= 0.8 * min(_hits(_TRIVIAL, text), 1)
    if counts["build"] and n_targets > 1:
        level += 0.5
    if n_files >= 3:
        level += 0.6
    if clauses >= 8:
        level += 0.3
    level += 0.3 * (len(text) > 350) + 0.3 * (len(text) > 700) + 0.5 * (len(text) > 1500)
    # Diagnosing a described symptom in quoted code is real work.
    if counts["debug"] and len(text) > 220:
        level += 0.3 + 0.2 * ("`" in text)
    if counts["explain"] and not (counts["build"] or counts["debug"]) and level < 1.5:
        level -= 0.3
    level = min(max(level, 0.0), 3.0)

    multi_file = min(0.15 + 0.2 * _hits(_MULTI_FILE, text) + 0.3 * bool(counts["build"])
                     + 0.4 * (n_files >= 2), 0.95)
    long_output = 0.9 if counts["build"] and level >= 2 else (0.5 if counts["build"] else 0.15)
    subtasks = ([f"build for {t}" for t in targets]
                if counts["build"] and len(targets) > 1 else [])

    top = max(task.values())
    second = sorted(task.values())[-2]
    # Sure when one type clearly leads and enough keywords fired.
    confidence = min(1.0, (top - second) * 2 + 0.1 * min(total, 4))
    return Tags(task=task, difficulty=_spread(level),
                predicates={"multi_file": round(multi_file, 3),
                            "multi_target": 0.95 if len(targets) > 1 else 0.05,
                            "long_output": long_output},
                source="rules", confidence=confidence, targets=targets,
                platforms=platforms, subtasks=subtasks)
