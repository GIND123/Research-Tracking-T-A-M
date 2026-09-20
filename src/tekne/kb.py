"""Knowledge base: a compiled gazetteer with hierarchy and synonymy.

The KB plays a deliberately limited role.  It is an *anchor*, not the target
vocabulary: a technology-tracking system whose output is constrained to a fixed
ontology can, by construction, never report an emerging technology.  So KB hits
raise confidence and supply a canonical id and a depth (our granularity proxy),
while everything else flows through as ``NIL`` and gets a corpus-local canonical
id from :mod:`tekne.link.canonical`.

Backing data is the Computer Science Ontology (Salatino et al., 2020) plus any
additional TSV gazetteers dropped into ``data/kb``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .lexicons import lemma_key


@dataclass(slots=True)
class KBEntry:
    entry_id: str
    label: str
    aliases: tuple[str, ...] = ()
    #: Distance from the ontology root; 0 is the root topic.
    depth: int = 0
    parents: tuple[str, ...] = ()
    kb: str = "cso"
    attested_from: int | None = None


@dataclass
class Gazetteer:
    """Longest-match dictionary lookup over lemma keys.

    Uses ``pyahocorasick`` when available (linear in text length, which matters
    once the KB has ~40k surface forms) and falls back to a dict lookup over
    candidate surfaces otherwise.
    """

    entries: dict[str, KBEntry] = field(default_factory=dict)
    #: lemma key -> entry id
    surface_index: dict[str, str] = field(default_factory=dict)
    _automaton: object | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    def contains(self, key: str) -> bool:
        return key in self.surface_index

    def lookup(self, surface: str) -> KBEntry | None:
        entry_id = self.surface_index.get(lemma_key(surface))
        return self.entries.get(entry_id) if entry_id else None

    def add(self, entry: KBEntry) -> None:
        self.entries[entry.entry_id] = entry
        for form in (entry.label, *entry.aliases):
            key = lemma_key(form)
            if len(key) < 3:
                continue
            # Never let a later, deeper entry steal a surface form from a
            # shallower one; ties go to the first writer for determinism.
            self.surface_index.setdefault(key, entry.entry_id)

    def build_automaton(self) -> None:
        try:
            import ahocorasick
        except ImportError:  # pragma: no cover - optional acceleration
            self._automaton = None
            return
        auto = ahocorasick.Automaton()
        for key, entry_id in self.surface_index.items():
            auto.add_word(key, (key, entry_id))
        if len(auto) == 0:
            self._automaton = None
            return
        auto.make_automaton()
        self._automaton = auto

    def scan(self, text: str) -> list[tuple[int, int, KBEntry]]:
        """Find KB surface forms in ``text``.

        ``text`` must already be lower-cased and whitespace-normalised by the
        caller so that offsets line up with the document; we do not transform it
        here precisely so offsets stay honest.
        """
        hits: list[tuple[int, int, KBEntry]] = []
        if self._automaton is None:
            return hits
        for end_index, (key, entry_id) in self._automaton.iter(text):  # type: ignore[union-attr]
            start = end_index - len(key) + 1
            if not _on_token_boundary(text, start, end_index + 1):
                continue
            entry = self.entries.get(entry_id)
            if entry:
                hits.append((start, end_index + 1, entry))
        return _longest_non_overlapping(hits)

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [
                {
                    "id": e.entry_id,
                    "label": e.label,
                    "aliases": list(e.aliases),
                    "depth": e.depth,
                    "parents": list(e.parents),
                    "kb": e.kb,
                    "attested_from": e.attested_from,
                }
                for e in self.entries.values()
            ]
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Gazetteer:
        gaz = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload["entries"]:
            gaz.add(
                KBEntry(
                    entry_id=row["id"],
                    label=row["label"],
                    aliases=tuple(row.get("aliases", ())),
                    depth=int(row.get("depth", 0)),
                    parents=tuple(row.get("parents", ())),
                    kb=row.get("kb", "cso"),
                    attested_from=row.get("attested_from"),
                )
            )
        gaz.build_automaton()
        return gaz

    @classmethod
    def empty(cls) -> Gazetteer:
        gaz = cls()
        gaz.build_automaton()
        return gaz


def _on_token_boundary(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not (text[start - 1].isalnum() or text[start - 1] in "-_")
    after_ok = end >= len(text) or not (text[end].isalnum() or text[end] in "-_")
    return before_ok and after_ok


def _longest_non_overlapping(
    hits: list[tuple[int, int, KBEntry]],
) -> list[tuple[int, int, KBEntry]]:
    """Keep the longest match at each position; ties broken by earlier start."""
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out: list[tuple[int, int, KBEntry]] = []
    last_end = -1
    for start, end, entry in hits:
        if start >= last_end:
            out.append((start, end, entry))
            last_end = end
    return out


# --- CSO ingestion ---------------------------------------------------------

_CSO_SUPER = "superTopicOf"
_CSO_RELATED = "relatedEquivalent"
_CSO_PREF = "preferentialEquivalent"
_CSO_LABEL = "rdf-schema#label"
_CSO_ROOT = "computer_science"


def build_from_cso(csv_path: Path, *, max_depth: int = 6) -> Gazetteer:
    """Compile the CSO triple dump into a gazetteer.

    ``depth`` is the shortest path from the ``computer_science`` root along
    ``superTopicOf``.  We use it as the granularity signal: depth <= 1 topics are
    research fields, deeper ones are techniques and artefacts.
    """
    children: dict[str, set[str]] = {}
    aliases: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    preferred: dict[str, str] = {}

    with Path(csv_path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            triple = _parse_triple(line)
            if not triple:
                continue
            subj, pred, obj = triple
            if _CSO_SUPER in pred:
                s, o = _topic_id(subj), _topic_id(obj)
                if s and o:
                    children.setdefault(s, set()).add(o)
                    labels.setdefault(s, _topic_label(s))
                    labels.setdefault(o, _topic_label(o))
            elif _CSO_RELATED in pred:
                s, o = _topic_id(subj), _topic_id(obj)
                if s and o:
                    aliases.setdefault(s, set()).add(_topic_label(o))
                    aliases.setdefault(o, set()).add(_topic_label(s))
            elif _CSO_PREF in pred:
                s, o = _topic_id(subj), _topic_id(obj)
                if s and o:
                    preferred[s] = o
                    aliases.setdefault(o, set()).add(_topic_label(s))
            elif _CSO_LABEL in pred:
                s = _topic_id(subj)
                if s:
                    labels[s] = _literal(obj) or _topic_label(s)

    depth = _bfs_depth(children, _CSO_ROOT, max_depth)
    parents: dict[str, set[str]] = {}
    for parent, kids in children.items():
        for kid in kids:
            parents.setdefault(kid, set()).add(parent)

    gaz = Gazetteer()
    topic_ids = set(labels) | set(children) | {c for kids in children.values() for c in kids}
    for topic in sorted(topic_ids):
        canonical = preferred.get(topic, topic)
        gaz.add(
            KBEntry(
                entry_id=f"cso:{canonical}",
                label=labels.get(canonical, _topic_label(canonical)),
                aliases=tuple(sorted(aliases.get(canonical, set()) | {_topic_label(topic)})),
                depth=depth.get(canonical, max_depth),
                parents=tuple(sorted(f"cso:{p}" for p in parents.get(canonical, set()))),
                kb="cso",
            )
        )
    gaz.build_automaton()
    return gaz


def add_tsv_gazetteer(gaz: Gazetteer, path: Path, *, kb: str, depth: int = 3) -> None:
    """Merge a ``id<TAB>label<TAB>alias|alias`` file into an existing gazetteer."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        entry_id, label = parts[0], parts[1]
        alias_field = parts[2] if len(parts) > 2 else ""
        year = int(parts[3]) if len(parts) > 3 and parts[3].strip().isdigit() else None
        gaz.add(
            KBEntry(
                entry_id=f"{kb}:{entry_id}",
                label=label,
                aliases=tuple(a for a in alias_field.split("|") if a),
                depth=depth,
                kb=kb,
                attested_from=year,
            )
        )


def _parse_triple(line: str) -> tuple[str, str, str] | None:
    parts = [p.strip() for p in line.rstrip("\n").split('","')]
    if len(parts) != 3:
        return None
    return (parts[0].lstrip('"'), parts[1], parts[2].rstrip('"'))


def _topic_id(uri: str) -> str | None:
    marker = "/topics/"
    idx = uri.find(marker)
    if idx < 0:
        return None
    return uri[idx + len(marker) :].strip("<>").strip()


def _topic_label(topic_id: str) -> str:
    return topic_id.replace("_", " ").strip()


def _literal(value: str) -> str:
    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        return ""
    return value.strip('"')


def _bfs_depth(children: dict[str, set[str]], root: str, max_depth: int) -> dict[str, int]:
    depth = {root: 0}
    frontier: Iterable[str] = [root]
    level = 0
    while frontier and level < max_depth:
        level += 1
        nxt: list[str] = []
        for node in frontier:
            for kid in children.get(node, ()):
                if kid not in depth:
                    depth[kid] = level
                    nxt.append(kid)
        frontier = nxt
    return depth
