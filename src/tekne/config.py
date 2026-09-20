"""Run configuration.

A run is defined by this object and nothing else: the digest of its serialised
form goes into every mention's provenance, so two results with the same
``config_digest`` were produced by the same pipeline.  That is also how the
ablation table is generated -- each row is a config, not a code branch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "TEKNE_"


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> dict[str, str]:
    """Read a ``.env`` file into the environment.

    Hand-rolled rather than a dependency: we need exactly ``KEY=value`` with
    comments, and a secrets loader is a thing worth being able to read in full.
    Values are not expanded and quotes are stripped only if balanced.
    """
    file = Path(path)
    loaded: dict[str, str] = {}
    if not file.is_file():
        return loaded
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


@dataclass
class ModelConfig:
    """The escalation cascade.

    Three roles, deliberately on three different models.  Proposal is the bulk
    workload and runs on the mid-tier model; verification is a narrow judgement
    over one sentence and runs on the cheapest; adjudication sees only the
    residue the first two disagreed on and runs on the strongest.  Using
    different models for proposing and verifying is not only a cost decision --
    a verifier that shares the proposer's weights shares its blind spots.
    """

    proposer: str = "claude-sonnet-5"
    verifier: str = "claude-haiku-4-5"
    adjudicator: str = "claude-opus-5"
    effort: str = "medium"


@dataclass
class BudgetConfig:
    max_llm_calls_per_doc: int = 12
    max_usd_per_doc: float = 0.25
    max_seconds_per_doc: float = 180.0
    #: Consecutive schema failures before the LLM tier is disabled for the run.
    circuit_breaker_failures: int = 5


@dataclass
class RecallConfig:
    use_patterns: bool = True
    use_abbrev: bool = True
    use_cvalue: bool = True
    use_gazetteer: bool = True
    use_llm: bool = True
    cvalue_threshold: float = 2.0
    max_chunk_tokens: int = 6
    #: Gate chunker output on the head-noun lexicon. Off by default; see
    #: recall/patterns.py for the measured recall cost of switching it on.
    require_tech_head: bool = False
    llm_max_windows: int = 3
    llm_max_input_chars: int = 9000


@dataclass
class GuardConfig:
    grounding: bool = True
    negative_lexicon: bool = True
    injection: bool = True
    consensus: bool = True
    consensus_min_proposers: int = 2
    evidence: bool = True
    temporal: bool = True
    provenance: bool = True
    verifier: bool = True
    allow_fuzzy_grounding: bool = False


@dataclass
class VerifyConfig:
    use_embedding: bool = True
    use_llm: bool = False
    batch_size: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Candidates with |margin| above this are decided without a model call.
    escalation_band: tuple[float, float] = (0.35, 0.65)
    adjudicate_disagreements: bool = True


@dataclass
class OutputConfig:
    #: Confidence below which a mention is withheld for review instead of emitted.
    abstain_below: float = 0.45
    #: Types that count as technology for the tracking corpus.
    tracked_types: tuple[str, ...] = ("artifact", "method", "material")
    keep_rejected: bool = True
    max_mentions_per_doc: int = 2000


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    kb_path: str | None = "data/kb/cso.json"
    cache_path: str | None = "runs/llm_cache.sqlite"
    seed: int = 20260101
    label: str = "default"

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> Config:
        load_dotenv()
        data: dict[str, Any] = {}
        if path:
            import yaml

            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls.from_dict(data)
        cfg.apply_env()
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        sections = {
            "models": ModelConfig,
            "budget": BudgetConfig,
            "recall": RecallConfig,
            "guards": GuardConfig,
            "verify": VerifyConfig,
            "output": OutputConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            payload = data.get(name) or {}
            valid = {f.name for f in fields(klass)}
            unknown = set(payload) - valid
            if unknown:
                raise ValueError(f"unknown keys in config section {name!r}: {sorted(unknown)}")
            kwargs[name] = klass(**payload)
        for scalar in ("kb_path", "cache_path", "seed", "label"):
            if scalar in data:
                kwargs[scalar] = data[scalar]
        return cls(**kwargs)

    def apply_env(self) -> None:
        """Environment overrides, for CI and for the ``.env`` cascade knobs."""
        mapping = {
            f"{ENV_PREFIX}PROPOSER_MODEL": ("models", "proposer"),
            f"{ENV_PREFIX}VERIFIER_MODEL": ("models", "verifier"),
            f"{ENV_PREFIX}ADJUDICATOR_MODEL": ("models", "adjudicator"),
            f"{ENV_PREFIX}KB_PATH": (None, "kb_path"),
            f"{ENV_PREFIX}CACHE_PATH": (None, "cache_path"),
        }
        for env_name, (section, attr) in mapping.items():
            value = os.environ.get(env_name)
            if not value:
                continue
            target = getattr(self, section) if section else self
            setattr(target, attr, value)

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a copy with dotted-path overrides applied (``guards.consensus=False``)."""
        data = self.to_dict()
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config.from_dict(data)

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def has_api_key(self) -> bool:
        return bool(
            os.environ.get("ANTHROPIC_API_KEY", "").strip()
            and "REPLACE_ME" not in os.environ.get("ANTHROPIC_API_KEY", "")
        )
