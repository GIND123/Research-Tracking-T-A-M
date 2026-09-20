"""TEKNE -- grounded extraction of technology mentions from papers and patents.

The name is from Greek *tekhne*, the root of "technology".
"""

from .config import Config
from .schema import (
    Document,
    ExtractionResult,
    Genre,
    Role,
    SectionKind,
    TechMention,
    TechType,
)

__version__ = "0.3.0"

__all__ = [
    "Config",
    "Document",
    "ExtractionResult",
    "Genre",
    "Role",
    "SectionKind",
    "TechMention",
    "TechType",
    "__version__",
]
