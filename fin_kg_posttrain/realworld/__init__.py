"""Extract structured knowledge from SAPerm real-world corpora → FT representation."""

from .loaders import RealCorpus, load_corpus, list_corpora
from .extract_kg import KnowledgeBundle, extract_bundle
from .pack import pack_for_ft, pack_sharegpt_rows

__all__ = [
    "RealCorpus",
    "load_corpus",
    "list_corpora",
    "KnowledgeBundle",
    "extract_bundle",
    "pack_for_ft",
    "pack_sharegpt_rows",
]
