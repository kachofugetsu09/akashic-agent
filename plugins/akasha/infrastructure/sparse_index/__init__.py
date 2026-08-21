"""Causal sparse turn index."""

from .builder import (
    AppendOnlyViolation,
    BuildConfig,
    BuildResult,
    EmbeddingAudit,
    EmbeddingIssue,
    SparseIndexRebuildRequired,
    audit_source_embeddings,
    build_sparse_index,
    sparse_index_state_sha256,
)

__all__ = [
    "AppendOnlyViolation",
    "BuildConfig",
    "BuildResult",
    "EmbeddingAudit",
    "EmbeddingIssue",
    "SparseIndexRebuildRequired",
    "audit_source_embeddings",
    "build_sparse_index",
    "sparse_index_state_sha256",
]
