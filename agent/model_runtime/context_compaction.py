"""Compatibility exports for immutable historical migrations.

Runtime compaction ownership lives in the ordinary ``compaction`` plugin.
"""

from agent.model_runtime.compaction_migration_v1 import (
    compaction_scope_id,
    compaction_source_ref,
)

__all__ = ["compaction_scope_id", "compaction_source_ref"]
