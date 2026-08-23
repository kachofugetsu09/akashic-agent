"""Read-only inventory and explicit handoff for the retired proactive island."""

from agent.migrations.proactive_island.handoff import (
    AdapterPlan,
    HandoffAdapter,
    HandoffBlocked,
    HandoffReport,
    HandoffStatus,
    TargetReceipt,
    apply_handoff,
    preflight_handoff,
)
from agent.migrations.proactive_island.inventory import (
    Inventory,
    InventoryBlock,
    LegacyFact,
    LegacyFactKind,
    inventory_workspace,
)

__all__ = [
    "AdapterPlan",
    "HandoffAdapter",
    "HandoffBlocked",
    "HandoffReport",
    "HandoffStatus",
    "Inventory",
    "InventoryBlock",
    "LegacyFact",
    "LegacyFactKind",
    "TargetReceipt",
    "apply_handoff",
    "inventory_workspace",
    "preflight_handoff",
]
