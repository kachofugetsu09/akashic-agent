"""旧 workloads 归档使用的执行 DTO 迁移桥。"""

from agent.plugin_composition.execution import (
    WorkloadEndpoint,
    WorkloadLease,
    WorkloadMode,
    WorkloadStartReceipt,
    WorkloadStartRequest,
    WorkloadStopReceipt,
    workload_spec_digest,
)

__all__ = [
    "WorkloadEndpoint",
    "WorkloadLease",
    "WorkloadMode",
    "WorkloadStartReceipt",
    "WorkloadStartRequest",
    "WorkloadStopReceipt",
    "workload_spec_digest",
]
