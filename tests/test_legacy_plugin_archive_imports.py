"""生产 stable 选择仍固定的归档 import 兼容测试。"""

from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG, build_runtime_catalog
from agent.plugin_composition.execution import WorkloadStartRequest
from plugins.stable_view.plugin import RUNTIME_CATALOG as STABLE_VIEW_CATALOG
from agent.workloads.model import WorkloadStartRequest as ArchivedWorkloadStartRequest


def test_stable_view_archive_uses_current_catalog_projection() -> None:
    assert STABLE_VIEW_CATALOG is RUNTIME_CATALOG
    assert build_runtime_catalog.__name__ == "build_runtime_catalog"


def test_workloads_archive_uses_current_execution_dto() -> None:
    assert ArchivedWorkloadStartRequest is WorkloadStartRequest
