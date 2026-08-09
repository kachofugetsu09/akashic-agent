from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from agent.control.models import TurnStatus
from agent.plugins.install import PluginInstallResult
from agent.plugins.manager import PluginManager
from infra.persistence.json_store import atomic_save_json, load_json

logger = logging.getLogger(__name__)

PluginOperationKind = Literal["install", "uninstall"]


@dataclass
class PendingPluginOperation:
    owner_turn_id: str
    plugin_id: str
    kind: PluginOperationKind
    generation_id: str = ""
    source_revision: str = ""
    reload_tx_id: str = ""
    validated: bool = False
    sealed: bool = False


class TurnPluginRollout:
    """Own turn-local plugin operations and resolve them after terminal release."""

    def __init__(
        self,
        manager: PluginManager,
        *,
        workspace: Path,
        uninstall: Callable[[str], Awaitable[dict[str, object]]],
    ) -> None:
        self._manager = manager
        self._uninstall = uninstall
        self._fact_path = Path(workspace) / "runtime" / "plugin-rollout-fact.json"
        self._pending: PendingPluginOperation | None = None
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._resolution_task: asyncio.Task[None] | None = None

    async def install(
        self,
        owner_turn_id: str,
        *,
        source: str,
        marketplace: str,
        ref_name: str,
        sparse_paths: list[str],
    ) -> tuple[PluginInstallResult, dict[str, object]]:
        """Stage one candidate and bind it to the calling active turn."""

        # 1. Turn identity is the authority boundary for revert and auto-commit.
        _require_owner_turn(owner_turn_id)
        async with self._lock:
            self._require_no_pending()
            result, status = await self._manager.install_candidate(
                source=source,
                marketplace=marketplace,
                ref_name=ref_name,
                sparse_paths=sparse_paths,
            )
            plugin_id = f"{result.plugin_name}@{result.marketplace}"
            if not result.staged_candidate:
                return result, status
            generation_id = str(status.get("candidate_generation_id") or "")
            reload_tx_id = str(status.get("candidate_reload_tx_id") or "")
            if not generation_id or not reload_tx_id:
                await self._manager.drop_candidate(plugin_id)
                raise RuntimeError("插件候选缺少 generation 或 reload transaction 身份")
            self._pending = PendingPluginOperation(
                owner_turn_id=owner_turn_id,
                plugin_id=plugin_id,
                kind="install",
                generation_id=generation_id,
                source_revision=result.source_revision,
                reload_tx_id=reload_tx_id,
            )
            if reload_tx_id:
                self._manager.annotate_reload(
                    reload_tx_id,
                    {
                        "event": "turn_operation_registered",
                        "owner_turn_id": owner_turn_id,
                        "operation": "install",
                    },
                )
            return result, status

    async def uninstall(self, owner_turn_id: str, plugin_id: str) -> dict[str, object]:
        """Register a reversible uninstall without touching manifest or code."""

        # 1. Registration changes no published state, so the current lease can finish.
        _require_owner_turn(owner_turn_id)
        normalized = plugin_id.strip()
        if not normalized:
            raise ValueError("缺少插件 ID")
        async with self._lock:
            self._require_no_pending()
            self._manager.require_installed_plugin(normalized)
            self._pending = PendingPluginOperation(
                owner_turn_id=owner_turn_id,
                plugin_id=normalized,
                kind="uninstall",
            )
        return {
            "pluginId": normalized,
            "operation": "uninstall",
            "publicationState": "pending_turn_end",
        }

    async def revert(self, owner_turn_id: str) -> dict[str, object]:
        """Cancel this turn's latest unsealed plugin operation."""

        # 1. Ownership and seal checks make cross-turn rollback unreachable.
        _require_owner_turn(owner_turn_id)
        async with self._lock:
            pending = self._require_owned_pending(owner_turn_id)
            if pending.sealed:
                raise RuntimeError("当前 turn 的插件操作已经封口，不能 revert")
            self._pending = None

        # 2. Candidate disposal may wait for child leases, so do it outside the lock.
        if pending.kind == "install":
            result = await self._manager.drop_candidate(pending.plugin_id)
            return {
                **result,
                "operation": "install",
                "reverted": True,
            }
        return {
            "plugin_id": pending.plugin_id,
            "operation": "uninstall",
            "reverted": True,
            "publication_state": "unchanged",
        }

    def child_binding(
        self,
        owner_turn_id: str,
        *,
        attached: bool,
    ) -> dict[str, str] | None:
        """Freeze the current candidate identity for one attached child."""

        pending = self._pending
        if (
            not attached
            or pending is None
            or pending.sealed
            or pending.kind != "install"
            or pending.owner_turn_id != owner_turn_id
        ):
            return None
        return {
            "runtime": "latest",
            "ownerTurnId": pending.owner_turn_id,
            "pluginId": pending.plugin_id,
            "generationId": pending.generation_id,
            "sourceRevision": pending.source_revision,
        }

    def turn_terminal(
        self,
        turn_id: str,
        status: TurnStatus,
        metadata: dict[str, object],
    ) -> None:
        """Record child evidence or seal a parent operation after lease release."""

        # 1. A causally bound child can validate only its frozen generation.
        owner_turn_id = str(metadata.get("_pluginRolloutOwnerTurnId") or "")
        if owner_turn_id:
            self._record_child_terminal(owner_turn_id, status, metadata)
            return

        # 2. The parent terminal seals once and hands slow work to a background task.
        pending = self._pending
        if pending is None or pending.owner_turn_id != turn_id or pending.sealed:
            return
        pending.sealed = True
        task = asyncio.create_task(
            self._resolve_parent(pending, status),
            name=f"plugin-turn-rollout:{turn_id}",
        )
        self._resolution_task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_for_turn_boundary(self) -> None:
        """Keep the next admitted turn behind a sealed rollout resolution."""

        task = self._resolution_task
        if task is not None:
            await asyncio.shield(task)

    async def shutdown(self) -> None:
        """Finish or cancel owned background resolutions during runtime shutdown."""

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def consume_fact(self) -> str:
        """Consume one runtime-owned rollout fact for the next user turn."""

        raw = load_json(
            self._fact_path,
            default=None,
            domain="plugin_rollout_fact",
        )
        if raw is None:
            return ""
        if not isinstance(raw, dict) or not isinstance(raw.get("message"), str):
            raise RuntimeError(f"插件 rollout fact 结构无效: {self._fact_path}")
        message = str(raw["message"])
        self._fact_path.unlink()
        return message

    def _record_child_terminal(
        self,
        owner_turn_id: str,
        status: TurnStatus,
        metadata: dict[str, object],
    ) -> None:
        pending = self._pending
        if (
            pending is None
            or pending.sealed
            or pending.kind != "install"
            or pending.owner_turn_id != owner_turn_id
        ):
            return
        generation_id = str(metadata.get("_pluginRolloutGenerationId") or "")
        source_revision = str(metadata.get("_pluginRolloutSourceRevision") or "")
        if (
            generation_id != pending.generation_id
            or source_revision != pending.source_revision
        ):
            logger.error(
                "plugin candidate child identity mismatch owner=%s expected=%s/%s actual=%s/%s",
                owner_turn_id,
                pending.generation_id,
                pending.source_revision,
                generation_id,
                source_revision,
            )
            return
        pending.validated = pending.validated or status is TurnStatus.COMPLETED
        if pending.reload_tx_id:
            self._manager.annotate_reload(
                pending.reload_tx_id,
                {
                    "event": "candidate_child_terminal",
                    "owner_turn_id": owner_turn_id,
                    "child_turn_id": str(metadata.get("turnId") or ""),
                    "status": status.value,
                    "identity_match": True,
                },
            )

    async def _resolve_parent(
        self,
        pending: PendingPluginOperation,
        status: TurnStatus,
    ) -> None:
        try:
            if status is not TurnStatus.COMPLETED:
                await self._cancel(pending, f"parent turn {status.value}")
            elif pending.kind == "install" and not pending.validated:
                await self._cancel(pending, "没有完成 attached programmatic 验证")
            elif pending.kind == "install":
                await self._manager.switch_ready(pending.plugin_id)
                self._write_fact(
                    f"{pending.plugin_id} 更新已经成功提交；本 turn 已加载新版本。"
                )
            else:
                await self._uninstall(pending.plugin_id)
                self._write_fact(
                    f"{pending.plugin_id} 已卸载；已安装代码已清理，plugin-data 保留。"
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception(
                "turn-owned plugin operation failed owner=%s plugin=%s kind=%s",
                pending.owner_turn_id,
                pending.plugin_id,
                pending.kind,
            )
            self._write_fact(
                f"{pending.plugin_id} 的 {pending.kind} 没有完成：{error}。"
                "Core 未把新结果标记为成功；请根据错误确认旧 endpoint 后再重试。"
            )
            if pending.kind == "install":
                try:
                    candidate_status = self._manager.candidate_status(
                        pending.plugin_id
                    )
                    if candidate_status.get("candidate_state") in {
                        "latest_ready",
                        "promoting",
                    }:
                        await self._manager.drop_candidate(pending.plugin_id)
                except Exception:
                    logger.exception(
                        "failed plugin rollout candidate cleanup owner=%s plugin=%s",
                        pending.owner_turn_id,
                        pending.plugin_id,
                    )
        finally:
            if self._pending is pending:
                self._pending = None
            if self._resolution_task is asyncio.current_task():
                self._resolution_task = None

    async def _cancel(self, pending: PendingPluginOperation, reason: str) -> None:
        if pending.kind == "install":
            await self._manager.drop_candidate(pending.plugin_id)
            self._write_fact(
                f"{pending.plugin_id} 没有切换：{reason}。原版本保持可用。"
            )
        logger.info(
            "turn-owned plugin operation cancelled owner=%s plugin=%s reason=%s",
            pending.owner_turn_id,
            pending.plugin_id,
            reason,
        )

    def _require_no_pending(self) -> None:
        pending = self._pending
        if pending is not None:
            raise RuntimeError(
                "已有 turn 持有插件操作: "
                f"turn={pending.owner_turn_id} plugin={pending.plugin_id} "
                f"operation={pending.kind}"
            )

    def _require_owned_pending(self, owner_turn_id: str) -> PendingPluginOperation:
        pending = self._pending
        if pending is None or pending.owner_turn_id != owner_turn_id:
            raise RuntimeError(
                "无法执行 revert：当前 turn 没有尚未提交的插件操作。"
                "revert 只能撤销本 turn 最近一次 install 或 uninstall，"
                "不能回滚上一 turn。"
            )
        return pending

    def _write_fact(self, message: str) -> None:
        atomic_save_json(
            self._fact_path,
            {"message": message},
            ensure_ascii=False,
            domain="plugin_rollout_fact",
        )


def _require_owner_turn(owner_turn_id: str) -> None:
    if not owner_turn_id.strip():
        raise ValueError("插件操作必须由当前 active turn 发起")
