"""PeerProcessManager：管理 peer agent 子进程的完整生命周期。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from core.net.http import HttpRequester, RequestBudget

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_S = 2.0
_SPAWN_POLL_INTERVAL_S = 1.0


@dataclass
class PeerProcessConfig:
    name: str
    base_url: str
    launcher: list[str]          # 拉起命令，如 ["uv", "run", "python", "-m", "app.a2a_server"]
    cwd: str | None = None       # 子进程工作目录，None 表示继承父进程
    health_path: str = "/health"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 10
    log_dir: str = "runtime/peer_agents"


@dataclass(frozen=True)
class PeerReady:
    """描述 ensure_ready 本次调用是否取得了新的进程 ownership。"""

    started_by_call: bool


class PeerProcessManager:
    """管理 peer agent 子进程的生命周期。"""

    def __init__(
        self,
        configs: list[PeerProcessConfig],
        requester: HttpRequester,
    ) -> None:
        self._validate_configs(configs)
        self._configs = {config.name: config for config in configs}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._requester = requester
        self._locks: dict[str, asyncio.Lock] = {
            config.name: asyncio.Lock() for config in configs
        }
        self._shutting_down = False

    @staticmethod
    def _validate_configs(configs: list[PeerProcessConfig]) -> None:
        """在构造时校验进程 ownership key 和启动前置条件。"""

        # 1. 配置名必须唯一且能安全映射到日志文件和 lock
        names: set[str] = set()
        for config in configs:
            if not isinstance(config.name, str) or not config.name.strip():
                raise ValueError("peer agent name 必须是非空字符串")
            if config.name in names:
                raise ValueError(f"重复 peer agent config name：{config.name!r}")
            if Path(config.name).name != config.name:
                raise ValueError(f"peer agent name 不能包含路径分隔符：{config.name!r}")
            names.add(config.name)

        # 2. 启动命令、工作目录和回收时限必须可执行
        for config in configs:
            if not isinstance(config.launcher, list) or not config.launcher:
                raise ValueError(f"peer agent {config.name!r} launcher 不能为空")
            if any(not isinstance(arg, str) or not arg.strip() for arg in config.launcher):
                raise ValueError(f"peer agent {config.name!r} launcher 参数必须是非空字符串")
            if not isinstance(config.base_url, str) or not config.base_url.strip():
                raise ValueError(f"peer agent {config.name!r} base_url 不能为空")
            if config.cwd is not None and not Path(config.cwd).is_dir():
                raise ValueError(f"peer agent {config.name!r} cwd 不是目录：{config.cwd!r}")
            if config.startup_timeout_s <= 0 or config.shutdown_timeout_s <= 0:
                raise ValueError(f"peer agent {config.name!r} 启动和关闭超时必须大于 0")
            if not isinstance(config.log_dir, str) or not config.log_dir.strip():
                raise ValueError(f"peer agent {config.name!r} log_dir 不能为空")
            log_dir = Path(config.log_dir)
            if log_dir.exists() and not log_dir.is_dir():
                raise ValueError(f"peer agent {config.name!r} log_dir 不是目录：{config.log_dir!r}")

    async def ensure_ready(self, name: str) -> PeerReady:
        """确保指定 agent 已启动且通过健康检查，并报告本次是否冷启动。"""

        cfg = self._configs.get(name)
        if cfg is None:
            raise ValueError(f"未知 peer agent: {name!r}")
        if self._shutting_down:
            raise RuntimeError("peer agent process manager 已进入关闭阶段")

        async with self._locks[name]:
            if self._shutting_down:
                raise RuntimeError("peer agent process manager 已进入关闭阶段")
            # 1. 先收回已退出但仍登记的旧 ownership，避免覆盖泄漏
            existing = self._procs.get(name)
            if existing is not None and existing.returncode is not None:
                await self._release_owned(name, existing, cfg.shutdown_timeout_s)

            # 2. 健康的外部进程或已管理进程都可以复用
            if await self._is_healthy(cfg):
                logger.debug("[PeerProcess] %s 已在线", name)
                return PeerReady(started_by_call=False)

            # 3. 健康检查失败时先释放旧进程，再取得新进程 ownership
            existing = self._procs.get(name)
            if existing is not None:
                await self._release_owned(name, existing, cfg.shutdown_timeout_s)
            logger.info("[PeerProcess] %s 未运行，开始冷启动", name)
            await self._spawn(cfg)
            logger.info("[PeerProcess] %s 启动成功", name)
            return PeerReady(started_by_call=True)

    async def terminate(self, name: str) -> None:
        """销毁指定 peer agent 进程；失败时保留 ownership 供重试。"""

        cfg = self._configs.get(name)
        if cfg is None:
            raise ValueError(f"未知 peer agent: {name!r}")
        async with self._locks[name]:
            proc = self._procs.get(name)
            if proc is None:
                return
            logger.info("[PeerProcess] 终止 %s (pid=%s)", name, proc.pid)
            await self._release_owned(name, proc, cfg.shutdown_timeout_s)

    async def shutdown_all(self) -> None:
        """尽量回收全部子进程，并显式保留所有清理失败。"""

        self._shutting_down = True
        names = list(self._configs.keys())
        if names:
            logger.info("[PeerProcess] 关闭所有子进程: %s", names)
        results = await asyncio.gather(
            *(self.terminate(name) for name in names),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("peer agent shutdown 失败", errors)

    # ── 内部方法 ──────────────────────────────────────────────

    async def _is_healthy(self, cfg: PeerProcessConfig) -> bool:
        """仅将连接、传输、超时和非 200 响应解释为未就绪。"""

        try:
            response = await self._requester.get(
                cfg.base_url.rstrip("/") + cfg.health_path,
                budget=RequestBudget(total_timeout_s=_HEALTH_TIMEOUT_S),
            )
        except httpx.UnsupportedProtocol:
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
            return False
        return response.status_code == 200

    async def _spawn(self, cfg: PeerProcessConfig) -> None:
        """启动子进程并在任意失败或取消时回收已取得的 ownership。"""

        # 1. 打开日志；创建或关闭失败都保留原始错误
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{cfg.name.replace(' ', '_')}.log"
        log_fp = log_path.open("ab")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cfg.launcher,
                stdout=log_fp,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cfg.cwd,
            )
        except BaseException as create_error:
            try:
                log_fp.close()
            except BaseException as close_error:
                errors: list[BaseException] = [create_error, close_error]
                try:
                    log_fp.close()
                except BaseException as retry_close_error:
                    errors.append(retry_close_error)
                raise BaseExceptionGroup(
                    f"{cfg.name} 创建进程和关闭日志均失败", errors
                ) from create_error
            raise
        self._procs[cfg.name] = proc
        # 子进程已经继承 stdout fd，父进程不再持有它的生命周期 ownership
        try:
            log_fp.close()
        except BaseException as close_error:
            errors: list[BaseException] = [close_error]
            try:
                await self._release_owned(cfg.name, proc, cfg.shutdown_timeout_s)
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
            try:
                log_fp.close()
            except BaseException as retry_close_error:
                errors.append(retry_close_error)
            if len(errors) > 1:
                raise BaseExceptionGroup(
                    f"{cfg.name} 关闭日志后清理失败", errors
                ) from close_error
            raise close_error

        # 2. 健康等待由本 manager 持有，提前退出、超时和调用方取消都走同一回收路径
        try:
            await self._wait_until_healthy(cfg, proc)
        except BaseException as startup_error:
            try:
                await self._release_owned(cfg.name, proc, cfg.shutdown_timeout_s)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    f"{cfg.name} 启动失败且清理失败",
                    [startup_error, cleanup_error],
                ) from startup_error
            raise

    async def _wait_until_healthy(
        self,
        cfg: PeerProcessConfig,
        proc: asyncio.subprocess.Process,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cfg.startup_timeout_s
        while True:
            if proc.returncode is not None:
                raise RuntimeError(f"{cfg.name} 启动后立即退出 (rc={proc.returncode})")
            if await self._is_healthy(cfg):
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError(f"{cfg.name} 启动超时（{cfg.startup_timeout_s}s）")
            await asyncio.sleep(min(_SPAWN_POLL_INTERVAL_S, remaining))

    async def _release_owned(
        self,
        name: str,
        proc: asyncio.subprocess.Process,
        timeout_s: int,
    ) -> None:
        """等待进程终止并移除 manager ownership。"""

        # 1. 只有进程完成 wait 后才允许释放登记，失败时保留重试所需 ownership
        if self._procs.get(name) is not proc:
            raise RuntimeError(f"peer agent {name!r} 进程 ownership 不一致")
        await self._kill(proc, timeout_s)

        # 2. 进程已 wait 完成后释放 manager ownership
        del self._procs[name]

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process, timeout_s: float) -> None:
        """SIGTERM 后等待，超时则 SIGKILL 并再次 wait。"""

        if proc.returncode is not None:
            await proc.wait()
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=float(timeout_s))
        except asyncio.TimeoutError:
            logger.warning("[PeerProcess] SIGTERM 超时，强制 SIGKILL pid=%d", proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
