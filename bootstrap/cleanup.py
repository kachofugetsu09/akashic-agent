from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_cleanup_steps(
    *steps: tuple[str, Callable[[], Awaitable[None]]],
) -> None:
    """逐项执行已取得资源的清理动作，并重新抛出首个失败。"""

    first_error: BaseException | None = None
    for name, step in steps:
        error: BaseException | None
        try:
            cleanup_task = asyncio.ensure_future(step())
        except asyncio.CancelledError as exc:
            error = exc
        except Exception as exc:
            error = exc
        else:
            error = await _wait_for_cleanup(name, cleanup_task)
        if error is None:
            continue
        if first_error is None:
            first_error = error
        logger.warning("cleanup step failed: %s: %s", name, error)
    if first_error is not None:
        raise first_error


async def _wait_for_cleanup(
    name: str,
    cleanup_task: asyncio.Future[None],
) -> BaseException | None:
    """调用方取消时仍等待当前清理动作完成。"""

    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError as exc:
        try:
            await cleanup_task
        except asyncio.CancelledError as cleanup_error:
            return cleanup_error
        except Exception as cleanup_error:
            return cleanup_error
        logger.debug("caller cancellation deferred until cleanup completed: %s", name)
        return exc
    except Exception as exc:
        return exc
    return None
