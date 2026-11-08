"""Read configured information once before persisting a Wake request."""
import asyncio
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from agent.plugin_composition import Context
from plugins.ledger.contract import json_value
from plugins.userinformation.contract import USER_INFORMATION_READ
from .api import InformationQuery

logger = logging.getLogger(__name__)
# 单项快照上限：随每次 Wake 请求归档并进入模型输入，超限整项标为不可用并记日志。
SNAPSHOT_LIMIT = 8192


async def snapshot_information(ctx: Context, queries: tuple[InformationQuery, ...]) -> tuple[dict[str, object], ...]:
    reader = ctx.require(USER_INFORMATION_READ)

    async def read(query: InformationQuery) -> dict[str, object]:
        evidence: dict[str, object] = {"request": query.model_dump(mode="json"),
                                       "queried_at": datetime.now(UTC).isoformat()}
        tag = f"provider={query.provider} query={query.query}"
        # 1. 读取失败是证据而不是准入失败；取消照常传播
        try:
            async with asyncio.timeout(5):
                result = json_value(await reader.query(query.provider, query.query, query.arguments))
        except TimeoutError:
            logger.warning("wake.information.timeout %s", tag)
            return {**evidence, "status": "unavailable", "reason": "timeout"}
        except Exception:
            logger.warning("wake.information.failed %s", tag, exc_info=True)
            return {**evidence, "status": "unavailable", "reason": "query_failed"}
        # 2. 超限不截断（来源语义未知），整项标不可用并写明大小，提示改用 source_ids 等参数缩小
        size = len(json.dumps(result, ensure_ascii=False).encode())
        if size > SNAPSHOT_LIMIT:
            logger.warning("wake.information.too_large %s bytes=%d limit=%d", tag, size, SNAPSHOT_LIMIT)
            return {**evidence, "status": "unavailable", "reason": "size_limit", "bytes": size}
        # 3. 来源未注册、来源自报失败也记日志，避免配置写错后长期静默
        if isinstance(result, Mapping) and ("error" in result or result.get("status") in {"unavailable", "too_large"}):
            logger.warning("wake.information.unavailable %s detail=%s", tag, str(result.get("error") or result.get("status"))[:200])
        return {**evidence, "response": result}

    snapshot = tuple(await asyncio.gather(*(read(query) for query in queries)))
    if queries:
        ok = sum(1 for item in snapshot if isinstance(r := item.get("response"), Mapping)
                 and "error" not in r and r.get("status") not in {"unavailable", "too_large"})
        logger.info("wake.information.snapshot queries=%d ok=%d", len(queries), ok)
    return snapshot
