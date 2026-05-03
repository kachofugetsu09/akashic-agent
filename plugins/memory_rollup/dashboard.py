from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.memory import MemoryStore
from memory2.store import MemoryStore2

CandidateTag = Literal["identity", "preference"]

_MAX_SOURCE_ITEMS = 180


# ── Pydantic Models ────────────────────────────────────────────────


class CommitItem(BaseModel):
    id: str
    tag: CandidateTag | None = None
    content: str | None = None


class CommitPayload(BaseModel):
    items: list[CommitItem]


class CandidateActionPayload(BaseModel):
    id: str


class GeneratePayload(BaseModel):
    limit: int = _MAX_SOURCE_ITEMS


# ── MemoryRollupReader ─────────────────────────────────────────────


class MemoryRollupReader:
    """从 memory2 的 profile/preference 中生成人类可确认的长期记忆候选。"""

    def __init__(self, plugin_dir: Path, workspace: Path) -> None:
        self.plugin_dir = plugin_dir
        self.workspace = workspace
        self.cache_path = workspace / "memory" / "memory_rollup_candidates.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────

    def overview(self) -> dict[str, Any]:
        """侧栏概览：各类型 active 数 + 候选数 + pending 预览。"""
        memory = self._memory2()
        try:
            # 1. 统计各 memory_type 的 active 条目数。
            counts = {
                item_type: memory.list_items_for_dashboard(
                    memory_type=item_type,
                    status="active",
                    page=1,
                    page_size=1,
                )[1]
                for item_type in ("profile", "preference")
            }
        finally:
            memory.close()
        # 2. 读取当前缓存候选数。
        cached = self._read_cached()
        return {
            "active_counts": counts,
            "candidate_count": len(cached),
            "pending_preview": self.pending_preview(),
        }

    def generate(self, limit: int = _MAX_SOURCE_ITEMS) -> list[dict[str, Any]]:
        """从 memory2 读取 source items，生成候选列表并写入缓存。"""
        # 1. 加载未被处理的 source items。
        source_items = self._load_source_items(limit=max(1, min(limit, 400)))
        # 2. 读当前 MEMORY.md 用于 overlap 检测。
        memory_text = MemoryStore(self.workspace).read_long_term()
        # 3. 按 memory_type + 原文分组。
        groups = _group_items(source_items)
        # 4. 每组构建一个候选对象。
        candidates = [
            _build_candidate(group, memory_text)
            for group in groups
            if group
        ]
        # 5. 排序：recommended 在前，同类内按 score 降序。
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                -int(item["reinforcement"]),
                -int(item["emotional_weight"]),
                str(item["updated_at"]),
            )
        )
        # 6. 写入缓存。
        _ = self.cache_path.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return candidates

    def candidates(self) -> list[dict[str, Any]]:
        """返回缓存候选；无缓存时自动触发 generate。"""
        cached = self._read_cached()
        if cached:
            return cached
        return self.generate()

    def commit(self, items: list[CommitItem]) -> dict[str, Any]:
        """将用户确认的候选写入 PENDING.md（带 - [tag] 格式）。"""
        cached = {str(item.get("id")): item for item in self.candidates()}
        pending_store = MemoryStore(self.workspace)
        appended = 0
        skipped = 0
        committed: list[dict[str, str]] = []

        for raw_item in items:
            # 1. 查找候选是否存在。
            candidate = cached.get(raw_item.id)
            if candidate is None:
                skipped += 1
                continue
            # 2. 确定 tag 和内容（优先使用用户编辑后的值）。
            tag = raw_item.tag or cast(CandidateTag, candidate.get("tag"))
            if tag not in {"identity", "preference"}:
                skipped += 1
                continue
            content = (raw_item.content or str(candidate.get("content") or "")).strip()
            if not content:
                skipped += 1
                continue
            # 3. 写入 PENDING.md（去重追加）。
            line = f"- [{tag}] {content}"
            ok = pending_store.append_pending_once(
                line,
                source_ref=f"memory_rollup:{raw_item.id}",
                kind="candidate",
            )
            if ok:
                appended += 1
                committed.append({"id": raw_item.id, "line": line})
            # 4. 标记 source items 并从缓存移除。
            self._mark_sources(candidate, raw_item.id, action="committed")
            self._remove_cached_candidate(raw_item.id)
            if not ok:
                skipped += 1

        return {
            "appended_count": appended,
            "skipped_count": skipped,
            "committed": committed,
            "pending_preview": pending_store.read_pending()[-4000:],
        }

    def ignore(self, candidate_id: str) -> dict[str, Any]:
        """标记候选为 ignored，后续不再为该 source 生成候选。"""
        candidate = self._candidate_by_id(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        self._mark_sources(candidate, candidate_id, action="ignored")
        self._remove_cached_candidate(candidate_id)
        return {"ignored": True, "id": candidate_id}

    def delete_sources(self, candidate_id: str) -> dict[str, Any]:
        """删除候选对应的 source memory items（不写入 PENDING）。"""
        candidate = self._candidate_by_id(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        source_ids = _candidate_source_ids(candidate)
        memory = self._memory2()
        try:
            deleted_count = memory.delete_items_batch(source_ids)
        finally:
            memory.close()
        self._remove_cached_candidate(candidate_id)
        return {"deleted_count": deleted_count, "id": candidate_id}

    def pending_preview(self) -> str:
        """返回 PENDING.md 末尾 4000 字符预览。"""
        return MemoryStore(self.workspace).read_pending()[-4000:]

    # ── Internal helpers ───────────────────────────────────────────

    def _memory2(self) -> MemoryStore2:
        return MemoryStore2(self.workspace / "memory" / "memory2.db")

    def _load_source_items(self, *, limit: int) -> list[dict[str, Any]]:
        """从 memory2 拉取未被 _rollup 标记的 active profile/preference。"""
        memory = self._memory2()
        try:
            items: list[dict[str, Any]] = []
            per_type = max(1, limit // 2)
            for item_type in ("preference", "profile"):
                # 1. 按 reinforcement 降序分页拉取。
                rows, _ = memory.list_items_for_dashboard(
                    memory_type=item_type,
                    status="active",
                    page=1,
                    page_size=per_type,
                    sort_by="reinforcement",
                    sort_order="desc",
                )
                for row in cast(list[dict[str, Any]], rows):
                    # 2. 取详情，跳过已带 _rollup 标记的。
                    detail = memory.get_item_for_dashboard(str(row.get("id") or ""))
                    if detail is None:
                        continue
                    extra_raw = detail.get("extra_json")
                    extra = cast(dict[str, object], extra_raw) if isinstance(extra_raw, dict) else {}
                    if extra.get("_rollup"):
                        continue
                    items.append({**row, "extra_json": extra})
            return items
        finally:
            memory.close()

    def _read_cached(self) -> list[dict[str, Any]]:
        """读取缓存 JSON，容错解析异常。"""
        if not self.cache_path.exists():
            return []
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list):
            return []
        return [
            cast(dict[str, Any], item)
            for item in cast(list[object], value)
            if isinstance(item, dict)
        ]

    def _candidate_by_id(self, candidate_id: str) -> dict[str, Any] | None:
        """按 id 查找缓存中的候选。"""
        for candidate in self.candidates():
            if str(candidate.get("id") or "") == candidate_id:
                return candidate
        return None

    def _mark_sources(
        self,
        candidate: dict[str, Any],
        candidate_id: str,
        *,
        action: Literal["committed", "ignored"],
    ) -> None:
        """在 memory2 的 extra_json._rollup 中标记处理状态，防止重复生成。"""
        source_ids = _candidate_source_ids(candidate)
        if not source_ids:
            return
        memory = self._memory2()
        acted_at = datetime.now(timezone.utc).isoformat()
        try:
            for source_id in source_ids:
                detail = memory.get_item_for_dashboard(source_id)
                if detail is None:
                    continue
                extra_raw = detail.get("extra_json")
                extra = dict(cast(dict[str, object], extra_raw)) if isinstance(extra_raw, dict) else {}
                extra["_rollup"] = {
                    "candidate_id": candidate_id,
                    "action": action,
                    "acted_at": acted_at,
                    "pending_source_ref": f"memory_rollup:{candidate_id}",
                }
                _ = memory.update_item_for_dashboard(source_id, extra_json=extra)
        finally:
            memory.close()

    def _remove_cached_candidate(self, candidate_id: str) -> None:
        """从缓存 JSON 中移除指定候选。"""
        cached = [
            item
            for item in self._read_cached()
            if str(item.get("id") or "") != candidate_id
        ]
        _ = self.cache_path.write_text(
            json.dumps(cached, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ── API Registration ───────────────────────────────────────────────


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> None:
    reader = MemoryRollupReader(plugin_dir, workspace)

    @app.get("/api/dashboard/memory-rollup/overview")
    def get_overview() -> dict[str, Any]:
        return reader.overview()

    @app.get("/api/dashboard/memory-rollup/candidates")
    def list_candidates() -> dict[str, Any]:
        items = reader.candidates()
        return {"items": items, "total": len(items)}

    @app.post("/api/dashboard/memory-rollup/generate")
    def generate_candidates(payload: GeneratePayload | None = None) -> dict[str, Any]:
        items = reader.generate(limit=payload.limit if payload is not None else _MAX_SOURCE_ITEMS)
        return {"items": items, "total": len(items)}

    @app.post("/api/dashboard/memory-rollup/commit")
    def commit_candidates(payload: CommitPayload) -> dict[str, Any]:
        if not payload.items:
            raise HTTPException(status_code=400, detail="请选择至少一条候选")
        return reader.commit(payload.items)

    @app.post("/api/dashboard/memory-rollup/ignore")
    def ignore_candidate(payload: CandidateActionPayload) -> dict[str, Any]:
        try:
            return reader.ignore(payload.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="候选不存在") from exc

    @app.post("/api/dashboard/memory-rollup/delete-sources")
    def delete_candidate_sources(payload: CandidateActionPayload) -> dict[str, Any]:
        try:
            return reader.delete_sources(payload.id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="候选不存在") from exc

    @app.get("/api/dashboard/memory-rollup/pending-preview")
    def get_pending_preview() -> dict[str, Any]:
        return {"pending_preview": reader.pending_preview()}


# ── 分组 ───────────────────────────────────────────────────────────


def _group_items(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """将 source items 按 memory_type 和清洗后的原文分组。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        summary = str(item.get("summary") or "")
        if not summary.strip():
            continue
        key = f"{item.get('memory_type')}:{_normalize_for_overlap(summary)}"
        groups.setdefault(key, []).append(item)
    return list(groups.values())


# ── 候选构建 ───────────────────────────────────────────────────────


def _build_candidate(group: list[dict[str, Any]], memory_text: str) -> dict[str, Any]:
    """对一组同原文的 source items 构建一个候选对象。"""
    first = group[0]
    memory_type = str(first.get("memory_type") or "")
    source_summaries = [str(item.get("summary") or "").strip() for item in group]
    source_ids = [str(item.get("id") or "") for item in group]
    tag = _candidate_tag(memory_type)
    content = _candidate_content(tag, source_summaries)
    metrics = _candidate_metrics(group)
    existing_overlap = _has_existing_overlap(memory_text, source_summaries)
    return {
        "id": _candidate_id(source_ids, content),
        "tag": tag,
        "title": _title(tag),
        "content": content,
        "score": metrics["score"],
        "reinforcement": metrics["reinforcement"],
        "emotional_weight": metrics["emotional_weight"],
        "updated_at": metrics["updated_at"],
        "evidence_count": len(group),
        "source_ids": source_ids,
        "source_summaries": source_summaries[:8],
        "existing_overlap": existing_overlap,
    }


def _candidate_tag(memory_type: str) -> CandidateTag:
    """确定候选 tag：只信任 memory2 原始类型。"""
    return "identity" if memory_type == "profile" else "preference"


def _title(tag: CandidateTag) -> str:
    """候选标题：只区分身份和偏好。"""
    return "身份信息" if tag == "identity" else "长期偏好"


# ── 内容生成 ───────────────────────────────────────────────────────


def _candidate_content(
    tag: CandidateTag,
    summaries: list[str],
) -> str:
    """从真实 source summaries 生成候选正文。"""
    # 1. 清洗所有 summary。
    cleaned = [_clean_summary(s) for s in summaries]
    # 2. 取最长的一条为主干。
    longest = max(cleaned, key=len)
    # 3. identity tag 补"用户"前缀使句式统一。
    if tag == "identity" and not longest.startswith("用户"):
        longest = f"用户的稳定身份信息是{longest}"
    return longest


def _candidate_metrics(group: list[dict[str, Any]]) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    best_score = -1.0
    best_item = group[0]
    for item in group:
        reinforcement = _int_field(item, "reinforcement")
        emotional_weight = _int_field(item, "emotional_weight")
        updated_at = _parse_updated_at(item.get("updated_at"))
        score = _recall_hotness_score(
            reinforcement,
            updated_at,
            now,
            emotional_weight=emotional_weight,
        )
        if score > best_score:
            best_score = score
            best_item = item
    return {
        "score": round(max(best_score, 0.0) * 100, 2),
        "reinforcement": _int_field(best_item, "reinforcement"),
        "emotional_weight": _int_field(best_item, "emotional_weight"),
        "updated_at": str(best_item.get("updated_at") or ""),
    }


def _int_field(item: dict[str, Any], key: str) -> int:
    value = item.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str | float):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _parse_updated_at(value: object) -> datetime:
    text = str(value or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _recall_hotness_score(
    reinforcement: int,
    updated_at: datetime,
    now: datetime,
    half_life_days: float = 14.0,
    emotional_weight: int = 0,
) -> float:
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    effective_half_life = max(
        half_life_days * (1.0 + 0.5 * max(0, min(10, emotional_weight)) / 10.0),
        0.1,
    )
    freq = 1.0 / (1.0 + math.exp(-math.log1p(max(0, reinforcement))))
    age_days = max((now - updated_at).total_seconds() / 86400.0, 0.0)
    recency = math.exp(-math.log(2) / effective_half_life * age_days)
    return freq * recency


# ── 重叠检测 ───────────────────────────────────────────────────────


def _has_existing_overlap(memory_text: str, summaries: list[str]) -> bool:
    """检查候选内容是否与 MEMORY.md 已有内容重叠。"""
    normalized_memory = _normalize_for_overlap(memory_text)
    if not normalized_memory:
        return False
    for summary in summaries:
        normalized = _normalize_for_overlap(summary)
        if len(normalized) >= 12 and normalized[:24] in normalized_memory:
            return True
    return False


def _normalize_for_overlap(text: str) -> str:
    """去空格、转小写，用于模糊匹配。"""
    return "".join(text.lower().split())


# ── 文本清洗 ───────────────────────────────────────────────────────


def _clean_summary(text: str) -> str:
    """去多余空白和标点前缀后缀。"""
    return " ".join(text.split()).strip(" -。；;")


# ── ID 生成 ────────────────────────────────────────────────────────


def _candidate_id(source_ids: list[str], content: str) -> str:
    """用 source_ids + content 的 SHA1 前 16 位作为候选 ID。"""
    raw = "\n".join(source_ids) + "\n" + content
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _candidate_source_ids(candidate: dict[str, Any]) -> list[str]:
    """从候选对象中安全提取 source_ids。"""
    source_ids_raw = candidate.get("source_ids")
    if not isinstance(source_ids_raw, list):
        return []
    return [
        source_id
        for source_id in [
            str(source_id or "").strip()
            for source_id in cast(list[object], source_ids_raw)
        ]
        if source_id
    ]
