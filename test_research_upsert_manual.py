#!/usr/bin/env python3
"""验证 research_* 字段在同一 tick_id 的后续阶段能正确更新落库"""

import sqlite3
import tempfile
from pathlib import Path

from core.observe.db import open_db
from core.observe.events import ProactiveDecisionTrace
from core.observe.writer import _write_proactive_decision


def test_research_fields_upsert():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = open_db(db_path)

        # 第一次写入：gate_and_sense 阶段，无 research 数据
        trace1 = ProactiveDecisionTrace(
            tick_id="test_tick_001",
            session_key="test_session",
            stage="gate_and_sense",
        )
        _write_proactive_decision(conn, trace1, "2026-03-18T10:00:00Z")

        # 查询第一次写入结果
        row1 = conn.execute(
            """
            SELECT stage, research_status, research_rounds_used, research_tools_called,
                   research_evidence_count, research_reason, fact_claims_count
            FROM proactive_decisions WHERE tick_id = ?
            """,
            ("test_tick_001",),
        ).fetchone()
        print(f"第一次写入后: {row1}")
        assert row1[0] == "gate_and_sense"
        assert row1[1] is None  # research_status
        assert row1[2] is None  # research_rounds_used

        # 第二次写入：research 阶段，填充 research 数据
        trace2 = ProactiveDecisionTrace(
            tick_id="test_tick_001",
            session_key="test_session",
            stage="research",
            research_status="success",
            research_rounds_used=3,
            research_tools_called=["web_search", "web_fetch"],
            research_evidence_count=2,
            research_reason="",
            fact_claims_count=5,
        )
        _write_proactive_decision(conn, trace2, "2026-03-18T10:00:05Z")

        # 查询第二次写入结果
        row2 = conn.execute(
            """
            SELECT stage, research_status, research_rounds_used, research_tools_called,
                   research_evidence_count, research_reason, fact_claims_count
            FROM proactive_decisions WHERE tick_id = ?
            """,
            ("test_tick_001",),
        ).fetchone()
        print(f"第二次写入后: {row2}")
        assert row2[0] == "research"
        assert row2[1] == "success"
        assert row2[2] == 3
        assert row2[3] == '["web_search", "web_fetch"]'
        assert row2[4] == 2
        assert row2[5] == ""
        assert row2[6] == 5

        # 第三次写入：compose 阶段，research 数据应保留
        trace3 = ProactiveDecisionTrace(
            tick_id="test_tick_001",
            session_key="test_session",
            stage="compose",
        )
        _write_proactive_decision(conn, trace3, "2026-03-18T10:00:10Z")

        # 查询第三次写入结果
        row3 = conn.execute(
            """
            SELECT stage, research_status, research_rounds_used, research_tools_called,
                   research_evidence_count, research_reason, fact_claims_count
            FROM proactive_decisions WHERE tick_id = ?
            """,
            ("test_tick_001",),
        ).fetchone()
        print(f"第三次写入后: {row3}")
        assert row3[0] == "compose"
        # research 字段应保留第二次写入的值
        assert row3[1] == "success"
        assert row3[2] == 3
        assert row3[3] == '["web_search", "web_fetch"]'
        assert row3[4] == 2
        assert row3[5] == ""
        assert row3[6] == 5

        conn.close()
        print("✅ research_* 字段 upsert 测试通过")


if __name__ == "__main__":
    test_research_fields_upsert()
