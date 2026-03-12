"""Tests for JobStore persistence."""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from agent.scheduler import JobStore, ScheduledJob
from tests.conftest import make_job


class TestJobStoreLoadSave:
    def test_load_empty_when_file_missing(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        assert store.load() == []

    def test_save_and_load_roundtrip(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        job = make_job(name="test-job")
        store.save({job.id: job})

        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].id == job.id
        assert loaded[0].name == "test-job"
        assert loaded[0].channel == job.channel
        assert loaded[0].chat_id == job.chat_id

    def test_fire_at_preserved_with_timezone(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        fire_at = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
        job = make_job(fire_at=fire_at)
        store.save({job.id: job})

        loaded = store.load()
        assert loaded[0].fire_at == fire_at
        assert loaded[0].fire_at.tzinfo is not None

    def test_multiple_jobs(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        jobs = {j.id: j for j in [make_job(name=f"job-{i}") for i in range(3)]}
        store.save(jobs)

        loaded = store.load()
        assert len(loaded) == 3
        names = {j.name for j in loaded}
        assert names == {"job-0", "job-1", "job-2"}

    def test_update_run_count_persisted(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        job = make_job()
        job.run_count = 5
        store.save({job.id: job})

        loaded = store.load()
        assert loaded[0].run_count == 5

    def test_delete_job(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        jobs = {j.id: j for j in [make_job(name=f"job-{i}") for i in range(2)]}
        store.save(jobs)

        # Delete one
        first_id = list(jobs.keys())[0]
        del jobs[first_id]
        store.save(jobs)

        loaded = store.load()
        assert len(loaded) == 1

    def test_survives_restart(self, tmp_path):
        """Data persists after creating a new JobStore instance."""
        path = tmp_path / "jobs.json"
        job = make_job(name="persistent")

        store1 = JobStore(path)
        store1.save({job.id: job})

        # New instance, simulating restart
        store2 = JobStore(path)
        loaded = store2.load()
        assert len(loaded) == 1
        assert loaded[0].name == "persistent"

    def test_all_fields_roundtrip(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        job = ScheduledJob(
            trigger="every",
            tier="soft",
            fire_at=datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc),
            channel="telegram",
            chat_id="456",
            interval_seconds=3600,
            cron_expr=None,
            message=None,
            prompt="查询天气",
            name="weather",
            timezone="Asia/Shanghai",
            run_count=3,
            enabled=True,
        )
        store.save({job.id: job})
        loaded = store.load()[0]

        assert loaded.trigger == "every"
        assert loaded.tier == "soft"
        assert loaded.interval_seconds == 3600
        assert loaded.prompt == "查询天气"
        assert loaded.timezone == "Asia/Shanghai"
        assert loaded.run_count == 3
