from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import threading

import pytest

from plugins.scheduler import migration, store as store_module
from plugins.scheduler.store import JobStore, fire_key
from tests.conftest import make_job


def legacy_file(path, *jobs):
    value = [JobStore(path).encode_job(job) for job in jobs]
    raw = json.dumps(value, ensure_ascii=False, indent=4).encode()
    path.write_bytes(raw)
    return raw


def test_migration_preserves_original_bytes_and_jobs_after_lost_ack(tmp_path, monkeypatch):
    path = tmp_path / "schedules.json"
    original = make_job(name="每日报告", trigger="every", interval_seconds=60)
    before = legacy_file(path, original)
    recovery = tmp_path / "backup"
    write = migration.atomic_write_text

    def write_then_fail(*args, **kwargs):
        write(*args, **kwargs)
        raise OSError("crash after replace")

    with monkeypatch.context() as patch:
        patch.setattr(migration, "atomic_write_text", write_then_fail)
        with pytest.raises(OSError, match="crash after replace"):
            migration.migrate(path, recovery)
    assert json.loads((recovery / "manifest.json").read_text())["status"] == "prepared"
    assert (recovery / "schedules.before.json").read_bytes() == before
    assert JobStore(path).load() == [original]
    migration.migrate(path, recovery)
    assert json.loads((recovery / "manifest.json").read_text())["status"] == "complete"
    assert JobStore(path).load() == [original]

    # 正常运行产生新事实后，重复迁移不能把文件覆盖回首次候选。
    new = make_job(name="新任务")
    _ = JobStore(path).add("new", new, "created")
    after_runtime = path.read_bytes()
    migration.migrate(path, recovery)
    assert path.read_bytes() == after_runtime
    assert (recovery / "schedules.before.json").read_bytes() == before


def test_read_never_upgrades_legacy_and_corruption_never_creates_backup(tmp_path):
    path = tmp_path / "schedules.json"
    before = legacy_file(path, make_job())
    with pytest.raises(ValueError, match="先迁移"):
        JobStore(path).load()
    assert path.read_bytes() == before
    assert tuple(tmp_path.iterdir()) == (path,)
    path.write_text('[{"id":"a","id":"b"}]')
    with pytest.raises(ValueError, match="重复"):
        migration.migrate(path, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_operation_result_survives_replace_failure_without_recreating_cancelled_job(tmp_path, monkeypatch):
    path = tmp_path / "schedules.json"
    store = JobStore(path)
    job = make_job()
    save = store_module.atomic_save_json

    def save_then_fail(*args, **kwargs):
        save(*args, **kwargs)
        raise OSError("ack lost")

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "atomic_save_json", save_then_fail)
        with pytest.raises(OSError, match="ack lost"):
            store.add("create", job, "original response")
    restarted = JobStore(path)
    operation = restarted.read().operations["create"]
    assert operation.job_ids == (job.id,)
    assert restarted.add("create", job, "unused response") == operation
    _ = restarted.cancel("cancel", (job.id,))
    assert store.add("create", job, "unused response") == operation
    assert store.load() == []
    with pytest.raises(ValueError, match="不同请求"):
        store.add("create", replace(job, message="changed"), "wrong")


def test_cancel_replay_only_affects_the_original_matches(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    original = make_job(name="report")
    _ = store.add("first", original, "created")
    cancelled = store.cancel("cancel", (original.id,))
    later = make_job(name="report")
    _ = store.add("later", later, "created")
    assert store.cancel("cancel", (original.id,)) == cancelled
    assert store.load() == [later]


def test_two_store_instances_serialize_their_candidate_writes(tmp_path, monkeypatch):
    path = tmp_path / "schedules.json"
    first, second = JobStore(path), JobStore(path)
    entered, second_started, release = threading.Event(), threading.Event(), threading.Event()
    save = store_module.atomic_save_json
    original, later = make_job(), make_job()

    def blocked_save(path, data, **kwargs):
        if "first" in data["operations"] and "second" not in data["operations"]:
            entered.set()
            assert release.wait(5)
        save(path, data, **kwargs)

    def second_add():
        second_started.set()
        return second.add("second", later, "second")

    monkeypatch.setattr(store_module, "atomic_save_json", blocked_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first.add, "first", original, "first")
        try:
            assert entered.wait(5)
            two = pool.submit(second_add)
            assert second_started.wait(5)
        finally:
            release.set()
        assert one.result().job_ids == (original.id,)
        assert two.result().job_ids == (later.id,)
    assert {job.id for job in first.load()} == {original.id, later.id}
    assert set(first.read().operations) == {"first", "second"}


def test_pending_fire_survives_misfire_recovery_and_settles_once(tmp_path):
    path = tmp_path / "schedules.json"
    store = JobStore(path)
    now = datetime(2026, 9, 6, tzinfo=UTC)
    job = make_job(trigger="every", interval_seconds=60, fire_at=now)
    _ = store.add("create", job, "created")
    fire = store.start_fire(job)
    assert fire is not None
    restarted = JobStore(path)
    later = now + timedelta(hours=1)
    state = restarted.recover(later)
    assert state.fires[fire.key] == fire
    assert state.jobs[job.id].fire_at == now
    assert restarted.start_fire(job) == fire
    restarted.settle(fire.key, "delivered", now=later)
    after = path.read_bytes()
    restarted.settle(fire.key, "delivered", now=later + timedelta(hours=1))
    assert path.read_bytes() == after
    state = restarted.read()
    assert state.jobs[job.id].run_count == 1
    assert state.jobs[job.id].fire_at == later + timedelta(seconds=60)
    assert state.fires[fire.key].status == "delivered"


def test_unstarted_misfires_preserve_one_shot_terminal_and_advance_periodic(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    now = datetime(2026, 9, 6, tzinfo=UTC)
    late = make_job(fire_at=now - timedelta(seconds=301))
    grace = make_job(fire_at=now - timedelta(seconds=300))
    periodic = make_job(trigger="every", interval_seconds=60, fire_at=now - timedelta(hours=1))
    store.save({job.id: job for job in (late, grace, periodic)})
    state = store.recover(now)
    assert not state.jobs[late.id].enabled
    assert state.jobs[grace.id] == grace
    assert state.jobs[periodic.id].fire_at == now + timedelta(seconds=60)
    assert not state.fires


def test_cancelled_fire_is_retained_and_cannot_be_resurrected(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    job = make_job()
    _ = store.add("create", job, "created")
    fire = store.start_fire(job)
    assert fire is not None
    _ = store.cancel("cancel", (job.id,))
    assert store.start_fire(job) is None
    store.settle(fire.key, "delivered", now=job.fire_at)
    assert store.load() == []
    assert store.read().fires[fire_key(job)].status == "cancelled"


def test_capacity_failure_is_a_replayable_outcome(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    jobs = [make_job() for _ in range(10)]
    store.save({job.id: job for job in jobs})
    extra = make_job()
    failed = store.add("full", extra, "created")
    assert failed.outcome == "error" and "schedule_capacity_reached" in failed.response
    _ = store.cancel("free", (jobs[0].id,))
    assert store.add("full", extra, "created") == failed
    assert extra.id not in {job.id for job in store.load()}
