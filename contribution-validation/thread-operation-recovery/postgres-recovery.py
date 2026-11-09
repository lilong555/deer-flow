"""Real PostgreSQL row-lock checks against the source checkout in the current cwd."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from uuid import uuid4

backend = Path.cwd().resolve()
for path in (backend, backend / "packages/extension-api", backend / "packages/harness", backend / "tests"):
    sys.path.insert(0, str(path))

from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from support.postgres import asyncpg_test_url

from deerflow.persistence.run.model import CompletedRunSnapshotRow, RunChangeClockRow, RunRow
from deerflow.persistence.run.sql import RunRepository
from deerflow.runtime import RunManager, RunStatus, ThreadOperationKind
from deerflow.runtime.runs import manager as manager_module
from deerflow.runtime.runs.manager import ConflictError
from deerflow.runtime.user_context import reset_current_user, set_current_user

assert Path(manager_module.__file__).resolve().is_relative_to(backend), manager_module.__file__
logging.basicConfig(level=logging.WARNING)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
evidence = Path(__file__).resolve().parent
uri = json.loads((evidence / "postgres-fixture.json").read_text())["url"]


@asynccontextmanager
async def database(*, lock_timeout="40ms"):
    schema = "release_" + uuid4().hex
    engine = create_async_engine(
        asyncpg_test_url(uri), pool_size=3, max_overflow=0,
        connect_args={"server_settings": {"search_path": schema, "lock_timeout": lock_timeout}},
    )
    errors = []
    deleting = asyncio.Event()

    @event.listens_for(engine.sync_engine, "handle_error")
    def record_error(context):
        errors.append(getattr(context.original_exception, "sqlstate", None))

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def observe_delete(_conn, _cursor, statement, _parameters, _context, _many):
        if "FOR UPDATE" in statement.upper() and not _conn.info.get("release_probe_blocker"):
            deleting.set()

    blocker = None
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.run_sync(lambda c: RunRow.metadata.create_all(c, tables=[RunRow.__table__, RunChangeClockRow.__table__, CompletedRunSnapshotRow.__table__]))
        repository = RunRepository(async_sessionmaker(engine, expire_on_commit=False))
        manager = RunManager(store=repository)
        blocker = await engine.connect()
        blocker.info["release_probe_blocker"] = True
        blocker_pid = await blocker.scalar(text("SELECT pg_backend_pid()"))
        yield SimpleNamespace(manager=manager, repository=repository, blocker=blocker, errors=errors, deleting=deleting, engine=engine, blocker_pid=blocker_pid)
    finally:
        if blocker is not None:
            await blocker.rollback()
            blocker.info.pop("release_probe_blocker", None)
            await blocker.close()
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await engine.dispose()


async def hold_row(db, run_id):
    await db.blocker.execute(select(RunRow.run_id).where(RunRow.run_id == run_id).with_for_update())


async def expect_conflict(awaitable):
    try:
        await asyncio.wait_for(awaitable, timeout=3)
    except ConflictError:
        return
    raise AssertionError("The active or unreleased reservation must reject admission")


async def fail_release(db, kind, *, body_error=False):
    try:
        async with db.manager.reserve_thread_operation("thread", kind=kind, user_id="alice") as record:
            await hold_row(db, record.run_id)
            if body_error:
                raise ValueError("original body failure")
    except ValueError as exc:
        assert body_error and str(exc) == "original body failure"
    else:
        assert not body_error
    assert "55P03" in db.errors, db.errors
    row = await db.repository.get(record.run_id, user_id="alice")
    assert row["status"] == "pending" and row["lease_expires_at"] is None
    return record


async def recovery(kind, strategy, *, body_error=False, new_key=False):
    async with database() as db:
        assert not db.manager.heartbeat_enabled
        record = await fail_release(db, kind, body_error=body_error)
        if not new_key:
            await expect_conflict(db.manager.create_or_reject("thread", user_id="alice", multitask_strategy=strategy))
        await db.blocker.rollback()
        token = set_current_user(SimpleNamespace(id="unrelated-ambient-owner"))
        try:
            successor = await db.manager.create_or_reject("thread", user_id="alice", multitask_strategy=strategy, idempotency_key="new-request" if new_key else None)
        finally:
            reset_current_user(token)
        assert await db.repository.get(record.run_id, user_id="alice") is None
        assert [row["run_id"] for row in await db.repository.list_inflight()] == [successor.run_id]
        return {"sqlstates": db.errors, "successor_admitted": True}


async def durable_replay(strategy):
    async with database() as db:
        original = await db.manager.create_or_reject("thread", user_id="alice", idempotency_key="completed-request")
        await db.manager.set_status(original.run_id, RunStatus.success)
        await db.manager.cleanup(original.run_id, delay=0)
        finished = await fail_release(db, ThreadOperationKind.checkpoint_write)
        replay = await db.manager.create_or_reject("thread", user_id="alice", idempotency_key="completed-request", multitask_strategy=strategy)
        assert replay.run_id == original.run_id
        assert replay.status == RunStatus.success
        assert replay.idempotency_reused and replay.store_only
        assert (await db.repository.get(finished.run_id, user_id="alice"))["status"] == "pending"
        return {"sqlstates": db.errors, "durable_result_replayed": True}


async def active_control(strategy):
    async with database() as db:
        peer = RunManager(store=db.repository)
        async with db.manager.reserve_thread_operation("thread", kind=ThreadOperationKind.checkpoint_write, user_id="alice") as record:
            for contender in (db.manager, peer):
                await expect_conflict(contender.create_or_reject("thread", user_id="alice", multitask_strategy=strategy))
            assert (await db.repository.get(record.run_id, user_id="alice"))["status"] == "pending"
        assert await db.repository.list_inflight() == []
        return {"local_and_peer_rejected": True}


async def missing_old_row_preserves_peer():
    async with database() as db:
        record = await fail_release(db, ThreadOperationKind.checkpoint_write)
        await db.blocker.rollback()
        await db.repository.delete_thread_operation(record.run_id, user_id="alice")
        peer = RunManager(store=db.repository)
        successor = await peer.create_or_reject("thread", user_id="alice")
        await expect_conflict(db.manager.create_or_reject("thread", user_id="alice"))
        assert [row["run_id"] for row in await db.repository.list_inflight()] == [successor.run_id]
        return {"peer_row_preserved": True}


async def cancellation():
    async with database(lock_timeout="5000ms") as db:
        finished = None

        async def operation():
            nonlocal finished
            async with db.manager.reserve_thread_operation("thread", kind=ThreadOperationKind.checkpoint_write, user_id="alice") as finished:
                await hold_row(db, finished.run_id)
                db.deleting.clear()

        async def observe_database_lock_wait():
            async with db.engine.connect() as inspector:
                while not await inspector.scalar(text("SELECT count(*) FROM pg_stat_activity WHERE :blocker_pid = ANY(pg_blocking_pids(pid))"), {"blocker_pid": db.blocker_pid}):
                    await asyncio.sleep(0.01)

        task = asyncio.create_task(operation())
        try:
            await asyncio.wait_for(db.deleting.wait(), timeout=3)
            await asyncio.wait_for(observe_database_lock_wait(), timeout=3)
        finally:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3)
            except asyncio.CancelledError:
                pass
        assert finished is not None
        assert (await db.repository.get(finished.run_id, user_id="alice"))["status"] == "pending"
        await db.blocker.rollback()
        successor = await db.manager.create_or_reject("thread", user_id="alice")
        assert await db.repository.get(finished.run_id, user_id="alice") is None
        assert [row["run_id"] for row in await db.repository.list_inflight()] == [successor.run_id]
        return {"database_lock_wait_observed": True, "cancelled_delete_recovered": True}


async def main():
    cases = []
    for kind in ThreadOperationKind:
        if kind == ThreadOperationKind.run:
            continue
        for strategy in ("reject", "interrupt", "rollback"):
            cases.append((f"recover/{kind.value}/{strategy}", lambda kind=kind, strategy=strategy: recovery(kind, strategy)))
    for strategy in ("reject", "interrupt", "rollback"):
        cases.append((f"active/{strategy}", lambda strategy=strategy: active_control(strategy)))
        cases.append((f"new-key/{strategy}", lambda strategy=strategy: recovery(ThreadOperationKind.checkpoint_write, strategy, new_key=True)))
        cases.append((f"durable-replay/{strategy}", lambda strategy=strategy: durable_replay(strategy)))
    cases.extend([
        ("body-error", lambda: recovery(ThreadOperationKind.checkpoint_write, "reject", body_error=True)),
        ("peer-after-external-release", missing_old_row_preserves_peer),
        ("cancelled-sql-delete", cancellation),
    ])
    results = []
    for name, case in cases:
        try:
            observation = await case()
        except Exception as exc:
            traceback.print_exc()
            result = {"case": name, "passed": False, "error_type": type(exc).__name__, "error": str(exc)}
        else:
            result = {"case": name, "passed": True, "observation": observation}
        results.append(result)
        print(json.dumps(result), flush=True)
    record = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=backend, text=True).strip(),
        "manager_source": str(Path(manager_module.__file__).resolve()),
        "manager_sha256": hashlib.sha256(Path(manager_module.__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": json.loads((evidence / "postgres-fixture.json").read_text()),
        "results": results,
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "failed": record["failed"]}), flush=True)
    return int(bool(record["failed"]))


raise SystemExit(asyncio.run(main()))
