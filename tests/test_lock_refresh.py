"""A running workflow's lock must be renewable, or a short lock timeout lets a peer take it over.

Edda only ever extends a workflow's lock if ``lock_refresh_interval`` is set. Without it, a workflow
that outlives ``lock_timeout_seconds`` looks stale to the auto-resume loop (``cleanup_stale_locks``)
even though its worker is alive, and a second worker would resume the same instance. The timeout
then has to be long, and a long timeout is how long a dead worker's workflows wait to be resumed.

Needs concurrent connections, so plain SQLite (one shared in-memory connection) is skipped.
"""

import asyncio

import pytest

from edda import WorkflowContext, activity, workflow
from edda.locking import workflow_lock
from edda.replay import ReplayEngine

LOCK_TIMEOUT_SECONDS = 1
RUN_SECONDS = 3.5


@activity
async def slow_step(ctx: WorkflowContext) -> str:
    await asyncio.sleep(RUN_SECONDS)
    return "done"


@workflow(lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
async def lock_refresh_slow_workflow(ctx: WorkflowContext) -> str:
    return await slow_step(ctx)


def needs_concurrent_connections(storage) -> None:
    dialect = storage.engine.dialect
    if dialect.name == "sqlite" and dialect.driver != "httpx":
        pytest.skip("the shared in-memory SQLite connection cannot run two sessions at once")


async def run_while_sweeping(storage, refresh_interval):
    """Run the slow workflow once while a second worker repeatedly sweeps for stale locks."""
    engine = ReplayEngine(
        storage=storage,
        service_name="svc",
        worker_id="owner",
        lock_refresh_interval=refresh_interval,
    )
    taken_over: list[str] = []
    running = True

    async def leader_sweep() -> None:
        while running:
            for found in await storage.cleanup_stale_locks():
                taken_over.append(found["instance_id"])
            await asyncio.sleep(0.2)

    sweeper = asyncio.create_task(leader_sweep())
    try:
        instance_id = await engine.start_workflow(
            workflow_name=lock_refresh_slow_workflow.name,
            workflow_func=lock_refresh_slow_workflow.func,
            input_data={},
            lock_timeout_seconds=LOCK_TIMEOUT_SECONDS,
        )
    finally:
        running = False
        await sweeper
    return instance_id, taken_over


async def test_without_refresh_a_live_workflow_looks_stale(db_storage):
    """The control: this is the hazard the setting exists to remove."""
    needs_concurrent_connections(db_storage)
    instance_id, taken_over = await run_while_sweeping(db_storage, refresh_interval=None)

    assert instance_id in taken_over


async def test_with_refresh_a_live_workflow_is_left_alone(db_storage):
    needs_concurrent_connections(db_storage)
    instance_id, taken_over = await run_while_sweeping(db_storage, refresh_interval=0.3)

    assert taken_over == []
    instance = await db_storage.get_instance(instance_id)
    assert instance["status"] == "completed"
    assert instance["locked_by"] is None  # released at the end, as ever


class FlakyRefreshStorage:
    """Grants the lock, fails the first refresh the way a dropped connection does, then renews."""

    def __init__(self) -> None:
        self.refreshes = 0
        self.released = False

    async def try_acquire_lock(self, instance_id, worker_id, timeout_seconds) -> bool:
        return True

    async def refresh_lock(self, instance_id, worker_id) -> bool:
        self.refreshes += 1
        if self.refreshes == 1:
            raise ConnectionError("connection dropped")
        return True

    async def release_lock(self, instance_id, worker_id) -> None:
        self.released = True


async def test_a_failed_refresh_is_retried_and_the_lock_is_still_released():
    """One failed renewal must not end renewal: the lock would lapse and a peer would rerun the workflow."""
    storage = FlakyRefreshStorage()

    async with workflow_lock(storage, "inst-1", "owner", refresh_interval=0.01):
        await asyncio.sleep(0.15)

    assert storage.refreshes >= 3  # it kept renewing after the failure
    assert storage.released  # and the workflow's exit neither raised nor skipped the release
