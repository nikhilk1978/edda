"""
Storage backend for rqlite, a distributed SQLite database.

``RqliteStorage`` is ``SQLAlchemyStorage`` running on the rqlite driver in
``edda.storage.rqlite_dialect``. rqlite handles consensus and write-forwarding, so a
cluster of Edda workers can share state without NFS, LiteFS or a database server.

What differs from the other backends, and why:

**No row locks.** rqlite has no ``SELECT ... FOR UPDATE``. Every claim (workflow lock,
system lock, outbox event) is a single conditional ``UPDATE`` whose ``rowcount`` says
whether this worker won. rqlite applies writes one at a time through Raft, so exactly
one contender can match the ``WHERE`` clause.

**One clock, never the database's.** Leases and timestamps must agree across workers, and rqlite
cannot supply the time: it rewrites ``now`` in write requests to the receiving node's LOCAL time as
if it were UTC (measured: four hours off on a node in US Eastern), while a read gives true UTC.
So no statement sends ``now`` or ``CURRENT_TIMESTAMP``. Every time is a bound parameter from the
dialect's clock, which ``calibrate_clock`` sets against the cluster's UTC (read from a query, where it
is correct). Workers whose own clocks differ still agree with one another.

**Transactions are atomic batches.** rqlite cannot hold a transaction open across HTTP
requests. Inside an Edda transaction (``@activity``) writes are queued and sent as one
atomic request at commit. See ``edda.storage.rqlite_dialect`` for the consequences:
reads inside the transaction do not see its own uncommitted writes, and
``rowcount``/``lastrowid`` are unavailable until commit. Operations that need an exact
``rowcount`` (locks, claims) always run outside the transaction.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import String, and_, bindparam, delete, exists, func, or_, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from edda.storage.rqlite_dialect import (
    BUFFERED,
    RqliteDialect_httpx,
    register_rqlite_dialect,
)
from edda.storage.rqlite_dialect import (
    OperationalError as RqliteOperationalError,
)
from edda.storage.sqlalchemy_storage import (
    CURRENT_SCHEMA_VERSION,
    ChannelSubscription,
    OutboxEvent,
    SQLAlchemyStorage,
    SystemLock,
    WorkflowDefinition,
    WorkflowInstance,
    WorkflowTimerSubscription,
)

logger = logging.getLogger(__name__)

_OUTBOX_CLAIMABLE = ("pending", "failed")

T = TypeVar("T")

_CLOCK_RECALIBRATE_SECONDS = 300
_LOST_RESPONSE_ATTEMPTS = (
    3  # tries for an operation whose answer may be lost; one repeat is too thin
)

# Writes that set or delete rows to a fixed outcome: doing one twice leaves the same state, so a worker
# may safely repeat it when the answer to the first attempt was lost. Deliberately absent: anything that
# inserts (a repeat collides or duplicates) or counts (``mark_outbox_failed`` adds 1 to retry_count).
_IDEMPOTENT_WRITES = (
    "update_instance_status",
    "update_instance_activity",
    "mark_outbox_published",
    "mark_outbox_permanently_failed",
    "mark_outbox_invalid",
    "mark_outbox_expired",
    "remove_timer_subscription",
    "remove_message_subscription",
    "clear_compensations",
    "leave_group",
    "leave_all_groups",
    "unsubscribe_from_channel",
    "clear_channel_waiting_state",
    "delete_channel_message",
    "update_delivery_cursor",
    "cleanup_instance_subscriptions",
)


def _outcome_unknown(exc: BaseException) -> bool:
    """True if a request may have been applied before it failed (its response was lost).

    A request that provably never left (no node could be connected to) is not ambiguous.
    """
    orig = getattr(exc, "orig", None)
    return isinstance(orig, RqliteOperationalError) and orig.request_sent


def create_rqlite_engine(db_url: str, **engine_kwargs: Any) -> AsyncEngine:
    """
    Create an async SQLAlchemy engine for an rqlite cluster.

    Accepts ``rqlite://host:4001`` (HTTP) and ``rqlites://host:4001`` (HTTPS), with
    optional credentials (``rqlite://user:pass@host:4001``). Query parameters:

    - ``level``: read consistency (``none``, ``weak``, ``linearizable``, ``strong``,
      ``auto``); default ``weak``
    - ``host``: an additional node, repeatable (``?host=n2:4001&host=n3:4001``);
      tried in order when a node cannot be connected to
    - ``timeout``: request timeout in seconds; default 30
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    register_rqlite_dialect()
    if db_url.startswith("rqlites://"):
        separator = "&" if "?" in db_url else "?"
        db_url = "rqlite+httpx://" + db_url[len("rqlites://") :] + f"{separator}tls=true"
    elif db_url.startswith("rqlite://"):
        db_url = "rqlite+httpx://" + db_url[len("rqlite://") :]
    return create_async_engine(db_url, **engine_kwargs)


class RqliteStorage(SQLAlchemyStorage):
    """``SQLAlchemyStorage`` for rqlite: atomic claims instead of row locks."""

    def __init__(
        self,
        engine: AsyncEngine,
        notify_listener: Any | None = None,
        migrations_dir: str | None = None,
    ):
        if not isinstance(engine.dialect, RqliteDialect_httpx):
            raise ValueError(
                "RqliteStorage requires an engine created for an rqlite URL "
                "(see create_rqlite_engine); got dialect "
                f"{type(engine.dialect).__name__}"
            )
        super().__init__(engine, notify_listener=notify_listener, migrations_dir=migrations_dir)
        self._rqlite_dialect: RqliteDialect_httpx = engine.dialect
        self._clock_calibrated_at = float("-inf")
        # Same pool as ``engine``; connections from it queue writes until commit.
        self._buffered_engine = engine.execution_options(isolation_level=BUFFERED)

    # ------------------------------------------------------------------
    # Transactions and time
    # ------------------------------------------------------------------

    def _new_transaction_session(self) -> AsyncSession:
        return AsyncSession(self._buffered_engine, expire_on_commit=False)

    @property
    def _atomic_engine(self) -> AsyncEngine:
        return self._buffered_engine

    def _get_session_for_operation(
        self, is_lock_operation: bool = False, *, atomic: bool = False
    ) -> AsyncSession:
        """
        Plain sessions apply each statement at once. ``atomic`` units queue theirs and send them as
        one request on commit, so a crash between statements cannot leave half of them applied.

        Inside an Edda transaction the transaction's own session is already batched and is kept.
        """
        session = super()._get_session_for_operation(is_lock_operation)
        if atomic and not self._is_managed_session(session):
            return AsyncSession(self._buffered_engine, expire_on_commit=False)
        return session

    # ------------------------------------------------------------------
    # Time: a bound parameter from the calibrated clock, never SQL ``now``
    # ------------------------------------------------------------------

    def utc_now_text(self) -> str:
        """The cluster's current UTC time as stored text (what the parameter below carries)."""
        return self._rqlite_dialect.utc_now_text()

    def _now(self) -> Any:
        return bindparam(
            "now", callable_=self._rqlite_dialect.utc_now_text, type_=String, unique=True
        )

    def _get_current_time_expr(self) -> Any:
        return func.datetime(self._now())

    def _expires_in(self, seconds: Any) -> Any:
        """SQL expression for "cluster now + seconds" (``seconds`` may be a column)."""
        return func.datetime(self._now(), func.printf("+%d seconds", seconds))

    async def calibrate_clock(self) -> float:
        """
        Measure this machine's clock against the cluster's UTC and return the offset in seconds.

        A read of ``'now'`` is correct UTC on the node that answers. Of three samples the one with
        the shortest round trip is used, because its midpoint is the best estimate of when the
        cluster read its clock. Call again if clocks drift; lock claims do so every few minutes.
        """
        best: tuple[float, float] | None = None
        for _ in range(3):
            started = time.time()
            async with AsyncSession(self.engine) as session:
                millis = (
                    await session.execute(
                        text("SELECT CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)")
                    )
                ).scalar_one()
            finished = time.time()
            sample = (finished - started, millis / 1000.0 - (started + finished) / 2)
            if best is None or sample[0] < best[0]:
                best = sample
        assert best is not None
        self._rqlite_dialect.clock_offset = best[1]
        self._clock_calibrated_at = time.monotonic()
        return best[1]

    async def _keep_clock_fresh(self) -> None:
        if time.monotonic() - self._clock_calibrated_at > _CLOCK_RECALIBRATE_SECONDS:
            await self.calibrate_clock()

    async def initialize(self) -> None:
        await self.calibrate_clock()  # before anything is written: rows carry the cluster's time
        await super().initialize()

    async def _initialize_schema_version(self) -> None:
        """
        Record the schema version on a fresh database, in one statement.

        Every node runs this when it starts. The base class counts the rows and then inserts, so
        nodes booting together on an empty cluster all see "empty" and all but one fail on the
        primary key. ``INSERT ... WHERE NOT EXISTS`` decides in the same statement it writes.
        """
        async with AsyncSession(self.engine) as session:
            await session.execute(
                text(
                    "INSERT INTO schema_version (version, applied_at, description) "
                    "SELECT :version, :now, :description "
                    "WHERE NOT EXISTS (SELECT 1 FROM schema_version)"
                ),
                {
                    "version": CURRENT_SCHEMA_VERSION,
                    "now": self.utc_now_text(),
                    "description": "Initial schema with workflow_definitions",
                },
            )
            await session.commit()

    async def create_instance(self, instance_id: str, *args: Any, **kwargs: Any) -> None:
        """
        Create an instance; if the response is lost, find out whether it was created.

        The id is the primary key, so a repeat after a success would fail. Instead look for the row:
        present means an attempt landed; absent means none did, and it is tried again.
        """
        for attempt in range(_LOST_RESPONSE_ATTEMPTS):
            try:
                await super().create_instance(instance_id, *args, **kwargs)
                return
            except DBAPIError as exc:
                if not _outcome_unknown(exc) or attempt == _LOST_RESPONSE_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "create_instance response lost (%s); checking whether it was applied", exc.orig
                )
            async with self._session_scope(
                self._get_session_for_operation(is_lock_operation=True)
            ) as session:
                created = (
                    await session.execute(
                        select(WorkflowInstance.instance_id).where(
                            WorkflowInstance.instance_id == instance_id
                        )
                    )
                ).first()
            if created is not None:
                return

    async def upsert_workflow_definition(
        self,
        workflow_name: str,
        source_hash: str,
        source_code: str,
    ) -> None:
        """
        Insert or update a definition in one statement.

        The base class looks for the row and then inserts it, so two nodes that start the same
        workflow for the first time both see "missing" and the second INSERT fails on the
        primary key. ``ON CONFLICT DO UPDATE`` leaves nothing between the check and the write.
        """
        session = self._get_session_for_operation()
        async with self._session_scope(session) as session:
            await session.execute(
                sqlite_insert(WorkflowDefinition)
                .values(
                    workflow_name=workflow_name, source_hash=source_hash, source_code=source_code
                )
                .on_conflict_do_update(
                    index_elements=["workflow_name", "source_hash"],
                    set_={"source_code": source_code},
                )
            )
            await self._commit_if_not_in_transaction(session)

    # ------------------------------------------------------------------
    # Workflow locks: one conditional UPDATE decides the winner
    # ------------------------------------------------------------------

    async def _claim_once(
        self, claim: Callable[[], Awaitable[bool]], holds: Callable[[], Awaitable[bool]]
    ) -> bool:
        """
        Run a conditional claim and settle the case where its response was lost.

        The claim may or may not have been applied. Ask the cluster whether this worker holds the
        lock now; if not, claim again. Repeating is safe because a claim only succeeds on a free or
        expired lock. Without this, an applied claim reported as a failure leaves a lock held by a
        worker that believes it has none, until the lease runs out. A few attempts are made, because
        a repeat can lose its answer too.
        """
        for attempt in range(_LOST_RESPONSE_ATTEMPTS):
            try:
                return await claim()
            except DBAPIError as exc:
                if not _outcome_unknown(exc) or attempt == _LOST_RESPONSE_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Claim response lost (%s); checking whether it was applied", exc.orig
                )
            if await holds():
                return True
        raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises

    async def _retry_idempotent(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run an idempotent write again if its response was lost, a few times at most."""
        for attempt in range(_LOST_RESPONSE_ATTEMPTS):
            try:
                return await operation()
            except DBAPIError as exc:
                if not _outcome_unknown(exc) or attempt == _LOST_RESPONSE_ATTEMPTS - 1:
                    raise
                logger.warning("Response lost (%s); repeating the idempotent operation", exc.orig)
        raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises

    async def _holds_workflow_lock(self, instance_id: str, worker_id: str) -> bool:
        async with self._session_scope(
            self._get_session_for_operation(is_lock_operation=True)
        ) as session:
            row = (
                await session.execute(
                    select(WorkflowInstance.locked_by).where(
                        and_(
                            WorkflowInstance.instance_id == instance_id,
                            WorkflowInstance.locked_by == worker_id,
                            func.datetime(WorkflowInstance.lock_expires_at)
                            > self._get_current_time_expr(),
                        )
                    )
                )
            ).first()
            return row is not None

    async def _holds_system_lock(self, lock_name: str, worker_id: str) -> bool:
        async with self._session_scope(
            self._get_session_for_operation(is_lock_operation=True)
        ) as session:
            row = (
                await session.execute(
                    select(SystemLock.locked_by).where(
                        and_(
                            SystemLock.lock_name == lock_name,
                            SystemLock.locked_by == worker_id,
                            func.datetime(SystemLock.lock_expires_at)
                            > self._get_current_time_expr(),
                        )
                    )
                )
            ).first()
            return row is not None

    async def try_acquire_lock(
        self,
        instance_id: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        await self._keep_clock_fresh()
        return await self._claim_once(
            lambda: self._claim_workflow_lock(instance_id, worker_id, timeout_seconds),
            lambda: self._holds_workflow_lock(instance_id, worker_id),
        )

    async def _claim_workflow_lock(
        self,
        instance_id: str,
        worker_id: str,
        timeout_seconds: int,
    ) -> bool:
        """
        Claim the instance with a single conditional UPDATE.

        The row is claimed when it is unlocked or its lease has expired. Exactly one
        concurrent caller sees ``rowcount == 1``. Priority for the lease length is
        the instance's own ``lock_timeout_seconds``, then ``timeout_seconds``.
        """
        session = self._get_session_for_operation(is_lock_operation=True)
        async with self._session_scope(session) as session:
            now = self._get_current_time_expr()
            result = await session.execute(
                update(WorkflowInstance)
                .where(
                    and_(
                        WorkflowInstance.instance_id == instance_id,
                        or_(
                            WorkflowInstance.locked_by.is_(None),
                            and_(
                                WorkflowInstance.lock_expires_at.is_not(None),
                                func.datetime(WorkflowInstance.lock_expires_at) < now,
                            ),
                        ),
                    )
                )
                .values(
                    locked_by=worker_id,
                    locked_at=now,
                    lock_expires_at=self._expires_in(
                        func.coalesce(WorkflowInstance.lock_timeout_seconds, timeout_seconds)
                    ),
                    updated_at=now,
                )
            )
            await session.commit()
            return bool(result.rowcount == 1)  # type: ignore[attr-defined]

    async def refresh_lock(
        self, instance_id: str, worker_id: str, timeout_seconds: int = 300
    ) -> bool:
        return await self._retry_idempotent(
            lambda: self._refresh_lock_once(instance_id, worker_id, timeout_seconds)
        )

    async def release_lock(self, instance_id: str, worker_id: str) -> None:
        await self._retry_idempotent(
            lambda: super(RqliteStorage, self).release_lock(instance_id, worker_id)
        )

    async def release_system_lock(self, lock_name: str, worker_id: str) -> None:
        await self._retry_idempotent(
            lambda: super(RqliteStorage, self).release_system_lock(lock_name, worker_id)
        )

    async def _refresh_lock_once(
        self, instance_id: str, worker_id: str, timeout_seconds: int
    ) -> bool:
        """Extend the lease if this worker still holds it (single UPDATE)."""
        session = self._get_session_for_operation(is_lock_operation=True)
        async with self._session_scope(session) as session:
            now = self._get_current_time_expr()
            result = await session.execute(
                update(WorkflowInstance)
                .where(
                    and_(
                        WorkflowInstance.instance_id == instance_id,
                        WorkflowInstance.locked_by == worker_id,
                    )
                )
                .values(
                    locked_at=now,
                    lock_expires_at=self._expires_in(
                        func.coalesce(WorkflowInstance.lock_timeout_seconds, timeout_seconds)
                    ),
                    updated_at=now,
                )
            )
            await session.commit()
            return bool(result.rowcount and result.rowcount > 0)  # type: ignore[attr-defined]

    async def cleanup_stale_locks(self) -> list[dict[str, str]]:
        """
        Release expired leases and return the workflows that should be resumed.

        Each release re-checks expiry in its own conditional UPDATE, so a worker that
        refreshed its lease between the scan and the release is left alone and its
        workflow is not reported for resumption.
        """
        session = self._get_session_for_operation(is_lock_operation=True)
        async with self._session_scope(session) as session:
            now = self._get_current_time_expr()
            is_stale = and_(
                WorkflowInstance.locked_by.is_not(None),
                WorkflowInstance.lock_expires_at.is_not(None),
                func.datetime(WorkflowInstance.lock_expires_at) < now,
                WorkflowInstance.framework == "python",
            )
            candidates = (
                await session.execute(
                    select(
                        WorkflowInstance.instance_id,
                        WorkflowInstance.workflow_name,
                        WorkflowInstance.source_hash,
                        WorkflowInstance.status,
                    ).where(is_stale)
                )
            ).all()

            workflows_to_resume: list[dict[str, str]] = []
            for row in candidates:
                released = await session.execute(
                    update(WorkflowInstance)
                    .where(and_(WorkflowInstance.instance_id == row.instance_id, is_stale))
                    .values(locked_by=None, locked_at=None, lock_expires_at=None, updated_at=now)
                )
                if released.rowcount != 1:  # type: ignore[attr-defined]
                    continue  # lease was refreshed or released meanwhile
                if row.status in ("running", "compensating"):
                    workflows_to_resume.append(
                        {
                            "instance_id": str(row.instance_id),
                            "workflow_name": str(row.workflow_name),
                            "source_hash": str(row.source_hash),
                            "status": str(row.status),
                        }
                    )

            await session.commit()
            return workflows_to_resume

    async def try_acquire_system_lock(
        self,
        lock_name: str,
        worker_id: str,
        timeout_seconds: int = 60,
    ) -> bool:
        return await self._claim_once(
            lambda: self._claim_system_lock(lock_name, worker_id, timeout_seconds),
            lambda: self._holds_system_lock(lock_name, worker_id),
        )

    async def _claim_system_lock(
        self,
        lock_name: str,
        worker_id: str,
        timeout_seconds: int,
    ) -> bool:
        """System lock with cluster-clock expiry (see class docstring)."""
        session = self._get_session_for_operation(is_lock_operation=True)
        async with self._session_scope(session) as session:
            await session.execute(
                sqlite_insert(SystemLock)
                .values(lock_name=lock_name, locked_by=None, locked_at=None, lock_expires_at=None)
                .on_conflict_do_nothing(index_elements=["lock_name"])
            )

            now = self._get_current_time_expr()
            result = await session.execute(
                update(SystemLock)
                .where(
                    and_(
                        SystemLock.lock_name == lock_name,
                        or_(
                            SystemLock.locked_by.is_(None),
                            SystemLock.locked_by == worker_id,  # renewal by the holder
                            func.datetime(SystemLock.lock_expires_at) <= now,
                        ),
                    )
                )
                .values(
                    locked_by=worker_id,
                    locked_at=now,
                    lock_expires_at=self._expires_in(int(timeout_seconds)),
                )
            )
            await session.commit()
            return bool(result.rowcount == 1)  # type: ignore[attr-defined]

    async def cancel_instance(self, instance_id: str, cancelled_by: str) -> bool:
        """
        Cancel a running or waiting workflow, all-or-nothing.

        The base class marks the row cancelled, checks ``rowcount`` to see whether it won a race,
        and then deletes the wait registrations in further statements. Here the status change and
        the cleanup are ONE batch, and each statement is guarded on the status that was read, so a
        worker that lost a race changes nothing. Whether this call won is read back afterwards,
        from the metadata only this call wrote.
        """
        cancellable = (
            "running",
            "waiting_for_event",
            "waiting_for_timer",
            "waiting_for_message",
            "compensating",
        )
        async with self._session_scope(
            self._get_session_for_operation(is_lock_operation=True)
        ) as session:
            row = (
                await session.execute(
                    select(WorkflowInstance.status).where(
                        WorkflowInstance.instance_id == instance_id
                    )
                )
            ).one_or_none()
            if row is None or row[0] not in cancellable:
                return False
            current_status = row[0]

        metadata = json.dumps(
            {
                "cancelled_by": cancelled_by,
                "cancelled_at": datetime.now(UTC).isoformat(),
                "previous_status": current_status,
            }
        )
        still_current = exists().where(
            and_(
                WorkflowInstance.instance_id == instance_id,
                WorkflowInstance.status == current_status,
            )
        )
        batch = AsyncSession(self._buffered_engine, expire_on_commit=False)
        try:
            if current_status == "waiting_for_timer":
                await batch.execute(
                    delete(WorkflowTimerSubscription).where(
                        and_(WorkflowTimerSubscription.instance_id == instance_id, still_current)
                    )
                )
            if current_status in ("waiting_for_event", "waiting_for_message"):
                await batch.execute(
                    update(ChannelSubscription)
                    .where(and_(ChannelSubscription.instance_id == instance_id, still_current))
                    .values(activity_id=None, timeout_at=None)
                )
            await batch.execute(
                update(WorkflowInstance)
                .where(
                    and_(
                        WorkflowInstance.instance_id == instance_id,
                        WorkflowInstance.status == current_status,
                    )
                )
                .values(
                    status="cancelled",
                    output_data=metadata,
                    locked_by=None,
                    locked_at=None,
                    lock_expires_at=None,
                    updated_at=func.now(),
                )
            )
            await batch.commit()
        finally:
            await batch.close()

        async with self._session_scope(
            self._get_session_for_operation(is_lock_operation=True)
        ) as session:
            outcome = (
                await session.execute(
                    select(WorkflowInstance.status, WorkflowInstance.output_data).where(
                        WorkflowInstance.instance_id == instance_id
                    )
                )
            ).one_or_none()
        return outcome is not None and outcome[0] == "cancelled" and outcome[1] == metadata

    # ------------------------------------------------------------------
    # Outbox: claim events one conditional UPDATE at a time
    # ------------------------------------------------------------------

    async def get_pending_outbox_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Claim up to ``limit`` pending/failed outbox events for publishing.

        Candidates are read oldest-first, then each is claimed with
        ``UPDATE ... SET status='processing' WHERE event_id=? AND status IN (...)``.
        Only the worker that sees ``rowcount == 1`` publishes the event; a worker that
        loses the race simply skips it, so the result may hold fewer than ``limit``.
        """
        session = self._get_session_for_operation(is_lock_operation=True)
        async with self._session_scope(session) as session:
            candidates = (
                (
                    await session.execute(
                        select(OutboxEvent)
                        .where(OutboxEvent.status.in_(_OUTBOX_CLAIMABLE))
                        .order_by(OutboxEvent.created_at.asc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

            claimed = []
            for row in candidates:
                result = await session.execute(
                    update(OutboxEvent)
                    .where(
                        and_(
                            OutboxEvent.event_id == row.event_id,
                            OutboxEvent.status.in_(_OUTBOX_CLAIMABLE),
                        )
                    )
                    .values(status="processing")
                )
                if result.rowcount == 1:  # type: ignore[attr-defined]
                    claimed.append(row)
            await session.commit()

            return [self._outbox_row_to_dict(row) for row in claimed]

    @staticmethod
    def _outbox_row_to_dict(row: OutboxEvent) -> dict[str, Any]:
        import json

        return {
            "event_id": row.event_id,
            "event_type": row.event_type,
            "event_source": row.event_source,
            "event_data": (
                row.event_data_binary
                if row.data_type == "binary"
                else json.loads(row.event_data)  # type: ignore[arg-type]
            ),
            "content_type": row.content_type,
            "created_at": row.created_at.isoformat(),
            "status": "processing",
            "retry_count": row.retry_count,
            "last_error": row.last_error,
        }


def _retrying_idempotent(name: str) -> Callable[..., Awaitable[Any]]:
    original = getattr(SQLAlchemyStorage, name)

    async def method(self: RqliteStorage, *args: Any, **kwargs: Any) -> Any:
        return await self._retry_idempotent(lambda: original(self, *args, **kwargs))

    method.__name__ = name
    method.__doc__ = original.__doc__
    return method


for _name in _IDEMPOTENT_WRITES:
    setattr(RqliteStorage, _name, _retrying_idempotent(_name))
