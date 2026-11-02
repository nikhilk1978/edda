"""
Tests for the rqlite backend: HTTP client, SQLAlchemy driver and RqliteStorage.

These run against ``tests/rqlite_fake.py``, an in-process model of an rqlite cluster
whose requests interleave like real network calls. The fake is a model of rqlite, not
rqlite: set EDDA_TEST_RQLITE_URL (e.g. rqlite://localhost:4001) to run the shared
``db_storage`` suites against a real cluster as well.
"""

import asyncio
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx2
import pytest
import pytest_asyncio
from sqlalchemy import (
    Column,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError, NotSupportedError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from edda.storage.migrations import apply_dbmate_migrations
from edda.storage.rqlite_client import (
    RqliteAuthError,
    RqliteClient,
    RqliteConnectionError,
    RqliteStatementError,
    encode_param,
)
from edda.storage.rqlite_dialect import (
    BUFFERED,
    RqliteDialect_httpx,
    is_read_statement,
    register_rqlite_dialect,
)
from edda.storage.rqlite_storage import RqliteStorage, create_rqlite_engine
from edda.storage.sqlalchemy_storage import Base
from tests.rqlite_fake import FakeRqlite

SCHEMA_DIR = Path(__file__).parent.parent / "schema" / "db" / "migrations"

TXN_CONTROL = ("BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE")


def all_statements(fake: FakeRqlite) -> list[str]:
    return [sql for request in fake.requests for sql, _ in request.statements]


@pytest.fixture
def fake():
    cluster = FakeRqlite()
    yield cluster
    cluster.close()


@pytest_asyncio.fixture
async def engine(fake):
    eng = create_rqlite_engine(
        "rqlite://localhost:4001", connect_args={"transport": fake.transport()}
    )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def storage(fake, engine):
    await apply_dbmate_migrations(engine, SCHEMA_DIR)
    store = RqliteStorage(engine)
    await store.upsert_workflow_definition("wf", "hash", "async def wf(ctx): pass")
    yield store
    await store.close()


async def make_instance(storage: RqliteStorage, instance_id: str = "inst-1", **kw) -> str:
    await storage.create_instance(
        instance_id=instance_id,
        workflow_name="wf",
        source_hash="hash",
        owner_service="svc",
        input_data={},
        **kw,
    )
    return instance_id


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class TestRqliteClient:
    def test_encode_param(self):
        assert encode_param(None) is None
        assert encode_param("x") == "x"
        assert encode_param(True) == 1
        assert encode_param(b"\x00\xff") == [0, 255]
        with pytest.raises(TypeError):
            encode_param(object())

    async def test_blob_round_trip(self, fake):
        client = RqliteClient("http://n1:4001", transport=fake.transport())
        await client.execute(["CREATE TABLE t (id INTEGER PRIMARY KEY, data BLOB, note TEXT)"])
        await client.execute(
            [("INSERT INTO t (data, note) VALUES (?, ?)", (b"\x00\x01\xfe", "hi"))]
        )
        result = await client.query(("SELECT data, note FROM t", ()))
        assert result.rows == [(b"\x00\x01\xfe", "hi")]
        await client.aclose()

    async def test_fails_over_when_node_unreachable(self, fake):
        fake.down_hosts.add("n1")
        client = RqliteClient(["http://n1:4001", "http://n2:4001"], transport=fake.transport())
        await client.query(("SELECT 1", ()))
        assert {r.host for r in fake.requests} == {"n2"}
        # The working node is now preferred: no wasted attempt on the dead one.
        before = len(fake.requests)
        await client.query(("SELECT 1", ()))
        assert len(fake.requests) == before + 1
        await client.aclose()

    async def test_all_nodes_down(self, fake):
        fake.down_hosts.update({"n1", "n2"})
        client = RqliteClient(["http://n1:4001", "http://n2:4001"], transport=fake.transport())
        with pytest.raises(RqliteConnectionError) as info:
            await client.query(("SELECT 1", ()))
        assert info.value.request_sent is False
        await client.aclose()

    async def test_write_is_not_retried_after_it_was_sent(self, fake):
        """An ambiguous failure must not be replayed: the write may have been applied."""

        def drop(request):
            raise httpx2.ReadError("connection reset", request=None)

        fake.before_request.append(drop)
        client = RqliteClient(["http://n1:4001", "http://n2:4001"], transport=fake.transport())
        with pytest.raises(RqliteConnectionError) as info:
            await client.execute([("INSERT INTO nope VALUES (1)", ())])
        assert info.value.request_sent is True
        assert len(fake.requests) == 1  # no retry on the second node
        await client.aclose()

    async def test_basic_auth(self):
        cluster = FakeRqlite(credentials=("user", "secret"))
        good = RqliteClient(
            "http://n1:4001", username="user", password="secret", transport=cluster.transport()
        )
        await good.query(("SELECT 1", ()))
        bad = RqliteClient(
            "http://n1:4001", username="user", password="wrong", transport=cluster.transport()
        )
        with pytest.raises(RqliteAuthError):
            await bad.query(("SELECT 1", ()))
        await good.aclose()
        await bad.aclose()
        cluster.close()

    async def test_statement_error_is_reported(self, fake):
        client = RqliteClient("http://n1:4001", transport=fake.transport())
        with pytest.raises(RqliteStatementError, match="no such table"):
            await client.query(("SELECT * FROM missing", ()))
        await client.aclose()


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 1", True),
        ("  select * from t", True),
        ("-- note\nSELECT 1", True),
        ("PRAGMA table_info(t)", True),
        ("WITH c AS (SELECT 1) SELECT * FROM c", True),
        ("WITH c AS (SELECT 1) UPDATE t SET a = 1", False),
        ("INSERT INTO t VALUES (1)", False),
        ("UPDATE t SET updated_at = 1", False),
        ("DELETE FROM t", False),
        ("CREATE TABLE t (a INT)", False),
    ],
)
def test_read_write_classification(sql, expected):
    assert is_read_statement(sql) is expected


# ---------------------------------------------------------------------------
# SQLAlchemy driver
# ---------------------------------------------------------------------------

metadata = MetaData()
items = Table(
    "items",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(50), unique=True),
    Column("payload", LargeBinary, nullable=True),
)


@pytest_asyncio.fixture
async def items_engine(fake, engine):
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    return engine


class TestDriverAutocommit:
    """Default mode: every statement is sent immediately with exact results."""

    async def test_dialect_looks_like_sqlite(self, engine):
        assert isinstance(engine.dialect, RqliteDialect_httpx)
        # Edda branches on dialect.name == "sqlite"; rqlite must take those branches.
        assert engine.dialect.name == "sqlite"

    async def test_rowcount_and_lastrowid_are_exact(self, items_engine):
        async with AsyncSession(items_engine) as session:
            result = await session.execute(insert(items).values(name="a"))
            assert result.inserted_primary_key == (1,)
            result = await session.execute(
                update(items).where(items.c.name == "a").values(name="b")
            )
            assert result.rowcount == 1
            result = await session.execute(
                update(items).where(items.c.name == "zzz").values(name="c")
            )
            assert result.rowcount == 0
            await session.commit()

    async def test_blob_and_unique_violation(self, items_engine):
        async with AsyncSession(items_engine) as session:
            await session.execute(insert(items).values(name="a", payload=b"\x01\x02"))
            await session.commit()
            row = (await session.execute(select(items.c.payload))).scalar_one()
            assert row == b"\x01\x02"
            with pytest.raises(IntegrityError):
                await session.execute(insert(items).values(name="a"))

    async def test_savepoint_needs_buffered_mode(self, items_engine):
        async with items_engine.connect() as conn:
            with pytest.raises(NotSupportedError):
                await conn.begin_nested()

    async def test_executemany_is_one_atomic_request(self, fake, items_engine):
        fake.requests.clear()
        async with items_engine.begin() as conn:
            await conn.execute(insert(items), [{"name": "a"}, {"name": "b"}])
        batches = [r for r in fake.writes() if len(r.statements) == 2]
        assert len(batches) == 1 and batches[0].transaction

        fake.requests.clear()
        with pytest.raises(IntegrityError):
            async with items_engine.begin() as conn:
                await conn.execute(insert(items), [{"name": "c"}, {"name": "a"}])  # "a" exists
        # all-or-nothing: "c" must not have been applied
        assert fake.table_rows("SELECT name FROM items WHERE name = 'c'") == []


class TestDriverBuffered:
    """BUFFERED mode: writes queue up and are applied as one atomic batch on commit."""

    @pytest_asyncio.fixture
    async def buffered(self, items_engine):
        return items_engine.execution_options(isolation_level=BUFFERED)

    async def test_nothing_is_sent_until_commit(self, fake, buffered):
        fake.requests.clear()
        async with AsyncSession(buffered) as session:
            await session.execute(insert(items).values(name="a"))
            await session.execute(insert(items).values(name="b"))
            assert fake.writes() == []
            await session.commit()
        writes = fake.writes()
        assert len(writes) == 1
        assert writes[0].transaction and len(writes[0].statements) == 2
        assert fake.table_rows("SELECT name FROM items ORDER BY id") == [("a",), ("b",)]

    async def test_rollback_discards_everything(self, fake, buffered):
        fake.requests.clear()
        async with AsyncSession(buffered) as session:
            await session.execute(insert(items).values(name="a"))
            await session.rollback()
        assert fake.writes() == []
        assert fake.table_rows("SELECT * FROM items") == []

    async def test_failed_commit_applies_nothing(self, fake, buffered):
        async with AsyncSession(buffered) as session:
            await session.execute(insert(items).values(name="a"))
            await session.commit()
        async with AsyncSession(buffered) as session:
            await session.execute(insert(items).values(name="fresh"))
            await session.execute(insert(items).values(name="a"))  # violates UNIQUE
            with pytest.raises(IntegrityError):
                await session.commit()
        assert fake.table_rows("SELECT name FROM items ORDER BY id") == [("a",)]

    async def test_reads_do_not_see_uncommitted_writes(self, buffered):
        async with AsyncSession(buffered) as session:
            await session.execute(insert(items).values(name="a"))
            assert (await session.execute(select(items))).all() == []  # documented limitation
            await session.commit()
            assert len((await session.execute(select(items))).all()) == 1

    async def test_savepoint_rollback_drops_only_inner_writes(self, fake, buffered):
        async with AsyncSession(buffered) as session:
            await session.begin()
            await session.execute(insert(items).values(name="outer"))
            inner = await session.begin_nested()
            await session.execute(insert(items).values(name="inner"))
            await inner.rollback()
            await session.execute(insert(items).values(name="after"))
            await session.commit()
        assert fake.table_rows("SELECT name FROM items ORDER BY id") == [("outer",), ("after",)]

    async def test_released_savepoint_keeps_writes(self, fake, buffered):
        async with AsyncSession(buffered) as session:
            await session.begin()
            inner = await session.begin_nested()
            await session.execute(insert(items).values(name="inner"))
            await inner.commit()
            await session.commit()
        assert fake.table_rows("SELECT name FROM items") == [("inner",)]

    async def test_connection_returns_to_autocommit_after_use(self, fake, items_engine, buffered):
        async with AsyncSession(buffered) as session:
            await session.execute(insert(items).values(name="a"))
            await session.commit()
        fake.requests.clear()
        async with AsyncSession(items_engine) as session:  # same pool, default mode
            await session.execute(insert(items).values(name="b"))
        # applied immediately, not queued behind a commit that never comes
        assert len(fake.writes()) == 1 and not fake.writes()[0].transaction

    async def test_node_never_receives_transaction_control_statements(self, fake, buffered):
        async with AsyncSession(buffered) as session:
            await session.begin()
            await session.execute(insert(items).values(name="a"))
            inner = await session.begin_nested()
            await session.execute(insert(items).values(name="b"))
            await inner.rollback()
            await session.commit()
        offending = [s for s in all_statements(fake) if s.lstrip().upper().startswith(TXN_CONTROL)]
        assert offending == []


# ---------------------------------------------------------------------------
# Timestamp defaults (replication safety)
# ---------------------------------------------------------------------------


def time_default_columns(table):
    return [
        c
        for c in table.columns
        if c.server_default is not None and "now" in str(c.server_default.arg).lower()
    ]


@pytest.mark.parametrize(
    "table",
    [t for t in Base.metadata.sorted_tables if time_default_columns(t)],
    ids=lambda t: t.name,
)
def test_omitted_timestamp_defaults_are_filled_with_the_cluster_clock(engine, table):
    """rqlite does not apply column DEFAULTs uniformly, and rewrites ``now`` to the node's local time,
    so the INSERT carries each timestamp as a parameter taken from the dialect's clock."""
    timed = {c.key for c in time_default_columns(table)}
    keys = [c.key for c in table.columns if c.key not in timed]
    compiled = insert(table).compile(dialect=engine.dialect, column_keys=keys)
    assert "CURRENT_TIMESTAMP" not in str(compiled)
    clock_params = [name for name in compiled.binds if name.startswith("now")]
    assert len(clock_params) == len(timed)


def test_explicit_timestamp_is_respected(engine):
    table = Base.metadata.tables["workflow_instances"]
    keys = [c.key for c in table.columns]  # everything supplied, including started_at
    compiled = insert(table).compile(dialect=engine.dialect, column_keys=keys)
    assert not [name for name in compiled.binds if name.startswith("now")]


def test_other_backends_are_untouched():
    plain = create_async_engine("sqlite+aiosqlite:///:memory:").dialect
    table = Base.metadata.tables["workflow_instances"]
    keys = [c.key for c in table.columns if c.key not in ("started_at", "updated_at")]
    assert "CURRENT_TIMESTAMP" not in str(insert(table).compile(dialect=plain, column_keys=keys))


# ---------------------------------------------------------------------------
# RqliteStorage: claims instead of row locks
# ---------------------------------------------------------------------------


class TestWorkflowLocks:
    async def test_exactly_one_of_many_concurrent_workers_wins(self, storage):
        await make_instance(storage)
        results = await asyncio.gather(
            *[storage.try_acquire_lock("inst-1", f"worker-{n}") for n in range(10)]
        )
        assert sum(results) == 1

    async def test_lock_blocks_others_until_released(self, storage):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "a")
        assert not await storage.try_acquire_lock("inst-1", "b")
        await storage.release_lock("inst-1", "a")
        assert await storage.try_acquire_lock("inst-1", "b")

    async def test_unknown_instance_cannot_be_locked(self, storage):
        assert not await storage.try_acquire_lock("nope", "a")

    async def test_expired_lease_can_be_taken_over(self, fake, storage):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "a")
        fake.db.execute(
            "UPDATE workflow_instances SET lock_expires_at = datetime('now', '-5 seconds')"
        )
        assert await storage.try_acquire_lock("inst-1", "b")
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("b",)]

    async def test_unexpired_lease_is_respected(self, fake, storage):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "a", timeout_seconds=600)
        assert not await storage.try_acquire_lock("inst-1", "b")

    async def test_instance_timeout_overrides_default(self, fake, storage):
        await make_instance(storage, lock_timeout_seconds=42)
        assert await storage.try_acquire_lock("inst-1", "a", timeout_seconds=300)
        (seconds,) = fake.table_rows(
            "SELECT CAST(strftime('%s', lock_expires_at) AS INTEGER) - "
            "CAST(strftime('%s', locked_at) AS INTEGER) FROM workflow_instances"
        )[0]
        assert seconds == 42

    async def test_refresh_only_for_the_holder(self, fake, storage):
        await make_instance(storage)
        await storage.try_acquire_lock("inst-1", "a", timeout_seconds=10)
        fake.db.execute(
            "UPDATE workflow_instances SET lock_expires_at = datetime('now', '+1 second')"
        )
        assert await storage.refresh_lock("inst-1", "a", timeout_seconds=600)
        (remaining,) = fake.table_rows(
            "SELECT CAST(strftime('%s', lock_expires_at) AS INTEGER) - "
            "CAST(strftime('%s', 'now') AS INTEGER) FROM workflow_instances"
        )[0]
        assert remaining > 500
        assert not await storage.refresh_lock("inst-1", "intruder")

    async def test_cleanup_releases_only_expired_locks(self, fake, storage):
        await make_instance(storage, "stale")
        await make_instance(storage, "fresh")
        await storage.try_acquire_lock("stale", "a")
        await storage.try_acquire_lock("fresh", "b")
        fake.db.execute(
            "UPDATE workflow_instances SET lock_expires_at = datetime('now', '-1 minute') "
            "WHERE instance_id = 'stale'"
        )
        resumed = await storage.cleanup_stale_locks()
        assert [w["instance_id"] for w in resumed] == ["stale"]
        assert fake.table_rows(
            "SELECT instance_id, locked_by FROM workflow_instances ORDER BY instance_id"
        ) == [("fresh", "b"), ("stale", None)]
        assert await storage.cleanup_stale_locks() == []

    async def test_cleanup_does_not_steal_a_refreshed_lease(self, fake, storage):
        """A lease renewed between the scan and the release must survive."""
        await make_instance(storage)
        await storage.try_acquire_lock("inst-1", "a")
        fake.db.execute(
            "UPDATE workflow_instances SET lock_expires_at = datetime('now', '-1 minute')"
        )

        released = False

        def renew_after_scan(request):
            nonlocal released
            # The scan (a read) has happened; the holder renews before the release lands.
            if request.path == "/db/query" and not released:
                released = True
                fake.db.execute(
                    "UPDATE workflow_instances SET lock_expires_at = datetime('now', '+5 minutes')"
                )

        fake.before_request.append(renew_after_scan)
        assert await storage.cleanup_stale_locks() == []
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("a",)]

    async def test_lease_times_follow_the_cluster_clock_not_this_machines(self, fake, storage):
        """A worker whose clock disagrees with the cluster's must still write leases the cluster honours."""
        fake.read_clock_skew_seconds = (
            120  # the cluster's UTC clock is two minutes ahead of this machine
        )
        await storage.calibrate_clock()
        assert 119 < storage.engine.dialect.clock_offset < 121
        await make_instance(storage)

        assert await storage.try_acquire_lock("inst-1", "a", timeout_seconds=300)

        (expires_in,) = fake.table_rows(
            "SELECT CAST(strftime('%s', lock_expires_at) AS INTEGER) - CAST(strftime('%s', 'now') AS INTEGER)"
            " FROM workflow_instances"
        )[0]
        assert 300 + 115 < expires_in < 300 + 125  # cluster time (+120s) plus the 300s lease

    async def test_a_lease_is_judged_on_the_same_clock_it_was_written_on(self, fake, storage):
        fake.read_clock_skew_seconds = 120
        await storage.calibrate_clock()
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "a", timeout_seconds=30)

        assert await storage.try_acquire_lock("inst-1", "b") is False  # not expired
        assert await storage.cleanup_stale_locks() == []


class TestWorkflowDefinitions:
    async def test_concurrent_first_starts_of_one_workflow_do_not_collide(self, fake, storage):
        """Several nodes registering the same workflow at once must all succeed.

        The base class checks for the row and then inserts, so two nodes that both see "missing"
        collide on the primary key.
        """
        await asyncio.gather(
            *[
                storage.upsert_workflow_definition("fresh_wf", "hash-1", f"source {n}")
                for n in range(10)
            ]
        )
        rows = fake.table_rows(
            "SELECT workflow_name, source_hash FROM workflow_definitions WHERE workflow_name = 'fresh_wf'"
        )
        assert rows == [("fresh_wf", "hash-1")]

    async def test_upsert_updates_the_source_of_an_existing_definition(self, fake, storage):
        await storage.upsert_workflow_definition("fresh_wf", "hash-1", "old")
        await storage.upsert_workflow_definition("fresh_wf", "hash-1", "new")
        assert fake.table_rows(
            "SELECT source_code FROM workflow_definitions WHERE workflow_name = 'fresh_wf'"
        ) == [("new",)]


class TestConcurrentStartup:
    async def test_many_nodes_initializing_one_empty_cluster(self, fake):
        """Every node runs initialize() when it boots; they must not trip over each other."""
        engines = [
            create_rqlite_engine(
                "rqlite://localhost:4001", connect_args={"transport": fake.transport()}
            )
            for _ in range(5)
        ]
        storages = [RqliteStorage(engine) for engine in engines]

        await asyncio.gather(*(s.initialize() for s in storages))

        assert fake.table_rows("SELECT count(*) FROM schema_version") == [(1,)]
        versions = fake.table_rows("SELECT version FROM schema_migrations")
        assert len(versions) == len(set(versions)) >= 1
        await asyncio.gather(*(s.close() for s in storages))


async def waiting_on_channel(storage, instance_id, channel="jobs", mode="competing"):
    """An instance that has subscribed to ``channel`` and is parked waiting for a message."""
    await make_instance(storage, instance_id)
    await storage.update_instance_status(instance_id, "running")
    worker = f"worker-{instance_id}"
    assert await storage.try_acquire_lock(instance_id, worker)
    await storage.subscribe_to_channel(instance_id, channel, mode)
    await storage.register_channel_receive_and_release_lock(
        instance_id, worker, channel, activity_id="receive:1"
    )


def transactional_writes(fake):
    """The write requests that were applied as one atomic batch, as statement-count per request."""
    return [len(r.statements) for r in fake.writes() if r.transaction]


class TestAtomicUnitsOfWork:
    """Edda documents several storage operations as one transaction. rqlite has no interactive
    transactions, so each must reach the cluster as ONE ``?transaction`` request, or a crash between
    its statements leaves half of it applied."""

    async def test_sleep_registration_is_one_request(self, fake, storage):
        await make_instance(storage, "sleeper")
        assert await storage.try_acquire_lock("sleeper", "w1")
        fake.requests.clear()

        await storage.register_timer_subscription_and_release_lock(
            "sleeper",
            "w1",
            "timer-1",
            datetime.now(UTC) + timedelta(seconds=30),
            activity_id="sleep:1",
        )

        assert transactional_writes(fake) == [
            3
        ]  # subscription, current activity, status + lock release
        assert len(fake.writes()) == 1  # and nothing outside that batch
        assert fake.table_rows("SELECT status, locked_by FROM workflow_instances") == [
            ("waiting_for_timer", None)
        ]
        assert fake.table_rows("SELECT timer_id FROM workflow_timer_subscriptions") == [
            ("timer-1",)
        ]

    async def test_wait_registration_is_one_request(self, fake, storage):
        await make_instance(storage, "waiter")
        await storage.update_instance_status("waiter", "running")
        assert await storage.try_acquire_lock("waiter", "w1")
        await storage.subscribe_to_channel("waiter", "jobs", "broadcast")
        fake.requests.clear()

        await storage.register_channel_receive_and_release_lock(
            "waiter", "w1", "jobs", activity_id="receive:1"
        )

        assert transactional_writes(fake) == [2]  # subscription activity, status + lock release
        assert len(fake.writes()) == 1
        assert fake.table_rows("SELECT status, locked_by FROM workflow_instances") == [
            ("waiting_for_message", None)
        ]

    async def test_archiving_history_is_one_request(self, fake, storage):
        await make_instance(storage)
        for n in range(3):
            await storage.append_history("inst-1", f"act:{n}", "ActivityCompleted", {"n": n})
        fake.requests.clear()

        assert await storage.archive_history("inst-1") == 3

        assert transactional_writes(fake) == [4]  # three archive rows, one delete
        assert len(fake.writes()) == 1
        assert fake.table_rows("SELECT COUNT(*) FROM workflow_history") == [(0,)]
        assert fake.table_rows("SELECT COUNT(*) FROM workflow_history_archive") == [(3,)]

    async def test_delivering_a_message_is_one_request(self, fake, storage):
        await waiting_on_channel(storage, "recv", mode="broadcast")
        message_id = await storage.publish_to_channel("jobs", {"n": 1})
        fake.requests.clear()

        result = await storage.deliver_channel_message(
            "recv", "jobs", message_id, {"n": 1}, {}, "w2"
        )

        assert result is not None
        assert transactional_writes(fake) == [
            3
        ]  # history, subscription (cursor + activity), status
        assert fake.table_rows("SELECT status FROM workflow_instances") == [("running",)]
        assert fake.table_rows("SELECT event_type FROM workflow_history") == [
            ("ChannelMessageReceived",)
        ]

    async def test_delivery_of_a_direct_message_is_one_request(self, fake, storage):
        await waiting_on_channel(storage, "recv", mode="broadcast")
        fake.requests.clear()

        result = await storage.deliver_message("recv", "jobs", {"n": 1}, {}, worker_id="w2")

        assert result is not None
        assert 3 in transactional_writes(fake)  # history, subscription, status in one batch
        assert fake.table_rows("SELECT event_type FROM workflow_history") == [
            ("ChannelMessageReceived",)
        ]

    async def test_losing_competing_consumer_leaves_no_trace(self, fake, storage):
        """Two instances compete for one message. The loser's claim insert fails, and with it must go
        everything else the loser wrote, or its replay finds a message it never received."""
        await waiting_on_channel(storage, "a")
        await waiting_on_channel(storage, "b")
        message_id = await storage.publish_to_channel("jobs", {"n": 1})

        first = await storage.deliver_channel_message("a", "jobs", message_id, {"n": 1}, {}, "w1")
        second = await storage.deliver_channel_message("b", "jobs", message_id, {"n": 1}, {}, "w2")

        assert first is not None and second is None
        assert fake.table_rows("SELECT instance_id FROM workflow_history") == [("a",)]
        assert fake.table_rows(
            "SELECT instance_id, status FROM workflow_instances ORDER BY instance_id"
        ) == [("a", "running"), ("b", "waiting_for_message")]
        subscription = await storage.get_channel_subscription("b", "jobs")
        assert subscription["activity_id"] == "receive:1"  # still waiting

    async def test_concurrent_competing_deliveries_pick_one_winner(self, fake, storage):
        names = [f"c{n}" for n in range(6)]
        for name in names:
            await waiting_on_channel(storage, name)
        message_id = await storage.publish_to_channel("jobs", {"n": 1})

        results = await asyncio.gather(
            *[
                storage.deliver_channel_message(n, "jobs", message_id, {"n": 1}, {}, f"w-{n}")
                for n in names
            ]
        )

        assert sum(r is not None for r in results) == 1
        assert len(fake.table_rows("SELECT * FROM workflow_history")) == 1
        waiting = fake.table_rows(
            "SELECT COUNT(*) FROM workflow_instances WHERE status = 'waiting_for_message'"
        )
        assert waiting == [(5,)]

    async def test_cancelling_a_sleeper_is_one_request(self, fake, storage):
        await make_instance(storage, "sleeper")
        assert await storage.try_acquire_lock("sleeper", "w1")
        await storage.register_timer_subscription_and_release_lock(
            "sleeper",
            "w1",
            "timer-1",
            datetime.now(UTC) + timedelta(seconds=30),
            activity_id="sleep:1",
        )
        fake.requests.clear()

        assert await storage.cancel_instance("sleeper", "alice") is True

        assert transactional_writes(fake) == [2]  # drop the timer, mark cancelled
        assert len(fake.writes()) == 1
        assert fake.table_rows("SELECT status FROM workflow_instances") == [("cancelled",)]
        assert fake.table_rows("SELECT COUNT(*) FROM workflow_timer_subscriptions") == [(0,)]

    async def test_cancelling_races_have_one_winner(self, fake, storage):
        await waiting_on_channel(storage, "target", mode="broadcast")

        results = await asyncio.gather(
            *[storage.cancel_instance("target", f"user-{n}") for n in range(5)]
        )

        assert sum(results) == 1
        assert fake.table_rows("SELECT status FROM workflow_instances") == [("cancelled",)]

    async def test_cancelling_a_finished_workflow_is_refused(self, fake, storage):
        await make_instance(storage)
        await storage.update_instance_status("inst-1", "completed")
        assert await storage.cancel_instance("inst-1", "alice") is False
        assert await storage.cancel_instance("missing", "alice") is False


def lose_next_response(fake, path="/db/execute"):
    """The next ``path`` request is applied by the cluster, but its answer never arrives."""
    fired = []

    def hook(record):
        if record.path == path and not fired:
            fired.append(record)
            raise httpx2.ReadError("connection reset before the response", request=None)

    fake.after_request.append(hook)
    return fired


class TestAmbiguousOutcomes:
    """A write whose response is lost may or may not have been applied. For a claim that matters:
    treating an applied claim as failed leaves the lock held by a worker that thinks it has none."""

    async def test_lock_acquired_but_response_lost_is_still_acquired(self, fake, storage):
        await make_instance(storage)
        lost = lose_next_response(fake)

        assert await storage.try_acquire_lock("inst-1", "w1") is True

        assert lost, "the injected failure never fired"
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("w1",)]
        assert await storage.try_acquire_lock("inst-1", "w2") is False  # and it is a real lock

    async def test_lock_request_lost_before_it_applied_is_retried(self, fake, storage):
        await make_instance(storage)
        fired = []

        def drop_before_apply(record):
            if record.path == "/db/execute" and not fired:
                fired.append(record)
                raise httpx2.ReadError("connection reset", request=None)

        fake.before_request.append(drop_before_apply)

        # Not held after the first attempt, so the claim is repeated and wins.
        assert await storage.try_acquire_lock("inst-1", "w1") is True
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("w1",)]

    async def test_a_retried_claim_cannot_take_a_lock_someone_else_holds(self, fake, storage):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "w1")
        lose_next_response(fake)

        assert await storage.try_acquire_lock("inst-1", "w2") is False
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("w1",)]

    async def test_system_lock_acquired_but_response_lost_is_still_acquired(self, fake, storage):
        await storage.try_acquire_system_lock("task", "w0", 60)
        await storage.release_system_lock("task", "w0")
        lose_next_response(fake)  # the INSERT OR IGNORE goes first; lose whichever write comes next

        assert await storage.try_acquire_system_lock("task", "w1", 60) is True
        assert await storage.try_acquire_system_lock("task", "w2", 60) is False

    async def test_release_whose_response_was_lost_still_releases(self, fake, storage):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "w1")
        lose_next_response(fake)

        await storage.release_lock("inst-1", "w1")

        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [(None,)]

    async def test_refresh_whose_response_was_lost_reports_the_truth(self, fake, storage):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "w1", timeout_seconds=5)
        lose_next_response(fake)

        assert await storage.refresh_lock("inst-1", "w1", timeout_seconds=600) is True

    async def test_a_commit_whose_response_was_lost_is_a_success(self, fake, storage):
        """The activity's batch was applied; only the answer was lost. Repeating the activity would hit
        its own history row, so the commit must report success."""
        await make_instance(storage)
        await storage.begin_transaction()
        await storage.append_history("inst-1", "act:0", "ActivityCompleted", {})
        await storage.commit_transaction()  # also creates the marker table, once
        await storage.begin_transaction()
        await storage.append_history("inst-1", "act:1", "ActivityCompleted", {})
        lost = lose_next_response(fake)

        await storage.commit_transaction()  # must not raise

        assert lost, "the injected failure never fired"
        assert fake.table_rows("SELECT activity_id FROM workflow_history ORDER BY id") == [
            ("act:0",),
            ("act:1",),
        ]
        assert not storage.in_transaction()

    async def test_a_commit_that_never_arrived_is_a_failure(self, fake, storage):
        await make_instance(storage)
        await storage.begin_transaction()
        await storage.append_history("inst-1", "act:0", "ActivityCompleted", {})
        await storage.commit_transaction()
        await storage.begin_transaction()
        await storage.append_history("inst-1", "act:1", "ActivityCompleted", {})
        fired = []

        def drop_before_apply(record):
            if record.path == "/db/execute" and not fired:
                fired.append(record)
                raise httpx2.ReadError("connection reset", request=None)

        fake.before_request.append(drop_before_apply)

        with pytest.raises(OperationalError):
            await storage.commit_transaction()
        await storage.rollback_transaction()

        assert fake.table_rows("SELECT activity_id FROM workflow_history") == [("act:0",)]

    async def test_old_commit_markers_are_purged(self, fake, storage):
        await make_instance(storage)
        await storage.begin_transaction()
        await storage.append_history("inst-1", "act:0", "ActivityCompleted", {})
        await storage.commit_transaction()  # creates the table
        fake.db.execute("INSERT INTO edda_commit_markers VALUES ('ancient', '2020-01-01 00:00:00')")
        storage.engine.dialect.dbapi.batches = 99  # the next batch is the 100th

        await storage.begin_transaction()
        await storage.append_history("inst-1", "act:1", "ActivityCompleted", {})
        await storage.commit_transaction()

        tokens = [t for (t,) in fake.table_rows("SELECT token FROM edda_commit_markers")]
        assert "ancient" not in tokens and len(tokens) == 2

    async def test_a_status_update_whose_response_was_lost_is_repeated(self, fake, storage):
        """The end of every workflow is a status write. Losing its answer must not fail the workflow."""
        await make_instance(storage)
        lost = lose_next_response(fake)

        await storage.update_instance_status("inst-1", "completed", {"result": 1})

        assert lost, "the injected failure never fired"
        assert fake.table_rows("SELECT status FROM workflow_instances") == [("completed",)]

    async def test_creating_an_instance_whose_response_was_lost_is_not_repeated_into_a_collision(
        self, fake, storage
    ):
        lose_next_response(fake)

        await make_instance(storage)  # must neither raise nor insert twice

        assert fake.table_rows("SELECT COUNT(*) FROM workflow_instances") == [(1,)]

    async def test_creating_an_instance_whose_request_never_arrived_is_repeated(
        self, fake, storage
    ):
        fired = []

        def drop_before_apply(record):
            if record.path == "/db/execute" and not fired:
                fired.append(record)
                raise httpx2.ReadError("connection reset", request=None)

        fake.before_request.append(drop_before_apply)

        await make_instance(storage)

        assert fake.table_rows("SELECT instance_id FROM workflow_instances") == [("inst-1",)]

    async def test_a_counter_is_not_repeated(self, fake, storage):
        """mark_outbox_failed adds 1 to retry_count; repeating it after it applied would add 2."""
        await storage.add_outbox_event("e1", "t", "svc", {}, "application/json")
        lose_next_response(fake)

        with pytest.raises(OperationalError):
            await storage.mark_outbox_failed("e1", "boom")

        assert fake.table_rows("SELECT retry_count FROM outbox_events") == [(1,)]

    async def test_a_read_on_a_dropped_connection_is_repeated(self, fake, storage):
        await make_instance(storage)
        fired = []

        def drop_read(record):
            if record.path == "/db/query" and len(fired) < 2:
                fired.append(record)
                raise httpx2.ReadError("connection reset", request=None)

        fake.before_request.append(drop_read)

        assert await storage.get_instance("inst-1") is not None
        assert len(fired) == 2  # it took three tries

    async def test_errors_that_are_not_ambiguous_still_raise(self, fake, storage):
        await make_instance(storage)
        fake.down_hosts.add("localhost")  # nothing is reachable: the request provably never left

        with pytest.raises(Exception, match="connect"):
            await storage.try_acquire_lock("inst-1", "w1")


class TestNoWallClockInWrites:
    """A real node rewrites ``now`` in a write to its OWN LOCAL time, written as if UTC (measured:
    four hours early on a node in US Eastern). The fake does the same, so any statement that sends
    ``now`` or ``CURRENT_TIMESTAMP`` in a write stores a wrong time here too."""

    async def test_no_write_statement_carries_now(self, fake, storage):
        await make_instance(storage)
        fake.requests.clear()
        await storage.try_acquire_lock("inst-1", "a")
        await storage.refresh_lock("inst-1", "a")
        await storage.try_acquire_system_lock("task", "a")
        await storage.append_history("inst-1", "act:1", "ActivityCompleted", {})
        await storage.add_outbox_event("e1", "t", "svc", {}, "application/json")
        await storage.update_instance_status("inst-1", "running")
        await storage.update_instance_activity("inst-1", "act:2")
        await storage.get_pending_outbox_events()
        await storage.mark_outbox_published("e1")
        await storage.cleanup_stale_locks()
        await storage.cancel_instance("inst-1", "alice")
        await storage.release_lock("inst-1", "a")
        await storage.release_system_lock("task", "a")

        writes = [sql for r in fake.writes() for sql, _ in r.statements]
        assert writes
        offenders = [sql for sql in writes if re.search(r"CURRENT_TIMESTAMP|'now'", sql, re.I)]
        assert offenders == []

    async def test_stored_times_are_utc_even_though_the_node_rewrites_now_to_local_time(
        self, fake, storage
    ):
        await make_instance(storage)
        assert await storage.try_acquire_lock("inst-1", "a")
        await storage.append_history("inst-1", "act:1", "ActivityCompleted", {})

        for sql in (
            "SELECT started_at FROM workflow_instances",
            "SELECT updated_at FROM workflow_instances",
            "SELECT locked_at FROM workflow_instances",
            "SELECT created_at FROM workflow_history",
        ):
            (stored,) = fake.table_rows(
                f"SELECT CAST(strftime('%s', t) AS INTEGER) FROM ({sql.replace(' FROM', ' AS t FROM', 1)})"
            )[0]
            assert abs(stored - datetime.now(UTC).timestamp()) < 5, sql


class TestSystemLocks:
    async def test_single_winner_and_renewal(self, storage):
        results = await asyncio.gather(
            *[storage.try_acquire_system_lock("cleanup", f"w{n}", 60) for n in range(8)]
        )
        assert sum(results) == 1
        winner = f"w{results.index(True)}"
        assert await storage.try_acquire_system_lock("cleanup", winner, 60)  # renewal
        assert not await storage.try_acquire_system_lock("cleanup", "other", 60)

    async def test_expired_system_lock_is_taken_over(self, fake, storage):
        assert await storage.try_acquire_system_lock("cleanup", "a", 60)
        fake.db.execute("UPDATE system_locks SET lock_expires_at = datetime('now', '-1 second')")
        assert await storage.try_acquire_system_lock("cleanup", "b", 60)

    async def test_release(self, storage):
        await storage.try_acquire_system_lock("cleanup", "a", 60)
        await storage.release_system_lock("cleanup", "a")
        assert await storage.try_acquire_system_lock("cleanup", "b", 60)


class TestOutboxClaims:
    async def add_events(self, storage, count):
        for i in range(count):
            await storage.add_outbox_event(
                event_id=str(uuid.uuid4()),
                event_type=f"t.{i}",
                event_source="svc",
                event_data={"i": i},
                content_type="application/json",
            )

    async def test_concurrent_workers_never_share_an_event(self, storage):
        await self.add_events(storage, 12)

        async def drain(limit):
            claimed = []
            for _ in range(6):
                batch = await storage.get_pending_outbox_events(limit=limit)
                if not batch:
                    break
                claimed += [e["event_id"] for e in batch]
            return claimed

        results = await asyncio.gather(drain(5), drain(5), drain(5))
        flat = [eid for claimed in results for eid in claimed]
        assert len(flat) == len(set(flat)) == 12

    async def test_claimed_events_are_marked_processing(self, fake, storage):
        await self.add_events(storage, 3)
        events = await storage.get_pending_outbox_events(limit=10)
        assert {e["status"] for e in events} == {"processing"}
        assert fake.table_rows("SELECT DISTINCT status FROM outbox_events") == [("processing",)]
        assert await storage.get_pending_outbox_events(limit=10) == []

    async def test_failed_events_can_be_claimed_again(self, storage):
        await self.add_events(storage, 1)
        (event,) = await storage.get_pending_outbox_events()
        await storage.mark_outbox_failed(event["event_id"], "boom")
        (again,) = await storage.get_pending_outbox_events()
        assert again["event_id"] == event["event_id"] and again["retry_count"] == 1

    async def test_oldest_first_and_limit(self, storage):
        await self.add_events(storage, 5)
        batch = await storage.get_pending_outbox_events(limit=2)
        assert [e["event_data"]["i"] for e in batch] == [0, 1]


# ---------------------------------------------------------------------------
# Edda transactions on rqlite
# ---------------------------------------------------------------------------


class TestEddaTransactions:
    async def test_activity_writes_commit_as_one_atomic_request(self, fake, storage):
        await make_instance(storage)
        fake.requests.clear()
        await storage.begin_transaction()
        await storage.append_history("inst-1", "a:1", "ActivityCompleted", {"ok": 1})
        await storage.append_history("inst-1", "a:2", "ActivityCompleted", {"ok": 2})
        await storage.add_outbox_event("evt-1", "t", "svc", {"x": 1}, "application/json")
        assert fake.writes() == []  # nothing leaves the process before commit
        await storage.commit_transaction()

        (request,) = fake.writes()
        assert request.transaction
        assert [s.split()[2] for s, _ in request.statements] == [
            "workflow_history",
            "workflow_history",
            "outbox_events",
        ]
        # Database-generated ids are real and in insertion order.
        assert fake.table_rows("SELECT id, activity_id FROM workflow_history ORDER BY id") == [
            (1, "a:1"),
            (2, "a:2"),
        ]

    async def test_rollback_persists_nothing(self, fake, storage):
        await make_instance(storage)
        fake.requests.clear()
        await storage.begin_transaction()
        await storage.append_history("inst-1", "a:1", "ActivityCompleted", {})
        await storage.add_outbox_event("evt-1", "t", "svc", {}, "application/json")
        await storage.rollback_transaction()
        assert fake.writes() == []
        assert fake.table_rows("SELECT * FROM workflow_history") == []
        assert fake.table_rows("SELECT * FROM outbox_events") == []

    async def test_failed_commit_leaves_no_partial_state(self, fake, storage):
        """The transactional-outbox guarantee: history and outbox land together or not at all."""
        await make_instance(storage)
        await storage.add_outbox_event("dup", "t", "svc", {}, "application/json")
        # Same flow as Context.transaction(): a failed commit is followed by a rollback.
        await storage.begin_transaction()
        with pytest.raises(IntegrityError):
            try:
                await storage.append_history("inst-1", "a:1", "ActivityCompleted", {})
                await storage.add_outbox_event(
                    "dup", "t", "svc", {}, "application/json"
                )  # PK clash
                await storage.commit_transaction()
            except Exception:
                await storage.rollback_transaction()
                raise
        assert fake.table_rows("SELECT * FROM workflow_history") == []
        assert not storage.in_transaction()
        assert storage.engine.pool.checkedout() == 0  # nothing leaked

    async def test_nested_transaction_rolls_back_to_savepoint(self, fake, storage):
        await make_instance(storage)
        await storage.begin_transaction()
        await storage.append_history("inst-1", "outer", "ActivityCompleted", {})
        await storage.begin_transaction()
        await storage.append_history("inst-1", "inner", "ActivityCompleted", {})
        await storage.rollback_transaction()
        await storage.commit_transaction()
        assert fake.table_rows("SELECT activity_id FROM workflow_history") == [("outer",)]

    async def test_post_commit_callbacks_run_only_after_commit(self, storage):
        await make_instance(storage)
        calls = []

        async def callback():
            calls.append("ran")

        await storage.begin_transaction()
        storage.register_post_commit_callback(callback)
        await storage.append_history("inst-1", "a:1", "ActivityCompleted", {})
        assert calls == []
        await storage.commit_transaction()
        assert calls == ["ran"]

    async def test_locks_taken_inside_a_transaction_apply_immediately(self, fake, storage):
        """Claims need an exact rowcount, so they must not be queued behind the commit."""
        await make_instance(storage)
        await storage.begin_transaction()
        assert await storage.try_acquire_lock("inst-1", "a")
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("a",)]
        assert not await storage.try_acquire_lock("inst-1", "b")
        await storage.rollback_transaction()
        assert fake.table_rows("SELECT locked_by FROM workflow_instances") == [("a",)]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_url_schemes(self):
        register_rqlite_dialect()
        plain = create_rqlite_engine("rqlite://user:pw@h1:4001/?level=strong&host=h2:4001")
        assert plain.url.drivername == "rqlite+httpx"
        args = plain.dialect.create_connect_args(plain.url)[1]
        assert args["base_urls"] == ["http://h1:4001", "http://h2:4001"]
        assert (args["username"], args["password"], args["level"]) == ("user", "pw", "strong")

        secure = create_rqlite_engine("rqlites://h1:4001")
        assert secure.dialect.create_connect_args(secure.url)[1]["base_urls"] == ["https://h1:4001"]

    def test_storage_rejects_other_engines(self):
        with pytest.raises(ValueError, match="rqlite"):
            RqliteStorage(create_async_engine("sqlite+aiosqlite:///:memory:"))

    async def test_edda_app_selects_rqlite_storage(self):
        from edda import EddaApp

        app = EddaApp(service_name="svc", db_url="rqlite://localhost:4001")
        assert isinstance(app.storage, RqliteStorage)
        await app.storage.close()

    async def test_migrations_use_sqlite_scripts(self, fake, engine):
        applied = await apply_dbmate_migrations(engine, SCHEMA_DIR)
        assert applied
        assert fake.table_rows(
            "SELECT count(*) FROM sqlite_master WHERE name = 'workflow_instances'"
        ) == [(1,)]
        # idempotent: a second worker starting up applies nothing and does not fail
        assert await apply_dbmate_migrations(engine, SCHEMA_DIR) == []

    async def test_pool_check_query_works(self, engine):
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar() == 1
