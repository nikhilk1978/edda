"""
Pytest configuration and fixtures for Edda tests.
"""

import os
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from edda.serialization.json import JSONSerializer
from edda.storage.migrations import apply_dbmate_migrations
from edda.storage.sqlalchemy_storage import SQLAlchemyStorage

# Path to schema migrations (relative to project root)
SCHEMA_DIR = Path(__file__).parent.parent / "schema" / "db" / "migrations"


def get_database_urls() -> dict[str, str]:
    """Get database connection URLs from environment variables."""
    return {
        "sqlite": "sqlite+aiosqlite:///:memory:",
        "postgresql": os.getenv(
            "EDDA_TEST_POSTGRES_URL",
            "postgresql+asyncpg://edda:edda_test_password@localhost:5432/edda_test",
        ),
        "mysql": os.getenv(
            "EDDA_TEST_MYSQL_URL",
            "mysql+aiomysql://edda:edda_test_password@localhost:3306/edda_test",
        ),
    }


# Tables in the order they can be emptied (children first, for foreign keys).
EDDA_TABLES = (
    "channel_message_claims",
    "channel_delivery_cursors",
    "channel_messages",
    "channel_subscriptions",
    "workflow_timer_subscriptions",
    "workflow_group_memberships",
    "workflow_compensations",
    "workflow_history_archive",
    "workflow_history",
    "outbox_events",
    "workflow_instances",
    "workflow_definitions",
    "system_locks",
)


async def open_rqlite_storage():
    """A migrated ``RqliteStorage`` plus a ``cleanup`` coroutine function.

    Uses the real cluster in EDDA_TEST_RQLITE_URL if set (its rows are deleted afterwards, so use a
    dedicated one), otherwise a fresh in-process fake (tests/rqlite_fake.py) per call.
    """
    from edda.storage.rqlite_storage import RqliteStorage, create_rqlite_engine

    real_url = os.getenv("EDDA_TEST_RQLITE_URL")
    fake = None
    if real_url:
        engine = create_rqlite_engine(real_url)
    else:
        from tests.rqlite_fake import FakeRqlite

        fake = FakeRqlite()
        engine = create_rqlite_engine(
            "rqlite://localhost:4001", connect_args={"transport": fake.transport()}
        )

    await apply_dbmate_migrations(engine, SCHEMA_DIR)
    storage = RqliteStorage(engine)
    await storage.upsert_workflow_definition(
        workflow_name="test_workflow",
        source_hash="abc123def456",
        source_code="async def test_workflow(ctx): pass",
    )

    async def cleanup() -> None:
        if real_url:
            # One atomic request: on a real cluster every request is a network round trip.
            async with AsyncSession(storage._atomic_engine) as session:
                for table in EDDA_TABLES:
                    await session.execute(text(f"DELETE FROM {table}"))
                # Let autoincrement ids start again, as they do on a fresh database.
                await session.execute(text("DELETE FROM sqlite_sequence"))
                await session.commit()
        await storage.close()
        if fake is not None:
            fake.close()

    return storage, cleanup


@pytest_asyncio.fixture(params=["sqlite", "postgresql", "mysql", "rqlite"])
async def db_storage(request):
    """
    Parametrized fixture that creates storage for SQLite, PostgreSQL, MySQL, and rqlite.

    This fixture will run each test 4 times (once for each database).
    Uses Testcontainers for PostgreSQL and MySQL. rqlite runs against an in-process
    fake cluster (tests/rqlite_fake.py), or a real cluster when EDDA_TEST_RQLITE_URL
    is set (e.g. rqlite://localhost:4001); its tables are emptied afterwards.
    Schema is created using dbmate migration files from schema/db/migrations/.
    """
    db_type = request.param

    if db_type == "rqlite":
        storage, cleanup = await open_rqlite_storage()
        yield storage
        await cleanup()

    elif db_type == "sqlite":
        # SQLite: in-memory database
        # Use StaticPool to ensure all connections share the same in-memory database
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            poolclass=StaticPool,
        )

        # Apply migrations from SQL files
        await apply_dbmate_migrations(engine, SCHEMA_DIR)

        storage = SQLAlchemyStorage(engine)

        # Create a sample workflow definition for tests
        await storage.upsert_workflow_definition(
            workflow_name="test_workflow",
            source_hash="abc123def456",
            source_code="async def test_workflow(ctx): pass",
        )

        yield storage
        await storage.close()

    elif db_type == "postgresql":
        # PostgreSQL with Testcontainers
        try:
            import asyncpg  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("asyncpg not installed")

        try:
            from testcontainers.postgres import PostgresContainer
        except ModuleNotFoundError:
            pytest.skip("testcontainers not installed")

        with PostgresContainer(
            "postgres:17", username="edda", password="edda_test_password", dbname="edda_test"
        ) as postgres:
            db_url = postgres.get_connection_url().replace("psycopg2", "asyncpg")
            # Use READ COMMITTED to match production and avoid snapshot issues
            engine = create_async_engine(db_url, echo=False, isolation_level="READ COMMITTED")

            # Apply migrations from SQL files
            await apply_dbmate_migrations(engine, SCHEMA_DIR)

            storage = SQLAlchemyStorage(engine)

            await storage.upsert_workflow_definition(
                workflow_name="test_workflow",
                source_hash="abc123def456",
                source_code="async def test_workflow(ctx): pass",
            )

            yield storage

            # Cleanup (order matters due to foreign key constraints)
            async with AsyncSession(storage.engine) as session:
                await session.execute(text("DELETE FROM channel_message_claims"))
                await session.execute(text("DELETE FROM channel_delivery_cursors"))
                await session.execute(text("DELETE FROM channel_messages"))
                await session.execute(text("DELETE FROM channel_subscriptions"))
                await session.execute(text("DELETE FROM workflow_timer_subscriptions"))
                await session.execute(text("DELETE FROM workflow_group_memberships"))
                await session.execute(text("DELETE FROM workflow_compensations"))
                await session.execute(text("DELETE FROM workflow_history_archive"))
                await session.execute(text("DELETE FROM workflow_history"))
                await session.execute(text("DELETE FROM outbox_events"))
                await session.execute(text("DELETE FROM workflow_instances"))
                await session.execute(text("DELETE FROM workflow_definitions"))
                await session.commit()

            await storage.close()

    elif db_type == "mysql":
        # MySQL with Testcontainers
        try:
            import aiomysql  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("aiomysql not installed")

        try:
            from testcontainers.mysql import MySqlContainer
        except ModuleNotFoundError:
            pytest.skip("testcontainers not installed")

        with MySqlContainer(
            "mysql:9", username="edda", password="edda_test_password", dbname="edda_test"
        ) as mysql:
            db_url = mysql.get_connection_url().replace("mysql://", "mysql+aiomysql://")
            # Use READ COMMITTED instead of REPEATABLE READ to avoid snapshot issues with SKIP LOCKED
            engine = create_async_engine(db_url, echo=False, isolation_level="READ COMMITTED")

            # Apply migrations from SQL files
            await apply_dbmate_migrations(engine, SCHEMA_DIR)

            storage = SQLAlchemyStorage(engine)

            await storage.upsert_workflow_definition(
                workflow_name="test_workflow",
                source_hash="abc123def456",
                source_code="async def test_workflow(ctx): pass",
            )

            yield storage

            # Cleanup (order matters due to foreign key constraints)
            async with AsyncSession(storage.engine) as session:
                await session.execute(text("DELETE FROM channel_message_claims"))
                await session.execute(text("DELETE FROM channel_delivery_cursors"))
                await session.execute(text("DELETE FROM channel_messages"))
                await session.execute(text("DELETE FROM channel_subscriptions"))
                await session.execute(text("DELETE FROM workflow_timer_subscriptions"))
                await session.execute(text("DELETE FROM workflow_group_memberships"))
                await session.execute(text("DELETE FROM workflow_compensations"))
                await session.execute(text("DELETE FROM workflow_history_archive"))
                await session.execute(text("DELETE FROM workflow_history"))
                await session.execute(text("DELETE FROM outbox_events"))
                await session.execute(text("DELETE FROM workflow_instances"))
                await session.execute(text("DELETE FROM workflow_definitions"))
                await session.commit()

            await storage.close()


@pytest_asyncio.fixture
async def sqlite_storage():
    """Create an in-memory SQLite storage for testing.

    EDDA_TEST_STORAGE=rqlite runs every test that uses this fixture on ``RqliteStorage`` instead
    (the in-process fake, or the cluster in EDDA_TEST_RQLITE_URL; see ``open_rqlite_storage``).
    """
    if os.getenv("EDDA_TEST_STORAGE") == "rqlite":
        storage, cleanup = await open_rqlite_storage()
        yield storage
        await cleanup()
        return

    # Use StaticPool to ensure all connections share the same in-memory database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
    )

    # Apply migrations from SQL files
    await apply_dbmate_migrations(engine, SCHEMA_DIR)

    storage = SQLAlchemyStorage(engine)

    # Create a sample workflow definition for tests
    await storage.upsert_workflow_definition(
        workflow_name="test_workflow",
        source_hash="abc123def456",  # Must match sample_workflow_data
        source_code="async def test_workflow(ctx): pass",
    )

    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def postgresql_storage():
    """Create a PostgreSQL storage for testing with Testcontainers."""
    # Check if asyncpg driver is installed
    try:
        import asyncpg  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("asyncpg driver not installed. " "Install with: uv sync --extra postgresql")

    # Try to use Testcontainers
    try:
        from testcontainers.postgres import PostgresContainer
    except ModuleNotFoundError:
        pytest.skip("testcontainers not installed. " "Install with: uv sync --extra dev")

    # Start PostgreSQL container
    with PostgresContainer(
        "postgres:17", username="edda", password="edda_test_password", dbname="edda_test"
    ) as postgres:
        # Get connection URL and replace psycopg2 with asyncpg
        db_url = postgres.get_connection_url().replace("psycopg2", "asyncpg")
        engine = create_async_engine(db_url, echo=False, isolation_level="READ COMMITTED")

        # Apply migrations from SQL files
        await apply_dbmate_migrations(engine, SCHEMA_DIR)

        storage = SQLAlchemyStorage(engine)

        # Create a sample workflow definition for tests
        await storage.upsert_workflow_definition(
            workflow_name="test_workflow",
            source_hash="abc123def456",
            source_code="async def test_workflow(ctx): pass",
        )

        yield storage

        # Cleanup: Truncate all tables (order matters due to foreign key constraints)
        async with AsyncSession(storage.engine) as session:
            await session.execute(text("DELETE FROM channel_message_claims"))
            await session.execute(text("DELETE FROM channel_delivery_cursors"))
            await session.execute(text("DELETE FROM channel_messages"))
            await session.execute(text("DELETE FROM channel_subscriptions"))
            await session.execute(text("DELETE FROM workflow_timer_subscriptions"))
            await session.execute(text("DELETE FROM workflow_group_memberships"))
            await session.execute(text("DELETE FROM workflow_compensations"))
            await session.execute(text("DELETE FROM workflow_history_archive"))
            await session.execute(text("DELETE FROM workflow_history"))
            await session.execute(text("DELETE FROM outbox_events"))
            await session.execute(text("DELETE FROM workflow_instances"))
            await session.execute(text("DELETE FROM workflow_definitions"))
            await session.commit()

        await storage.close()
    # Container automatically stopped here


@pytest_asyncio.fixture
async def mysql_storage():
    """Create a MySQL storage for testing with Testcontainers."""
    # Check if aiomysql driver is installed
    try:
        import aiomysql  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("aiomysql driver not installed. " "Install with: uv sync --extra mysql")

    # Try to use Testcontainers
    try:
        from testcontainers.mysql import MySqlContainer
    except ModuleNotFoundError:
        pytest.skip("testcontainers not installed. " "Install with: uv sync --extra dev")

    # Start MySQL container
    with MySqlContainer(
        "mysql:9", username="edda", password="edda_test_password", dbname="edda_test"
    ) as mysql:
        # Get connection URL and add aiomysql driver
        db_url = mysql.get_connection_url().replace("mysql://", "mysql+aiomysql://")
        engine = create_async_engine(db_url, echo=False, isolation_level="READ COMMITTED")

        # Apply migrations from SQL files
        await apply_dbmate_migrations(engine, SCHEMA_DIR)

        storage = SQLAlchemyStorage(engine)

        # Create a sample workflow definition for tests
        await storage.upsert_workflow_definition(
            workflow_name="test_workflow",
            source_hash="abc123def456",
            source_code="async def test_workflow(ctx): pass",
        )

        yield storage

        # Cleanup: Truncate all tables (order matters due to foreign key constraints)
        async with AsyncSession(storage.engine) as session:
            await session.execute(text("DELETE FROM channel_message_claims"))
            await session.execute(text("DELETE FROM channel_delivery_cursors"))
            await session.execute(text("DELETE FROM channel_messages"))
            await session.execute(text("DELETE FROM channel_subscriptions"))
            await session.execute(text("DELETE FROM workflow_timer_subscriptions"))
            await session.execute(text("DELETE FROM workflow_group_memberships"))
            await session.execute(text("DELETE FROM workflow_compensations"))
            await session.execute(text("DELETE FROM workflow_history_archive"))
            await session.execute(text("DELETE FROM workflow_history"))
            await session.execute(text("DELETE FROM outbox_events"))
            await session.execute(text("DELETE FROM workflow_instances"))
            await session.execute(text("DELETE FROM workflow_definitions"))
            await session.commit()

        await storage.close()
    # Container automatically stopped here


@pytest.fixture
def json_serializer():
    """Create a JSON serializer for testing."""
    return JSONSerializer()


@pytest.fixture
def sample_workflow_data():
    """Sample workflow instance data for testing."""
    return {
        "instance_id": "test-instance-123",
        "workflow_name": "test_workflow",
        "source_hash": "abc123def456",  # Mock hash for testing
        "owner_service": "test-service",
        "input_data": {"order_id": "order-123", "amount": 100},
    }


@pytest.fixture
def sample_event_data():
    """Sample event data for testing."""
    return {
        "event_type": "payment.completed",
        "event_source": "payment-service",
        "event_data": {"order_id": "order-123", "payment_id": "pay-456"},
    }


@pytest_asyncio.fixture
async def create_test_instance(sqlite_storage):
    """
    Helper fixture to create workflow instances with proper workflow definitions.

    Returns a function that can be called to create instances.
    Usage:
        instance_id = await create_test_instance(
            instance_id="test-123",
            workflow_name="my_workflow",
            input_data={"key": "value"}
        )
    """

    async def _create_instance(
        instance_id: str,
        workflow_name: str,
        input_data: dict,
        owner_service: str = "test-service",
        source_hash: str | None = None,
        source_code: str | None = None,
    ) -> str:
        """Create a test instance with workflow definition."""
        # Use default values if not provided
        if source_hash is None:
            source_hash = "test-hash-" + workflow_name
        if source_code is None:
            source_code = f"async def {workflow_name}(ctx): pass"

        # Ensure workflow definition exists
        await sqlite_storage.upsert_workflow_definition(
            workflow_name=workflow_name,
            source_hash=source_hash,
            source_code=source_code,
        )

        # Create instance
        await sqlite_storage.create_instance(
            instance_id=instance_id,
            workflow_name=workflow_name,
            source_hash=source_hash,
            owner_service=owner_service,
            input_data=input_data,
        )

        return instance_id

    return _create_instance
