"""
SQLAlchemy dialect and asyncio driver for rqlite.

This lets ``SQLAlchemyStorage`` run unchanged on an rqlite cluster: SQL is compiled
by SQLAlchemy's SQLite dialect and each statement is carried over HTTP by
``RqliteClient``. Register it with ``register_rqlite_dialect()`` and use a URL like
``rqlite+httpx://user:pass@host:4001``.

Execution modes (selected through SQLAlchemy's ``isolation_level``):

``AUTOCOMMIT`` (default)
    Every statement is sent as soon as it is executed. ``rowcount`` and ``lastrowid``
    are exact, which compare-and-set statements (``UPDATE ... WHERE ...``) rely on.

``BUFFERED``
    Write statements are queued on the connection and sent as ONE atomic request
    (``/db/execute?transaction``) on ``commit()``. rqlite cannot hold a transaction
    open across HTTP requests, so this is how a multi-statement unit of work stays
    all-or-nothing. Consequences, by design:

    - ``rowcount`` is ``-1`` and ``lastrowid`` is ``None`` for queued writes.
    - Reads do not see the connection's own uncommitted writes.
    - A constraint violation surfaces at ``commit()``, not at ``execute()``.
    - SAVEPOINTs are emulated client-side by truncating the queue.
    - ORM objects whose autoincrement id the database must generate are sent as Core
      INSERTs and detached from the session (see ``_insert_generated_pk_rows_as_core``),
      so ``obj.id`` stays ``None``. Use client-generated keys (e.g. UUIDs) if you need
      the id before commit.
"""

from __future__ import annotations

import collections
import re
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Integer, String, bindparam, event, insert, pool
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects import registry
from sqlalchemy.dialects.sqlite.base import SQLiteCompiler, SQLiteDialect
from sqlalchemy.engine import AdaptedConnection
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import Session
from sqlalchemy.sql import functions
from sqlalchemy.util.concurrency import await_only

from edda.storage.rqlite_client import (
    ExecuteResult,
    RqliteAuthError,
    RqliteClient,
    RqliteConnectionError,
    RqliteError,
    RqliteStatementError,
    Statement,
)

AUTOCOMMIT = "AUTOCOMMIT"
BUFFERED = "BUFFERED"

# Every atomic batch carries one row in this table, written in the same transaction. If the answer to a
# batch is lost, the row says whether the batch was applied: a lost response is otherwise
# indistinguishable from a lost request, and repeating a batch that did apply fails or duplicates.
MARKER_TABLE = "edda_commit_markers"
_MARKER_DDL = (
    f"CREATE TABLE IF NOT EXISTS {MARKER_TABLE} (token TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
)
_MARKER_INSERT = f"INSERT INTO {MARKER_TABLE} (token, created_at) VALUES (?, ?)"
_MARKER_LOOKUP = f"SELECT 1 FROM {MARKER_TABLE} WHERE token = ?"
_MARKER_PURGE = f"DELETE FROM {MARKER_TABLE} WHERE created_at < ?"
_MARKER_KEEP_SECONDS = 600  # only needed while a lost answer is being resolved: seconds
_MARKER_PURGE_EVERY = 100  # batches


# ---------------------------------------------------------------------------
# DBAPI exceptions (PEP 249)
# ---------------------------------------------------------------------------


class Error(Exception):
    pass


class InterfaceError(Error):
    pass


class DatabaseError(Error):
    pass


class DataError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    # Set when the connection is unusable and should be discarded from the pool.
    disconnect = False
    # True when the request may have been applied before the failure: the outcome is unknown.
    request_sent = False


class IntegrityError(DatabaseError):
    pass


class InternalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass


_INTEGRITY_MARKERS = ("constraint failed", "unique constraint", "constraint violation")


def map_error(exc: RqliteError) -> Error:
    """Translate a client error into the matching DBAPI exception."""
    if isinstance(exc, RqliteConnectionError):
        err = OperationalError(str(exc))
        err.disconnect = True
        err.request_sent = exc.request_sent
        return err
    if isinstance(exc, RqliteAuthError):
        return OperationalError(str(exc))
    message = str(exc)
    lowered = message.lower()
    if any(marker in lowered for marker in _INTEGRITY_MARKERS):
        return IntegrityError(message)
    if isinstance(exc, RqliteStatementError):
        # sqlite3 reports missing tables, syntax errors, etc. as OperationalError.
        return OperationalError(message)
    return DatabaseError(message)


# ---------------------------------------------------------------------------
# Statement classification
# ---------------------------------------------------------------------------

_LEADING_COMMENTS = r"^\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*"
_READ_RE = re.compile(_LEADING_COMMENTS + r"(select|pragma|explain|values)\b", re.I | re.S)
_WITH_RE = re.compile(_LEADING_COMMENTS + r"with\b", re.I | re.S)
_WRITE_WORD_RE = re.compile(r"\b(insert|update|delete|replace)\b", re.I)
_SAVEPOINT_RE = re.compile(r"^\s*SAVEPOINT\s+(\S+)\s*$", re.I)
_RELEASE_RE = re.compile(r"^\s*RELEASE\s+(?:SAVEPOINT\s+)?(\S+)\s*$", re.I)
_ROLLBACK_TO_RE = re.compile(r"^\s*ROLLBACK\s+TO\s+(?:SAVEPOINT\s+)?(\S+)\s*$", re.I)
_TXN_CONTROL_RE = re.compile(r"^\s*(BEGIN|COMMIT|END|ROLLBACK)\b", re.I)


def is_read_statement(sql: str) -> bool:
    """True if ``sql`` only reads (goes to ``/db/query``)."""
    if _READ_RE.match(sql):
        return True
    if _WITH_RE.match(sql):
        return _WRITE_WORD_RE.search(sql) is None
    return False


# ---------------------------------------------------------------------------
# asyncio DBAPI adapter
# ---------------------------------------------------------------------------


class RqliteCursor:
    """PEP 249 cursor whose statements are carried by an ``RqliteClient``."""

    __slots__ = (
        "_adapt_connection",
        "_rows",
        "arraysize",
        "description",
        "lastrowid",
        "rowcount",
    )

    server_side = False

    def __init__(self, adapt_connection: RqliteConnection) -> None:
        self._adapt_connection = adapt_connection
        self._rows: collections.deque[Any] = collections.deque()
        self.arraysize = 1
        self.description: list[tuple[Any, ...]] | None = None
        self.lastrowid: int | None = None
        self.rowcount = -1

    async def _async_soft_close(self) -> None:
        return

    def close(self) -> None:
        self._rows.clear()

    def setinputsizes(self, *inputsizes: Any) -> None:
        pass

    def execute(self, operation: str, parameters: Sequence[Any] | None = None) -> None:
        conn = self._adapt_connection
        try:
            conn.await_(self._execute(operation, tuple(parameters or ())))
        except RqliteError as exc:
            raise map_error(exc) from exc

    def executemany(self, operation: str, seq_of_parameters: Sequence[Sequence[Any]]) -> None:
        conn = self._adapt_connection
        statements = [(operation, tuple(p)) for p in seq_of_parameters]
        try:
            conn.await_(self._execute_many(statements))
        except RqliteError as exc:
            raise map_error(exc) from exc

    async def _execute(self, operation: str, params: tuple[Any, ...]) -> None:
        conn = self._adapt_connection
        self.description = None
        self._rows.clear()

        if await conn._handle_control_statement(operation):
            self.rowcount = -1
            self.lastrowid = None
            return

        if is_read_statement(operation):
            result = await conn.client.query((operation, params))
            self.description = [
                (name, None, None, None, None, None, None) for name in result.columns
            ]
            self._rows = collections.deque(result.rows)
            self.rowcount = -1
            self.lastrowid = None
        elif conn.buffered:
            conn.buffer.append((operation, params))
            self.rowcount = -1
            self.lastrowid = None
        else:
            (result_w,) = await conn.client.execute([(operation, params)])
            self.rowcount = result_w.rows_affected
            self.lastrowid = result_w.last_insert_id

    async def _execute_many(self, statements: Sequence[Statement]) -> None:
        conn = self._adapt_connection
        self.description = None
        self._rows.clear()
        if not statements:
            return
        if conn.buffered:
            conn.buffer.extend(statements)
            self.rowcount = -1
            self.lastrowid = None
            return
        # One atomic request, matching sqlite3's executemany semantics.
        results = await conn.apply_atomically(statements)
        self.rowcount = sum(r.rows_affected for r in results)
        self.lastrowid = results[-1].last_insert_id

    def __iter__(self) -> Any:
        while self._rows:
            yield self._rows.popleft()

    def fetchone(self) -> Any | None:
        return self._rows.popleft() if self._rows else None

    def fetchmany(self, size: int | None = None) -> list[Any]:
        count = self.arraysize if size is None else size
        return [self._rows.popleft() for _ in range(min(count, len(self._rows)))]

    def fetchall(self) -> list[Any]:
        rows = list(self._rows)
        self._rows.clear()
        return rows


class RqliteConnection(AdaptedConnection):
    """PEP 249 connection: owns one HTTP client and the BUFFERED-mode write queue."""

    await_ = staticmethod(await_only)

    __slots__ = ("buffer", "buffered", "client", "dbapi", "savepoints")

    def __init__(self, dbapi: RqliteDBAPI, client: RqliteClient) -> None:
        self.dbapi = dbapi
        self._connection = client  # type: ignore[assignment]
        self.client = client
        self.buffered = False
        self.buffer: list[Statement] = []
        self.savepoints: dict[str, int] = {}

    def cursor(self, server_side: bool = False) -> RqliteCursor:  # noqa: ARG002
        return RqliteCursor(self)

    def execute(self, operation: str, parameters: Sequence[Any] | None = None) -> RqliteCursor:
        cursor = self.cursor()
        cursor.execute(operation, parameters)
        return cursor

    def set_mode(self, mode: str) -> None:
        """Switch between AUTOCOMMIT and BUFFERED. Pending writes are never dropped."""
        if mode == self.mode:
            return
        if self.buffer:
            raise NotSupportedError("Cannot change execution mode with uncommitted writes")
        self.buffered = mode == BUFFERED

    @property
    def mode(self) -> str:
        return BUFFERED if self.buffered else AUTOCOMMIT

    async def _handle_control_statement(self, operation: str) -> bool:
        """Handle SAVEPOINT/RELEASE/ROLLBACK TO. Returns True if ``operation`` was one."""
        if _TXN_CONTROL_RE.match(operation) and not _ROLLBACK_TO_RE.match(operation):
            raise NotSupportedError(
                "rqlite has no interactive transactions; use BUFFERED mode for "
                "multi-statement atomicity"
            )
        match = _SAVEPOINT_RE.match(operation)
        if match:
            self._require_buffered("SAVEPOINT")
            self.savepoints[match.group(1)] = len(self.buffer)
            return True
        match = _ROLLBACK_TO_RE.match(operation)
        if match:
            self._require_buffered("ROLLBACK TO SAVEPOINT")
            name = match.group(1)
            if name not in self.savepoints:
                raise OperationalError(f"no such savepoint: {name}")
            mark = self.savepoints[name]
            del self.buffer[mark:]
            # Savepoints opened after this one are discarded; this one stays open.
            for later in [n for n, m in self.savepoints.items() if m > mark and n != name]:
                del self.savepoints[later]
            return True
        match = _RELEASE_RE.match(operation)
        if match:
            self._require_buffered("RELEASE SAVEPOINT")
            self.savepoints.pop(match.group(1), None)
            return True
        return False

    def _require_buffered(self, what: str) -> None:
        if not self.buffered:
            raise NotSupportedError(
                f"{what} needs BUFFERED mode: rqlite cannot keep a transaction open "
                "across HTTP requests"
            )

    def commit(self) -> None:
        if not self.buffer:
            return
        batch, self.buffer = self.buffer, []
        self.savepoints.clear()
        try:
            self.await_(self.apply_atomically(batch))
        except RqliteError as exc:
            raise map_error(exc) from exc

    async def apply_atomically(self, batch: Sequence[Statement]) -> list[ExecuteResult]:
        """
        Send ``batch`` as one atomic request, and settle the case where its response is lost.

        A marker row rides in the batch. If the request fails after it may have reached the cluster,
        look the marker up: present means the batch was applied and the commit succeeded, absent
        means it was not and the failure stands. (A request still in flight when the client gave up
        can be applied later; that residual window is the price of not holding transactions open.)
        """
        token = uuid.uuid4().hex
        await self._ensure_marker_table()
        now = datetime.now(UTC)
        extra: list[Statement] = [(_MARKER_INSERT, (token, now.strftime("%Y-%m-%d %H:%M:%S.%f")))]
        self.dbapi.batches += 1
        if self.dbapi.batches % _MARKER_PURGE_EVERY == 0:
            cutoff = (now - timedelta(seconds=_MARKER_KEEP_SECONDS)).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
            extra.append((_MARKER_PURGE, (cutoff,)))
        try:
            results = await self.client.execute([*batch, *extra], transaction=True)
        except RqliteConnectionError as exc:
            if not exc.request_sent:
                raise
            found = await self.client.query((_MARKER_LOOKUP, (token,)))
            if not found.rows:
                raise
            # Applied. The results of the caller's own statements are not known; none is needed
            # (a buffered write never reports a row count).
            return [ExecuteResult(rows_affected=-1, last_insert_id=None) for _ in batch]
        return results[: len(batch)]

    async def _ensure_marker_table(self) -> None:
        if not self.dbapi.marker_table_ready:
            await self.client.execute([_MARKER_DDL])
            self.dbapi.marker_table_ready = True

    def rollback(self) -> None:
        self.buffer = []
        self.savepoints.clear()

    def close(self) -> None:
        self.buffer = []
        self.savepoints.clear()
        self.await_(self.client.aclose())


class RqliteDBAPI:
    """Module-like object SQLAlchemy treats as the DBAPI."""

    paramstyle = "qmark"
    apilevel = "2.0"
    threadsafety = 2

    # Reported to SQLAlchemy's SQLite dialect. Kept below 3.35 so SQLAlchemy never
    # emits RETURNING, which Edda does not use and older rqlite nodes lack.
    sqlite_version = "3.34.0"
    sqlite_version_info = (3, 34, 0)

    # Shared by every connection of one engine: whether the marker table exists, and a batch counter.
    marker_table_ready = False
    batches = 0

    Error = Error
    InterfaceError = InterfaceError
    DatabaseError = DatabaseError
    DataError = DataError
    OperationalError = OperationalError
    IntegrityError = IntegrityError
    InternalError = InternalError
    ProgrammingError = ProgrammingError
    NotSupportedError = NotSupportedError

    @staticmethod
    def Binary(value: bytes) -> bytes:  # noqa: N802 - PEP 249 name
        return bytes(value)

    def connect(self, **kw: Any) -> RqliteConnection:
        client = RqliteClient(
            kw["base_urls"],
            username=kw.get("username"),
            password=kw.get("password"),
            level=kw.get("level", "weak"),
            timeout=kw.get("timeout", 30.0),
            transport=kw.get("transport"),
        )
        return RqliteConnection(self, client)


# ---------------------------------------------------------------------------
# SQL compilation
# ---------------------------------------------------------------------------


def _is_time_default(column: Any) -> bool:
    server_default = column.server_default
    arg = getattr(server_default, "arg", None)
    return isinstance(arg, functions.now)


class RqliteCompiler(SQLiteCompiler):
    """
    SQLite compiler that keeps wall-clock time out of the SQL it sends.

    rqlite replicates statements, so a statement must carry the same time on every node. It
    achieves that by rewriting ``CURRENT_TIMESTAMP`` and ``datetime('now')`` in WRITE requests to a
    concrete value before the statement enters the Raft log. Measured on a real cluster, that value
    is the receiving node's LOCAL time written as if it were UTC (a node in US Eastern stored times
    four hours early), while the same expression in a read is correct UTC. Comparing the two, or
    mixing either with a Python UTC timestamp, gives wrong answers.

    So no statement carries ``now``: it is a bound parameter taken from this dialect's clock (see
    ``RqliteDialect_httpx.utc_now_text``), which the storage calibrates against the cluster. Column
    ``DEFAULT``s are not rewritten by rqlite either, so omitted timestamp columns are filled the
    same way, in ``visit_insert``.
    """

    def visit_now_func(self, fn: Any, **kw: Any) -> str:  # noqa: ARG002
        param = bindparam("now", callable_=self.dialect.utc_now_text, type_=String, unique=True)  # type: ignore[attr-defined]
        return str(self.process(param, **kw))

    def visit_insert(
        self,
        insert_stmt: Any,
        visited_bindparam: Any = None,
        visiting_cte: Any = None,
        **kw: Any,
    ) -> Any:
        table = insert_stmt.table
        if insert_stmt.select is None and not insert_stmt._multi_values:
            provided = set(self.column_keys or ())
            provided.update(getattr(key, "key", key) for key in (insert_stmt._values or {}))
            missing = {
                column.key: functions.now()
                for column in table.columns
                if column.key not in provided
                and column.default is None
                and _is_time_default(column)
            }
            if missing:
                insert_stmt = insert_stmt.values(**missing)
        return super().visit_insert(  # type: ignore[no-untyped-call]
            insert_stmt, visited_bindparam=visited_bindparam, visiting_cte=visiting_cte, **kw
        )


# ---------------------------------------------------------------------------
# Dialect
# ---------------------------------------------------------------------------


class RqliteDialect_httpx(SQLiteDialect):  # noqa: N801 - SQLAlchemy dialect naming
    # ``name`` stays "sqlite" (inherited) so every ``dialect.name == "sqlite"`` branch
    # in SQLAlchemy and in Edda applies to rqlite too: it is SQLite underneath.
    driver = "httpx"
    supports_statement_cache = True
    is_async = True
    supports_server_side_cursors = False
    # BUFFERED writes report rowcount -1, so ORM row-count verification must be off.
    supports_sane_rowcount = False
    supports_sane_multi_rowcount = False
    statement_compiler = RqliteCompiler

    # Seconds to add to this machine's clock to get the cluster's UTC clock; set by the storage.
    clock_offset: float = 0.0

    def utc_now_text(self) -> str:
        """The cluster's current UTC time, in the format SQLite's DateTime columns use."""
        moment = datetime.fromtimestamp(time.time() + self.clock_offset, UTC)
        return moment.strftime("%Y-%m-%d %H:%M:%S.%f")

    _isolation_lookup = {AUTOCOMMIT: None, BUFFERED: None}  # type: ignore[assignment]

    @classmethod
    def import_dbapi(cls) -> RqliteDBAPI:  # type: ignore[override]
        return RqliteDBAPI()

    @classmethod
    def get_pool_class(cls, url: URL) -> type[pool.Pool]:  # noqa: ARG003
        return pool.AsyncAdaptedQueuePool

    def create_connect_args(self, url: URL) -> tuple[list[Any], dict[str, Any]]:
        query = dict(url.query)
        scheme = "https" if str(query.get("tls", "")).lower() in ("1", "true", "yes") else "http"
        hosts = [f"{url.host or 'localhost'}:{url.port or 4001}"]
        extra = query.get("host", ())
        hosts.extend([extra] if isinstance(extra, str) else list(extra))
        kwargs: dict[str, Any] = {
            "base_urls": [f"{scheme}://{h}" for h in hosts],
            "username": url.username,
            "password": url.password,
            "level": str(query.get("level", "weak")),
            "timeout": float(str(query.get("timeout", 30))),
        }
        return [], kwargs

    def get_driver_connection(self, connection: Any) -> Any:
        return connection._connection

    def _get_server_version_info(self, connection: Any) -> Any:  # noqa: ARG002
        return self.dbapi.sqlite_version_info  # type: ignore[union-attr]

    def get_isolation_level_values(  # type: ignore[override]
        self, dbapi_connection: Any  # noqa: ARG002
    ) -> list[str]:
        return [AUTOCOMMIT, BUFFERED]

    def set_isolation_level(self, dbapi_connection: Any, level: Any) -> None:
        dbapi_connection.set_mode(level)

    def get_isolation_level(self, dbapi_connection: Any) -> str:  # type: ignore[override]
        return str(dbapi_connection.mode)

    def detect_autocommit_setting(self, dbapi_conn: Any) -> bool:
        return not dbapi_conn.buffered

    def on_connect(self) -> None:
        return None

    def is_disconnect(self, e: Any, connection: Any, cursor: Any) -> bool:  # noqa: ARG002
        return isinstance(e, OperationalError) and e.disconnect


# ---------------------------------------------------------------------------
# ORM support for BUFFERED mode
# ---------------------------------------------------------------------------


def _is_buffered_rqlite(session: Session) -> bool:
    try:
        bind = session.get_bind()
    except Exception:  # unbound session, mapper-less lookup, ...
        return False
    if not isinstance(getattr(bind, "dialect", None), RqliteDialect_httpx):
        return False
    return bool(bind.get_execution_options().get("isolation_level") == BUFFERED)


def _has_db_generated_pk(state: Any) -> bool:
    """True if the object's only primary key is an autoincrement column left unset."""
    mapper = state.mapper
    pk_columns = mapper.primary_key
    if len(pk_columns) != 1:
        return False
    column = pk_columns[0]
    if not isinstance(column.type, Integer) or column.autoincrement not in (True, "auto"):
        return False
    return mapper.get_property_by_column(column).key not in state.dict


def _insert_generated_pk_rows_as_core(
    session: Session,
    flush_context: Any,  # noqa: ARG001
    instances: Any,  # noqa: ARG001
) -> None:
    """
    Let the ORM add rows whose primary key the database generates, in BUFFERED mode.

    The ORM must learn a new row's autoincrement id from the INSERT itself, but a
    queued write has no id until the batch is applied at commit. Those objects are
    therefore sent as Core INSERTs (id omitted, so SQLite assigns real, monotonic ids
    in batch order) and removed from the session. The caller's object keeps
    ``id = None`` instead of getting a wrong value.
    """
    if not session.new or not _is_buffered_rqlite(session):
        return
    pending = [obj for obj in session.new if _has_db_generated_pk(sa_inspect(obj))]
    pending.sort(key=lambda obj: getattr(sa_inspect(obj), "insert_order", 0) or 0)
    for obj in pending:
        state = sa_inspect(obj)
        values = {
            column.key: state.dict[prop.key]
            for prop in state.mapper.column_attrs
            for column in prop.columns
            if state.dict.get(prop.key) is not None
        }
        session.execute(insert(state.mapper.local_table).values(**values))
        session.expunge(obj)


event.listen(Session, "before_flush", _insert_generated_pk_rows_as_core)


def register_rqlite_dialect() -> None:
    """Make ``rqlite+httpx://`` URLs resolvable by SQLAlchemy. Safe to call repeatedly."""
    registry.register("rqlite.httpx", "edda.storage.rqlite_dialect", "RqliteDialect_httpx")


dialect = RqliteDialect_httpx
