"""
In-process fake of an rqlite cluster, for tests.

It is an ``httpx2.MockTransport`` handler backed by one in-memory SQLite database,
so no network, Docker or ``rqlited`` binary is needed. It models the parts of the
rqlite HTTP API that Edda's driver depends on:

- ``POST /db/execute`` with optional ``?transaction`` (atomic batch, stops at the
  first error, rolls the whole batch back)
- ``POST /db/query`` with ``level`` and ``blob_array``
- each request is applied atomically, but concurrent callers interleave between
  requests (the fake yields to the event loop while a request is "in flight")
- no interactive transactions: BEGIN/COMMIT/ROLLBACK/SAVEPOINT are rejected, so a
  driver that tried to hold a transaction open across requests fails loudly
- optional basic auth, and nodes that can be marked unreachable

It is a model of rqlite, not rqlite itself: behaviour it does not cover (Raft
elections, follower forwarding, statement rewriting) must be verified against a real
cluster. See ``EDDA_TEST_RQLITE_URL`` in ``tests/test_storage_rqlite.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2

_CONTROL_RE = re.compile(r"^\s*(BEGIN|COMMIT|END|ROLLBACK|SAVEPOINT|RELEASE)\b", re.I)
_READ_RE = re.compile(r"^\s*(select|pragma|explain|values)\b", re.I)
_WITH_RE = re.compile(r"^\s*with\b", re.I)
_WRITE_WORD_RE = re.compile(r"\b(insert|update|delete|replace)\b", re.I)
_NOW_RE = re.compile(r"CURRENT_TIMESTAMP|'now'", re.I)


@dataclass
class RecordedRequest:
    host: str
    path: str
    transaction: bool
    level: str | None
    statements: list[tuple[str, list[Any]]]


@dataclass
class FakeRqlite:
    """One logical rqlite cluster reachable at any of several host names."""

    credentials: tuple[str, str] | None = None
    down_hosts: set[str] = field(default_factory=set)
    requests: list[RecordedRequest] = field(default_factory=list)
    # What a real node does to ``now`` in a WRITE request: it substitutes its own LOCAL time, as text that
    # SQLite then reads as UTC. Measured on a real cluster in US Eastern: stored times were 4 hours early.
    # Reads are not rewritten and give true UTC. ``None`` disables the rewrite.
    write_now_offset_seconds: float | None = -4 * 3600
    # How far this cluster's UTC clock is ahead of the machine running the tests, as seen by reads.
    read_clock_skew_seconds: float = 0.0
    # Callables run before each request is applied; may raise to inject a failure.
    before_request: list[Any] = field(default_factory=list)
    # Callables run after a request was applied, before the answer goes back; raising loses the
    # response while the change stays applied, the case a client cannot tell from a lost request.
    after_request: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._handle)

    def close(self) -> None:
        self.db.close()

    # -- request helpers --------------------------------------------------

    def writes(self) -> list[RecordedRequest]:
        """Write requests, without the driver's own bookkeeping (the commit marker and its table)."""
        out = []
        for request in self.requests:
            if request.path != "/db/execute":
                continue
            kept = [
                (sql, params)
                for sql, params in request.statements
                if "edda_commit_markers" not in sql
            ]
            if kept:
                out.append(replace(request, statements=kept))
        return out

    def reads(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == "/db/query"]

    def table_rows(self, sql: str, *params: Any) -> list[tuple[Any, ...]]:
        """Inspect the cluster state directly, bypassing the HTTP layer."""
        return list(self.db.execute(sql, params).fetchall())

    # -- HTTP handler -----------------------------------------------------

    async def _handle(self, request: httpx2.Request) -> httpx2.Response:
        # The request is "on the wire": let other tasks run, as they would over a real
        # network. Without this, concurrent callers would execute strictly one after
        # another and races between a read and a later write could never show up.
        await asyncio.sleep(0)
        response = self._apply(request)
        await asyncio.sleep(0)
        return response

    def _apply(self, request: httpx2.Request) -> httpx2.Response:
        """Apply one request atomically (rqlite serialises writes through Raft)."""
        host = request.url.host
        if host in self.down_hosts:
            raise httpx2.ConnectError(f"{host} is down", request=request)
        if self.credentials is not None and not self._authorised(request):
            return httpx2.Response(401, text="unauthorized")

        statements = [self._decode_statement(s) for s in json.loads(request.content)]
        params = request.url.params
        transaction = "transaction" in params
        record = RecordedRequest(
            host=host,
            path=request.url.path,
            transaction=transaction,
            level=params.get("level"),
            statements=statements,
        )
        self.requests.append(record)
        for hook in self.before_request:
            hook(record)

        if request.url.path == "/db/execute":
            response = self._execute(statements, transaction)
        elif request.url.path == "/db/query":
            response = self._query(statements, blob_array="blob_array" in params)
        else:
            return httpx2.Response(404, text="not found")
        for hook in self.after_request:
            hook(record)
        return response

    def _authorised(self, request: httpx2.Request) -> bool:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("basic "):
            return False
        decoded = base64.b64decode(header[6:]).decode()
        return decoded == ":".join(self.credentials or ("", ""))

    @staticmethod
    def _decode_statement(raw: Any) -> tuple[str, list[Any]]:
        if isinstance(raw, str):
            return raw, []
        sql, *params = raw
        # A JSON array parameter is rqlite's way of passing a BLOB.
        return sql, [bytes(p) if isinstance(p, list) else p for p in params]

    def _execute(
        self, statements: list[tuple[str, list[Any]]], transaction: bool
    ) -> httpx2.Response:
        results: list[dict[str, Any]] = []
        failed = False
        if transaction:
            self.db.execute("BEGIN")
        for sql, params in statements:
            if _CONTROL_RE.match(sql):
                results.append({"error": "transaction control statements are not supported"})
                failed = True
                break
            try:
                cursor = self.db.execute(
                    self._rewrite_now(sql, self.write_now_offset_seconds), params
                )
            except sqlite3.Error as exc:
                results.append({"error": str(exc)})
                failed = True
                break
            results.append({"last_insert_id": cursor.lastrowid, "rows_affected": cursor.rowcount})
        if transaction:
            self.db.execute("ROLLBACK" if failed else "COMMIT")
        return httpx2.Response(200, json={"results": results})

    def _query(
        self, statements: list[tuple[str, list[Any]]], *, blob_array: bool
    ) -> httpx2.Response:
        results: list[dict[str, Any]] = []
        for sql, params in statements:
            is_read = bool(_READ_RE.match(sql)) or (
                bool(_WITH_RE.match(sql)) and not _WRITE_WORD_RE.search(sql)
            )
            if not is_read:
                results.append({"error": "attempt to change database via query operation"})
                break
            try:
                cursor = self.db.execute(
                    self._rewrite_now(sql, self.read_clock_skew_seconds or None), params
                )
                rows = cursor.fetchall()
            except sqlite3.Error as exc:
                results.append({"error": str(exc)})
                break
            columns = [d[0] for d in (cursor.description or [])]
            values = [[self._encode_cell(c, blob_array) for c in row] for row in rows]
            results.append({"columns": columns, "types": ["" for _ in columns], "values": values})
        return httpx2.Response(200, json={"results": results})

    @staticmethod
    def _rewrite_now(sql: str, offset_seconds: float | None) -> str:
        """Replace ``now`` with a literal time ``offset_seconds`` from UTC (DDL is left as written)."""
        if offset_seconds is None or sql.lstrip().upper().startswith("CREATE"):
            return sql
        moment = datetime.now(UTC) + timedelta(seconds=offset_seconds)
        return _NOW_RE.sub(f"'{moment:%Y-%m-%d %H:%M:%S}'", sql)

    @staticmethod
    def _encode_cell(cell: Any, blob_array: bool) -> Any:
        if isinstance(cell, bytes):
            return list(cell) if blob_array else base64.b64encode(cell).decode()
        return cell
