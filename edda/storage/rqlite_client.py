"""
Minimal asynchronous HTTP client for rqlite.

rqlite (https://rqlite.io) is a distributed SQLite database. Consensus, leader
election and write-forwarding are handled by the cluster itself, so a client only
has to translate SQL into HTTP requests:

- ``POST /db/execute``  - writes, optionally as one atomic batch (``?transaction``)
- ``POST /db/query``    - reads, with a configurable consistency ``level``

This module is deliberately free of SQLAlchemy so it can be tested on its own.
``edda.storage.rqlite_dialect`` builds the SQLAlchemy driver on top of it.

Retry policy: a request is retried on another node only when the connection could
not be established at all (the request provably never reached a server). Requests
that fail after being sent are never retried, because a write may already have been
applied; the caller decides what to do with an ambiguous failure.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx2

logger = logging.getLogger(__name__)

_READ_ATTEMPTS = 3

# A statement is its SQL text plus positional (qmark) parameters.
Statement = tuple[str, Sequence[Any]]


class RqliteError(Exception):
    """Base class for all rqlite client errors."""


class RqliteConnectionError(RqliteError):
    """No node could be reached, or the connection broke before a response arrived."""

    def __init__(self, message: str, *, request_sent: bool = False) -> None:
        super().__init__(message)
        # True when the request may have reached a server (outcome is ambiguous).
        self.request_sent = request_sent


class RqliteAuthError(RqliteError):
    """The cluster rejected the credentials (HTTP 401/403)."""


class RqliteStatementError(RqliteError):
    """A statement was rejected by SQLite or rqlite."""

    def __init__(self, message: str, *, statement_index: int | None = None) -> None:
        super().__init__(message)
        self.statement_index = statement_index


@dataclass(frozen=True)
class ExecuteResult:
    """Outcome of one write statement."""

    rows_affected: int
    last_insert_id: int | None


@dataclass(frozen=True)
class QueryResult:
    """Outcome of one read statement."""

    columns: list[str]
    rows: list[tuple[Any, ...]]


def encode_param(value: Any) -> Any:
    """Convert a Python value to something rqlite accepts as a JSON parameter."""
    if value is None or isinstance(value, (str, int, float)):
        # bool is an int subclass; SQLite stores it as 0/1 either way.
        return int(value) if isinstance(value, bool) else value
    if isinstance(value, (bytes, bytearray, memoryview)):
        # rqlite maps a JSON array of byte values onto a BLOB parameter.
        return list(bytes(value))
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Unsupported rqlite parameter type: {type(value).__name__}")


def encode_statement(statement: Statement | str) -> str | list[Any]:
    """Encode one statement (SQL text, or SQL plus parameters) for the request body."""
    if isinstance(statement, str):
        return statement
    sql, params = statement
    if not params:
        return sql
    return [sql, *(encode_param(p) for p in params)]


def _decode_value(value: Any) -> Any:
    # With ``blob_array`` set, rqlite returns BLOB cells as arrays of byte values.
    # No other SQLite storage class is ever returned as a JSON array.
    if isinstance(value, list):
        return bytes(value)
    return value


def _raise_for_result_error(result: dict[str, Any], index: int) -> None:
    error = result.get("error")
    if error is None and isinstance(result.get("error_v2"), dict):
        error = result["error_v2"].get("message")
    if error:
        raise RqliteStatementError(str(error), statement_index=index)


class RqliteClient:
    """
    Asynchronous rqlite client.

    Args:
        base_urls: One or more node URLs (``http://host:4001``). The first is
            preferred; the others are tried when a node cannot be connected to.
        username: Optional HTTP basic-auth user.
        password: Optional HTTP basic-auth password.
        level: Read consistency level (``none``, ``weak``, ``linearizable``,
            ``strong`` or ``auto``). Defaults to ``weak`` (read from the leader).
        timeout: Per-request timeout in seconds.
        transport: Optional ``httpx2`` transport, used by tests to inject a fake node.
    """

    def __init__(
        self,
        base_urls: str | Iterable[str],
        *,
        username: str | None = None,
        password: str | None = None,
        level: str = "weak",
        timeout: float = 30.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
        if not urls:
            raise ValueError("At least one rqlite node URL is required")
        self._base_urls = [u.rstrip("/") for u in urls]
        self._level = level
        auth = httpx2.BasicAuth(username, password or "") if username else None
        self._client = httpx2.AsyncClient(auth=auth, timeout=timeout, transport=transport)
        self._preferred = 0

    @property
    def level(self) -> str:
        return self._level

    async def aclose(self) -> None:
        await self._client.aclose()

    async def execute(
        self, statements: Sequence[Statement | str], *, transaction: bool = False
    ) -> list[ExecuteResult]:
        """
        Run write statements.

        With ``transaction=True`` the whole batch is applied atomically through the
        Raft log: either every statement takes effect or none does.
        """
        if not statements:
            return []
        params = {"transaction": ""} if transaction else None
        payload = await self._post("/db/execute", [encode_statement(s) for s in statements], params)
        results = payload.get("results", [])
        out: list[ExecuteResult] = []
        for index, result in enumerate(results):
            _raise_for_result_error(result, index)
            out.append(
                ExecuteResult(
                    rows_affected=int(result.get("rows_affected", 0)),
                    last_insert_id=result.get("last_insert_id"),
                )
            )
        if len(out) != len(statements):
            raise RqliteStatementError(
                f"rqlite returned {len(out)} results for {len(statements)} statements"
            )
        return out

    async def query(self, statement: Statement, *, level: str | None = None) -> QueryResult:
        """Run a single read statement. A read changes nothing, so a dropped connection is retried."""
        params: dict[str, str] = {"level": level or self._level, "blob_array": ""}
        body = [encode_statement(statement)]
        for attempt in range(_READ_ATTEMPTS):
            try:
                payload = await self._post("/db/query", body, params)
                break
            except RqliteConnectionError:
                if attempt == _READ_ATTEMPTS - 1:
                    raise
                logger.warning("rqlite read failed on a dropped connection; trying again")
        else:  # pragma: no cover - the loop breaks or raises
            raise AssertionError("unreachable")
        results = payload.get("results", [])
        if not results:
            raise RqliteStatementError("rqlite returned no result for query")
        result = results[0]
        _raise_for_result_error(result, 0)
        columns = list(result.get("columns") or [])
        raw_rows = result.get("values") or []
        rows = [tuple(_decode_value(cell) for cell in row) for row in raw_rows]
        return QueryResult(columns=columns, rows=rows)

    async def _post(
        self, path: str, body: list[Any], params: dict[str, str] | None
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        count = len(self._base_urls)
        for offset in range(count):
            index = (self._preferred + offset) % count
            url = self._base_urls[index] + path
            try:
                response = await self._client.post(url, json=body, params=params)
            except httpx2.ConnectError as exc:
                # Never reached a server: safe to try the next node, even for writes.
                logger.warning("rqlite node %s unreachable: %s", self._base_urls[index], exc)
                last_error = exc
                continue
            except httpx2.TimeoutException as exc:
                raise RqliteConnectionError(
                    f"rqlite request to {url} timed out", request_sent=True
                ) from exc
            except httpx2.TransportError as exc:
                raise RqliteConnectionError(
                    f"rqlite request to {url} failed: {exc}", request_sent=True
                ) from exc

            self._preferred = index
            return self._parse_response(response)

        raise RqliteConnectionError(f"Could not connect to any rqlite node: {last_error}")

    @staticmethod
    def _parse_response(response: httpx2.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise RqliteAuthError(f"rqlite rejected credentials (HTTP {response.status_code})")
        if response.status_code >= 500:
            raise RqliteConnectionError(
                f"rqlite node error (HTTP {response.status_code}): {response.text[:200]}",
                request_sent=True,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RqliteStatementError(
                f"rqlite returned a non-JSON response (HTTP {response.status_code})"
            ) from exc
        if response.status_code >= 400:
            message = payload.get("error") if isinstance(payload, dict) else None
            raise RqliteStatementError(
                message or f"rqlite request failed (HTTP {response.status_code})"
            )
        if not isinstance(payload, dict):
            raise RqliteStatementError("Unexpected rqlite response shape")
        return payload
