# rqlite (distributed SQLite)

[rqlite](https://rqlite.io) is a distributed database built on SQLite. A cluster elects a leader with Raft, forwards writes to it, and replicates every change to all nodes. Edda can use it as its database, so a group of workers can share workflow state **without NFS, LiteFS, or a database server**. Every node keeps its data on local disk.

```python
from edda import EddaApp

app = EddaApp(
    service_name="order-service",
    db_url="rqlite://localhost:4001",
)
```

No extra install is needed: the driver uses `httpx2`, which Edda already depends on.

## Connecting

| URL | Meaning |
|---|---|
| `rqlite://host:4001` | HTTP |
| `rqlites://host:4001` | HTTPS |
| `rqlite://user:pass@host:4001` | HTTP basic auth |

Query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `level` | `weak` | Read consistency: `none`, `weak`, `linearizable`, `strong`, `auto`. Leave at `weak` or stricter for workflow state; `none` can return stale data. |
| `host` | – | An additional node, repeatable: `?host=node2:4001&host=node3:4001`. Tried in order when a node cannot be connected to. |
| `timeout` | `30` | Request timeout in seconds. |

```python
db_url = "rqlite://user:pass@node1:4001?host=node2:4001&host=node3:4001&level=linearizable"
```

A request is retried on another node only when the connection could not be established at all. A request that was already sent may have been applied, so it is not blindly replayed; see [Lost responses](#lost-responses).

## How it differs from PostgreSQL and MySQL

### Claims are single `UPDATE`s, not row locks

rqlite has no `SELECT ... FOR UPDATE`. Wherever Edda needs exactly one worker to win, it runs one conditional `UPDATE` and checks `rowcount`:

- **Workflow locks**: `UPDATE ... SET locked_by = ? WHERE instance_id = ? AND (locked_by IS NULL OR lease expired)`
- **System locks** (background-task leadership): the same pattern.
- **Outbox events**: each event is claimed with `UPDATE ... SET status = 'processing' WHERE event_id = ? AND status IN ('pending', 'failed')`. A worker that loses the race skips that event, so a poll may return fewer than `limit` events.

rqlite applies writes one at a time through Raft, so at most one contender matches the `WHERE` clause.

### Time comes from a calibrated clock, never from SQL `now`

rqlite replicates statements, so it rewrites `now` and `CURRENT_TIMESTAMP` in **write** requests to a fixed value before the statement enters the Raft log. Measured on a real cluster (rqlite v10.6.6, node in US Eastern time), that value is the node's *local* time written as if it were UTC, four hours early, while the same expression in a read is correct UTC. Comparing the two, or either with a UTC timestamp from Python, gives wrong answers. (Stale-lock cleanup silently found nothing.)

So Edda never sends `now` in a write. Every timestamp (`created_at`, `updated_at`, lock expiry, ...) is a bound parameter from a clock that `RqliteStorage` calibrates against the cluster's UTC (read from a query, where it is correct) at start-up and every few minutes. Workers whose own clocks differ still agree with each other and with the cluster. Running `rqlited` with `TZ=UTC` would also hide the problem, but nothing here depends on it.

In your own tables, do the same: use `func.now()` (Edda's compiler turns it into the calibrated parameter) or pass `storage.utc_now_text()`, and avoid `DEFAULT CURRENT_TIMESTAMP` and `datetime('now')` in writes.

### Transactions are atomic batches

rqlite cannot keep a transaction open across HTTP requests. Inside an Edda transaction (every `@activity`, and `ctx.transaction()`), writes are **queued and sent as one atomic request at commit**. The activity result, its outbox events and your `ctx.session` writes still succeed or fail together, which is what the transactional outbox needs.

The trade-offs of queueing:

- **Reads inside the transaction do not see its own uncommitted writes.** Read first, then write.
- **`rowcount` and `lastrowid` are unavailable until commit.** Edda's own claims run outside the transaction for this reason.
- **A constraint violation is raised at commit**, not at the statement that caused it.
- **ORM objects whose id the database generates keep `id = None`** after `flush()`. Use client-generated keys (for example UUIDs) if you need the id before commit.
- Savepoints (nested `ctx.transaction()`) are supported and roll back only the inner writes.

Each batch also carries one row in a small table, `edda_commit_markers`, created on first use and purged after ten minutes. See [Lost responses](#lost-responses).

### Edda's own multi-step operations are atomic too

These operations are documented as one transaction on the other backends, and each reaches the cluster as one request here: `register_timer_subscription_and_release_lock` (what `sleep` uses), `register_channel_receive_and_release_lock`, `deliver_message`, `deliver_channel_message`, `archive_history` and `cancel_instance` (whose cleanup statements are guarded on the status that was read, so a worker that lost a race changes nothing).

Three cleanup operations still run as several requests: `cleanup_instance_subscriptions`, `delete_channel_message` and `cleanup_old_channel_messages`. Each step is a delete that a retry repeats, so stopping halfway is safe.

### Nodes can start together

Every node runs `initialize()` and registers its workflows at boot. The other backends check for the schema-version row and the workflow definition and then insert them, so nodes that boot at the same moment can collide on the primary key. On rqlite each is a single statement that decides and writes together (`INSERT ... WHERE NOT EXISTS`, `INSERT ... ON CONFLICT DO UPDATE`), so any number of nodes can start against an empty cluster at once.

### Lost responses

A write can be applied by the cluster while its answer never arrives. The worker sees an error and cannot tell that from a write that never happened. Tested by dropping about one write response in thirteen, after the cluster had applied the write, while workers ran a whole scenario:

- **Batches** (activity commits, atomic operations) carry a marker row. After a lost response the driver looks the marker up: present means the batch was applied and the commit succeeded. This is what stops Edda from repeating an activity whose history row already exists.
- **Lock claims** ask the cluster whether this worker now holds the lock, and claim again if not. A claim only succeeds on a free or expired lock, so repeating it is safe.
- **Reads** are repeated on a dropped connection (a read changes nothing).
- **Idempotent writes** (setting a status, deleting a subscription, releasing a lock, and the start-up upserts of the schema version and workflow definitions) are repeated, up to three attempts in all.
- **`create_instance`** checks whether the row exists before repeating.
- **Counters and other inserts are not repeated**: `mark_outbox_failed` adds 1 to a counter, and `add_outbox_event`, `append_history` and `publish_to_channel` outside a transaction would collide. They raise.

Limits: an outbox claim whose response is lost leaves the event `processing` (there is no column saying who claimed it), and a request that times out on the client but is still being processed can be applied after the client concluded it was not.

### Detecting a dead worker

Edda renews a running workflow's lock only if you ask it to. By default it never does, so a workflow that runs longer than its `lock_timeout_seconds` looks stale to the auto-resume loop and can be taken over while its worker is alive. That forces a long timeout (a scanner uses 3600 s), and the timeout is how long a dead worker's workflows wait before anyone resumes them.

Set `lock_refresh_interval` on `EddaApp` (or `ReplayEngine`) to renew the lock while the workflow runs, and the timeout can be short:

```python
app = EddaApp(service_name="scanner", db_url="rqlite://...", lock_refresh_interval=10)

@workflow(lock_timeout_seconds=45)
async def scan(ctx, ...): ...
```

Choose an interval of about a third of the timeout. On rqlite this is the only signal that a worker died (there is no NFS lease or connection to watch).

Measured on a real 3-node cluster (rqlite v10.6.6) with a 6 s timeout renewed every 2 s: a worker killed with `SIGKILL` mid-scan was reported stale by a peer 6.4 s later, and a live worker that kept renewing ran for minutes, far past its timeout, without being reported. Without renewal the same live worker is reported stale.

## Limitations

- **Throughput.** Every write is a Raft round trip, and an outbox poll costs one request per claimed event. rqlite suits moderate write rates, not thousands of workflow steps per second.
- **No LISTEN/NOTIFY.** Workers poll. `use_listen_notify=True` is PostgreSQL-only.
- **User tables with `DATETIME`/`TIMESTAMP` columns.** rqlite may return such columns as RFC 3339 strings (`2026-01-01T12:00:00Z`). Edda's own tables use `TEXT` and are not affected, but SQLAlchemy's SQLite `DateTime` type can misread that format in your tables. Declare those columns as `TEXT`, or parse them yourself.
- **Not verified.** Leader failover, network partitions, node restarts, TLS and authentication have not been exercised.
- **Quorum.** A cluster needs a majority of its voters (three tolerate one failure). Without a quorum every workflow stalls.

## Testing

Most of the suite runs against an in-process model of rqlite (`tests/rqlite_fake.py`), which needs no Docker or `rqlited`. It models the HTTP API, and since a model can only be as right as what its author knew, it also reproduces what a real node was measured to do (it rewrites `now` in writes to local time, as above). Two switches run more against a real cluster:

```bash
# the shared storage tests and anything that uses the db_storage fixture
EDDA_TEST_RQLITE_URL=rqlite://localhost:4001 uv run pytest tests -k rqlite

# every test that uses the sqlite_storage fixture, on rqlite (the fake, or the cluster if the URL is set)
EDDA_TEST_STORAGE=rqlite uv run pytest tests
```

!!! warning "Use a dedicated cluster"
    With `EDDA_TEST_RQLITE_URL` set, the tests **delete all rows** from Edda's tables when they finish. Never point it at a cluster that holds real data.

Verified against a real 3-node cluster (rqlite v10.6.6, all nodes on one VM, plain HTTP, no authentication), through one follower: the storage tests, the timer, channel and stale-lock tests, and a multi-node scan scenario including a killed worker and write responses dropped after being applied. Real requests there took about 0.5 s each, so those runs are slow, and the fake is what runs in CI.
