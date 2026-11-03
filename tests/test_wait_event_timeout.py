"""wait_event() reports a timeout as EventTimeoutError, as its documentation says.

receive(), which wait_event() is built on, raises a plain TimeoutError when the message subscription
times out (replayed from the MessageTimeout history row that _check_expired_message_subscriptions()
writes). Workflows written against wait_event() catch EventTimeoutError; if the plain TimeoutError
escaped, none of those handlers would run and every timed-out wait would fail its workflow instead.
"""

import pytest
import pytest_asyncio

from edda.channels import EventTimeoutError, subscribe, wait_event
from edda.context import WorkflowContext


@pytest_asyncio.fixture
async def waiting_instance(sqlite_storage, create_test_instance):
    instance_id = "wait-event-timeout-001"
    await create_test_instance(
        instance_id=instance_id,
        workflow_name="test_workflow",
        owner_service="test-service",
        input_data={},
    )
    await sqlite_storage.update_instance_status(instance_id, "running")
    # What _check_expired_message_subscriptions() records when the wait times out
    await sqlite_storage.append_history(
        instance_id=instance_id,
        activity_id="receive_payment.completed:1",
        event_type="MessageTimeout",
        event_data={
            "_error": True,
            "error_type": "TimeoutError",
            "error_message": "Message on channel 'payment.completed' did not arrive within timeout",
            "channel": "payment.completed",
        },
    )
    return instance_id


async def replaying_context(storage, instance_id: str) -> WorkflowContext:
    ctx = WorkflowContext(
        instance_id=instance_id,
        workflow_name="test_workflow",
        storage=storage,
        worker_id="worker-1",
        is_replaying=True,
    )
    await subscribe(ctx, "payment.completed", mode="broadcast")
    await ctx._load_history()
    ctx._activity_call_counters.clear()
    return ctx


async def test_a_timed_out_wait_raises_event_timeout_error(sqlite_storage, waiting_instance):
    ctx = await replaying_context(sqlite_storage, waiting_instance)

    with pytest.raises(EventTimeoutError) as exc_info:
        await wait_event(ctx, event_type="payment.completed", timeout_seconds=60)

    assert exc_info.value.event_type == "payment.completed"
    assert exc_info.value.timeout_seconds == 60
    assert isinstance(exc_info.value.__cause__, TimeoutError)


async def test_code_that_catches_timeout_error_still_catches_it(sqlite_storage, waiting_instance):
    ctx = await replaying_context(sqlite_storage, waiting_instance)

    with pytest.raises(TimeoutError):
        await wait_event(ctx, event_type="payment.completed", timeout_seconds=60)
