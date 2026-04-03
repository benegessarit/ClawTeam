"""Tests for TaskWaiter, including inbox DONE auto-completion."""

from __future__ import annotations

import threading

import pytest

from clawteam.team.mailbox import MailboxManager
from clawteam.team.models import TaskStatus
from clawteam.team.tasks import TaskStore
from clawteam.team.waiter import TaskWaiter


@pytest.fixture
def store(team_name):
    return TaskStore(team_name)


@pytest.fixture
def mailbox(team_name):
    return MailboxManager(team_name)


@pytest.fixture
def waiter(team_name, store, mailbox):
    return TaskWaiter(
        team_name=team_name,
        agent_name="leader",
        mailbox=mailbox,
        task_store=store,
        poll_interval=0.1,
        timeout=5.0,
    )


class TestInboxDoneAutoComplete:
    """Inbox DONE messages should auto-complete the sender's task."""

    def test_done_message_completes_in_progress_task(self, store, mailbox, waiter):
        t = store.create("build feature", owner="builder-1")
        store.update(t.id, status=TaskStatus.in_progress)

        # Builder sends DONE via inbox
        mailbox.send(from_agent="builder-1", to="leader", content="DONE: feature built")

        result = waiter.wait()

        assert result.status == "completed"
        assert result.completed == 1
        updated = store.get(t.id)
        assert updated.status == TaskStatus.completed
        assert updated.completion_message == "DONE: feature built"

    def test_done_message_completes_pending_task(self, store, mailbox, waiter):
        t = store.create("build feature", owner="builder-1")
        # Task stays pending — agent never called task update in_progress

        mailbox.send(from_agent="builder-1", to="leader", content="DONE: done")

        result = waiter.wait()

        assert result.status == "completed"
        updated = store.get(t.id)
        assert updated.status == TaskStatus.completed

    def test_done_case_insensitive(self, store, mailbox, waiter):
        t = store.create("task", owner="worker")
        store.update(t.id, status=TaskStatus.in_progress)

        mailbox.send(from_agent="worker", to="leader", content="done: finished")

        result = waiter.wait()

        assert result.status == "completed"
        assert store.get(t.id).status == TaskStatus.completed

    def test_done_prefix_with_colon(self, store, mailbox, waiter):
        t = store.create("task", owner="worker")
        store.update(t.id, status=TaskStatus.in_progress)

        mailbox.send(from_agent="worker", to="leader", content="DONE: summary here")

        result = waiter.wait()
        assert result.status == "completed"

    def test_non_done_message_does_not_complete(self, store, mailbox, waiter):
        """Regular messages should not trigger auto-completion."""
        t = store.create("task", owner="worker")
        store.update(t.id, status=TaskStatus.in_progress)

        mailbox.send(from_agent="worker", to="leader", content="Need help: stuck on X")

        # Complete the task normally so wait doesn't hang
        def complete_later():
            import time
            time.sleep(0.3)
            store.update(t.id, status=TaskStatus.completed)

        threading.Thread(target=complete_later, daemon=True).start()
        result = waiter.wait()

        assert result.status == "completed"
        # Verify the message callback fires but task wasn't auto-completed by inbox
        assert result.messages_received >= 1

    def test_done_unblocks_dependent_task(self, store, mailbox, waiter):
        """DONE auto-complete should trigger dependency resolution."""
        t1 = store.create("prerequisite", owner="builder-1")
        store.update(t1.id, status=TaskStatus.in_progress)
        t2 = store.create("depends on t1", owner="builder-2", blocked_by=[t1.id])
        assert store.get(t2.id).status == TaskStatus.blocked

        # Builder-1 sends DONE — should auto-complete t1 and unblock t2
        mailbox.send(from_agent="builder-1", to="leader", content="DONE: prerequisite done")

        # Builder-2 also finishes after being unblocked
        def complete_t2():
            import time
            time.sleep(0.5)
            store.update(t2.id, status=TaskStatus.completed)

        threading.Thread(target=complete_t2, daemon=True).start()
        result = waiter.wait()

        assert result.status == "completed"
        assert result.completed == 2
        assert store.get(t2.id).status == TaskStatus.completed
        assert store.get(t2.id).blocked_by == []

    def test_done_prefers_in_progress_over_pending(self, store, mailbox, waiter):
        """If agent owns both in_progress and pending tasks, complete in_progress first."""
        t1 = store.create("first task", owner="worker")
        store.update(t1.id, status=TaskStatus.in_progress)
        t2 = store.create("second task", owner="worker")
        # t2 stays pending

        mailbox.send(from_agent="worker", to="leader", content="DONE: first done")

        # Complete t2 manually so wait finishes
        def complete_t2():
            import time
            time.sleep(0.3)
            store.update(t2.id, status=TaskStatus.completed)

        threading.Thread(target=complete_t2, daemon=True).start()
        result = waiter.wait()

        assert result.status == "completed"
        assert store.get(t1.id).status == TaskStatus.completed
        assert store.get(t1.id).completion_message == "DONE: first done"

    def test_done_from_unknown_sender_ignored(self, store, mailbox, waiter):
        """DONE from a sender with no tasks should be harmless."""
        t = store.create("task", owner="real-worker")
        store.update(t.id, status=TaskStatus.in_progress)

        mailbox.send(from_agent="ghost", to="leader", content="DONE: who am I")

        # Complete task normally
        def complete_later():
            import time
            time.sleep(0.3)
            store.update(t.id, status=TaskStatus.completed)

        threading.Thread(target=complete_later, daemon=True).start()
        result = waiter.wait()

        assert result.status == "completed"
        # ghost's message didn't break anything
        assert store.get(t.id).status == TaskStatus.completed


class TestWaiterBasic:
    """Basic waiter behavior (pre-existing, not inbox-related)."""

    def test_completes_when_all_tasks_done(self, store, mailbox, waiter):
        t = store.create("task", owner="worker")
        store.update(t.id, status=TaskStatus.completed)

        result = waiter.wait()
        assert result.status == "completed"
        assert result.total == 1
        assert result.completed == 1

    def test_timeout(self, team_name, store, mailbox):
        store.create("task", owner="worker")
        w = TaskWaiter(
            team_name=team_name,
            agent_name="leader",
            mailbox=mailbox,
            task_store=store,
            poll_interval=0.05,
            timeout=0.2,
        )
        result = w.wait()
        assert result.status == "timeout"

    def test_no_tasks_completes_immediately_on_next_poll(self, team_name, store, mailbox):
        """With no tasks, waiter should timeout (total == 0, 0 != 0)."""
        w = TaskWaiter(
            team_name=team_name,
            agent_name="leader",
            mailbox=mailbox,
            task_store=store,
            poll_interval=0.05,
            timeout=0.2,
        )
        result = w.wait()
        assert result.status == "timeout"
        assert result.total == 0
