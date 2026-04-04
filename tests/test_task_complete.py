"""Tests for `clawteam task complete` command and failed task status."""

from __future__ import annotations

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


# --- TaskStatus.failed in store ---

class TestFailedStatus:
    def test_update_to_failed(self, store):
        t = store.create("build feature", owner="builder")
        store.update(t.id, status=TaskStatus.in_progress)
        store.update(t.id, status=TaskStatus.failed, completion_message="PR not merged")

        updated = store.get(t.id)
        assert updated.status == TaskStatus.failed
        assert updated.completion_message == "PR not merged"

    def test_failed_releases_lock(self, store):
        t = store.create("build feature", owner="builder")
        store.update(t.id, status=TaskStatus.in_progress, caller="builder")
        assert store.get(t.id).locked_by == "builder"

        store.update(t.id, status=TaskStatus.failed)
        assert store.get(t.id).locked_by == ""

    def test_failed_tracks_duration(self, store):
        t = store.create("build feature", owner="builder")
        store.update(t.id, status=TaskStatus.in_progress)
        store.update(t.id, status=TaskStatus.failed)

        updated = store.get(t.id)
        assert "duration_seconds" in updated.metadata

    def test_failed_resolves_dependents(self, store):
        t1 = store.create("first")
        t2 = store.create("second", blocked_by=[t1.id])
        assert store.get(t2.id).status == TaskStatus.blocked

        store.update(t1.id, status=TaskStatus.failed)

        # t2 should be unblocked (pending) since t1 is terminal
        updated_t2 = store.get(t2.id)
        assert updated_t2.status == TaskStatus.pending
        assert t1.id not in updated_t2.blocked_by


# --- TaskWaiter with failed tasks ---

class TestWaiterFailedStatus:
    def test_waiter_treats_failed_as_terminal(self, team_name, store, mailbox):
        t = store.create("build feature", owner="builder")
        store.update(t.id, status=TaskStatus.failed, completion_message="PR check failed")

        waiter = TaskWaiter(
            team_name=team_name,
            agent_name="leader",
            mailbox=mailbox,
            task_store=store,
            poll_interval=0.1,
            timeout=2.0,
        )
        result = waiter.wait()

        assert result.status == "completed"
        assert result.failed == 1
        assert result.completed == 0
        assert result.total == 1

    def test_waiter_mixed_completed_and_failed(self, team_name, store, mailbox):
        t1 = store.create("task A", owner="w1")
        t2 = store.create("task B", owner="w2")

        store.update(t1.id, status=TaskStatus.completed, completion_message="DONE: A")
        store.update(t2.id, status=TaskStatus.failed, completion_message="PR not merged")

        waiter = TaskWaiter(
            team_name=team_name,
            agent_name="leader",
            mailbox=mailbox,
            task_store=store,
            poll_interval=0.1,
            timeout=2.0,
        )
        result = waiter.wait()

        assert result.status == "completed"
        assert result.completed == 1
        assert result.failed == 1
        assert result.total == 2

    def test_waiter_completion_messages_include_failed(self, team_name, store, mailbox):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.failed, completion_message="verify failed")

        waiter = TaskWaiter(
            team_name=team_name,
            agent_name="leader",
            mailbox=mailbox,
            task_store=store,
            poll_interval=0.1,
            timeout=2.0,
        )
        result = waiter.wait()

        assert "verify failed" in result.completion_messages


# --- task complete CLI (via store, not CLI runner) ---

class TestTaskCompleteLogic:
    """Test the core logic of task complete: find in_progress tasks by agent, mark completed."""

    def test_complete_marks_in_progress_tasks(self, store):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.in_progress)

        # Simulate what `clawteam task complete` does
        tasks = store.list_tasks()
        targets = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        for target in targets:
            store.update(target.id, status=TaskStatus.completed, completion_message="DONE: built")

        assert store.get(t.id).status == TaskStatus.completed
        assert store.get(t.id).completion_message == "DONE: built"

    def test_complete_is_idempotent(self, store):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.completed, completion_message="first")

        # Second call: no in_progress tasks to find
        tasks = store.list_tasks()
        targets = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        assert len(targets) == 0  # no-op

    def test_complete_skips_pending_tasks(self, store):
        t = store.create("build", owner="builder")
        # Task is pending, not in_progress — should not be touched

        tasks = store.list_tasks()
        targets = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        assert len(targets) == 0

        # Pending task untouched
        assert store.get(t.id).status == TaskStatus.pending

    def test_complete_with_verify_failure_marks_failed(self, store):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.in_progress)

        # Simulate verify failure → mark failed
        tasks = store.list_tasks()
        targets = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        for target in targets:
            store.update(target.id, status=TaskStatus.failed, completion_message="PR not merged")

        assert store.get(t.id).status == TaskStatus.failed

    def test_complete_only_affects_specified_agent(self, store):
        t1 = store.create("task A", owner="alice")
        t2 = store.create("task B", owner="bob")
        store.update(t1.id, status=TaskStatus.in_progress)
        store.update(t2.id, status=TaskStatus.in_progress)

        # Complete only alice's tasks
        tasks = store.list_tasks()
        targets = [t for t in tasks if t.owner == "alice" and t.status == TaskStatus.in_progress]
        for target in targets:
            store.update(target.id, status=TaskStatus.completed)

        assert store.get(t1.id).status == TaskStatus.completed
        assert store.get(t2.id).status == TaskStatus.in_progress


# --- lifecycle on-exit doesn't reset completed/failed ---

class TestLifecycleOnExitSafety:
    def test_on_exit_does_not_reset_completed(self, store):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.completed, completion_message="done")

        # Simulate lifecycle on-exit: only finds in_progress tasks
        tasks = store.list_tasks()
        abandoned = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        assert len(abandoned) == 0  # completed task is NOT abandoned

    def test_on_exit_does_not_reset_failed(self, store):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.failed, completion_message="verify failed")

        tasks = store.list_tasks()
        abandoned = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        assert len(abandoned) == 0  # failed task is NOT abandoned

    def test_on_exit_resets_truly_abandoned(self, store):
        t = store.create("build", owner="builder")
        store.update(t.id, status=TaskStatus.in_progress)

        # Simulate lifecycle on-exit: in_progress IS abandoned
        tasks = store.list_tasks()
        abandoned = [t for t in tasks if t.owner == "builder" and t.status == TaskStatus.in_progress]
        assert len(abandoned) == 1

        for ab in abandoned:
            store.update(ab.id, status=TaskStatus.pending)

        assert store.get(t.id).status == TaskStatus.pending
