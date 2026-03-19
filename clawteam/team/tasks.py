"""Task store for shared team task management.

This module re-exports from :mod:`clawteam.store` for backward compatibility.
New code should import from ``clawteam.store`` directly.
"""

# Lazy imports to avoid circular dependency:
# store/base.py -> team/models.py -> team/__init__.py -> team/tasks.py -> store/base.py


def __getattr__(name: str):
    if name == "TaskStore":
        from clawteam.store.file import FileTaskStore
        return FileTaskStore
    if name == "TaskLockError":
        from clawteam.store.base import TaskLockError
        return TaskLockError
    if name == "BaseTaskStore":
        from clawteam.store.base import BaseTaskStore
        return BaseTaskStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TaskStore", "TaskLockError", "BaseTaskStore"]  # noqa: F822
