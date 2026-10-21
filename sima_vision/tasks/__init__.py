"""The task registry, and how a fourth application joins it.

A task is one application: what to do with a frame once the MLA has finished
with it. Everything before that point -- config, assets, the Neat graph, the
pull loop, the sinks -- is shared, so a new one is a single module subclassing
:class:`~sima_vision.tasks.base.Task`, and the subcommand, its flags, its config
sections and its place in ``--help`` all follow from that.

Two ways in:

**In this package.** Write the module and add the class to :data:`BUILTIN`.

**In your own package.** Advertise an entry point, and `pip install` it next to
this one::

    [project.entry-points."sima_vision.tasks"]
    count = "my_package.count:CountTask"

which is enough for ``sima-vision count`` to exist, with the shared flags, the
asset handling and the automatic setup already attached. Nothing is edited here.
A plugin that fails to import is reported and skipped rather than taking the
whole CLI down with it: one broken third-party package must not stop
``sima-vision detect`` from running.
"""

from __future__ import annotations

from ..console import console
from .base import Task
from .detect import DetectTask
from .fall import FallTask
from .retail import RetailTask
from .segment import SegmentTask

#: Where a separate distribution advertises the tasks it adds.
ENTRY_POINT_GROUP = "sima_vision.tasks"

#: The applications that ship with this package, in ``--help`` order.
BUILTIN: tuple[type[Task], ...] = (DetectTask, SegmentTask, FallTask, RetailTask)


def load_plugins(taken: set[str]) -> dict[str, type[Task]]:
    """Tasks advertised by other installed distributions.

    Args:
        taken: Names already spoken for. A plugin claiming one is skipped, so a
            third-party package cannot quietly replace ``detect`` with its own.

    Returns:
        Name -> task class, for everything that loaded and looked right.
    """
    try:
        from importlib.metadata import entry_points

        found = list(entry_points(group=ENTRY_POINT_GROUP))
    except Exception as exc:  # pragma: no cover - a broken metadata directory
        console.warn(f"could not read the {ENTRY_POINT_GROUP} entry points: {exc}")
        return {}

    plugins: dict[str, type[Task]] = {}
    for entry in found:
        try:
            candidate = entry.load()
        except Exception as exc:
            console.warn(f"task plugin {entry.name!r} failed to import and was skipped: {exc}")
            continue
        if not isinstance(candidate, type) or not issubclass(candidate, Task):
            console.warn(f"task plugin {entry.name!r} is not a Task subclass, skipping")
            continue
        name = candidate.name or entry.name
        if name in taken or name in plugins:
            console.warn(f"task plugin {entry.name!r} wants the name {name!r}, which is taken")
            continue
        plugins[name] = candidate
    return plugins


def build_registry() -> dict[str, type[Task]]:
    """Built-ins first, then whatever else is installed."""
    registry = {task.name: task for task in BUILTIN}
    registry.update(load_plugins(set(registry)))
    return registry


#: Subcommand name -> task class. Built once, at import.
TASKS: dict[str, type[Task]] = build_registry()

__all__ = [
    "BUILTIN",
    "ENTRY_POINT_GROUP",
    "TASKS",
    "DetectTask",
    "FallTask",
    "RetailTask",
    "SegmentTask",
    "Task",
    "build_registry",
    "load_plugins",
]
