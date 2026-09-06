"""The Python API: the same three verbs the command line has.

Every keyword these accept is derived from the CLI's own flags, so
``--blur-strength 81`` and ``blur_strength=81`` cannot drift apart -- there is
one table and it is built from the parser at import time.

    from sima_vision import run, validate

    validate("detect", conf=0.5)
    run("detect", source="clip.h264", model="yolo26m-det.tar.gz", conf=0.5)

``validate`` needs no board and touches no network. ``run`` needs the DevKit,
because that is where the MLA is.
"""

from __future__ import annotations

import argparse
from functools import cache
from pathlib import Path

from .tasks import TASKS


def _task(name: str):
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}. Choose one of: {', '.join(TASKS)}")
    return TASKS[name]()


def _alias_table(task) -> tuple[dict[str, str], set[str]]:
    """Cached by task class -- the flags cannot change between calls."""
    return _build_alias_table(type(task))


@cache
def _build_alias_table(task_cls) -> tuple[dict[str, str], set[str]]:
    """Map Python keyword -> config path, straight off this task's CLI flags.

    Returns the table and the set of keywords whose boolean has to be flipped:
    ``--send`` turns ``alerts.dry_run`` *off*, so ``send=True`` must write
    False. ``--no-save`` is registered as ``save`` for the same reason, in the
    other direction -- nobody wants to write ``no_save=True``.
    """
    from .cli import add_shared_arguments

    task = task_cls()
    parser = argparse.ArgumentParser(add_help=False)
    add_shared_arguments(parser)
    task.add_arguments(parser.add_argument_group("task"))

    aliases: dict[str, str] = {}
    inverted: set[str] = set()
    for action in parser._actions:
        if "." not in action.dest:
            continue
        turns_off = getattr(action, "const", None) is False
        for option in action.option_strings:
            negative = option.startswith("--no-")
            name = option.lstrip("-").replace("-", "_")
            if negative:
                name = name[3:]          # no_save -> save
            # Two flags claiming one keyword would silently hide a setting --
            # `--video PATH` and `--no-video` both wanted `video` once, and the
            # path became unreachable from Python.
            claimed = aliases.get(name)
            if claimed is not None and claimed != action.dest:
                raise RuntimeError(
                    f"{task.name}: {option} wants the keyword {name!r}, which "
                    f"already means {claimed!r}. Rename one of the flags."
                )
            aliases[name] = action.dest
            # `--send` reads as "do send", but it clears a dry_run flag.
            if turns_off and not negative:
                inverted.add(name)
    return aliases, inverted


def settings_to_overrides(task, settings: dict) -> dict:
    """Translate Python keywords into the dotted config paths the loader takes.

    Dotted keys are passed through untouched, so anything the aliases do not
    cover is still reachable::

        run("detect", **{"runtime.output_buffers": 2})

    Raises:
        TypeError: On a keyword that is neither an alias nor a dotted path,
            listing the near misses.
    """
    aliases, inverted = _alias_table(task)
    overrides: dict = {}
    for key, value in settings.items():
        if "." in key:
            overrides[key] = value
            continue
        path = aliases.get(key)
        if path is None:
            near = sorted(name for name in aliases if key in name or name in key)
            hint = f" Did you mean: {', '.join(near[:5])}?" if near else ""
            raise TypeError(f"{task.name}() got an unexpected setting {key!r}.{hint}")
        overrides[path] = not value if key in inverted else value
    return overrides


def load(task: str, config: str | Path | None = None, use_config_file: bool = True,
         **settings):
    """Resolve a configuration the way the CLI does, and validate it.

    Args:
        task: ``detect``, ``segment`` or ``fall``.
        config: Path to a config file, or None to look for ``./config.yaml``.
        use_config_file: False ignores any file, like ``--no-config``.
        **settings: Anything the CLI takes, as a keyword.

    Returns:
        A validated config for the task.
    """
    handle = _task(task)
    return handle.load(
        Path(config) if config else None,
        settings_to_overrides(handle, settings),
        use_file=use_config_file,
    )


def validate(task: str, config: str | Path | None = None, **settings):
    """Check a configuration without a board. Raises ValueError if it is wrong.

    Returns:
        The resolved config, so it can be inspected.
    """
    return load(task, config, **settings)


def run(task: str, config: str | Path | None = None, **settings) -> int:
    """Run one task to completion. **Needs the DevKit.**

    A clip or model archive that is missing is downloaded into ``assets/``
    first; see :mod:`sima_vision.assets`. ``validate`` resolves the same paths
    and never fetches anything.

    Args:
        task: ``detect``, ``segment`` or ``fall``.
        config: Path to a config file, or None to look for ``./config.yaml``.
        **settings: Anything the CLI takes, as a keyword.

    Returns:
        The number of frames processed.
    """
    import os

    from . import __version__
    from .bootstrap import detect_environment, ensure_runtime
    from .console import console
    from .runloop import Stopper

    handle = _task(task)
    cfg = load(task, config, **settings)

    # The same steps, in the same order, as the command line: `run("detect")`
    # and `sima-vision detect` are one code path with two front doors.
    console.banner(f"sima-vision {__version__}", task)
    with console.step("Checking the environment", "check") as step:
        env = detect_environment()
        step.done(env.summary())
    ensure_runtime(env)

    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
    if cfg.save_enable:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    return handle.run(cfg, Stopper())
