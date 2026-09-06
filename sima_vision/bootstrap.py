"""Everything that used to be a setup command, done by the run itself.

There used to be four of them -- `setup board`, `setup network`, `fetch`,
`doctor` -- and between them they were a checklist you had to know existed
before the thing you actually typed would work. None of them are here now. What
they did happens on the way into a run, in order, saying what it is doing:

    pip install sima-vision
    sima-vision detect

The three pieces a run needs are found rather than demanded:

* **pyneat** talks to the MLA. It is an aarch64 wheel from the Palette SDK, not
  something on PyPI, and `sima-cli sdk setup` puts it in a virtualenv of its own
  -- usually `~/pyneat` -- which is never the one pip installed *this* into. So
  it is looked for, and its site-packages goes on `sys.path` ahead of ours,
  which also picks up the numpy<2 it was compiled against. If it is genuinely
  not installed, a wheel left on the board by the SDK is installed from disk.
* **numpy and OpenCV** draw the overlay. The board ships both in
  `/usr/lib/python3*/dist-packages`, so that goes on the path too. Only a board
  missing them installs anything, and then with numpy pinned below 2, because
  2.x breaks pyneat and every `simaai-*` package with it.
* **the model and the clip** are downloaded on first use. See
  :mod:`sima_vision.assets`.

Nothing here runs for `--validate`, which is the point of `--validate`.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import runtime
from .console import console

#: Points at the pyneat virtualenv when it is somewhere this cannot guess.
PYNEAT_ENV = "SIMA_VISION_PYNEAT"

#: Set to 0 to make a run look but never install. The search and the path
#: injection still happen; only pip is held back.
INSTALL_ENV = "SIMA_VISION_AUTO_INSTALL"

#: A pip index carrying a pyneat wheel, if your site publishes one. With this
#: set, a board without pyneat installs it the same way it installs anything.
INDEX_ENV = "SIMA_VISION_PYNEAT_INDEX"

#: Where the pyneat venv ends up, tried first because they cost one stat each.
#: `~/pyneat` is what `sima-cli sdk setup` creates; `/media/nvme` is where you
#: are told to put it by hand, the board's root filesystem being too small.
PYNEAT_HOMES = (
    "~/pyneat",
    "/media/nvme/neat/pyneat",
    "/media/nvme/pyneat",
    "/opt/pyneat",
)

#: Searched when none of the above has it. A board set up by hand, or by a
#: different SDK version, puts it somewhere else entirely, and a fixed list of
#: guesses is exactly the thing that then says "not found" about a venv sitting
#: two directories away.
SEARCH_ROOTS = (
    "~",
    "/opt",
    "/media/nvme",
    "/media/nvme/neat",
    "/usr/local",
    "/srv",
    "/data",
)

#: Wheel names the Palette SDK leaves behind, in the order they are preferred.
WHEEL_PATTERNS = ("pyneat-*.whl", "pyneat*.whl")

#: The Neat core version this package is written against, and the one command
#: that installs it on the board. `sima-cli` holds the community.sima.ai login
#: that the download needs, which is why this is not something pip can do.
NEAT_VERSION = "0.3.0"
NEAT_INSTALL = f"sima-cli neat install core@v{NEAT_VERSION}"

#: numpy 2.x breaks pyneat and every simaai-* package, so the cap is not a
#: preference. OpenCV is headless because nothing here opens a window and the
#: GUI build wants X libraries the board does not have.
IMAGING_REQUIREMENTS = ("numpy>=1.24,<2", "opencv-python-headless>=4.7,<5")


# ---------------------------------------------------------------------------
# Filesystem helpers that answer "no" instead of raising
# ---------------------------------------------------------------------------


def is_dir(path: Path) -> bool:
    """`Path.is_dir()` that treats an unreadable directory as "not it".

    pathlib only swallows ENOENT, ENOTDIR, EBADF and ELOOP; EACCES comes
    straight back out. This walks directories nobody promised us access to --
    on the board plenty of them are root's -- and one unreadable path would
    otherwise end a run with a traceback rather than with the search moving on.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def globs(path: Path, pattern: str) -> list[Path]:
    """`Path.glob()` with the same promise, for the same reason."""
    try:
        return sorted(path.glob(pattern))
    except OSError:
        return []


# ---------------------------------------------------------------------------
# What machine is this
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Environment:
    """Enough about this machine to decide what a failure means.

    The distinction that matters is board or not. Missing pyneat on a DevKit is
    a thing to go and fix; missing pyneat on a laptop is simply what a laptop
    is, and the answer there is `--validate` or `push` rather than an install.
    """

    on_board: bool
    machine: str
    python: str
    executable: str
    dist_packages: tuple[str, ...]

    def summary(self) -> str:
        where = "Modalix DevKit" if self.on_board else f"{sys.platform} host"
        return f"{where}  {self.machine}  python {self.python}"


def detect_environment() -> Environment:
    """Work out where we are, from the two things that are actually diagnostic.

    A DevKit is aarch64 Linux with the board's own `dist-packages` present. An
    x86 SDK container has the second without the first, and must not be mistaken
    for a board: pyneat there would be the wrong architecture entirely.
    """
    import platform

    dist = tuple(sorted(glob.glob("/usr/lib/python3*/dist-packages")))
    machine = platform.machine() or "unknown"
    on_board = (
        sys.platform.startswith("linux")
        and machine in {"aarch64", "arm64"}
        and bool(dist)
    )
    return Environment(
        on_board=on_board,
        machine=machine,
        python=platform.python_version(),
        executable=sys.executable,
        dist_packages=dist,
    )


# ---------------------------------------------------------------------------
# Finding pyneat
# ---------------------------------------------------------------------------


def find_pyneat_env() -> tuple[Path | None, str]:
    """Locate a pyneat virtualenv this interpreter could actually import from.

    Returns:
        A ``(site_packages, note)`` pair. ``site_packages`` is None when there
        is nothing usable, and ``note`` always says why in one line, so the
        step and the error can print the same explanation.

    The version check is the point. pyneat is a compiled extension built for one
    CPython, so putting a 3.10 venv on a 3.12 path swaps ModuleNotFoundError for
    an undefined-symbol crash out of the dynamic linker, which is a far worse
    thing to hand someone.
    """
    want = f"python{sys.version_info.major}.{sys.version_info.minor}"
    override = os.environ.get(PYNEAT_ENV, "")
    wrong_version: list[str] = []

    def look_in(root: Path) -> Path | None:
        """Is there a pyneat for *this* Python under this venv root?"""
        site = root / "lib" / want / "site-packages"
        if globs(site, "pyneat*"):
            return site
        # There, but built for another CPython. Name the interpreter that can
        # use it rather than leaving someone to work it out.
        for other in globs(root / "lib", "python*"):
            if other.name != want and globs(other / "site-packages", "pyneat*"):
                wrong_version.append(f"{root}/bin/python3 ({other.name})")
        return None

    if override:
        root = Path(override).expanduser()
        found = look_in(root) if is_dir(root) else None
        if found:
            return found, f"using pyneat from {root}"
        if wrong_version:
            return None, f"found pyneat, but built for {wrong_version[0]}, not {want}"
        return None, f"${PYNEAT_ENV} is {override}, which has no pyneat for {want}"

    for home in PYNEAT_HOMES:
        root = Path(home).expanduser()
        if is_dir(root):
            found = look_in(root)
            if found:
                return found, f"using pyneat from {root}"

    # Nothing where it is supposed to be, so go and look. One level down from a
    # handful of roots finds every venv anyone actually makes, and stops well
    # short of walking the filesystem.
    for base in SEARCH_ROOTS:
        parent = Path(base).expanduser()
        if not is_dir(parent):
            continue
        try:
            children = sorted(child for child in parent.iterdir() if is_dir(child))
        except OSError:                       # unreadable, not our problem
            continue
        for child in children:
            if not is_dir(child / "lib"):     # not a venv, skip cheaply
                continue
            found = look_in(child)
            if found:
                return found, f"using pyneat from {child}"

    if wrong_version:
        return None, (
            f"found pyneat, but built for {', '.join(sorted(set(wrong_version)))} "
            f"and this is {want}"
        )
    return None, f"no pyneat for {want} anywhere under {', '.join(SEARCH_ROOTS)}"


def find_pyneat_wheel() -> Path | None:
    """A pyneat wheel the SDK left on this board, newest name first.

    `sima-cli sdk setup` copies the wheel over before installing it, and on most
    boards it is still sitting there. Installing from it is the one way a board
    can get pyneat without the PC being involved again.
    """
    found: list[Path] = []
    for base in SEARCH_ROOTS:
        root = Path(base).expanduser()
        if not is_dir(root):
            continue
        for pattern in WHEEL_PATTERNS:
            found.extend(globs(root, pattern))
            for child in globs(root, "*"):
                if is_dir(child):
                    found.extend(globs(child, pattern))
    # Sorted by name, so pyneat-0.3.0 loses to pyneat-0.10.0 only if someone
    # zero-pads. Good enough: boards carry one wheel, not a history of them.
    return sorted(set(found))[-1] if found else None


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------


def installs_allowed() -> bool:
    return os.environ.get(INSTALL_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def venv_python(site: Path) -> Path | None:
    """The interpreter owning a site-packages directory, if it has one.

    ``site`` is ``<root>/lib/pythonX.Y/site-packages``, so the venv root is
    three up. Two up is ``<root>/lib``, and ``<root>/lib/bin/python3`` helps
    nobody.
    """
    for name in ("python3", "python"):
        candidate = site.parents[2] / "bin" / name
        if candidate.exists():
            return candidate
    return None


def shell_line(command: list[str]) -> str:
    """A command echoed so it can be pasted back.

    Both halves of that matter here. `numpy>=1.24,<2` unquoted is a shell
    redirect, and on Windows the interpreter's path routinely has a space in it,
    so the line printed for someone to copy has to survive being copied.
    """
    return " ".join(
        token if re.fullmatch(r"[\w@%+=:,./-]+", token) else '"' + token + '"'
        for token in command
    )


def pip_complaint(output: str) -> list[str]:
    """The lines of a failed pip run that say what went wrong.

    pip ends every failure with two or three lines of generic advice -- "See
    above for output", "This is an issue with the package mentioned above" --
    so a plain tail shows the boilerplate and hides the reason. Its actual
    diagnosis is on the lines it marks ERROR.
    """
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    errors = [line for line in lines if line.lstrip().startswith(("ERROR", "error:"))]
    return (errors or lines)[-3:]


def pip_install(python: str | Path, args: list[str], step) -> bool:
    """Run one pip install, quietly unless it fails.

    pip's output while it is working is a wall of resolver noise that says
    nothing, so it is captured. A failure prints the part of it that is a
    diagnosis, because at that point it is the only thing worth reading.
    """
    if not installs_allowed():
        step.note(f"${INSTALL_ENV}=0, so nothing is being installed")
        return False
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check", *args]
    step.note("$ " + shell_line(command))
    try:
        result = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False,
        )
    except OSError as exc:
        step.note(f"could not run pip: {exc}")
        return False
    if result.returncode == 0:
        return True
    for line in pip_complaint(result.stderr or result.stdout or ""):
        step.note(line)
    return False


# ---------------------------------------------------------------------------
# The three steps
# ---------------------------------------------------------------------------


def missing_pyneat_message(env: Environment, note: str) -> str:
    """What to say when pyneat is not here and cannot be put here."""
    if not env.on_board:
        return (
            "pyneat is missing, and this is not a DevKit.\n"
            "It is an aarch64 wheel from the Palette SDK, so inference only runs on "
            "the board.\n"
            "From here you can still:\n"
            "  sima-vision detect --validate      check the settings, no hardware at all\n"
            "  sima-vision push clip.h264         send a clip over, then run it there"
        )
    return (
        f"pyneat is missing: {note}.\n"
        "It comes with the Neat core, which is not on PyPI. Install it here, on the "
        "board:\n"
        f"  sima-cli login\n  {NEAT_INSTALL}\n"
        "Then run this command again. If pyneat is installed but somewhere unusual:\n"
        f"  export {PYNEAT_ENV}=/path/to/the/venv\n"
        "If `sima-cli` is not on the board either, pairing never finished: run\n"
        "`sima-cli sdk setup --devkit <this board's ip>` from the PC that pairs with it."
    )


def is_neat_runtime(module) -> bool:
    """Is this the SDK's pyneat, or the unrelated one of the same name?

    `pyneat` on PyPI is a NEAT neuroevolution library. Nothing stops it being
    installed in the same interpreter, and binding it would get us all the way
    to building a graph before failing on an attribute, with a message about
    genomes. These three names are the ones every part of this package reaches
    for immediately.
    """
    return all(hasattr(module, name) for name in ("Graph", "Model", "ModelOptions"))


def import_pyneat() -> object | None:
    """`import pyneat`, but only admitting the one that talks to the MLA."""
    try:
        import pyneat
    except Exception:
        # Not just ImportError: an aarch64 extension loaded by the wrong CPython
        # comes out of the dynamic linker, and a half-installed one can raise
        # anything at all on the way up.
        return None
    return pyneat if is_neat_runtime(pyneat) else None


def locate_pyneat() -> tuple[object | None, str]:
    """Import pyneat from wherever it is, putting its venv on the path if needed.

    An interpreter that already has it is never second-guessed: the copy it has
    is the copy its numpy was built against.

    Returns:
        A ``(module, note)`` pair. ``module`` is None when there is nothing to
        import, and ``note`` always says where it came from or why it did not.
    """
    import importlib

    module = import_pyneat()
    if module is not None:
        return module, "already importable"

    site, note = find_pyneat_env()
    if site is None:
        return None, note

    # Ahead of the current environment: that venv also holds the numpy<2 pyneat
    # was compiled against, and that is the one it has to get. Dropping any
    # already-imported impostor first, or the import below is a no-op that hands
    # the wrong module back out of sys.modules.
    sys.modules.pop("pyneat", None)
    path = str(site)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    importlib.invalidate_caches()

    module = import_pyneat()
    if module is None:                        # pragma: no cover - a broken venv
        return None, f"{note}, but it did not import as the Neat runtime"
    return module, note


def ensure_pyneat(env: Environment, step) -> None:
    """Make pyneat importable, installing it from the board's own wheel if not.

    Raises:
        ImportError: When there is no pyneat and no way to get one, carrying the
            instructions for this particular machine.
    """
    module, note = locate_pyneat()
    if module is None and env.on_board and install_pyneat(step):
        module, note = locate_pyneat()
    if module is None:
        raise ImportError(missing_pyneat_message(env, note))
    runtime.pyneat = module
    step.done(f"{getattr(module, '__version__', 'version unknown')}  {note}")


def install_pyneat(step) -> bool:
    """Last resort on a board: install pyneat from a wheel, or from an index.

    Both are the SDK's own artefact, reached differently, and neither invents a
    source: the wheel has to already be on the board -- `sima-cli sdk setup`
    copies it over before installing it, and it is usually still there -- and
    the index has to have been configured.

    Returns:
        Whether pip reported success. Whether that produced something importable
        is the caller's question, and it answers it by looking again.
    """
    wheel = find_pyneat_wheel()
    if wheel is not None:
        step.detail(f"not installed; found the SDK wheel at {wheel}")
        if pip_install(sys.executable, [str(wheel)], step):
            return True

    index = os.environ.get(INDEX_ENV, "").strip()
    if index:
        step.detail(f"not installed; trying ${INDEX_ENV}")
        if pip_install(sys.executable, ["--index-url", index, "pyneat"], step):
            return True
    return False


def ensure_imaging(env: Environment, step) -> None:
    """Make numpy and OpenCV importable, from the board's copies for preference.

    The board ships both in `/usr/lib/python3*/dist-packages`, which is on no
    venv's path, so that directory goes on `sys.path` first. Installing them
    with pip instead would pull numpy 2.x over the board's own copy and break
    every `simaai-*` package, which is why the pip fallback pins numpy below 2
    and only runs when the import genuinely fails.

    Raises:
        ImportError: When neither is available and pip could not supply them.
    """
    for path in env.dist_packages:
        if path not in sys.path:
            sys.path.append(path)

    missing = [name for name in ("numpy", "cv2") if not _importable(name)]
    if missing:
        step.detail(f"missing {', '.join(missing)}; installing them")
        pip_install(imaging_target(), list(IMAGING_REQUIREMENTS), step)

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            f"{exc}\n"
            "A run draws an overlay on every frame, so it needs numpy and OpenCV.\n"
            "On the DevKit both come from the board's own system packages. Install "
            "them with:\n"
            f"  {shell_line([sys.executable, '-m', 'pip', 'install', *IMAGING_REQUIREMENTS])}"
        ) from None

    runtime.np, runtime.cv2 = np, cv2
    runtime.FONT = cv2.FONT_HERSHEY_SIMPLEX
    step.done(f"numpy {np.__version__}  opencv {cv2.__version__}")


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def imaging_target() -> str:
    """Which interpreter should own numpy and OpenCV: whichever one pyneat is in.

    pyneat is a compiled extension linked against a particular numpy. Installing
    numpy into *this* interpreter when pyneat is being borrowed from another
    venv puts the new copy first on the path and hands pyneat an ABI it was not
    built for, which fails at the first tensor rather than at the import.
    """
    module = getattr(runtime.pyneat, "__file__", "") or ""
    if module and not module.startswith(sys.prefix):
        site = Path(module).resolve().parent.parent   # .../lib/pythonX.Y/site-packages
        python = venv_python(site)
        if python is not None:
            return str(python)
    return sys.executable


def ensure_runtime(env: Environment | None = None) -> Environment:
    """The whole of what used to be setup, as two numbered steps.

    Safe to call twice: the second call finds everything already bound and
    returns without printing.

    Returns:
        The detected :class:`Environment`, which the caller has usually already
        printed as step one.
    """
    env = env or detect_environment()
    if runtime.pyneat is not None and runtime.cv2 is not None:
        return env
    with console.step("Loading the Neat runtime", "runtime") as step:
        ensure_pyneat(env, step)
    with console.step("Loading numpy and OpenCV", "imaging") as step:
        ensure_imaging(env, step)
    return env
