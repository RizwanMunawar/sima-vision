"""What the user actually sees.

A run narrates itself, one line per thing it is doing, in the order it does
them::

    Checking environment ...
      Modalix DevKit  aarch64  python 3.11.2
    Loading the Neat runtime ...
      pyneat 0.3.0
    Loading the model ...
      yolo26 -> YoloV26, 80 classes  (45.4s)
    Processing video ...

Each line opens with an icon where the terminal can render one. That is the
whole reason :meth:`Console.icon` exists: Python picks an encoder for stdout the
moment it is redirected, and on a Windows console that is often cp437, which has
no emoji at all. Printing one there does not look wrong, it raises
UnicodeEncodeError and takes the run down with it. So every icon is declared
with an ASCII stand-in and the stream is asked, once, which of the two it can
carry.

Four channels, and which one a line goes to is a decision about the reader:

* :meth:`Console.step` and :meth:`Console.write` are *progress*. ``--quiet``
  drops them, because someone who passed ``--quiet`` is not reading along.
* :meth:`Console.report` is a *result*. It survives ``--quiet``: a run that
  said nothing at all would be a broken one.
* :meth:`Console.warn` is something to act on, so it also survives, and stays
  on stdout in step order -- a warning read out of sequence is half a message.
* :meth:`Console.error` is the only thing on stderr.
"""

from __future__ import annotations

import os
import sys
import time

#: Set to 0/false to strip colour, 1/true to force it on. Unset means "decide".
COLOR_ENV = "SIMA_VISION_COLOR"

#: Set to anything non-empty to drop everything below WARN, as `--quiet` does.
QUIET_ENV = "SIMA_VISION_QUIET"

#: How far a step's own lines are indented under it.
BODY = "  "

#: ``name -> (icon, ASCII stand-in)``. The stand-in is not a fallback nobody
#: sees: a redirected run on Windows is cp437, and that is the common case for
#: anyone piping output into a file to paste into an issue.
ICONS: dict[str, tuple[str, str]] = {
    "check": ("\U0001f50e", "*"),      # magnifying glass
    "runtime": ("\U0001f9e9", "*"),    # puzzle piece
    "imaging": ("\U0001f5bc", "*"),    # framed picture
    "assets": ("\U0001f4e6", "*"),     # package
    "video": ("\U0001f3ac", "*"),      # clapper board
    "model": ("\U0001f9e0", "*"),      # brain
    "build": ("\U00002699", "*"),      # gear
    "run": ("\U000025b6", ">"),        # play
    "export": ("\U0001f4e4", "*"),     # outbox
    "compile": ("\U0001f527", "*"),    # wrench
    "download": ("\U00002b07", "v"),   # down arrow
    "ok": ("\U00002713", "ok"),        # check mark
    "fail": ("\U00002717", "x"),       # ballot x
}


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enable_windows_ansi(stream) -> bool:
    """Turn on VT processing for a Windows console, reporting whether it took.

    Windows 10 and 11 render ANSI perfectly well, but only once the console mode
    says so. Without this the escapes are printed literally, which is worse than
    no colour at all.
    """
    try:
        import ctypes
        from ctypes import wintypes

        handle = ctypes.windll.kernel32.GetStdHandle(-12 if stream is sys.stderr else -11)
        mode = wintypes.DWORD()
        if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # pragma: no cover - any failure here just means no colour
        return False


def want_color(stream) -> bool:
    """Whether to emit escapes at all.

    ``$NO_COLOR`` wins over everything, as its convention requires.
    ``$SIMA_VISION_COLOR`` then forces the answer either way, which is how a CI
    log gets colour on purpose and a pipe gets it never.
    """
    override = os.environ.get(COLOR_ENV)
    if override is not None and override.strip():
        return _truthy(override)
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if os.name == "nt":
        return _enable_windows_ansi(stream)
    return True


class Style:
    """The handful of colours this uses, or empty strings when it uses none."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""

    def paint(self, text: str, colour: str) -> str:
        return f"{colour}{text}{self.reset}" if self.enabled and colour else text


def human_bytes(size: float) -> str:
    """`118.4 MB`. Decimal units, because that is what download pages quote."""
    for unit in ("B", "KB", "MB"):
        if size < 1000:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000.0
    return f"{size:.1f} GB"


def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


class Step:
    """One line saying what is being done, and what came of it.

    Entered as a context manager so the line appears *before* the slow part runs
    and the outcome lands after it. A step that raises is marked failed on the
    way out, so an exception never leaves an unfinished line as the last thing
    on screen.

    The outcome goes on a line of its own rather than rewriting the first one.
    By the time a step finishes there may be output from pyneat or from pip
    printed underneath it, and moving the cursor back over that would eat it.
    """

    def __init__(self, console: Console, message: str, icon: str = "") -> None:
        self.console = console
        self.message = message
        self.icon = icon
        self.started = time.perf_counter()
        self._closed = False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def begin(self) -> Step:
        """Print the line now, so a slow step is never a silent one."""
        mark = self.console.icon(self.icon) if self.icon else ""
        lead = f"{mark} " if mark else ""
        self.console.write(f"{lead}{self.message} ...")
        return self

    def detail(self, text: str) -> None:
        """A continuation line, indented under the step."""
        for line in str(text).splitlines() or [""]:
            self.console.write(f"{BODY}{line}")

    def note(self, text: str) -> None:
        """The same, dimmed: true and worth having, but not worth reading."""
        style = self.console.style
        for line in str(text).splitlines() or [""]:
            self.console.write(f"{BODY}{style.paint(line, style.dim)}")

    def done(self, summary: str = "", timed: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if self.console.active_step is self:
            self.console.active_step = None
        if not summary:
            return
        style = self.console.style
        mark = style.paint(self.console.icon("ok"), style.green)
        when = f" {style.paint('(' + human_time(self.elapsed) + ')', style.dim)}" if timed else ""
        self.console.write(f"{BODY}{mark} {summary}{when}")

    def __enter__(self) -> Step:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.console.active_step is self:
            self.console.active_step = None
        if exc_type is not None and not self._closed:
            self._closed = True
            style = self.console.style
            # Not forced: it closes off the step line visually, and under
            # --quiet there is no step line for it to close. The error itself
            # is printed either way, and that is the part that matters.
            self.console.write(f"{BODY}{style.paint(self.console.icon('fail'), style.red)}")
        return False


class Console:
    """The single place anything user-facing is printed.

    Held as a module-level singleton (:data:`console`) rather than threaded
    through every call: the alternative is one more argument on thirty
    functions that all want the same object.
    """

    def __init__(self, stream=None, quiet: bool | None = None) -> None:
        self._stream = stream
        self.quiet = bool(os.environ.get(QUIET_ENV)) if quiet is None else quiet
        self._styled = (None, Style(False))
        self._fancy: tuple[object, bool] = (None, False)
        #: The step currently open, so a note from deep in the call stack lands
        #: under it instead of at the left margin. Set by `step`, cleared when
        #: that step is closed.
        self.active_step: Step | None = None

    @property
    def stream(self):
        """Whatever stdout is *now*.

        Resolved per access rather than captured in ``__init__``, because this
        object is a module-level singleton built at import: anything that
        replaces ``sys.stdout`` afterwards -- a shell redirect, a pytest
        capture, ``contextlib.redirect_stdout`` -- would otherwise be writing to
        a stream nothing reads.
        """
        return self._stream if self._stream is not None else sys.stdout

    @property
    def style(self) -> Style:
        """Colours for the current stream, decided once per stream.

        ``want_color`` asks the OS about console modes, so it is worth not
        repeating per line; keying the cache on the stream means a redirect
        still gets its own honest answer.
        """
        stream = self.stream
        cached_for, style = self._styled
        if cached_for is not stream:
            style = Style(want_color(stream))
            self._styled = (stream, style)
        return style

    # -- configuration ------------------------------------------------------

    def configure(self, quiet: bool | None = None, stream=None) -> None:
        """Settle the output policy once argv has been parsed."""
        if stream is not None:
            self._stream = stream
            self._styled = (None, Style(False))
        if quiet is not None:
            self.quiet = quiet

    def icon(self, name: str) -> str:
        """The icon for *name*, or its ASCII stand-in on a stream that cannot
        carry one.

        Asked of the stream rather than of the platform. ``PYTHONIOENCODING``,
        a redirect and a Windows console all change the answer independently of
        what OS this is, and getting it wrong is not a cosmetic bug: encoding a
        character the stream cannot represent raises UnicodeEncodeError from
        inside ``print``, which ends the run.
        """
        fancy, plain = ICONS.get(name, ("", ""))
        return fancy if fancy and self.supports_icons else plain

    @property
    def supports_icons(self) -> bool:
        """Whether this stream can encode the icons. Decided once per stream."""
        stream = self.stream
        cached_for, answer = self._fancy
        if cached_for is not stream:
            answer = self._can_encode(stream)
            self._fancy = (stream, answer)
        return answer

    @staticmethod
    def _can_encode(stream) -> bool:
        encoding = getattr(stream, "encoding", None)
        if not encoding:
            # No encoding attribute means an in-memory stream, as in the tests,
            # which takes str straight through and can carry anything.
            return not hasattr(stream, "buffer")
        try:
            "".join(fancy for fancy, _ in ICONS.values()).encode(encoding)
        except (UnicodeEncodeError, LookupError):
            return False
        return True

    # -- primitives ---------------------------------------------------------

    def write(self, text: str = "", force: bool = False) -> None:
        if self.quiet and not force:
            return
        print(text, file=self.stream, flush=True)

    def banner(self, title: str, subtitle: str = "") -> None:
        style = self.style
        line = style.paint(title, style.bold)
        if subtitle:
            line += f"  {style.paint(subtitle, style.cyan)}"
        self.write()
        self.write(line)
        self.write()

    def step(self, message: str, icon: str = "") -> Step:
        """Announce one piece of work. `message` is a phrase, not a label."""
        self.active_step = Step(self, message, icon).begin()
        return self.active_step

    def info(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.write(f"  {line}")

    def note(self, text: str) -> None:
        """An aside, indented under the open step if there is one.

        Callers four levels down from a step -- building the source graph, say --
        have no step to hand and should not be given one just to say a sentence.
        """
        indent = BODY if self.active_step is not None else "  "
        for line in str(text).splitlines() or [""]:
            self.write(f"{indent}{self.style.paint(line, self.style.dim)}")

    def success(self, text: str) -> None:
        self.write(f"  {self.style.paint('ok', self.style.green)}  {text}")

    def report(self, text: str) -> None:
        """A result, not progress. Survives ``--quiet``.

        The line between this and :meth:`info` is what someone asked for versus
        how it was arrived at. ``--quiet`` exists to keep the first and drop the
        second, so a run that says nothing at all would be a broken one.
        """
        for line in str(text).splitlines() or [""]:
            self.write(f"  {line}", force=True)

    def warn(self, text: str) -> None:
        """Warnings survive --quiet: something to act on is not noise."""
        for n, line in enumerate(str(text).splitlines() or [""]):
            head = self.style.paint("warn", self.style.yellow) if n == 0 else "    "
            self.write(f"  {head}  {line}", force=True)

    def error(self, text: str) -> None:
        """Errors go to stderr, always, whatever --quiet says."""
        style = Style(want_color(sys.stderr))
        for n, line in enumerate(str(text).splitlines() or [""]):
            head = style.paint("ERROR", style.red) if n == 0 else "     "
            print(f"  {head}  {line}", file=sys.stderr, flush=True)

    # -- progress -----------------------------------------------------------

    def progress(self, name: str, done: int, total: int) -> None:
        """A one-line download meter, redrawn in place on a terminal only.

        Piped to a file the carriage return does nothing useful and the same
        line lands three hundred times, so a non-terminal gets silence here and
        the single completion line from the caller instead.
        """
        if self.quiet or not self.style.enabled:
            return
        # Name first. It was last, after a fixed-width bar and two sizes, which
        # is exactly the part a narrow terminal cuts off -- leaving a meter that
        # does not say what it is measuring: `9.6 MB / 13.3 MB  peo`.
        if total:
            filled = int(20 * done / total)
            bar = "#" * filled + "-" * (20 - filled)
            text = f"{name}  {bar}  {human_bytes(done)} / {human_bytes(total)}"
        else:
            text = f"{name}  {human_bytes(done)}"
        line = f"{BODY}{text}"
        # Remembered so the erase covers what was actually drawn. A fixed width
        # left the tail of a longer line on screen, and the next line printed
        # over the front of it: `yolo26n-det-bf16-mla_tess-b1.tar.gz  (20.6 MB)`
        # came out with a stray `1.tar.gzz` hanging off the end.
        self._drawn = max(getattr(self, "_drawn", 0), len(line))
        print(f"\r{line}", end="", file=self.stream, flush=True)

    def progress_done(self) -> None:
        if not self.quiet and self.style.enabled:
            width = max(getattr(self, "_drawn", 0), 78)
            print("\r" + " " * width + "\r", end="", file=self.stream, flush=True)
        self._drawn = 0


#: The one console. Commands call :meth:`Console.configure` on it at startup.
console = Console()
