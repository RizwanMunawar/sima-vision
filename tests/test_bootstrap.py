"""Automatic setup: finding pyneat, installing it, and binding the imaging half.

`pip install sima-vision` puts this package wherever pip was pointed. `sima-cli
sdk setup` puts pyneat in a virtualenv of its own at ~/pyneat, and nothing puts
that on the default path -- so the two are almost never in the same place, and
the whole promise of "install it, then run it" rests on the run reconciling them
without being asked to.

These build the board's directory layout under tmp_path and check the search
picks the right one, refuses the wrong one, installs when it can and explains
itself when it cannot. The version check is the part that matters most: pyneat
is a compiled extension, so putting a 3.10 venv on a 3.12 path trades a clear
import error for an undefined-symbol crash out of the dynamic linker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sima_vision import bootstrap, runtime
from sima_vision.bootstrap import Environment
from sima_vision.console import BODY, ICONS, Console

THIS = f"python{sys.version_info.major}.{sys.version_info.minor}"
OTHER = "python3.7" if THIS != "python3.7" else "python3.8"

BOARD = Environment(
    on_board=True, machine="aarch64", python="3.11.2",
    executable="/home/sima/pyneat/bin/python3",
    dist_packages=("/usr/lib/python3.11/dist-packages",),
)
LAPTOP = Environment(
    on_board=False, machine="AMD64", python="3.13.1",
    executable=sys.executable, dist_packages=(),
)


def make_venv(root: Path, version: str, with_pyneat: bool = True) -> Path:
    """The parts of a virtualenv the search actually looks at."""
    site = root / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    (root / "bin").mkdir(exist_ok=True)
    # The interpreter itself, because `venv_python` will only name one that is
    # actually there rather than one that ought to be.
    (root / "bin" / "python3").write_text("", encoding="utf-8")
    if with_pyneat:
        (site / "pyneat").mkdir()
        # Not empty: the import has to come out looking like the Neat runtime,
        # or it is taken for the unrelated PyPI package of the same name.
        (site / "pyneat" / "__init__.py").write_text(
            "class Graph: pass" + chr(10)
            + "class Model: pass" + chr(10)
            + "class ModelOptions: pass" + chr(10),
            encoding="utf-8",
        )
    return site


def fake_pyneat(name: str = "pyneat"):
    """A stand-in that passes the "is this the SDK's pyneat" check.

    There is an unrelated `pyneat` on PyPI, so a bare module object is no longer
    accepted; see `bootstrap.is_neat_runtime`.
    """
    module = type(sys)(name)
    module.Graph = module.Model = module.ModelOptions = object
    return module


class FakeStep:
    """A console Step that records instead of printing."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def detail(self, text) -> None:
        self.lines.append(str(text))

    note = detail

    def done(self, summary="", timed=False) -> None:
        self.lines.append(str(summary))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture(autouse=True)
def no_real_env(monkeypatch, tmp_path):
    """Nothing on the developer's own machine may answer these."""
    monkeypatch.delenv(bootstrap.PYNEAT_ENV, raising=False)
    monkeypatch.delenv(bootstrap.INDEX_ENV, raising=False)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "absent"),))


# -- finding it --


def test_the_env_var_wins(tmp_path, monkeypatch):
    site = make_venv(tmp_path / "custom", THIS)
    monkeypatch.setenv(bootstrap.PYNEAT_ENV, str(tmp_path / "custom"))
    found, note = bootstrap.find_pyneat_env()
    assert found == site
    assert "custom" in note


def test_the_usual_locations_are_searched_in_order(tmp_path, monkeypatch):
    first = make_venv(tmp_path / "pyneat", THIS)
    make_venv(tmp_path / "nvme", THIS)
    monkeypatch.setattr(
        bootstrap, "PYNEAT_HOMES", (str(tmp_path / "pyneat"), str(tmp_path / "nvme"))
    )
    found, _ = bootstrap.find_pyneat_env()
    assert found == first


def test_a_home_that_is_not_there_is_skipped(tmp_path, monkeypatch):
    site = make_venv(tmp_path / "real", THIS)
    monkeypatch.setattr(
        bootstrap, "PYNEAT_HOMES", (str(tmp_path / "gone"), str(tmp_path / "real"))
    )
    assert bootstrap.find_pyneat_env()[0] == site


def test_a_venv_without_pyneat_is_not_taken(tmp_path, monkeypatch):
    """Some other venv at the same path must not be mistaken for this one."""
    make_venv(tmp_path / "pyneat", THIS, with_pyneat=False)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    found, note = bootstrap.find_pyneat_env()
    assert found is None
    assert THIS in note


def test_a_venv_for_another_python_is_refused_by_name(tmp_path, monkeypatch):
    """A compiled extension for the wrong CPython crashes the linker, not the import."""
    make_venv(tmp_path / "pyneat", OTHER)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    found, note = bootstrap.find_pyneat_env()
    assert found is None
    assert OTHER in note and THIS in note
    assert "bin/python3" in note, "say which interpreter can use it"


def test_the_right_version_wins_when_a_venv_holds_several(tmp_path, monkeypatch):
    make_venv(tmp_path / "pyneat", OTHER)
    site = make_venv(tmp_path / "pyneat", THIS)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    assert bootstrap.find_pyneat_env()[0] == site


def test_a_wrong_env_var_says_it_was_the_env_var(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv(bootstrap.PYNEAT_ENV, str(tmp_path / "empty"))
    found, note = bootstrap.find_pyneat_env()
    assert found is None
    assert bootstrap.PYNEAT_ENV in note


def test_the_env_var_stops_the_search(tmp_path, monkeypatch):
    """Pointing somewhere wrong must not silently fall back to somewhere right."""
    make_venv(tmp_path / "default", THIS)
    (tmp_path / "explicit").mkdir()
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "default"),))
    monkeypatch.setenv(bootstrap.PYNEAT_ENV, str(tmp_path / "explicit"))
    assert bootstrap.find_pyneat_env()[0] is None


# -- what machine is this --


def test_a_laptop_is_not_mistaken_for_a_board(monkeypatch):
    monkeypatch.setattr(bootstrap.glob, "glob", lambda _p: [])
    assert bootstrap.detect_environment().on_board is False


def test_an_x86_container_with_dist_packages_is_not_a_board(monkeypatch):
    """The SDK container has the board's directory layout and the wrong CPU."""
    import platform

    monkeypatch.setattr(bootstrap.glob, "glob", lambda _p: ["/usr/lib/python3.11/dist-packages"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert bootstrap.detect_environment().on_board is False


def test_an_aarch64_linux_with_dist_packages_is_a_board(monkeypatch):
    import platform

    monkeypatch.setattr(bootstrap.glob, "glob", lambda _p: ["/usr/lib/python3.11/dist-packages"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    env = bootstrap.detect_environment()
    assert env.on_board is True
    assert "DevKit" in env.summary()


# -- what it tells you --


def test_the_message_off_board_points_at_the_board():
    message = bootstrap.missing_pyneat_message(LAPTOP, "no pyneat virtualenv found")
    assert "not a DevKit" in message
    # Advice that does not parse costs a run to discover: `watch` was named
    # here and no longer exists. Check every command it suggests against the
    # parser rather than against a name written down twice.
    from sima_vision.cli import build_parser

    available = {
        name
        for action in build_parser()._actions
        for name in (getattr(action, "choices", None) or ())
    }
    named = {
        line.split()[1]
        for line in message.splitlines()
        if line.strip().startswith("sima-vision ")
    }
    assert named, "say what can still be done from here"
    assert named <= available, f"names commands that do not exist: {sorted(named - available)}"


def test_the_message_on_board_gives_the_install_command():
    message = bootstrap.missing_pyneat_message(BOARD, "no pyneat virtualenv found")
    assert "sima-cli sdk setup --devkit" in message
    assert bootstrap.PYNEAT_ENV in message


def test_every_message_is_ascii():
    """These print on a board whose console encoding is nobody's guess."""
    for env in (BOARD, LAPTOP):
        for note in ("no pyneat virtualenv found", "found pyneat, but built for x"):
            text = bootstrap.missing_pyneat_message(env, note)
            assert all(ord(ch) < 128 for ch in text), text


# -- loading --


@pytest.fixture
def unbound(monkeypatch):
    """Pretend nothing has been imported yet, and undo it afterwards."""
    monkeypatch.setattr(runtime, "pyneat", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    runtime.pyneat = None
    sys.modules.pop("pyneat", None)


def test_a_working_interpreter_is_never_interfered_with(monkeypatch, unbound):
    """If `import pyneat` succeeds, no venv is searched for or put on the path.

    The board's `/usr/lib/python3*/dist-packages` is still added, because that
    is where its OpenCV lives and every run needs it. What must not happen is a
    second environment appearing underneath an interpreter that was already
    working.
    """
    before = list(sys.path)

    def refuse():
        raise AssertionError("must not search when the plain import works")

    monkeypatch.setattr(bootstrap, "find_pyneat_env", refuse)
    monkeypatch.setitem(sys.modules, "pyneat", fake_pyneat())
    bootstrap.ensure_runtime()

    added = [p for p in sys.path if p not in before]
    assert not any("site-packages" in p for p in added), added


def test_the_venv_goes_ahead_of_the_current_environment(tmp_path, monkeypatch, unbound, capsys):
    """It holds the numpy<2 pyneat was built against, which has to win."""
    site = make_venv(tmp_path / "pyneat", THIS)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    monkeypatch.delitem(sys.modules, "pyneat", raising=False)

    bootstrap.ensure_runtime()

    assert sys.path[0] == str(site), "ahead of everything, not appended"
    assert runtime.pyneat.__file__.startswith(str(site))
    # It says so, because a path appearing from nowhere is worse than a line.
    assert "using pyneat from" in capsys.readouterr().out


def test_a_missing_pyneat_off_board_raises_the_explanation(monkeypatch, unbound):
    monkeypatch.delitem(sys.modules, "pyneat", raising=False)
    monkeypatch.setattr(bootstrap, "find_pyneat_env", lambda: (None, "nothing found"))
    with pytest.raises(ImportError, match="pyneat"):
        bootstrap.ensure_runtime(LAPTOP)


def test_a_missing_pyneat_off_board_does_not_try_to_install(monkeypatch, unbound):
    """Installing an aarch64 wheel on a laptop could only ever fail confusingly."""
    monkeypatch.delitem(sys.modules, "pyneat", raising=False)
    monkeypatch.setattr(bootstrap, "find_pyneat_env", lambda: (None, "nothing found"))
    monkeypatch.setattr(
        bootstrap, "install_pyneat",
        lambda step: pytest.fail("must not install off the board"),
    )
    with pytest.raises(ImportError):
        bootstrap.ensure_runtime(LAPTOP)


#: Not a real path anywhere, on purpose. A test that names the board's actual
#: dist-packages passes or fails on whether the host happens to have it and
#: whether it is already on sys.path, neither of which is what is being tested.
FAKE_DIST = "/nowhere/sima-vision-test/dist-packages"


def test_whatever_the_glob_finds_goes_on_the_path(monkeypatch, unbound):
    """The board's OpenCV branch, exercised on every platform.

    Faking the glob with a path that exists nowhere and is on no `sys.path`
    removes the host from the question entirely: on Windows the real directory
    does not exist, and on Linux it is already on the path, so either one makes
    this pass for a reason that has nothing to do with the behaviour.
    """
    monkeypatch.setattr(bootstrap.glob, "glob", lambda _p: [FAKE_DIST])
    monkeypatch.setitem(sys.modules, "pyneat", fake_pyneat())
    before = list(sys.path)
    bootstrap.ensure_runtime()
    assert [p for p in sys.path if p not in before] == [FAKE_DIST]


def test_a_path_already_present_is_not_added_twice(monkeypatch, unbound):
    """Repeated runs in one process must not grow sys.path."""
    monkeypatch.setattr(sys, "path", [FAKE_DIST, *sys.path])
    monkeypatch.setattr(bootstrap.glob, "glob", lambda _p: [FAKE_DIST])
    monkeypatch.setitem(sys.modules, "pyneat", fake_pyneat())
    before = list(sys.path)
    bootstrap.ensure_runtime()
    assert sys.path == before


def test_a_second_call_does_nothing(monkeypatch, unbound, capsys):
    """`run()` and the CLI both call it; the second one must be silent."""
    monkeypatch.setitem(sys.modules, "pyneat", fake_pyneat())
    bootstrap.ensure_runtime()
    capsys.readouterr()
    bootstrap.ensure_runtime()
    assert capsys.readouterr().out == ""


def test_an_unrelated_package_called_pyneat_is_not_mistaken_for_this_one(
    monkeypatch, unbound
):
    """`pyneat` on PyPI is a neuroevolution library. Binding it fails much later."""
    impostor = type(sys)("pyneat")
    impostor.Genome = object
    monkeypatch.setitem(sys.modules, "pyneat", impostor)
    monkeypatch.setattr(bootstrap, "find_pyneat_env", lambda: (None, "nothing found"))
    with pytest.raises(ImportError):
        bootstrap.ensure_runtime(LAPTOP)


def test_a_pyneat_that_raises_on_import_is_not_fatal(monkeypatch, unbound):
    """A wrong-architecture extension comes out of the linker, not as ImportError."""
    import builtins

    real = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "pyneat":
            raise OSError("undefined symbol: PyUnicode_AsUTF8")
        return real(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pyneat", raising=False)
    monkeypatch.setattr(builtins, "__import__", explode)
    assert bootstrap.import_pyneat() is None


# -- installing --


def test_a_wheel_left_by_the_sdk_is_found(tmp_path, monkeypatch):
    wheel = tmp_path / "pyneat-0.3.0-cp311-cp311-linux_aarch64.whl"
    wheel.write_bytes(b"PK")
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    assert bootstrap.find_pyneat_wheel() == wheel


def test_a_wheel_one_level_down_is_found(tmp_path, monkeypatch):
    (tmp_path / "sdk").mkdir()
    wheel = tmp_path / "sdk" / "pyneat-0.3.0-cp311-cp311-linux_aarch64.whl"
    wheel.write_bytes(b"PK")
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    assert bootstrap.find_pyneat_wheel() == wheel


def test_no_wheel_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    assert bootstrap.find_pyneat_wheel() is None


def test_installs_can_be_turned_off(monkeypatch):
    """A locked-down board must be able to say "look, but do not touch"."""
    monkeypatch.setenv(bootstrap.INSTALL_ENV, "0")
    assert bootstrap.installs_allowed() is False
    step = FakeStep()
    monkeypatch.setattr(
        bootstrap.subprocess, "run",
        lambda *a, **k: pytest.fail("nothing may be installed"),
    )
    assert bootstrap.pip_install(sys.executable, ["anything"], step) is False
    assert bootstrap.INSTALL_ENV in step.text


def test_the_install_command_is_printed_before_it_runs(monkeypatch):
    """Something writing to another virtualenv has to say so first."""
    calls = []

    class Result:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, **k: calls.append(cmd) or Result())
    step = FakeStep()
    assert bootstrap.pip_install("/pyneat/bin/python3", ["numpy<2"], step) is True
    assert calls[0][:4] == ["/pyneat/bin/python3", "-m", "pip", "install"]
    assert "/pyneat/bin/python3 -m pip install" in step.text


def test_a_failed_install_reports_the_tail_of_pips_output(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "line one\nline two\nERROR: no matching distribution"

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, **k: Result())
    step = FakeStep()
    assert bootstrap.pip_install(sys.executable, ["pyneat"], step) is False
    assert "no matching distribution" in step.text


def test_the_reason_is_shown_not_pips_closing_boilerplate(monkeypatch):
    """pip signs off with three lines of generic advice. Those are not the reason."""
    class Result:
        returncode = 1
        stdout = ""
        stderr = (
            "Collecting numpy" + chr(10)
            + "ERROR: Could not find a version that satisfies the requirement numpy<2"
            + chr(10)
            + chr(10)
            + "See above for output." + chr(10)
            + "note: This is an issue with the package mentioned above, not pip."
        )

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, **k: Result())
    step = FakeStep()
    assert bootstrap.pip_install(sys.executable, ["numpy<2"], step) is False
    assert "Could not find a version" in step.text
    assert "See above for output" not in step.text


def test_the_echoed_command_can_be_pasted_back():
    """`numpy>=1.24,<2` unquoted is a redirect, and Windows paths have spaces."""
    line = bootstrap.shell_line(
        ["C:/Program Files/py.exe", "-m", "pip", "install", "numpy>=1.24,<2"]
    )
    assert '"numpy>=1.24,<2"' in line
    assert '"C:/Program Files/py.exe"' in line
    assert " -m pip install " in line, "ordinary tokens are left alone"


def test_imaging_is_installed_into_the_venv_that_holds_pyneat(tmp_path, monkeypatch):
    """numpy has to land beside the extension it was compiled against, not here."""
    site = make_venv(tmp_path / "pyneat", THIS)
    fake = fake_pyneat()
    fake.__file__ = str(site / "pyneat" / "__init__.py")
    monkeypatch.setattr(runtime, "pyneat", fake)
    assert bootstrap.imaging_target() == str(tmp_path / "pyneat" / "bin" / "python3")


def test_imaging_falls_back_to_this_interpreter(monkeypatch):
    fake = fake_pyneat()
    fake.__file__ = str(Path(sys.prefix) / "lib" / "pyneat" / "__init__.py")
    monkeypatch.setattr(runtime, "pyneat", fake)
    assert bootstrap.imaging_target() == sys.executable


def test_the_venv_interpreter_is_named_not_its_lib_directory(tmp_path):
    """site-packages is three levels down, and `<root>/lib/bin/python3` helps nobody."""
    site = make_venv(tmp_path / "pyneat", THIS)
    found = bootstrap.venv_python(site)
    assert found is not None
    assert found.parent.parent == tmp_path / "pyneat"


# -- searching, when it is not where it is supposed to be --


def test_a_venv_somewhere_unexpected_is_still_found(tmp_path, monkeypatch):
    """A fixed list of guesses says "not found" about a venv two dirs away.

    This is what a board set up by hand looks like: pyneat is real, it is just
    not at ~/pyneat, and a lookup that only knows the usual places gives up
    without looking.
    """
    site = make_venv(tmp_path / "some-other-name", THIS)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", ("/nowhere",))
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    found, note = bootstrap.find_pyneat_env()
    assert found == site
    assert "some-other-name" in note


def test_the_known_places_win_over_the_search(tmp_path, monkeypatch):
    """One stat beats a directory walk, and the usual place is usually right."""
    known = make_venv(tmp_path / "known", THIS)
    make_venv(tmp_path / "searched" / "other", THIS)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(tmp_path / "known"),))
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path / "searched"),))
    assert bootstrap.find_pyneat_env()[0] == known


def test_the_search_does_not_descend_into_everything(tmp_path, monkeypatch):
    """One level down only. A deep tree must not turn this into a find(1)."""
    buried = tmp_path / "a" / "b" / "c" / "deep"
    make_venv(buried, THIS)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", ("/nowhere",))
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    assert bootstrap.find_pyneat_env()[0] is None


def test_a_directory_that_is_not_a_venv_is_skipped_cheaply(tmp_path, monkeypatch):
    (tmp_path / "just-a-folder").mkdir()
    (tmp_path / "just-a-folder" / "notes.txt").write_text("hi", encoding="utf-8")
    site = make_venv(tmp_path / "real", THIS)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", ("/nowhere",))
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    assert bootstrap.find_pyneat_env()[0] == site


@pytest.mark.parametrize("failing", ["is_dir", "iterdir", "glob"])
def test_an_unreadable_path_does_not_stop_the_search(tmp_path, monkeypatch, failing):
    """A directory we are not allowed to read must be skipped, not fatal.

    pathlib swallows ENOENT, ENOTDIR, EBADF and ELOOP; EACCES comes straight
    back out. The board has plenty of root-owned directories under the places
    this searches, and one of them raising would end a run with a traceback.

    Faked rather than made with real permissions: chmod means nothing on
    Windows and nothing at all when the tests run as root, so a genuinely
    unreadable directory is not something every machine can produce.
    """
    site = make_venv(tmp_path / "real", THIS)
    forbidden = tmp_path / "locked"
    forbidden.mkdir()

    real = getattr(Path, failing)

    def refuse(self, *args, **kwargs):
        if str(self).startswith(str(forbidden)):
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, failing, refuse)
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", (str(forbidden),))
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))

    assert bootstrap.find_pyneat_env()[0] == site


def test_the_not_found_note_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "PYNEAT_HOMES", ("/nowhere",))
    monkeypatch.setattr(bootstrap, "SEARCH_ROOTS", (str(tmp_path),))
    _, note = bootstrap.find_pyneat_env()
    assert str(tmp_path) in note, "an empty 'not found' leaves nothing to act on"
    assert THIS in note, "and it has to say which Python it wanted"


# -- the console the steps print through --


def test_the_console_follows_a_redirected_stdout(capsys):
    """It is a singleton built at import, so it must not capture stdout then."""
    Console().info("hello")
    assert "hello" in capsys.readouterr().out


def test_quiet_keeps_warnings_and_drops_steps(capsys):
    console = Console(quiet=True)
    console.info("routine")
    console.step("thing", "doing it")
    console.warn("something you should know")
    out = capsys.readouterr().out
    assert "routine" not in out and "doing it" not in out
    assert "something you should know" in out


def test_a_step_says_what_it_is_doing_before_it_does_it(capsys):
    """A slow step must not be a silent one, so the line comes first."""
    console = Console()
    step = console.step("Loading the model", "model")
    assert "Loading the model ..." in capsys.readouterr().out
    step.done("80 classes")
    assert "80 classes" in capsys.readouterr().out


def test_a_step_that_raises_says_so(capsys):
    console = Console()
    with pytest.raises(ValueError):
        with console.step("Loading the model", "model"):
            raise ValueError("nope")
    out = capsys.readouterr().out
    assert "Loading the model ..." in out
    assert console.icon("fail") in out


def test_a_note_from_inside_a_step_is_indented_under_it(capsys):
    """Callers deep in the stack have no step to hand; the console knows.

    `raw H.264 elementary stream, demuxer bypassed` is printed four calls below
    the step that is running, and landed at the left margin in the middle of an
    indented block.
    """
    console = Console()
    with console.step("Building the pipeline", "build"):
        console.note("raw H.264 elementary stream, demuxer bypassed")
    line = [ln for ln in capsys.readouterr().out.splitlines() if "demuxer" in ln][0]
    assert line.startswith(BODY + "raw"), repr(line)


def test_a_note_outside_a_step_is_not_indented(capsys):
    console = Console()
    with console.step("Building the pipeline", "build"):
        pass
    console.note("afterwards")
    line = [ln for ln in capsys.readouterr().out.splitlines() if "afterwards" in ln][0]
    assert line.startswith("  ") and not line.startswith("   "), repr(line)


def test_a_stream_that_cannot_encode_icons_gets_plain_ascii():
    """The icons are the one thing here that is not ASCII, so they are the risk.

    Python picks an encoder for stdout the moment it is redirected. On a Windows
    console that is often cp437, and printing an emoji there does not degrade --
    it raises UnicodeEncodeError from inside `print` and takes the run down. The
    stand-ins are what test_portability is really relying on.
    """
    import io

    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp437", newline="")
    console = Console(stream=legacy)
    assert console.supports_icons is False

    console.banner("sima-vision 2.0.0", "detect")
    step = console.step("Loading the model", "model")
    step.detail("detail")
    step.note("note")
    step.done("done", timed=True)
    console.warn("warn")
    legacy.flush()

    text = legacy.buffer.getvalue().decode("cp437")
    assert all(ord(ch) < 128 for ch in text), text
    assert "Loading the model ..." in text


def test_a_stream_that_can_encode_them_gets_the_icons():
    import io

    modern = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
    console = Console(stream=modern)
    assert console.supports_icons is True
    console.step("Loading the model", "model")
    modern.flush()
    assert ICONS["model"][0] in modern.buffer.getvalue().decode("utf-8")


def test_every_icon_declares_an_ascii_stand_in():
    """A missing stand-in is invisible until someone redirects the output."""
    for name, (fancy, plain) in ICONS.items():
        assert fancy, name
        assert plain, f"{name} has no ASCII stand-in"
        assert all(ord(ch) < 128 for ch in plain), name
