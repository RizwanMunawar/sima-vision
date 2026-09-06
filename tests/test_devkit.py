"""push and pull: the scp wrappers.

No network and no ssh binary. What is being tested is the command line each one
builds and the directory it runs from, because that is where the platform
differences live: a Windows drive letter looks like a hostname to scp.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sima_vision import devkit
from sima_vision.cli import main

HOST = "sima@192.168.137.50"


@pytest.fixture(autouse=True)
def tools(monkeypatch, tmp_path):
    """Pretend ssh and scp are installed, and forget any real DevKit address.

    Also run from a scratch directory. Nothing here writes into it any more --
    the command that did, `watch`, is gone -- but a test that runs from the
    repository root is one accident away from leaving something in it, which is
    how `sima-vision.sdp` came to be committed once.
    """
    monkeypatch.setattr(devkit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv(devkit.DEVKIT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)


class Calls(list):
    """The subprocess calls made, with the `ls` answer to hand back on `listing`."""

    listing: dict


@pytest.fixture
def calls(monkeypatch):
    """Record every subprocess, and answer `ls` with whatever is queued."""
    seen = Calls()
    listing = {"out": ""}

    class Result:
        returncode = 0

        def __init__(self):
            self.stdout = listing["out"]
            self.stderr = ""

    def fake_run(command, cwd=None, check=False, **kwargs):
        seen.append({"command": command, "cwd": Path(cwd) if cwd else None})
        return Result()

    monkeypatch.setattr(devkit.subprocess, "run", fake_run)
    seen.listing = listing
    return seen


# ── the host ──


def test_the_host_can_come_from_the_environment(monkeypatch, calls):
    monkeypatch.setenv(devkit.DEVKIT_ENV, HOST)
    assert devkit.resolve_host(None) == HOST
    # And the flag still wins over it.
    assert devkit.resolve_host("sima@other") == "sima@other"


def test_no_host_anywhere_explains_how_to_set_one():
    with pytest.raises(SystemExit) as caught:
        devkit.resolve_host(None)
    message = str(caught.value)
    assert devkit.DEVKIT_ENV in message
    assert "--host" in message


def test_a_missing_ssh_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(devkit.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit, match="OpenSSH"):
        devkit.require("ssh")


# ── push ──


def test_push_runs_from_the_files_own_folder(tmp_path, calls):
    """The whole point: scp never sees a path, so a `D:` is never a hostname."""
    clip = tmp_path / "clips" / "a.h264"
    clip.parent.mkdir()
    clip.write_bytes(b"x")

    assert devkit.run_push([clip], HOST, "~/") == 0
    (call,) = calls
    assert call["cwd"] == clip.parent
    assert call["command"] == ["/usr/bin/scp", "-r", "a.h264", f"{HOST}:~/"]
    assert not any(":" in token for token in call["command"][2:-1])


def test_push_groups_by_folder(tmp_path, calls):
    """Two folders means two scp calls, each run from its own."""
    for folder, name in (("one", "a.h264"), ("two", "b.tar.gz")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_bytes(b"x")

    paths = [tmp_path / "one" / "a.h264", tmp_path / "two" / "b.tar.gz"]
    assert devkit.run_push(paths, HOST, "~/") == 0
    assert len(calls) == 2
    assert {call["cwd"] for call in calls} == {tmp_path / "one", tmp_path / "two"}


def test_push_sends_two_files_from_one_folder_in_one_call(tmp_path, calls):
    for name in ("a.h264", "b.h264"):
        (tmp_path / name).write_bytes(b"x")
    paths = [tmp_path / "a.h264", tmp_path / "b.h264"]
    assert devkit.run_push(paths, HOST, "~/") == 0
    (call,) = calls
    assert call["command"][2:4] == ["a.h264", "b.h264"]


def test_push_checks_the_files_exist_first(tmp_path, calls):
    with pytest.raises(SystemExit, match="not found"):
        devkit.run_push([tmp_path / "nope.h264"], HOST, "~/")
    assert not calls, "nothing should have been attempted"


def test_push_reports_a_failed_transfer(tmp_path, monkeypatch):
    (tmp_path / "a.h264").write_bytes(b"x")

    class Failed:
        returncode = 255

    monkeypatch.setattr(devkit.subprocess, "run", lambda *a, **k: Failed())
    assert devkit.run_push([tmp_path / "a.h264"], HOST, "~/") == 255


# ── pull ──


def test_pull_asks_what_exists_then_fetches_only_that(tmp_path, calls):
    calls.listing["out"] = "detections.mp4\nframes\n"
    assert devkit.run_pull([], HOST, tmp_path) == 0

    ask, fetch = calls
    assert ask["command"][0] == "/usr/bin/ssh"
    assert "ls -d --" in ask["command"][2]
    # Every candidate is offered, and only the two that came back are fetched.
    assert "segmentation.mp4" in ask["command"][2]
    assert fetch["command"] == [
        "/usr/bin/scp", "-r",
        f"{HOST}:detections.mp4", f"{HOST}:frames", ".",
    ]
    assert fetch["cwd"] == tmp_path


def test_pull_runs_from_the_destination(tmp_path, calls):
    """Same reason as push: a Windows destination would carry a colon."""
    calls.listing["out"] = "detections.mp4\n"
    into = tmp_path / "results"
    assert devkit.run_pull([], HOST, into) == 0
    assert calls[1]["cwd"] == into
    assert calls[1]["command"][-1] == "."
    assert into.is_dir(), "the destination is created"


def test_pull_can_name_one_file(tmp_path, calls):
    calls.listing["out"] = "falls.mp4\n"
    assert devkit.run_pull(["falls.mp4"], HOST, tmp_path) == 0
    assert "falls.mp4" in calls[0]["command"][2]
    assert "detections.mp4" not in calls[0]["command"][2]


def test_pull_says_so_when_the_board_has_nothing(tmp_path, calls, capsys):
    calls.listing["out"] = ""
    assert devkit.run_pull([], HOST, tmp_path) == 1
    assert "nothing to pull" in capsys.readouterr().err
    assert len(calls) == 1, "no scp should follow an empty listing"


def test_pull_surfaces_an_unreachable_board(tmp_path, monkeypatch):
    class Unreachable:
        returncode = 255
        stdout = ""
        stderr = "ssh: connect to host 192.168.137.50 port 22: No route to host"

    monkeypatch.setattr(devkit.subprocess, "run", lambda *a, **k: Unreachable())
    with pytest.raises(SystemExit, match="No route to host"):
        devkit.run_pull([], HOST, tmp_path)


def test_a_missing_name_is_not_an_error_from_ls(tmp_path, monkeypatch):
    """`ls -d a b` exits 1 or 2 when some are missing. That is the normal case."""
    class Partial:
        returncode = 2
        stdout = "frames\n"
        stderr = "ls: cannot access 'falls.mp4': No such file or directory"

    monkeypatch.setattr(devkit.subprocess, "run", lambda *a, **k: Partial())
    assert devkit.remote_paths(HOST, ["frames", "falls.mp4"]) == ["frames"]


# ── through the CLI ──


def test_the_cli_reaches_both(tmp_path, monkeypatch, calls):
    monkeypatch.setenv(devkit.DEVKIT_ENV, HOST)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("model: {}", encoding="utf-8")

    assert main(["push", "config.yaml"]) == 0
    calls.listing["out"] = "detections.mp4\n"
    assert main(["pull", "--into", "out"]) == 0
    assert (tmp_path / "out").is_dir()


def test_the_cli_reports_a_missing_host_without_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.h264").write_bytes(b"x")
    assert main(["push", "a.h264"]) == 1
    assert devkit.DEVKIT_ENV in capsys.readouterr().err


def test_subprocess_is_never_run_with_a_shell():
    """A filename with a space or a quote must not become shell syntax."""
    source = Path(devkit.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert subprocess is not None  # the module under test uses the real one
