"""The parts that behave differently on Windows, macOS and Linux.

Two things bite here, and neither shows up on the developer's own machine:

1. **Console encoding.** Python encodes stdout with the locale codepage the
   moment it is redirected, so a single non-ASCII character in something the app
   prints turns `sima-vision detect > log.txt` on a Windows console into a
   UnicodeEncodeError. That is what these subprocess runs are for: they force a
   legacy codepage and check every off-board command survives it.
2. **Path separators.** A config or a flag can carry `C:\\clips\\a.h264`, and
   `\\` is an escape character in some of the places a path passes through.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sima_vision import assets
from sima_vision.cli import main

#: Everything that runs without a board, and therefore on any machine.
OFFBOARD = [
    ["--help"],
    ["--version"],
    ["detect", "--no-config", "--validate"],
    ["segment", "--no-config", "--validate"],
    ["fall", "--no-config", "--validate"],
    ["fall", "--no-config", "--model", "m.tar.gz", "--source", "c.h264", "--validate"],
    ["detect", "--no-config", "--validate", "--quiet"],
    ["push", "--help"],
    ["pull", "--help"],
]


#: The checkout, which has to win over anything installed. See `run_cli`.
REPO = Path(__file__).resolve().parents[1]


def run_cli(argv: list[str], encoding: str, cwd: Path) -> subprocess.CompletedProcess:
    """The CLI in a child process, with stdout forced to a given codepage.

    A child, not `main()`, because the failure being guarded against is in the
    encoder attached to a real pipe. pytest's captured stdout does not have one.

    PYTHONPATH is the checkout, and that is not decoration. The child runs from
    tmp_path, so without it `python -m sima_vision.cli` imports whatever copy of
    the package happens to be installed -- which on a developer's machine is
    usually an older one. These tests passed locally against a build that still
    had `watch` for exactly that reason, and only CI, where the install is
    `pip install -e .`, noticed.
    """
    env = {
        **os.environ,
        "PYTHONIOENCODING": encoding,
        "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    return subprocess.run(
        [sys.executable, "-m", "sima_vision.cli", *argv],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=cwd, check=False, timeout=120,
    )


def test_the_child_runs_this_checkout_and_not_an_installed_copy(tmp_path):
    """Otherwise these tests grade the wrong package.

    The child runs from tmp_path, so import resolution falls to whatever is on
    sys.path -- and a developer with `pip install sima-vision` in site-packages
    gets that copy, silently, for every case below.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import sima_vision, sys; sys.stdout.write(sima_vision.__file__)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        cwd=tmp_path, check=False, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout).resolve().is_relative_to(REPO), result.stdout


@pytest.mark.parametrize("argv", OFFBOARD, ids=lambda a: " ".join(a))
def test_output_survives_a_legacy_codepage(argv, tmp_path):
    """cp437 is the old Windows console default and has no smart punctuation.

    Anything that still encodes cleanly here encodes anywhere.
    """
    result = run_cli(argv, "cp437", tmp_path)
    assert "UnicodeEncodeError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("argv", OFFBOARD, ids=lambda a: " ".join(a))
def test_output_survives_ascii(argv, tmp_path):
    """The strictest case there is, and the one a bare Docker image often gets."""
    result = run_cli(argv, "ascii", tmp_path)
    assert "UnicodeEncodeError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr


def test_no_colour_escapes_reach_a_pipe():
    """A redirected run is read by people and by grep, and neither wants ANSI.

    The check runs in a child because the decision is made from the stream
    itself: pytest's captured stdout is not a terminal either, but a real pipe
    is the case that matters and only a subprocess has one.
    """
    result = run_cli(["detect", "--no-config", "--validate"], "utf-8", Path.cwd())
    assert result.returncode == 0, result.stderr
    assert chr(27) not in result.stdout, "an escape sequence reached a pipe"


def test_a_windows_style_path_is_passed_through_verbatim():
    r"""`--source C:\clips\a.h264` must arrive spelled exactly as it was written.

    Only Windows can resolve that string to a file, so turning it into a path
    and comparing is a Windows-only assertion. What holds everywhere is that
    nothing on the way mangles it: no escape processing, no separator
    rewriting, and `is_url` does not read the drive letter's colon as a scheme.
    """
    from sima_vision.tasks import TASKS

    source = r"C:\clips\a.h264"
    model = r"D:\packs\yolo26m-det.tar.gz"
    assert not assets.is_url(source), "a drive letter is not a URL scheme"

    cfg = TASKS["detect"]().load(
        None, {"source.uri": source, "model.path": model}, use_file=False
    )
    assert cfg.source_uri == source
    assert cfg.model_path == model


def test_a_native_path_to_a_real_file_resolves(tmp_path, monkeypatch):
    """The same journey with this platform's own separator, and a file at the end."""
    monkeypatch.chdir(tmp_path)
    clip = tmp_path / "clips" / "a.h264"
    clip.parent.mkdir()
    clip.write_bytes(b"\x00\x00\x00\x01")

    from sima_vision.tasks import TASKS

    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )
    assert Path(cfg.source_uri) == clip
    assert assets.ensure_source(cfg.source_uri) == str(clip), "an existing file is not fetched"


def test_the_assets_directory_takes_an_absolute_path(tmp_path, monkeypatch):
    """Including a Windows one, where the drive letter is part of it."""
    monkeypatch.setenv(assets.ASSETS_ENV, str(tmp_path / "shared"))
    assert assets.models_dir() == tmp_path / "shared" / "models"
    # as_posix() is what reaches the config, and it must still point at the
    # same place once Path() has read it back.
    assert Path(assets.default_model_path("detect")).parent == tmp_path / "shared" / "models"


def test_validate_needs_no_network_and_no_board(tmp_path, monkeypatch):
    """The whole off-board story depends on this staying true."""
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("--validate must not open a connection")

    monkeypatch.setattr(assets.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(assets.subprocess, "run", forbidden)

    for name in ("detect", "segment", "fall"):
        assert main([name, "--no-config", "--validate"]) == 0
    assert not (tmp_path / "assets").exists(), "nothing should have been downloaded"
