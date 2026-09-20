"""Moving files between your PC and the DevKit.

Two commands wrap `scp` so the awkward parts stop being yours:

  sima-vision push clip.h264      your PC  ->  the board
  sima-vision pull                the board  ->  your PC

The awkward parts, in order of how much time each one costs:

1. On Windows, `scp D:\\clips\\a.h264 sima@ip:~` fails with "could not resolve
   hostname d:", because scp reads everything before the first colon as a host.
   So a push never passes a full local path: it groups the files by folder and
   runs scp from inside each one, with bare filenames.
2. A pull of "whatever the run produced" cannot be written as one scp, because
   scp fails the whole transfer on a name that is not there and the outputs
   depend on which task ran. So the names are listed over ssh first, and only
   what exists is fetched.

The host is `user@address`, from --host or $SIMA_VISION_DEVKIT. Authentication
is ssh's own business: an agent, a key, or it asks. Nothing here handles
passwords, and nothing here stores one.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from . import __version__
from .console import console, human_bytes

#: Saves retyping `--host` on every command.
DEVKIT_ENV = "SIMA_VISION_DEVKIT"

#: What a run leaves in its working directory, across all three tasks. `pull`
#: with no arguments asks for these and takes whatever is actually there, so it
#: does not need to be told which task ran.
OUTPUTS = (
    "detections.mp4", "segmentation.mp4", "falls.mp4",
    "detections.avi", "segmentation.avi", "falls.avi",
    "frames", "config.yaml",
)


def resolve_host(host: str | None) -> str:
    """The board to talk to, from the flag or the environment."""
    found = host or os.environ.get(DEVKIT_ENV, "")
    if not found:
        raise SystemExit(
            "no DevKit address. Pass --host, or set it once for the session:\n\n"
            "  export SIMA_VISION_DEVKIT=sima@192.168.137.50      # macOS, Linux\n"
            '  $env:SIMA_VISION_DEVKIT = "sima@192.168.137.50"    # PowerShell\n\n'
            "The board takes a DHCP address that changes between reboots. The\n"
            "serial console prints it, and so does `arp -a` on the PC it is\n"
            "cabled to."
        )
    return found.strip()


def require(tool: str) -> str:
    """Find `ssh` or `scp`, or explain how to get them on this platform."""
    found = shutil.which(tool)
    if found:
        return found
    raise SystemExit(
        f"`{tool}` is not on PATH, and this command is a wrapper around it.\n"
        "  Windows 10/11: Settings > Apps > Optional features > OpenSSH Client\n"
        "  macOS: it ships with the system\n"
        "  Debian/Ubuntu: sudo apt install openssh-client"
    )


def report(command: list[str], cwd: Path | None = None) -> None:
    """Echo the ssh or scp being run. These wrap other programs, and hiding
    which one would make their failures unattributable."""
    where = f"  (from {cwd})" if cwd is not None else ""
    style = console.style
    console.write(f"  {style.paint('$', style.dim)} {' '.join(command)}{where}")


def remote_paths(host: str, names) -> list[str]:
    """Which of `names` exist in the board's home directory.

    One `ls` rather than one probe per name, and a missing name is not an error
    here: `pull` is asking a vague question on purpose.
    """
    listing = " ".join(shlex.quote(str(name)) for name in names)
    command = [require("ssh"), host, f"ls -d -- {listing} 2>/dev/null"]
    result = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if result.returncode not in (0, 1, 2):
        # 1 and 2 are `ls` saying "some of those are not there", which is fine.
        # Anything else is ssh itself failing, and its message is the useful one.
        raise SystemExit(f"could not reach {host}:\n{result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_push(paths: list[Path], host: str | None, dest: str) -> int:
    """Copy local files or folders to the board.

    Grouped by parent folder and run from inside it, which is what keeps a
    Windows drive letter from being read as a hostname. See the module docstring.
    """
    host = resolve_host(host)
    console.banner(f"sima-vision {__version__}", f"push -> {host}")
    scp = require("scp")

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            "not found: " + ", ".join(str(p) for p in missing)
        )

    groups: dict[Path, list[str]] = defaultdict(list)
    for path in paths:
        resolved = path.resolve()
        groups[resolved.parent].append(resolved.name)

    target = f"{host}:{dest}"
    for parent, names in groups.items():
        command = [scp, "-r", *names, target]
        report(command, parent)
        result = subprocess.run(command, cwd=parent, check=False)  # noqa: S603
        if result.returncode != 0:
            return result.returncode

    console.write()
    console.success(f"copied {sum(len(n) for n in groups.values())} item(s) to {target}")
    return 0


def run_pull(names: list[str], host: str | None, into: Path) -> int:
    """Copy results back from the board.

    With no names, asks for everything a run of any task could have left and
    takes what is there.
    """
    host = resolve_host(host)
    console.banner(f"sima-vision {__version__}", f"pull <- {host}")
    scp = require("scp")
    wanted = names or list(OUTPUTS)

    found = remote_paths(host, wanted)
    if not found:
        listed = ", ".join(wanted[:4]) + ("..." if len(wanted) > 4 else "")
        console.error(
            f"nothing to pull: {host} has none of {listed}\n"
            "Run a task there first, or name the file you want."
        )
        return 1

    into.mkdir(parents=True, exist_ok=True)
    sources = [f"{host}:{shlex.quote(name)}" for name in found]
    # cwd=into with "." as the destination, for the same reason push runs from
    # the file's own folder: a Windows destination path would carry a colon.
    command = [scp, "-r", *sources, "."]
    report(command, into)
    result = subprocess.run(command, cwd=into, check=False)  # noqa: S603
    if result.returncode != 0:
        return result.returncode

    console.write()
    console.success(f"pulled into {into.resolve()}:")
    for name in found:
        local = into / name
        size = human_bytes(local.stat().st_size) if local.is_file() else ""
        console.info(f"  {name}" + (f"  ({size})" if size else ""))
    return 0
