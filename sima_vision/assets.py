"""Where the sample clips and model archives live, and how they get there.

There is one ``assets/`` directory for the whole project rather than one inside
each task folder. A clip is a clip: the same 13 MB of people walking through a
mall feeds ``detect``, ``segment`` and ``fall``, and the detect archive is
shared by ``detect`` and ``fall`` outright. Three copies of it said nothing that
one copy does not.

``--source`` and ``--model`` therefore take one of three things:

1. a local path -- used as given
2. an ``http(s)`` URL -- downloaded into ``assets/`` once, then reused
3. nothing at all -- the task's default, downloaded on first run

Case 3 is what makes ``sima-vision detect`` work on its own, and it is the case
that matters: nobody should have to look up a model URL before their first run.
The clips are on a public GitHub release so they are simply fetched. The model
packs are behind a `community.sima.ai <https://community.sima.ai>`_ login -- the
download URL answers a plain GET with a 302 to ``auth.sima.ai`` -- so those go
through ``sima-cli``, which already holds that login, and fall back to printing
the command when it is not installed.

Nothing here runs at config time. ``--validate`` resolves the same paths and
never touches the network; only :meth:`Task.run
<sima_vision.tasks.base.Task.run>` calls :func:`ensure_assets`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from .console import console, human_bytes

#: Overrides where downloads land. Default ``./assets`` in the working directory.
ASSETS_ENV = "SIMA_VISION_ASSETS"

#: Clips and model packs alike, on one public GitHub release. Anything here is
#: a plain GET: no login, no `sima-cli`, and the same code path for both kinds.
SAMPLE_RELEASE = "https://github.com/RizwanMunawar/sima-vision/releases/download/0.0.1"

#: Clips on the release, by name. Two of these were missing from this table for
#: a while and naming either of them got you "source file not found" for a file
#: sitting on the very release this app downloads from -- which is what
#: `published_assets` now catches regardless of what is written here.
SAMPLE_VIDEOS = {
    "people-walking-outside-mall.h264": "1920x1080 @ 24 fps, 13.3 MB. The usual default",
    "people-walking-inside-mall.h264": "1920x1080 @ 30 fps, 1.2 MB. Quicker smoke test",
    "people-walking-in-street.mp4": "5.4 MB. Reframed to Annex-B on first use",
    "people-walking-small.mp4": "3.4 MB. The quickest thing here to try",
}

#: Model packs on that same release, by size. Nano is the default: 20 MB
#: downloads and starts far quicker than anything else here, which is what a
#: first run wants. Pass ``--model yolo26s-det-bf16-mla_tess-b1.tar.gz`` for
#: the small pack and it is fetched from here by name.
#:
#: The detection packs carry a ``-b1`` batch suffix and the segmentation packs
#: do not. That asymmetry is in the published filenames, not a mistake here.
RELEASE_MODELS = {
    "yolo26n-det-bf16-mla_tess-b1.tar.gz": "nano detection, 21 MB",
    "yolo26n-seg-bf16-mla_tess.tar.gz": "nano segmentation, 24 MB",
    "yolo26s-det-bf16-mla_tess-b1.tar.gz": "small detection, 37 MB",
    "yolo26s-seg-bf16-mla_tess.tar.gz": "small segmentation, 41 MB",
}

#: SHA-256 of every release asset, straight from the GitHub API.
#:
#: Worth having because the alternative is silent. A truncated or swapped
#: download is a file of plausible size that fails much later and somewhere
#: else -- a clip that decodes half way, a model pack that unpacks to
#: nonsense -- and a whole afternoon went into suspecting exactly that of a
#: file which turned out to be fine. Checking here answers it in a second.
RELEASE_SHA256 = {
    "people-walking-in-street.mp4":
        "f13617463afd41307e684d16d9c679b23dca13566decaf3d3ffcce5173ebf3ce",
    "people-walking-inside-mall.h264":
        "b65591bb027f7ca184feab29a8a4fc3c6620d632a549793604cdd7b414993b9b",
    "people-walking-outside-mall.h264":
        "72da5de46024028766b9f6df30de09560593f14ec3f4f9703394a28fda8d0140",
    "people-walking-small.mp4":
        "c217cb8a661fb735fcdef336019d2335576bff4ecaed8e89bbcc07eb9846cd00",
    "yolo26n-det-bf16-mla_tess-b1.tar.gz":
        "7c67ecbd823e128edb0fdc1d1bca47abea4d91c9843ff5c93101361081095bea",
    "yolo26n-seg-bf16-mla_tess.tar.gz":
        "b32ab8ee2c217f88165a2e17c2b8f6124a4a60776a2d5d9b5bd50ed6bbdf4e6a",
    "yolo26s-det-bf16-mla_tess-b1.tar.gz":
        "7bb911cbf356c352df4ecef8600b32e8f77baf945eb7788e6fc585665790bb17",
    "yolo26s-seg-bf16-mla_tess.tar.gz":
        "4aaa6068b1dd12d4774a8af917c169dcf2488a3214c84716d7956b85bd1c6cfc",
}

#: Where the SDK publishes packs that are not on the release. Reaching one of
#: these needs a community.sima.ai login, which is what ``sima-cli`` holds and
#: why that path still exists -- it is the fallback now, not the default.
MODEL_BASE = "https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix"


@dataclass(frozen=True)
class TaskAssets:
    """What one task runs on when it is given nothing.

    Attributes:
        model_dir: Model pack directory under :data:`MODEL_BASE`.
        model_file: Archive name, which is also its name inside ``assets/models``.
        clip: Sample clip name, a key of :data:`SAMPLE_VIDEOS`.
    """

    model_dir: str
    model_file: str
    clip: str


#: Task name -> its default model and clip. ``detect`` and ``fall`` share a head.
#:
#: All three defaults are packs on the GitHub release, so a first run needs no
#: login and no ``sima-cli``. Each task fetches only its own: ``detect`` and
#: ``fall`` share the detection pack, ``segment`` pulls the segmentation one.
CATALOGUE: dict[str, TaskAssets] = {
    "detect": TaskAssets(
        "yolo26-detection",
        "yolo26n-det-bf16-mla_tess-b1.tar.gz",
        "people-walking-outside-mall.h264",
    ),
    "segment": TaskAssets(
        "yolo26-segmentation",
        "yolo26n-seg-bf16-mla_tess.tar.gz",
        "people-walking-outside-mall.h264",
    ),
    "fall": TaskAssets(
        "yolo26-detection",
        "yolo26n-det-bf16-mla_tess-b1.tar.gz",
        "people-walking-inside-mall.h264",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Locations
# ─────────────────────────────────────────────────────────────────────────────


def assets_root() -> Path:
    """The ``assets/`` directory, honouring ``$SIMA_VISION_ASSETS``.

    Read on every call rather than cached at import, so setting the variable
    from a test -- or between two calls in one process -- takes effect.
    """
    return Path(os.environ.get(ASSETS_ENV) or "assets")


def videos_dir() -> Path:
    return assets_root() / "videos"


def models_dir() -> Path:
    return assets_root() / "models"


def default_model_path(task: str) -> str:
    """Where this task's model archive is expected, as a string for the config."""
    return (models_dir() / CATALOGUE[task].model_file).as_posix()


def default_source_uri(task: str) -> str:
    """Where this task's sample clip is expected, as a string for the config."""
    return (videos_dir() / CATALOGUE[task].clip).as_posix()


def release_url(name: str) -> str:
    """Where one published clip or model pack lives."""
    return f"{SAMPLE_RELEASE}/{name}"


#: The same release, as the API rather than as a download path. Asked only when
#: a name is not in the tables above, so the usual run makes no API call.
RELEASE_API = (
    "https://api.github.com/repos/RizwanMunawar/sima-vision/releases/tags/"
    + SAMPLE_RELEASE.rsplit("/", 1)[1]
)

#: Filled in by :func:`published_assets` on the first miss, so a run asks once.
_published: frozenset[str] | None = None


def published_assets() -> frozenset[str]:
    """Every file name on the release, or an empty set if it cannot be asked.

    The tables in this module are the fast path and the documentation: they let
    ``--validate`` stay offline and they say what each pack is for. What they
    cannot do is stay right. Four clips were on the release and two were listed,
    so naming either of the other two got you "source file not found" for a file
    sitting on the very release this app downloads from.

    So a name that is not in a table is not refused until the release itself has
    been asked. Failure here is not an error -- no network, rate limited, a
    private repository -- it just means the answer is the tables alone.
    """
    global _published
    if _published is not None:
        return _published
    names: set[str] = set()
    try:
        with urllib.request.urlopen(RELEASE_API, timeout=15) as response:  # noqa: S310
            payload = json.load(response)
        names = {asset["name"] for asset in payload.get("assets", [])}
    except Exception:  # noqa: BLE001 - any failure means "the tables alone"
        pass
    _published = frozenset(names)
    return _published


def on_release(name: str) -> bool:
    """Whether this archive is a documented pack, by the table alone.

    Deliberately offline. :func:`fetchable` is the one that may ask the release.
    """
    return name in RELEASE_MODELS


def worth_asking(path: Path) -> bool:
    """Whether a missing path could be a release asset at all.

    A bare filename is someone naming a published file: `--model
    yolo26s-det-bf16-mla_tess-b1.tar.gz`. So is anything already pointing into
    our own assets directory, which is where the defaults live.

    ``nowhere/mine.h264`` is neither. It is someone's own file, missing, and the
    honest answer is the error `check_source_file` gives -- not a round trip to
    GitHub first to confirm what the path already says.
    """
    return path.parent == Path() or path.parent in {videos_dir(), models_dir()}


def fetchable(path: Path, table: dict) -> bool:
    """Whether this missing file can be downloaded from the release.

    The table answers instantly and covers everything documented. Only a name it
    does not know, on a path that could plausibly be published, is worth asking
    the release itself -- which is how a file that is genuinely there but not
    yet tabulated gets fetched instead of refused.
    """
    if path.name in table:
        return True
    return worth_asking(path) and path.name in published_assets()


def model_url(task: str) -> str:
    entry = CATALOGUE[task]
    if on_release(entry.model_file):
        return release_url(entry.model_file)
    return f"{MODEL_BASE}/{entry.model_dir}/{entry.model_file}"


def model_command(task: str) -> str:
    """The one line that downloads the right model pack for a task.

    A run does this itself; this is the printable copy, for the case where
    ``sima-cli`` is not on PATH and a run cannot. ``-o`` rather than a ``cd``
    into ``assets/models``: naming the destination outright is the difference
    between a pack the config can see and one that landed wherever the shell
    happened to be.
    """
    models = models_dir().as_posix()
    return (
        f"mkdir -p {models} && "
        f"sima-cli download {model_url(task)} -o {default_model_path(task)}"
    )


def is_url(value: str) -> bool:
    """True for something to download. ``rtsp://`` is a stream, not a file."""
    return str(value).startswith(("http://", "https://"))


# ─────────────────────────────────────────────────────────────────────────────
# Downloading
# ─────────────────────────────────────────────────────────────────────────────


def say(step, text: str) -> None:
    """One line of progress, under the step that asked for it if there is one."""
    if step is not None:
        step.detail(text)
    else:
        console.info(text)


def download(url: str, out: Path, step=None) -> bool:
    """Fetch one file, reporting progress. Returns False on any HTTP failure."""
    if out.exists():
        say(step, f"have  {out}  ({human_bytes(out.stat().st_size)})")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    say(step, f"get   {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with part.open("wb") as handle:
                while chunk := response.read(1 << 16):
                    handle.write(chunk)
                    done += len(chunk)
                    console.progress(out.name, done, total)
        console.progress_done()
        # A server that closes early, or a proxy that truncates, ends the read
        # loop exactly like a finished transfer does. Without this the partial
        # file is renamed into place and every later run reuses it, because the
        # first thing this function does is trust a file that already exists.
        if total and done != total:
            part.unlink(missing_ok=True)
            console.error(
                f"{out.name}: got {done} of {total} bytes, the transfer was cut short"
            )
            return False
        wanted = RELEASE_SHA256.get(Path(url.split("?")[0]).name)
        if wanted and not verify_sha256(part, wanted):
            part.unlink(missing_ok=True)
            console.error(
                f"{out.name}: downloaded, but its contents are not what they "
                "should be.\n  The file is discarded rather than used. Try again; "
                "if it keeps happening,\n  something between here and GitHub is "
                "rewriting the download."
            )
            return False
        part.replace(out)
        say(step, f"got   {out}  ({human_bytes(done)})")
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        console.progress_done()
        part.unlink(missing_ok=True)
        console.error(f"{out.name}: {exc}")
        return False


def verify_sha256(path: Path, wanted: str) -> bool:
    """Whether the file hashes to ``wanted``. Read in blocks, not all at once.

    A model pack is tens of megabytes and this runs on a board where a careless
    allocation has already cost a run, so it is streamed.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest() == wanted


def cache_name(url: str) -> str:
    """A local filename for a URL that cannot collide with another URL's.

    ``--source https://a.example/clip.h264`` and the same name on another host
    are different videos. Keying the cache on the last path segment alone meant
    the second one silently ran the first one's footage, and the download was
    skipped because the file was already there.

    The digest goes before the extension rather than after, so the suffix still
    says what the file is and `.tar.gz` survives intact::

        clip.h264            ->  clip-1a2b3c4d.h264
        yolo26m-det.tar.gz   ->  yolo26m-det-1a2b3c4d.tar.gz
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    name = Path(url.split("?")[0]).name or "download"
    head, dot, tail = name.partition(".")
    return f"{head}-{digest}{dot}{tail}"


def fetch(url: str, out: Path, what: str, step=None) -> Path:
    """Download to ``out``, or raise. The insisting version of :func:`download`."""
    if not download(url, out, step):
        raise RuntimeError(
            f"could not download the {what} from {url}\n"
            f"  wanted: {out}\n"
            "Fetch it by hand and pass the local path instead."
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────


def custom_model_help(task: str) -> str:
    """How to run something other than the packs published here.

    Printed where a missing archive is discovered rather than buried in the
    README, because that is the moment somebody needs it.
    """
    published = "\n".join(
        f"    {name:<38} {what}" for name, what in RELEASE_MODELS.items()
    )
    return (
        "Published packs, fetched by name with no login:\n"
        f"{published}\n"
        f"    sima-vision {task} --model yolo26s-det-bf16-mla_tess-b1.tar.gz\n"
        "\n"
        "Your own model, in three steps:\n"
        "  1. Compile it for Modalix with the SiMa SDK. The result is a\n"
        "     .tar.gz pack holding an .elf and its preprocess contract.\n"
        "  2. Get it onto the board:  sima-vision push my-model.tar.gz\n"
        "  3. Run it:                 sima-vision "
        f"{task} --model my-model.tar.gz\n"
        "\n"
        "  A pack from a URL works too and is cached under assets/models:\n"
        f"    sima-vision {task} --model https://example.com/my-model.tar.gz\n"
        "  If its head is not yolo26, say so with --family, or the box decoder\n"
        "  reads the output tensor the wrong way and every detection is noise."
    )


def ensure_source(uri: str, source_type: str = "video", step=None) -> str:
    """Make ``source.uri`` name a file that exists, downloading if it has to.

    Args:
        uri: The resolved ``source.uri``: a path, an ``http(s)`` URL, or one of
            the sample clip paths the defaults fill in.
        source_type: Only ``video`` reads a file. An RTSP URL or a camera is
            handed back untouched.
        step: The console step to report under, if there is one.

    Returns:
        A local path, or ``uri`` unchanged when there is nothing to fetch. A
        path that is simply missing is *also* handed back unchanged, so the
        error comes from :func:`sima_vision.media.check_source_file`, which
        knows how to describe it.
    """
    if source_type != "video" or not uri:
        return uri
    if is_url(uri):
        return str(fetch(uri, videos_dir() / cache_name(uri), "source video", step))
    path = Path(uri)
    if path.exists():
        say(step, f"have  {uri}  ({human_bytes(path.stat().st_size)})")
        return uri
    # A default, a clip named by hand, or anything else the release publishes.
    # Falling through to "not found" for a file the app could have fetched in
    # two seconds is the kind of unhelpful that is worth one API call to avoid.
    if not fetchable(path, SAMPLE_VIDEOS):
        return uri
    if path.parent != Path():
        fetch(release_url(path.name), path, "sample clip", step)
        return uri
    # A bare name lands in assets/videos rather than wherever the shell happens
    # to be, which is what makes `--source people-walking-small.mp4` do the
    # obvious thing. as_posix keeps the string the shape the config uses.
    out = videos_dir() / path.name
    fetch(release_url(path.name), out, "sample clip", step)
    return out.as_posix()


def ensure_model(path: str, task: str, step=None) -> str:
    """Make ``model.path`` name an archive that exists, downloading if it has to.

    A URL is fetched directly. Anything already on disk is used as it stands.
    The remaining case is the default -- the task's own archive, not yet
    downloaded -- and that one goes through ``sima-cli``, because a plain GET on
    the pack URL answers with a login redirect rather than a tarball.

    Raises:
        RuntimeError: When the archive is missing and cannot be fetched, with
            the command to run by hand.
    """
    if is_url(path):
        return str(fetch(path, models_dir() / cache_name(path), "model archive", step))
    if not path:
        return path
    target = Path(path)
    if target.exists():
        say(step, f"have  {path}  ({human_bytes(target.stat().st_size)})")
        return path

    # A published pack, whether it is this task's default or one named by hand.
    # These are on the same release as the clips, so they are a plain GET and
    # need no login at all. Naming one by its bare filename lands it in
    # assets/models rather than wherever the shell happened to be, which is what
    # makes `--model yolo26s-det-bf16-mla_tess-b1.tar.gz` do the obvious thing.
    if fetchable(target, RELEASE_MODELS):
        out = target if target.parent != Path() else models_dir() / target.name
        fetch(release_url(target.name), out, "model pack", step)
        return str(out)

    entry = CATALOGUE.get(task)
    if entry is None or target.name != entry.model_file:
        # Not something this task knows how to fetch: a name we have no URL for.
        raise RuntimeError(
            f"model archive not found: {path}\n"
            f"  launched from: {Path.cwd()}\n"
            f"{custom_model_help(task)}"
        )

    url = model_url(task)
    if shutil.which("sima-cli") is None:
        raise RuntimeError(
            f"model archive not found: {path}\n"
            "The model packs need a community.sima.ai login, and `sima-cli` is "
            "not on PATH here,\nso it cannot be fetched for you. Run:\n\n"
            f"  sima-cli login\n  {model_command(task)}\n\n"
            "Or pass --model with an https URL you can reach."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    say(step, f"get   {url}")
    say(step, "      via sima-cli, which holds your community.sima.ai login")
    result = subprocess.run(  # noqa: S603
        # -o names the destination, so this does not depend on where sima-cli
        # would otherwise have put it or on what the working directory is.
        ["sima-cli", "download", url, "-o", str(target)],
        check=False,
        # Left alone, sima-cli opens with "a newer version is available, update
        # now? [Y/n]" and waits. Nothing is watching that prompt in the middle
        # of a run, and answering no aborts the download with it. Its own
        # message names this variable as the way off.
        env={**os.environ, "SIMA_CLI_CHECK_FOR_UPDATE": "0"},
    )
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"`sima-cli download` did not produce {target}\n"
            "It needs a community.sima.ai login, and the board needs a route to the "
            "internet\nthrough the PC it is cabled to. Run `sima-cli login` and try "
            "again, or download\nthe pack on your PC and `sima-vision push` it over."
        )
    say(step, f"got   {path}  ({human_bytes(target.stat().st_size)})")
    return path


def ensure_assets(cfg, task: str, step=None):
    """Resolve ``model.path`` and ``source.uri`` to files that exist.

    Called once, from :meth:`Task.run <sima_vision.tasks.base.Task.run>`, so
    that everything which does not run inference -- ``--validate`` and the
    Python ``validate()`` -- stays offline.

    Returns:
        The config, or a copy of it with the two paths replaced.
    """
    source = ensure_source(cfg.source_uri, cfg.source_type, step)
    model = ensure_model(cfg.model_path, task, step)
    if (source, model) == (cfg.source_uri, cfg.model_path):
        return cfg
    return replace(cfg, source_uri=source, model_path=model)
