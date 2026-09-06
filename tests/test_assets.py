"""Resolving `--source` and `--model` down to files that exist.

Nothing here reaches the network: `urlopen` and `sima-cli` are both replaced.
What is being tested is the decision -- use this, fetch that, refuse the other
-- not the download, which is the same `download()` `fetch` already used.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from sima_vision import assets
from sima_vision.tasks import TASKS

CLIP = "people-walking-outside-mall.h264"


@pytest.fixture
def here(tmp_path, monkeypatch):
    """Run in an empty directory, so `assets/` is this test's own."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def offline(monkeypatch):
    """Serve four bytes for any URL, and return the list of URLs asked for.

    Also clears the cached release listing. It is a module global, so one test
    that fills it would answer for every test after it.
    """
    asked: list[str] = []

    class Response:
        headers = {"Content-Length": "4"}

        def __init__(self):
            self.left = [b"data"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    def urlopen(url, timeout=0):
        asked.append(url)
        return Response()

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    # Four bytes cannot hash to a published digest, and these tests are about
    # the decision rather than the transfer. Verification has its own tests.
    monkeypatch.setattr(assets, "RELEASE_SHA256", {})
    monkeypatch.setattr(assets, "_published", None)
    return asked


# ── where things live ──


def test_the_default_assets_directory_is_beside_you(here):
    assert assets.assets_root() == Path("assets")
    assert assets.default_model_path("detect").startswith("assets/models/")
    assert assets.default_source_uri("fall").endswith("people-walking-inside-mall.h264")


def test_the_assets_directory_can_be_moved(here, monkeypatch):
    monkeypatch.setenv(assets.ASSETS_ENV, str(here / "shared"))
    assert assets.models_dir() == here / "shared" / "models"
    assert assets.default_model_path("segment").startswith((here / "shared").as_posix())


def test_every_task_has_a_model_and_a_clip():
    for name in TASKS:
        entry = assets.CATALOGUE[name]
        assert entry.clip in assets.SAMPLE_VIDEOS
        assert entry.model_file.endswith(".tar.gz")
        # Every default is on the public release, so a first run needs no
        # login. That is the whole point of the defaults being these packs.
        assert assets.on_release(entry.model_file), entry.model_file
        assert assets.model_url(name) == assets.release_url(entry.model_file)


def test_only_http_urls_count_as_downloadable():
    assert assets.is_url("https://example.com/clip.h264")
    assert assets.is_url("http://example.com/clip.h264")
    # An RTSP stream is opened, never fetched.
    assert not assets.is_url("rtsp://cam/live")
    assert not assets.is_url("assets/videos/clip.h264")


# ── sources ──


def test_a_source_url_is_downloaded_once(here, offline):
    first = assets.ensure_source("https://example.com/clip.h264")
    assert Path(first).parent == Path("assets/videos")
    assert Path(first).is_file()
    assert Path(first).suffix == ".h264", "the extension still says what it is"

    second = assets.ensure_source("https://example.com/clip.h264")
    assert second == first, "the same URL is the same file"
    assert len(offline) == 1, "the second call must reuse the file on disk"


def test_a_missing_sample_clip_is_fetched_from_the_release(here, offline):
    uri = assets.ensure_source(f"assets/videos/{CLIP}")
    assert uri == f"assets/videos/{CLIP}"
    assert Path(uri).is_file()
    assert offline == [f"{assets.SAMPLE_RELEASE}/{CLIP}"]


def test_an_existing_file_is_left_alone(here, offline):
    clip = here / "mine.h264"
    clip.write_bytes(b"\x00\x00\x00\x01")
    assert assets.ensure_source("mine.h264") == "mine.h264"
    assert not offline


def test_an_unknown_missing_file_is_left_for_the_real_error(here, offline):
    """check_source_file describes a missing path far better than we could."""
    assert assets.ensure_source("nowhere/mine.h264") == "nowhere/mine.h264"
    assert not offline


def test_a_stream_source_is_never_touched(here, offline):
    assert assets.ensure_source("rtsp://cam/live", "rtsp") == "rtsp://cam/live"
    assert assets.ensure_source("", "usb") == ""
    assert not offline


def test_a_failed_source_download_says_so(here, monkeypatch):
    def boom(url, timeout=0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(assets.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="could not download"):
        assets.ensure_source("https://example.com/clip.h264")
    assert not list(Path("assets").rglob("*.part"))


# ── models ──


def test_a_model_url_is_downloaded(here, offline):
    path = assets.ensure_model("https://example.com/det.tar.gz", "detect")
    assert Path(path).parent == Path("assets/models")
    assert Path(path).is_file()
    assert Path(path).name.endswith(".tar.gz"), "a double extension survives intact"


def test_an_existing_model_is_used_as_it_stands(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    archive = here / "det.tar.gz"
    archive.write_bytes(b"tar")
    assert assets.ensure_model("det.tar.gz", "detect") == "det.tar.gz"


def test_a_published_pack_needs_no_login_at_all(here, offline, monkeypatch):
    """The change this release move is for.

    Every default pack used to go through `sima-cli`, which needs a
    community.sima.ai login, so a first run stopped dead without one. They are
    on the same public release as the clips now, so they are a plain GET.
    """
    monkeypatch.setattr(assets.shutil, "which", lambda _name: None)   # no sima-cli
    monkeypatch.setattr(
        assets.subprocess, "run",
        lambda *a, **k: pytest.fail("a published pack must not shell out"),
    )
    path = assets.ensure_model(assets.default_model_path("detect"), "detect")

    assert Path(path).is_file()
    assert offline == [assets.release_url(assets.CATALOGUE["detect"].model_file)]


def test_a_pack_named_by_hand_is_fetched_into_the_assets_directory(here, offline):
    """`--model yolo26s-det-...tar.gz` should not need a URL or a path."""
    path = assets.ensure_model("yolo26s-det-bf16-mla_tess-b1.tar.gz", "detect")

    assert Path(path).parent == Path("assets/models"), "not the working directory"
    assert Path(path).is_file()
    assert offline == [assets.release_url("yolo26s-det-bf16-mla_tess-b1.tar.gz")]


def test_an_unpublished_pack_still_goes_through_sima_cli(here, monkeypatch):
    """The m packs are not on the release, so that path has to stay."""
    monkeypatch.setattr(assets.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        assets.CATALOGUE, "detect",
        assets.TaskAssets("yolo26-detection", "yolo26m-det-bf16-mla_tess-b1.tar.gz", CLIP),
    )
    with pytest.raises(RuntimeError) as caught:
        assets.ensure_model(assets.default_model_path("detect"), "detect")
    message = str(caught.value)
    assert "sima-cli login" in message
    assert assets.MODEL_BASE in message


def test_a_missing_model_is_fetched_with_sima_cli(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    monkeypatch.setitem(
        assets.CATALOGUE, "segment",
        assets.TaskAssets("yolo26-segmentation", "yolo26m-seg-bf16-mla_tess-b1.tar.gz",
                          CLIP),
    )
    seen = {}

    class Result:
        returncode = 0

    def fake_run(command, check=False, env=None):
        seen["command"] = command
        seen["env"] = env
        # -o names the destination, so the fake writes exactly where the real
        # sima-cli would rather than wherever the process happens to be.
        Path(command[-1]).write_bytes(b"tar")
        return Result()

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    path = assets.default_model_path("segment")
    assert assets.ensure_model(path, "segment") == path
    assert seen["command"] == [
        "sima-cli", "download", assets.model_url("segment"), "-o", str(Path(path)),
    ]
    assert Path(path).is_file()
    # Without this sima-cli opens with "update now? [Y/n]" and waits for an
    # answer nobody is there to give, then aborts the download with it.
    assert seen["env"]["SIMA_CLI_CHECK_FOR_UPDATE"] == "0"
    assert "PATH" in seen["env"], "the rest of the environment must survive"


def test_sima_cli_failing_is_reported(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    monkeypatch.setitem(
        assets.CATALOGUE, "fall",
        assets.TaskAssets("yolo26-detection", "yolo26m-det-bf16-mla_tess-b1.tar.gz",
                          CLIP),
    )

    class Result:
        returncode = 1

    monkeypatch.setattr(
        assets.subprocess, "run", lambda *a, **k: Result()
    )
    with pytest.raises(RuntimeError, match="sima-cli login"):
        assets.ensure_model(assets.default_model_path("fall"), "fall")


def test_a_model_we_have_no_url_for_is_not_guessed_at(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    with pytest.raises(RuntimeError, match="model archive not found"):
        assets.ensure_model("some/other-model.tar.gz", "detect")


# ── the two together ──


def test_ensure_assets_leaves_a_resolved_config_alone(here, offline):
    clip = here / "mine.h264"
    clip.write_bytes(b"\x00\x00\x00\x01")
    archive = here / "det.tar.gz"
    archive.write_bytes(b"tar")

    cfg = TASKS["detect"]().load(
        None, {"source.uri": "mine.h264", "model.path": "det.tar.gz"}, use_file=False
    )
    assert assets.ensure_assets(cfg, "detect") is cfg
    assert not offline


def test_ensure_assets_fills_in_both_defaults(here, offline, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: None)

    cfg = TASKS["detect"]().load(None, {}, use_file=False)
    resolved = assets.ensure_assets(cfg, "detect")
    assert Path(resolved.source_uri).is_file()
    assert Path(resolved.model_path).is_file()
    # Clip and pack, both from the release, both without sima-cli on PATH.
    assert offline == [
        f"{assets.SAMPLE_RELEASE}/{CLIP}",
        assets.release_url(assets.CATALOGUE["detect"].model_file),
    ]


def test_a_task_fetches_its_own_pack_and_no_others(here, offline, monkeypatch):
    """One pack per run, not the whole published set.

    detect and fall share the detection pack, segment pulls the segmentation
    one, and nothing pulls all four: a first run should start, not download a
    catalogue.
    """
    monkeypatch.setattr(assets.shutil, "which", lambda _name: None)

    cfg = TASKS["segment"]().load(None, {}, use_file=False)
    assets.ensure_assets(cfg, "segment")

    packs = [url for url in offline if url.endswith(".tar.gz")]
    assert packs == [assets.release_url("yolo26n-seg-bf16-mla_tess.tar.gz")]


# -- what is downloaded has to be what was published --


def test_a_download_that_is_not_what_was_published_is_discarded(here, monkeypatch):
    """A truncated or rewritten file used to be kept and reused forever.

    `download` only ever hashed the URL, to name the cache entry. So a bad
    transfer of the right length landed in assets/ and every later run trusted
    it, because the first thing that function does is believe a file that is
    already there. It fails much later and somewhere else -- a clip that
    decodes half way, a pack that unpacks to nonsense -- and an afternoon went
    into suspecting exactly that of a file which turned out to be fine.
    """
    class Response:
        headers = {"Content-Length": "4"}

        def __init__(self):
            self.left = [b"junk"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout=0: Response())
    with pytest.raises(RuntimeError, match="could not download"):
        assets.ensure_source(f"assets/videos/{CLIP}")

    assert not Path(f"assets/videos/{CLIP}").exists(), "a bad file must not be kept"
    assert not list(Path("assets").rglob("*.part"))


def test_content_that_matches_its_digest_is_kept(here, monkeypatch):
    """The other half: verification must not reject a good download."""
    import hashlib

    body = b"the real thing"
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setitem(assets.RELEASE_SHA256, CLIP, digest)

    class Response:
        headers = {"Content-Length": str(len(body))}

        def __init__(self):
            self.left = [body]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout=0: Response())
    assets.ensure_source(f"assets/videos/{CLIP}")
    assert Path(f"assets/videos/{CLIP}").read_bytes() == body


def test_every_published_asset_has_a_digest():
    """A pack added to the catalogue without one is silently unverified."""
    for name in assets.RELEASE_MODELS:
        assert name in assets.RELEASE_SHA256, name
    for name in assets.SAMPLE_VIDEOS:
        assert name in assets.RELEASE_SHA256, name


# -- the cache must not confuse two URLs, or accept half a file --


def test_two_urls_ending_in_the_same_name_are_two_files(here, monkeypatch):
    """Keying on the last path segment ran one host's video for another's."""
    bodies = {"host-a": b"AAAA", "host-b": b"BBBB"}
    fetched = []

    class Body:
        headers = {"Content-Length": "4"}

        def __init__(self, url):
            self.left = [bodies["host-a" if "host-a" in url else "host-b"]]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    def urlopen(url, timeout=0):
        fetched.append(url)
        return Body(url)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)

    a = assets.ensure_source("https://host-a.example/clip.h264")
    b = assets.ensure_source("https://host-b.example/clip.h264")

    assert a != b, "different URLs must not share a cache entry"
    assert len(fetched) == 2, "the second URL must actually be fetched"
    assert Path(a).read_bytes() == b"AAAA"
    assert Path(b).read_bytes() == b"BBBB"


def test_the_cache_name_is_stable_and_readable():
    once = assets.cache_name("https://example.com/clip.h264")
    assert once == assets.cache_name("https://example.com/clip.h264")
    assert once.startswith("clip-") and once.endswith(".h264")
    # A double extension is not split down the middle.
    assert assets.cache_name("https://x/yolo26m-det.tar.gz").endswith(".tar.gz")
    # A query string is not part of the filename.
    assert "?" not in assets.cache_name("https://x/clip.h264?token=abc")


def test_a_truncated_download_is_not_kept(here, monkeypatch, capsys):
    """A server that stops early ends the read loop exactly like success does.

    Accepting it renames a partial file into place, and every later run then
    reuses it, because an existing file is trusted without being re-checked.
    """
    class Truncated:
        headers = {"Content-Length": "13000000"}

        def __init__(self):
            self.left = [b"only the first bit"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: Truncated())

    assert assets.download("https://example.com/clip.h264", Path("a.h264")) is False
    assert not Path("a.h264").exists(), "no partial file may be left behind"
    assert not list(Path(".").glob("*.part"))
    assert "cut short" in capsys.readouterr().err


def test_a_truncated_source_raises_rather_than_running_on_it(here, monkeypatch):
    class Truncated:
        headers = {"Content-Length": "999"}

        def __init__(self):
            self.left = [b"short"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: Truncated())
    with pytest.raises(RuntimeError, match="could not download"):
        assets.ensure_source("https://example.com/clip.h264")


def test_a_length_the_server_does_not_give_is_still_accepted(here, monkeypatch):
    """Chunked responses carry no Content-Length. That is not an error."""
    class NoLength:
        headers: dict[str, str] = {}

        def __init__(self):
            self.left = [b"payload"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: NoLength())
    assert assets.download("https://example.com/clip.h264", Path("a.h264")) is True
    assert Path("a.h264").read_bytes() == b"payload"


# -- a name the release has but the tables do not --


def test_a_name_only_the_release_knows_is_still_fetched(here, offline, monkeypatch):
    """The tables drift. The release is what actually decides.

    Four clips were published and two were listed here, so naming either of the
    other two got "source file not found" for a file sitting on the very release
    this app downloads from.
    """
    monkeypatch.setattr(assets, "SAMPLE_VIDEOS", {})
    monkeypatch.setattr(
        assets, "published_assets", lambda: frozenset({"people-walking-small.mp4"})
    )
    uri = assets.ensure_source("people-walking-small.mp4")
    assert uri == "assets/videos/people-walking-small.mp4", "a bare name lands in assets"
    assert Path(uri).is_file()
    assert offline == [f"{assets.SAMPLE_RELEASE}/people-walking-small.mp4"]


def test_a_model_only_the_release_knows_is_still_fetched(here, offline, monkeypatch):
    monkeypatch.setattr(assets, "RELEASE_MODELS", {})
    monkeypatch.setattr(
        assets, "published_assets", lambda: frozenset({"yolo26x-det-bf16.tar.gz"})
    )
    path = assets.ensure_model("yolo26x-det-bf16.tar.gz", "detect")
    assert Path(path).is_file()
    assert offline == [f"{assets.SAMPLE_RELEASE}/yolo26x-det-bf16.tar.gz"]


def test_someone_elses_path_is_not_looked_up(here, offline, monkeypatch):
    """`nowhere/mine.h264` is a missing file of theirs, not a published name.

    Asking GitHub about it costs a round trip to confirm what the path already
    says, and the error `check_source_file` gives is the better answer anyway.
    """
    def refuse():
        raise AssertionError("must not ask the release about a path like this")

    monkeypatch.setattr(assets, "published_assets", refuse)
    assert assets.ensure_source("nowhere/mine.h264") == "nowhere/mine.h264"
    assert not offline


def test_our_own_assets_directory_is_worth_asking_about(monkeypatch):
    """The defaults live there, so a name under it could well be published."""
    monkeypatch.setenv(assets.ASSETS_ENV, "assets")
    assert assets.worth_asking(Path("clip.h264")) is True
    assert assets.worth_asking(assets.videos_dir() / "clip.h264") is True
    assert assets.worth_asking(assets.models_dir() / "pack.tar.gz") is True
    assert assets.worth_asking(Path("nowhere/mine.h264")) is False


def test_a_release_that_cannot_be_asked_is_not_an_error(monkeypatch):
    """No network, rate limited, private repo: the tables are then the answer."""
    def boom(url, timeout=0):
        raise OSError("no route to host")

    monkeypatch.setattr(assets, "_published", None)
    monkeypatch.setattr(assets.urllib.request, "urlopen", boom)
    assert assets.published_assets() == frozenset()


def test_the_release_is_asked_once_per_run(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *_a):
            return b'{"assets": [{"name": "a.tar.gz"}]}'

    def urlopen(url, timeout=0):
        calls.append(url)
        return Response()

    monkeypatch.setattr(assets, "_published", None)
    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    assert assets.published_assets() == frozenset({"a.tar.gz"})
    assert assets.published_assets() == frozenset({"a.tar.gz"})
    assert len(calls) == 1, "the listing is cached for the process"
