"""Config loading, CLI overrides and discovery.

None of this needs pyneat or a board, which is the point: a bad config should
fail on a laptop in under a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sima_vision.config import (
    BaseConfig,
    DrawConfig,
    TaskDefaults,
    apply_overrides,
    discover_config,
    load_base_config,
    validate_base,
)
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]

#: One exhaustive config file per task. Nothing ships these -- the package has
#: no starter configs to copy out any more, because a run needs no config at all
#: -- but a file that names *every* documented key is the only thing that proves
#: the loader still understands every documented key. So they live here, and
#: loading each one is the regression guard for the shared-core refactor.
FIXTURES = Path(__file__).parent / "configs"
FULL = [
    (name, FIXTURES / f"{name}.yaml")
    for name in TASKS
    if (FIXTURES / f"{name}.yaml").is_file()
]


def test_there_is_a_fixture_for_every_built_in_task():
    """A new task without one would silently skip the parametrised tests below."""
    from sima_vision.tasks import BUILTIN

    assert {name for name, _ in FULL} == {task.name for task in BUILTIN}


@pytest.mark.parametrize("name,path", FULL, ids=[n for n, _ in FULL])
def test_a_fully_specified_config_loads_and_validates(name, path):
    """Every key the docs mention must still load and pass its task's validation."""
    cfg = TASKS[name]().load(path, {})
    assert cfg.config_path == path
    assert cfg.model_path, "model.path should be set by the fixture"


@pytest.mark.parametrize("name,path", FULL, ids=[n for n, _ in FULL])
def test_a_fully_specified_config_describes(name, path):
    """--validate must be able to describe every one of them."""
    task = TASKS[name]()
    cfg = task.load(path, {})
    for line in task.describe(cfg):
        assert isinstance(line, str) and line


def test_defaults_alone_are_a_complete_config():
    """A TaskDefaults with no catalogue entry is the one case still missing a model.

    The three real tasks name themselves, so `model.path` defaults into
    `assets/`; see test_assets. A bare TaskDefaults stands for a task that has
    no default archive, and that one still has to be told.
    """
    cfg = load_base_config({}, None, TaskDefaults())
    assert isinstance(cfg, BaseConfig)
    with pytest.raises(ValueError, match="model.path must be set"):
        validate_base(cfg)


def test_flags_alone_are_enough_to_run():
    cfg = TASKS["detect"]().load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    assert cfg.model_path == "m.tar.gz"
    assert cfg.source_uri == "c.h264"
    assert cfg.config_path is None


# ── overrides ──


def test_apply_overrides_creates_missing_sections():
    raw = {}
    apply_overrides(raw, {"source.uri": "clip.h264", "decode.score_threshold": 0.7})
    assert raw == {"source": {"uri": "clip.h264"}, "decode": {"score_threshold": 0.7}}


def test_apply_overrides_reaches_nested_sections():
    raw = {}
    apply_overrides(raw, {"alerts.smtp.port": 465})
    assert raw["alerts"]["smtp"]["port"] == 465


def test_apply_overrides_ignores_none():
    """An unset flag must defer to the config file rather than blank it."""
    raw = {"source": {"uri": "from-file.h264"}}
    apply_overrides(raw, {"source.uri": None})
    assert raw["source"]["uri"] == "from-file.h264"


def test_apply_overrides_replaces_a_non_mapping():
    raw = {"source": "not-a-mapping"}
    apply_overrides(raw, {"source.uri": "clip.h264"})
    assert raw["source"] == {"uri": "clip.h264"}


def test_flags_beat_the_config_file():
    cfg = TASKS["detect"]().load(
        FIXTURES / "detect.yaml",
        {"decode.score_threshold": 0.9, "output.save.enable": False},
    )
    assert cfg.score_threshold == 0.9
    assert cfg.save_enable is False


def test_override_goes_through_validation():
    """A CLI flag cannot reach a state a config file could not."""
    with pytest.raises(ValueError, match=r"score_threshold"):
        TASKS["detect"]().load(
            None,
            {"model.path": "m.tar.gz", "source.uri": "c.h264", "decode.score_threshold": 5.0},
            use_file=False,
        )


# ── discovery ──


def test_discovery_finds_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("model:\n  path: a.tar.gz\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert discover_config(None) == tmp_path / "config.yaml"


def test_discovery_returns_none_when_there_is_nothing(tmp_path, monkeypatch):
    """No file is not an error: the defaults are a complete configuration."""
    monkeypatch.chdir(tmp_path)
    assert discover_config(None) is None


def test_explicit_config_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_config(tmp_path / "nope.yaml")


def test_a_config_in_the_working_directory_is_found_by_every_task(tmp_path, monkeypatch):
    """No task needs a config, but every task must read one that is sitting there."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  path: m.tar.gz\nsource:\n  uri: c.h264\n", encoding="utf-8"
    )
    for name in TASKS:
        cfg = TASKS[name]().load(None, {})
        assert cfg.config_path == tmp_path / "config.yaml"
        assert cfg.model_path == "m.tar.gz"


# ── scalar readers ──


def test_yaml_bare_on_off_folds_onto_tokens(tmp_path):
    """YAML 1.1 turns `enable: on` into True; the loader must fold it back."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  path: m.tar.gz\nsource:\n  uri: c.h264\n"
        "preprocess:\n  enable: on\n  resize:\n    enable: off\n",
        encoding="utf-8",
    )
    cfg = TASKS["detect"]().load(path, {})
    assert cfg.preprocess.enable == "on"
    assert cfg.preprocess.resize_enable == "off"


def test_bad_colour_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  path: m.tar.gz\nsource:\n  uri: c.h264\n"
        "visualization:\n  text_color: [0, 0, 300]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="between 0 and 255"):
        TASKS["detect"]().load(path, {})


def test_draw_defaults_differ_per_task():
    """The DrawConfig is shared, but its defaults are not."""
    detect = TASKS["detect"]().defaults.draw
    segment = TASKS["segment"]().defaults.draw
    assert detect.box_thickness == 3
    assert segment.box_thickness == 2
    assert detect.centre_dot is True
    assert segment.centre_dot is False
    assert isinstance(detect, DrawConfig)


def test_the_decoder_hands_over_every_frame_by_default():
    """`auto` tuning and a depth-1 runtime queue each dropped frames on a file.

    Measured on a DevKit with Neat 0.4.0 and the sample clip: `auto` let 245 of
    379 decoded frames through, and depth 1 lost more at the join whenever the
    recorder slowed the pull loop. Either one makes the recording choppy.
    """
    cfg = TASKS["detect"]().load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    assert cfg.decoder_tuning == "default"
    assert cfg.queue_depth >= 4


def test_an_unknown_decoder_tuning_is_refused():
    with pytest.raises(ValueError, match="decoder_tuning"):
        TASKS["detect"]().load(
            None,
            {"model.path": "m.tar.gz", "source.uri": "c.h264",
             "runtime.decoder_tuning": "fast"},
            use_file=False,
        )
