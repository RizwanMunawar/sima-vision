"""The Python API.

Its whole promise is that a CLI flag and a Python keyword are the same setting
under the same name, so most of these tests are about that not drifting.
"""

from __future__ import annotations

import pytest

import sima_vision
from sima_vision.api import _alias_table, settings_to_overrides
from sima_vision.cli import build_parser
from sima_vision.tasks import TASKS


def test_the_package_exports_its_verbs():
    for name in ("run", "validate", "load"):
        assert callable(getattr(sima_vision, name))



@pytest.mark.parametrize("name", list(TASKS))
def test_every_cli_flag_has_a_python_keyword(name):
    """If a flag exists, Python can set it. That is the whole contract."""
    task = TASKS[name]()
    aliases, _ = _alias_table(task)
    assert build_parser().parse_args([name, "--no-config", "--validate"]).command == name

    # Collect every dotted dest the subcommand can write...
    import argparse

    probe = argparse.ArgumentParser(add_help=False)
    from sima_vision.cli import add_shared_arguments

    add_shared_arguments(probe)
    task.add_arguments(probe.add_argument_group("task"))
    dests = {a.dest for a in probe._actions if "." in a.dest}
    # ...and check each is reachable from Python.
    assert dests <= set(aliases.values()), dests - set(aliases.values())


def test_negative_flags_become_positive_keywords():
    """Nobody should have to write no_save=True."""
    aliases, _ = _alias_table(TASKS["detect"]())
    assert aliases["save"] == "output.save.enable"
    assert aliases["video"] == "output.video.enable"
    assert "no_save" not in aliases


def test_a_plain_negative_flag_is_not_inverted_twice():
    """`--no-save` already means save=False; flipping it again would undo it."""
    _, inverted = _alias_table(TASKS["fall"]())
    assert "save" not in inverted


def test_keywords_map_to_config_paths():
    task = TASKS["detect"]()
    assert settings_to_overrides(task, {"conf": 0.4}) == {"decode.score_threshold": 0.4}
    assert settings_to_overrides(task, {"source": "c.h264"}) == {"source.uri": "c.h264"}
    assert settings_to_overrides(task, {"max_det": 7}) == {"decode.max_detections": 7}


def test_dotted_keys_pass_straight_through():
    """Anything the aliases miss is still reachable."""
    task = TASKS["detect"]()
    out = settings_to_overrides(task, {"runtime.output_buffers": 2})
    assert out == {"runtime.output_buffers": 2}


def test_an_unknown_keyword_suggests_the_near_miss():
    with pytest.raises(TypeError, match="conf"):
        settings_to_overrides(TASKS["detect"](), {"confidence": 0.5})


# ── the verbs ──


def test_validate_returns_the_resolved_config():
    cfg = sima_vision.validate(
        "detect", use_config_file=False, model="m.tar.gz", source="c.h264",
        conf=0.55, max_det=12, save=False,
    )
    assert cfg.score_threshold == 0.55
    assert cfg.max_detections == 12
    assert cfg.save_enable is False


def test_validate_raises_on_a_bad_setting():
    with pytest.raises(ValueError, match="score_threshold"):
        sima_vision.validate(
            "detect", use_config_file=False, model="m", source="c", conf=5.0
        )


def test_anonymise_keyword():
    cfg = sima_vision.validate(
        "segment", use_config_file=False, model="m", source="c",
        anonymise=True, keep_classes=["person"],
    )
    assert cfg.blur.invert is True
    assert cfg.blur.keep_classes == ("person",)


def test_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        sima_vision.validate("nope")




