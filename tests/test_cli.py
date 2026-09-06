"""The command surface: parsing, dispatch and the compatibility shims."""

from __future__ import annotations

from pathlib import Path

import pytest

from sima_vision import __version__
from sima_vision.cli import build_parser, collect_overrides, main
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]


def parse(argv):
    return build_parser().parse_args(argv)


def test_every_task_has_a_subcommand():
    parser = build_parser()
    for name in TASKS:
        args = parser.parse_args([name, "--no-config", "--validate"])
        assert args.command == name


def test_the_setup_commands_are_gone():
    """init, fetch, doctor and setup: a run does all four of those jobs now."""
    parser = build_parser()
    available = {
        name
        for action in parser._actions
        for name in (getattr(action, "choices", None) or ())
    }
    for name in ("init", "fetch", "doctor", "setup"):
        with pytest.raises(SystemExit):
            parser.parse_args([name])
        assert name not in available


def test_the_board_commands_are_push_and_pull_only():
    """`watch` and `remote` ran things on the board over ssh. Both are gone."""
    parser = build_parser()
    available = {
        name
        for action in parser._actions
        for name in (getattr(action, "choices", None) or ())
    }
    assert {"push", "pull"} <= available
    for name in ("watch", "remote"):
        assert name not in available
        with pytest.raises(SystemExit):
            parser.parse_args([name])


def test_preview_is_gone():
    """It drew synthetic detections. Nothing here invents data any more."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["preview"])
    assert "preview" not in parser.format_help()


def test_no_command_prints_help_and_fails():
    assert main([]) == 2


def test_version_is_the_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_shared_flags_map_to_config_paths():
    args = parse(["detect", "--source", "c.h264", "--conf", "0.4", "--max-det", "7"])
    overrides = collect_overrides(args)
    assert overrides["source.uri"] == "c.h264"
    assert overrides["decode.score_threshold"] == 0.4
    assert overrides["decode.max_detections"] == 7


def test_unset_flags_are_not_overrides():
    """Only what the user typed may override the file."""
    args = parse(["detect", "--source", "c.h264"])
    assert "decode.score_threshold" not in collect_overrides(args)


def test_negative_switches_override_to_false():
    args = parse(["detect", "--no-save", "--no-video"])
    overrides = collect_overrides(args)
    assert overrides["output.save.enable"] is False
    assert overrides["output.video.enable"] is False


def test_segment_flags():
    args = parse(["segment", "--anonymise", "--keep-classes", "person", "car",
                  "--blur-method", "pixelate", "--mask-threshold", "0.3"])
    overrides = collect_overrides(args)
    assert overrides["blur.invert"] is True
    assert overrides["blur.keep_classes"] == ["person", "car"]
    assert overrides["blur.method"] == "pixelate"
    assert overrides["segmentation.threshold"] == 0.3


def test_fall_flags_reach_the_nested_smtp_section():
    args = parse(["fall", "--alert-to", "a@x.com", "--smtp-port", "465", "--send"])
    overrides = collect_overrides(args)
    assert overrides["alerts.to"] == ["a@x.com"]
    assert overrides["alerts.smtp.port"] == 465
    assert overrides["alerts.dry_run"] is False


def test_alert_recipient_turns_alerts_on():
    cfg = TASKS["fall"]().load(
        None,
        {"model.path": "m.tar.gz", "source.uri": "c.h264", "alerts.to": ["a@x.com"]},
        use_file=False,
    )
    assert cfg.alerts.enable is True
    # ...but still a dry run, so nobody is emailed by accident.
    assert cfg.alerts.dry_run is True


def test_send_is_needed_to_actually_send():
    cfg = TASKS["fall"]().load(
        None,
        {
            "model.path": "m.tar.gz", "source.uri": "c.h264",
            "alerts.to": ["a@x.com"], "alerts.from": "b@x.com",
            "alerts.dry_run": False,
        },
        use_file=False,
    )
    assert cfg.alerts.enable is True
    assert cfg.alerts.dry_run is False


def test_config_and_no_config_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse(["detect", "--config", "a.yaml", "--no-config"])


def test_minimal_strips_the_sinks():
    task = TASKS["segment"]()
    cfg = task.load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    args = parse(["segment", "--minimal"])
    stripped = task.post_process(cfg, args)
    assert stripped.segment.masks == "off"
    assert stripped.blur.enable is False
    assert not (stripped.save_enable or stripped.video_enable or stripped.insight_enable)


def test_validate_exits_zero_without_a_board():
    code = main([
        "detect", "--no-config", "--model", "m.tar.gz", "--source", "c.h264", "--validate",
    ])
    assert code == 0


def test_a_bad_config_exits_one(capsys):
    code = main(["detect", "--no-config", "--conf", "5", "--validate"])
    assert code == 1
    assert "decode.score_threshold" in capsys.readouterr().err


def test_no_flags_at_all_still_validates(capsys):
    """Neither --model nor --source is required: both default into assets/."""
    assert main(["detect", "--no-config", "--validate"]) == 0
    out = capsys.readouterr().out
    assert "assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz" in out
    assert "assets/videos/people-walking-outside-mall.h264" in out


# -- what a run needs, and how it says so --


def test_the_model_command_is_runnable_for_every_task():
    """A run shells out to this when the pack is missing, so it has to be exact."""
    from sima_vision.assets import CATALOGUE, model_command

    for name in TASKS:
        assert name in CATALOGUE
        command = model_command(name)
        assert "sima-cli download" in command
        assert CATALOGUE[name].model_file in command
        # It must land where a run then looks for it.
        assert "assets/models" in command


def test_every_command_is_reachable():
    parser = build_parser()
    for name in [*TASKS, "push", "pull"]:
        assert name in parser.format_help()


def test_a_task_is_all_you_need_to_type():
    """No setup subcommand may stand between `pip install` and a run."""
    parser = build_parser()
    for name in TASKS:
        args = parser.parse_args([name])
        assert args.command == name
        assert args.validate is False


def test_quiet_is_available_on_every_task():
    for name in TASKS:
        assert parse([name, "--quiet"]).quiet is True


def test_validate_prints_through_the_console(capsys):
    """--validate is the one path with no board, so its output is the whole answer."""
    assert main(["detect", "--no-config", "--validate"]) == 0
    out = capsys.readouterr().out
    assert "config OK" in out
    assert "nothing was downloaded" in out
