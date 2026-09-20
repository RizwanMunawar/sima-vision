"""The command surface: parsing, dispatch and the compatibility shims."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sima_vision import __version__, cli
from sima_vision.cli import build_parser, collect_overrides, main
from sima_vision.console import console
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


def test_minimal_strips_the_sinks():
    task = TASKS["segment"]()
    cfg = task.load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    args = parse(["segment", "--minimal"])
    stripped = task.post_process(cfg, args)
    assert stripped.segment.masks == "off"
    assert stripped.blur.enable is False
    assert not (stripped.save_enable or stripped.video_enable)


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


def test_an_error_that_names_itself_is_not_headed_twice(capsys):
    """ultralytics prefixes its own messages, and a bad .pt is the common one.

    It came out as `ERROR  ERROR  best.pt is not a loadable checkpoint`, which
    reads as a bug in this program rather than a problem with the file.
    """
    console.error("ERROR  best.pt is not a loadable checkpoint")
    err = capsys.readouterr().err
    assert err.count("ERROR") == 1
    assert "best.pt is not a loadable checkpoint" in err


def test_an_ordinary_error_still_gets_its_heading(capsys):
    console.error("no such file: best.pt")
    err = capsys.readouterr().err
    assert "ERROR" in err and "no such file: best.pt" in err


# -- the frame-rate badge --

def test_the_badge_can_be_restyled_without_a_config_file():
    """Every knob was reachable only through `visualization.hud` in YAML.

    It had been configurable since the first version, which is not the same as
    being findable: the flag table listed `--no-hud` and nothing else, so the
    question people actually asked was whether it could be changed at all.
    """
    args = parse(["segment", "--hud-scale", "2.5", "--hud-thickness", "4",
                  "--hud-bg", "0,0,255", "--hud-color", "0,255,255",
                  "--hud-padding", "30"])
    assert collect_overrides(args) == {
        "visualization.hud.text_scale": 2.5,
        "visualization.hud.text_thickness": 4,
        "visualization.hud.bg_color": [0, 0, 255],
        "visualization.hud.text_color": [0, 255, 255],
        "visualization.hud.padding": 30,
    }


def test_the_badge_flags_are_on_every_app():
    """One overlay, one set of flags. detect and fall draw the same badge."""
    for name in TASKS:
        args = parse([name, "--hud-bg", "10,20,30"])
        assert collect_overrides(args)["visualization.hud.bg_color"] == [10, 20, 30]


def test_a_colour_is_three_channels_of_0_to_255():
    assert cli.bgr_colour("0,255,255") == [0, 255, 255]
    assert cli.bgr_colour(" 1 , 2 , 3 ") == [1, 2, 3]


@pytest.mark.parametrize("value", ["255,0", "1,2,3,4", "300,0,0", "red", "1,2,x", ""])
def test_a_colour_that_is_not_one_is_refused_with_the_reason(value):
    """A flag that reads correctly and paints the wrong colour is worse than
    one that refuses. `red` is not accepted precisely because it would have to
    mean 0,0,255 here, and nobody expects that of the word."""
    with pytest.raises(argparse.ArgumentTypeError):
        cli.bgr_colour(value)


def test_the_channel_order_matches_the_config_file():
    """BGR, because the config file and OpenCV are both BGR.

    Taking RGB on the flag and BGR in the YAML would make the same three
    numbers mean two different colours depending on where they were written.
    """
    import inspect

    doc = inspect.getdoc(cli.bgr_colour) or ""
    assert "BGR" in doc
    # Red is 0,0,255 in this order. If that ever flips, this is the canary.
    assert cli.bgr_colour("0,0,255") == [0, 0, 255]


# -- stills are opt-in; the video is the output --

def test_no_app_writes_stills_unless_asked():
    """The annotated video is what people came for.

    A run used to drop a still every 10 frames beside it, which on a 1080p clip
    is a few hundred JPEGs nobody asked for and everybody then deleted. Asserted
    across every app, because this is the kind of default that gets restored in
    one task and not the others.
    """
    for name in TASKS:
        cfg = TASKS[name]().load(
            None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
        )
        assert cfg.save_enable is False, name
        assert cfg.video_enable is True, name


@pytest.mark.parametrize(
    ("argv", "enabled"),
    [
        ([], None),                                   # not given: the file decides
        (["--save"], True),
        (["--save-every", "5"], True),                # asking the rate asks for stills
        (["--save-dir", "shots"], True),              # so does asking where
        (["--save-every", "0"], None),                # 0 disables via the rate
        (["--save-every", "5", "--no-save"], False),  # an explicit no still wins
        (["--no-save"], False),
    ],
)
def test_asking_where_or_how_often_asks_for_stills(argv, enabled):
    """`--save-every 5` on its own has to write something.

    Stills are off by default, so the flag would otherwise be accepted and
    change nothing -- and `--save-every 0` already carries that same
    enable/disable sense in the other direction.
    """
    args = parse(["detect", *argv])
    assert collect_overrides(args).get("output.save.enable") is enabled


def test_save_every_zero_writes_nothing_even_where_stills_are_on():
    """0 disables through the rate, so it does not need the enable flag too.

    Which is why it is the one `--save-every` value that implies nothing: a
    config file saying `enable: true` plus `--save-every 0` still writes no
    stills, and leaving the flag alone keeps that readable.
    """
    from sima_vision.sinks import wants_jpeg

    args = parse(["detect", "--save-every", "0"])
    cfg = TASKS["detect"]().load(
        Path(__file__).parent / "configs" / "detect.yaml", collect_overrides(args)
    )
    assert cfg.save_enable is True
    assert not any(wants_jpeg(cfg, index) for index in range(50))


def test_a_config_file_can_still_turn_stills_on():
    """The default moved; the setting did not. An existing config keeps working."""
    cfg = TASKS["detect"]().load(Path(__file__).parent / "configs" / "detect.yaml", {})
    assert cfg.save_enable is True
    assert cfg.save_every == 10


def test_the_rate_survives_the_default_moving():
    """`save_every` is the rate once stills are on, not a second off switch.

    Zeroing it as well as the enable flag would have made `--save` alone write
    nothing, which is the one thing that flag has to do.
    """
    args = parse(["detect", "--save"])
    cfg = TASKS["detect"]().load(None, collect_overrides(args), use_file=False)
    assert (cfg.save_enable, cfg.save_every) == (True, 10)
