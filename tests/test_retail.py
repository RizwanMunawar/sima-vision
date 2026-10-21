from types import SimpleNamespace

import pytest

from sima_vision.cli import build_parser, main
from sima_vision.sinks import load_labels
from sima_vision.tasks.detect import DetectRuntime
from sima_vision.tasks.retail import RetailRuntime, RetailTask


def test_only_selected_items_are_displayed(monkeypatch):
    cfg = RetailTask().load(None, {"retail.classes": ["bottle"]}, use_file=False)
    pipeline = SimpleNamespace(labels=load_labels(cfg.labels_path))
    item = {"class_id": pipeline.labels.index("bottle")}
    person = {"class_id": pipeline.labels.index("person")}
    frame = object()
    monkeypatch.setattr(DetectRuntime, "decode", lambda *args: (frame, [person, item], 0.0))
    runtime = RetailRuntime(cfg, pipeline)
    assert runtime.decode(pipeline, cfg, None, 1) == (frame, [item], 0.0)
    monkeypatch.setattr(DetectRuntime, "decode", lambda *args: (frame, [], 0.0))
    assert runtime.decode(pipeline, cfg, None, 2) == (frame, [], 0.0)
    assert runtime.summarise(pipeline, 2) == []


def test_custom_product_labels_and_all_classes(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("cereal\nmilk\n", encoding="utf-8")
    cfg = RetailTask().load(None, {"model.labels": str(labels), "retail.classes": []},
                            use_file=False)
    assert RetailRuntime(cfg, SimpleNamespace(labels=load_labels(cfg.labels_path))).classes is None


def test_unknown_item_is_rejected():
    with pytest.raises(ValueError, match="unknown class"):
        RetailTask().load(None, {"retail.classes": ["not-a-product"]}, use_file=False)


def test_cli_validation(capsys):
    assert main(["ret-analytics", "--no-config", "--validate"]) == 0
    assert "trolley items" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--checkout", "--confirm-frames", "--report"])
def test_checkout_options_are_removed(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ret-analytics", flag])


def test_discovered_fall_config_uses_retail_source_and_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "source:\n  uri: fall.h264\noutput:\n  video:\n    path: falls.mp4\n"
        "fall:\n  confirm_seconds: 2\ndecode:\n  score_threshold: 0.5\n",
        encoding="utf-8",
    )
    cfg = RetailTask().load(None, {})
    assert cfg.source_uri.endswith("shopping-trolley-items.mp4")
    assert cfg.video_path == "retail.mp4"
    assert cfg.score_threshold == 0.5
    overridden = RetailTask().load(None, {
        "source.uri": "my-trolley.mp4", "output.video.path": "my-retail.mp4",
    })
    assert overridden.source_uri == "my-trolley.mp4"
    assert overridden.video_path == "my-retail.mp4"
    explicit = RetailTask().load(config, {})
    assert explicit.source_uri == "fall.h264"
    assert explicit.video_path == "falls.mp4"
