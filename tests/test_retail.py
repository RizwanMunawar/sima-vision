from types import SimpleNamespace

import pytest

from sima_vision.cli import build_parser, main
from sima_vision.sinks import load_labels
from sima_vision.tasks.detect import DetectRuntime
from sima_vision.tasks.retail import BasketMenu, RetailRuntime, RetailTask, basket_rows, draw_basket


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


def test_default_keeps_all_detections_and_visible_counts(monkeypatch):
    cfg = RetailTask().load(None, {}, use_file=False)
    labels = ["banana", "tomato", "unknown product"]
    boxes = [{"class_id": 0}, {"class_id": 1}, {"class_id": 1}, {"class_id": 2}]
    pipeline = SimpleNamespace(labels=labels)
    monkeypatch.setattr(DetectRuntime, "decode", lambda *args: (None, boxes, 0.0))
    rt = RetailRuntime(cfg, pipeline)
    assert rt.decode(pipeline, cfg, None, 1)[1] == boxes
    assert basket_rows(boxes, labels, cfg.retail.prices) == [
        ("banana", 1, 0.60), ("tomato", 2, 0.75), ("unknown product", 1, None),
    ]
    assert basket_rows([], labels, cfg.retail.prices) == []


def test_retail_renders_real_boxes_and_white_menu():
    import numpy as np

    cfg = RetailTask().load(None, {}, use_file=False)
    labels = ["banana", "tomato"]
    pipeline = SimpleNamespace(labels=labels)
    boxes = [dict(class_id=0, x1=100, y1=200, x2=300, y2=400, score=0.9)]
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rt = RetailRuntime(cfg, pipeline)
    for _ in range(3):
        annotated = rt.render(cfg, pipeline, frame, boxes, 25)
    assert annotated.shape == frame.shape  # both hardware and OpenCV writers need this
    assert np.count_nonzero(annotated[200:400, :600]) > 0  # visible box in image area
    assert np.all(annotated[0, -1] == 255)
    assert np.any(annotated[20:200, 900:] < 100)  # menu text
    assert not np.any(frame)  # input remains untouched
    assert draw_basket(frame, []).shape == frame.shape


@pytest.mark.parametrize("price", [-1, float("nan"), True, "free"])
def test_invalid_price_is_rejected(price):
    with pytest.raises(ValueError, match="retail.prices"):
        RetailTask().load(None, {"retail.prices": {"tomato": price}}, use_file=False)


def test_menu_keeps_insertion_order_and_counts_through_misses():
    menu = BasketMenu()
    banana, apple = ("banana", 1, .2), ("apple", 1, 1.)
    for _ in range(3):
        rows = menu.update([banana])
    assert rows == [banana]
    for _ in range(3):
        rows = menu.update([apple, banana])
    assert rows == [banana, apple]  # no alphabetical reshuffling
    assert menu.update([]) == rows
    assert menu.update([apple]) == rows
    assert menu.update([("banana", 2, .2), apple]) == rows  # one-frame spike
    assert menu.update([banana, apple]) == rows
    for _ in range(3):
        rows = menu.update([apple, ("banana", 2, .2)])
    assert rows == [("banana", 2, .2), apple]
    for _ in range(10):
        assert menu.update([banana]) == rows  # no count flicker or repeat increments


def test_menu_drops_unconfirmed_new_items():
    menu = BasketMenu()
    assert menu.update([("banana", 1, .2)]) == []
    assert menu.update([]) == []
    assert not menu.pending
