import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sima_vision.cli import main
from sima_vision.sinks import load_labels
from sima_vision.tasks.retail import RetailAnalyzer, RetailConfig, RetailRuntime, RetailTask


def box(cls=39, x=10):
    return dict(x1=x, y1=10, x2=x + 10, y2=20, class_id=cls, score=0.9)


def test_confirmation_and_no_repeat_on_reentry():
    a = RetailAnalyzer(RetailConfig(checkout=(0, 0, .5, 1)), {39})
    for _ in range(10):
        a.update([box()], 100, 100)
    assert a.counts == {39: 1}
    a.update([], 100, 100)
    for _ in range(4):
        a.update([box()], 100, 100)
    assert a.counts == {39: 1}


def test_classes_do_not_share_tracks_and_counts_require_consecutive_frames():
    a = RetailAnalyzer(RetailConfig(), {39, 40})
    a.update([box()], 100, 100)
    a.update([], 100, 100)
    a.update([box()], 100, 100)
    a.update([box()], 100, 100)
    assert not a.counts
    a.update([box()], 100, 100)
    for _ in range(3):
        a.update([box(40)], 100, 100)
    assert a.counts == {39: 1, 40: 1}


def test_roi_filter_and_multiple_items():
    a = RetailAnalyzer(RetailConfig(checkout=(0, 0, .5, 1)), {39})
    for _ in range(4):
        visible = a.update([box(), box(x=30), box(x=80), box(0)], 100, 100)
    assert len(visible) == 3
    assert a.counts == {39: 2}


def test_expiration_bounds_memory():
    a = RetailAnalyzer(RetailConfig(max_age=1), None)
    a.update([box()], 100, 100)
    a.update([], 100, 100)
    a.update([], 100, 100)
    assert not a.items


@pytest.mark.parametrize("overrides", [
    {"retail.checkout": [0, 0, 2, 1]}, {"retail.confirm_frames": 0},
    {"retail.iou": 0}, {"retail.classes": ["not-a-product"]},
])
def test_validation(overrides):
    with pytest.raises(ValueError):
        RetailTask().load(None, overrides, use_file=False)


def test_cli_validation(capsys):
    assert main(["ret-analytics", "--no-config", "--classes", "bottle",
                 "--checkout", "0", "0", "1", "1", "--validate"]) == 0
    assert "purchase candidates" in capsys.readouterr().out


def test_report(tmp_path):
    cfg = RetailTask().load(None, {}, use_file=False)
    path = tmp_path / "out" / "report.json"
    cfg = replace(cfg, retail=replace(cfg.retail, report=str(path)))
    pipeline = SimpleNamespace(labels=load_labels(cfg.labels_path))
    runtime = RetailRuntime(cfg, pipeline)
    for _ in range(3):
        runtime.analyzer.update([box()], 100, 100)
    runtime.summarise(pipeline, 3)
    assert json.loads(path.read_text())["items"] == {"bottle": 1}
