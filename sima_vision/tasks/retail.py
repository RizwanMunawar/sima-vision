"""Retail checkout monitoring using class-aware item tracks.

Counts are visual purchase candidates, not payment confirmations or shopper identities.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..config import BaseConfig, TaskDefaults, _float, _int, _section, _str, _str_list
from ..samples import resolve_classes
from ..sinks import load_labels
from .detect import DETECT_DRAW, DetectRuntime, DetectTask
from .fall import box_iou


@dataclass(frozen=True)
class RetailConfig:
    classes: tuple[str, ...] = ("bottle", "banana", "apple", "orange")
    checkout: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
    confirm_frames: int = 3
    max_age: int = 30
    iou: float = 0.3
    report: str = "retail-report.json"


@dataclass(frozen=True)
class RetailAppConfig(BaseConfig):
    retail: RetailConfig = field(default_factory=RetailConfig)


@dataclass
class Item:
    box: dict
    misses: int = 0
    dwell: int = 0
    counted: bool = False


class RetailAnalyzer:
    """Count each track once after consecutive observations in the checkout ROI.

    Greedy IoU association can split tracks after occlusion or swap identical items.
    Only active tracks are retained, keeping memory bounded on live streams.
    """

    def __init__(self, cfg: RetailConfig, classes: set[int] | None):
        self.cfg = cfg
        self.classes = classes
        self.items: list[Item] = []
        self.counts: Counter = Counter()

    def update(self, boxes: list[dict], width: int, height: int) -> list[dict]:
        boxes = [b for b in boxes if self.classes is None or b["class_id"] in self.classes]
        pairs = sorted(
            ((box_iou(t.box, b), ti, bi)
             for ti, t in enumerate(self.items) for bi, b in enumerate(boxes)
             if t.box["class_id"] == b["class_id"]), reverse=True,
        )
        used_t, used_b = set(), set()
        visible = []
        for score, ti, bi in pairs:
            if score < self.cfg.iou:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self.items[ti].box = boxes[bi]
            self.items[ti].misses = 0
            visible.append(self.items[ti])
        for ti, item in enumerate(self.items):
            if ti not in used_t:
                item.misses += 1
                item.dwell = 0
        self.items = [t for t in self.items if t.misses <= self.cfg.max_age]
        for bi, box in enumerate(boxes):
            if bi not in used_b:
                item = Item(box=box)
                self.items.append(item)
                visible.append(item)
        x1, y1, x2, y2 = self.cfg.checkout
        for item in visible:
            b = item.box
            cx, cy = (b["x1"] + b["x2"]) / (2 * width), (b["y1"] + b["y2"]) / (2 * height)
            inside = x1 <= cx <= x2 and y1 <= cy <= y2
            item.dwell = item.dwell + 1 if inside else 0
            if item.dwell >= self.cfg.confirm_frames and not item.counted:
                self.counts[b["class_id"]] += 1
                item.counted = True
        return boxes


class RetailRuntime(DetectRuntime):
    unit = "items"

    def __init__(self, cfg, pipeline):
        self.cfg = cfg.retail
        self.analyzer = RetailAnalyzer(cfg.retail, resolve_classes(
            cfg.retail.classes, pipeline.labels, "retail.classes", cfg.labels_path,
        ))

    def decode(self, pipeline, cfg, sample, index):
        frame, boxes, elapsed = super().decode(pipeline, cfg, sample, index)
        boxes = self.analyzer.update(boxes, pipeline.frame_w, pipeline.frame_h)
        return frame, boxes, elapsed

    def render(self, cfg, pipeline, frame, results, fps):
        import cv2

        annotated = super().render(cfg, pipeline, frame, results, fps)
        height, width = annotated.shape[:2]
        x1, y1, x2, y2 = self.cfg.checkout
        cv2.rectangle(annotated, (int(x1 * (width - 1)), int(y1 * (height - 1))),
                      (int(x2 * (width - 1)), int(y2 * (height - 1))), (0, 200, 255), 2)
        return annotated

    def summarise(self, pipeline, processed):
        counts = {pipeline.labels[c] if 0 <= c < len(pipeline.labels) else str(c): n
                  for c, n in sorted(self.analyzer.counts.items())}
        report = {"type": "visual_purchase_candidates", "frames": processed,
                  "total": sum(counts.values()), "items": counts}
        path = Path(self.cfg.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return [f"purchase_candidates={report['total']}", f"report={path}"]


class RetailTask(DetectTask):
    name = "ret-analytics"
    help = "Monitor checkout items and report visual purchase candidates"
    config_class = RetailAppConfig
    defaults = TaskDefaults(task="ret-analytics", video_path="retail.mp4",
                            save_dir="retail-frames", draw=DETECT_DRAW)

    def add_arguments(self, parser):
        parser.add_argument("--classes", dest="retail.classes", nargs="+",
                            help="Product class names or ids; default bottle banana apple orange.")
        parser.add_argument("--checkout", dest="retail.checkout", nargs=4, type=float,
                            metavar=("X1", "Y1", "X2", "Y2"),
                            help="Normalized checkout rectangle; default whole frame.")
        parser.add_argument("--confirm-frames", dest="retail.confirm_frames", type=int,
                            help="Consecutive checkout observations before counting; default 3.")
        parser.add_argument("--report", dest="retail.report", help="JSON report path.")

    def extra_sections(self, raw):
        r = _section(raw, "retail")
        defaults = RetailConfig()
        for key in ("confirm_frames", "max_age"):
            if key in r and (isinstance(r[key], bool) or not isinstance(r[key], int)):
                raise ValueError(f"retail.{key} must be an integer")
        roi = r.get("checkout", defaults.checkout)
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("retail.checkout must contain four normalized coordinates")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in roi):
            raise ValueError("retail.checkout coordinates must be numeric")
        return {"retail": RetailConfig(
            classes=_str_list(r, "classes", defaults.classes), checkout=tuple(roi),
            confirm_frames=_int(r, "confirm_frames", 3), max_age=_int(r, "max_age", 30),
            iou=_float(r, "iou", 0.3), report=_str(r, "report", defaults.report),
        )}

    def validate(self, cfg):
        super().validate(cfg)
        r = cfg.retail
        x1, y1, x2, y2 = r.checkout
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("retail.checkout must satisfy 0 <= x1 < x2 <= 1 and y1 < y2")
        if r.confirm_frames < 1 or r.max_age < 0:
            raise ValueError("retail.confirm_frames must be positive and max_age nonnegative")
        if not math.isfinite(r.iou) or not 0 < r.iou <= 1:
            raise ValueError("retail.iou must be in (0, 1]")
        if not r.report.strip():
            raise ValueError("retail.report must name a JSON output file")
        resolve_classes(r.classes, load_labels(cfg.labels_path), "retail.classes", cfg.labels_path)

    def describe(self, cfg):
        return [f"retail: classes={cfg.retail.classes} checkout={cfg.retail.checkout}",
                f"visual purchase candidates only; report={cfg.retail.report}"]

    def runtime(self, cfg, pipeline):
        return RetailRuntime(cfg, pipeline)
