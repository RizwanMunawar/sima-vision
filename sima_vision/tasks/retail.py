"""Display detected trolley items using the shared detection workflow."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import runtime
from ..assets import default_source_uri
from ..config import (
    BaseConfig,
    TaskDefaults,
    _section,
    _str_list,
    discover_config,
    read_config_file,
)
from ..console import console
from ..samples import resolve_classes
from ..sinks import load_labels
from .detect import DETECT_DRAW, DetectRuntime, DetectTask

# Rough per-item US retail estimates, not a live price feed.
# Produce reference: https://www.walmart.com/cp/fresh-produce/976793
DEFAULT_PRICES = {
    "cherry": 0.10,       # one cherry
    "cantaloupe": 3.00,   # one melon
    "banana": 0.20,
    "cucumber": 0.75,
    "bread": 2.00,        # one loaf
    "snack": 1.50,        # one small packet
    "orange": 1.00,
    "avocados": 1.00,     # one avocado, even when the model label is plural
    "avocado": 1.00,
    "egg": 0.25,          # one egg
    "milk": 4.00,         # one gallon container
    "tomato": 0.25,
    "apple": 1.00,
    "bottle": 1.50,
    "carrot": 0.25,
}


@dataclass(frozen=True)
class RetailConfig:
    classes: tuple[str, ...] = ()
    prices: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))


def basket_rows(boxes, labels, prices):
    """Current visible quantities, never accumulated once per video frame."""
    counts = Counter(labels[b["class_id"]] if 0 <= b["class_id"] < len(labels)
                     else f"class {b['class_id']}" for b in boxes)
    return [(name, count, prices.get(name.lower())) for name, count in sorted(counts.items())]


def draw_basket(annotated, rows):
    """Fit the complete annotated image beside a white menu at recorder size."""
    cv2, np = runtime.cv2, runtime.np
    height, width = annotated.shape[:2]
    panel_w = max(1, round(width * 0.34))
    image_w = width - panel_w
    canvas = np.full_like(annotated, 255)
    scaled_h = max(1, round(height * image_w / width))
    if image_w > 0:
        canvas[:, :image_w] = 24
        top = (height - scaled_h) // 2
        canvas[top:top + scaled_h, :image_w] = cv2.resize(annotated, (image_w, scaled_h))
    scale = max(0.18, min(panel_w / 520, height / 800))
    margin = max(2, round(20 * scale))
    line_h = max(8, round(42 * scale))
    x, y = image_w + margin, margin + line_h

    def text(value, baseline, bold=False):
        size = scale * (0.85 if bold else 0.70)
        while cv2.getTextSize(value, runtime.FONT, size, 1)[0][0] > panel_w - 2 * margin:
            size *= 0.9
            if size < 0.05:
                break
        cv2.putText(canvas, value, (x, baseline), runtime.FONT, size,
                    (25, 25, 25), max(1, round(2 * scale)) if bold else 1, cv2.LINE_AA)

    text("TROLLEY ITEMS", y, True)
    y += line_h
    text("Estimated USD / item", y)
    y += line_h
    capacity = max(0, (height - y - 3 * line_h) // line_h)
    shown = rows[:capacity]
    if not rows:
        text("No items detected", y)
    for name, quantity, price in shown:
        amount = "price N/A" if price is None else f"${price:.2f} ea  ${quantity * price:.2f}"
        text(f"{quantity} x {name}  {amount}", y)
        y += line_h
    if len(rows) > len(shown):
        text(f"+ {len(rows) - len(shown)} more item types", y)
    total = sum(quantity * price for _, quantity, price in rows if price is not None)
    partial = any(price is None for _, _, price in rows)
    text(f"{'Known subtotal' if partial else 'Est. total'}: ${total:.2f}",
         height - 2 * line_h, True)
    text("Visible items / estimated prices", height - line_h)
    return canvas


@dataclass(frozen=True)
class RetailAppConfig(BaseConfig):
    retail: RetailConfig = field(default_factory=RetailConfig)


class RetailRuntime(DetectRuntime):
    unit = "items"

    def __init__(self, cfg, pipeline):
        self.classes = resolve_classes(
            cfg.retail.classes, pipeline.labels, "retail.classes", cfg.labels_path,
        )

    def decode(self, pipeline, cfg, sample, index):
        frame, boxes, elapsed = super().decode(pipeline, cfg, sample, index)
        if self.classes is not None:
            boxes = [box for box in boxes if box["class_id"] in self.classes]
        return frame, boxes, elapsed

    def render(self, cfg, pipeline, frame, results, fps):
        annotated = super().render(cfg, pipeline, frame, results, fps)
        rows = basket_rows(results, pipeline.labels, cfg.retail.prices)
        return draw_basket(annotated, rows)


class RetailTask(DetectTask):
    name = "ret-analytics"
    help = "Display items in a shopping trolley with names and confidence"
    config_class = RetailAppConfig
    defaults = TaskDefaults(task="ret-analytics", video_path="retail.mp4",
                            save_dir="retail-frames", draw=DETECT_DRAW)

    def load(self, path, overrides, use_file=True):
        resolved = discover_config(path) if use_file else None
        if resolved is not None and path is None:
            raw = read_config_file(resolved)
            video = _section(_section(raw, "output"), "video")
            fall_config = "fall" in raw or "tracking" in raw or Path(
                str(video.get("path", ""))
            ).name in {"fall.mp4", "falls.mp4"}
            if fall_config and "retail" not in raw:
                # An implicitly discovered fall config must not select the fall
                # demo and output for a retail run. Explicit flags still win.
                overrides = {
                    "source.type": "video",
                    "source.uri": default_source_uri(self.name),
                    "output.video.path": self.defaults.video_path,
                    "output.save.dir": self.defaults.save_dir,
                    **overrides,
                }
                console.warn(
                    f"{resolved} contains fall settings; using the retail sample and "
                    "output paths. Use --config to explicitly reuse that file."
                )
        return super().load(resolved, overrides, use_file=use_file)

    def add_arguments(self, parser):
        parser.add_argument("--classes", dest="retail.classes", nargs="+",
                            help="Optional item class names or ids; default all model classes.")

    def extra_sections(self, raw):
        retail = _section(raw, "retail")
        prices = dict(DEFAULT_PRICES)
        for name, price in _section(retail, "prices").items():
            if (isinstance(price, bool) or not isinstance(price, (float, int))
                    or not math.isfinite(price) or price < 0):
                raise ValueError("retail.prices values must be finite, nonnegative USD amounts")
            prices[str(name).lower()] = float(price)
        return {"retail": RetailConfig(
            classes=_str_list(retail, "classes", RetailConfig().classes),
            prices=prices,
        )}

    def validate(self, cfg):
        super().validate(cfg)
        resolve_classes(cfg.retail.classes, load_labels(cfg.labels_path),
                        "retail.classes", cfg.labels_path)

    def describe(self, cfg):
        return [f"trolley items: classes={cfg.retail.classes or 'all'}"]

    def runtime(self, cfg, pipeline):
        return RetailRuntime(cfg, pipeline)
