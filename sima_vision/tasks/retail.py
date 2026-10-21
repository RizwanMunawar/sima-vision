"""Display detected trolley items using the shared detection workflow."""

from __future__ import annotations

import colorsys
import hashlib
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
    return [(name, count, prices.get(name.lower())) for name, count in counts.items()]


class BasketMenu:
    """Append confirmed products once and hold counts through detection misses.

    Quantities are confirmed simultaneous counts, not sums across video frames.
    The menu holds its basket until the run ends; it does not infer removals.
    """

    def __init__(self, confirm_frames=3):
        self.confirm_frames = confirm_frames
        self.items = {}
        self.pending = {}

    def update(self, rows):
        observed = {name: (quantity, price) for name, quantity, price in rows}
        for name in list(self.pending):
            if name not in observed:
                del self.pending[name]
        for name, (quantity, price) in observed.items():
            previous = self.items.get(name, (0, price))[0]
            if quantity <= previous:
                self.pending.pop(name, None)
                continue
            target, hits = self.pending.get(name, (quantity, 0))
            hits = hits + 1 if quantity == target else 1
            self.pending[name] = (quantity, hits)
        # Pending insertion order preserves first appearance even when the
        # detector returns the boxes in a different order on the next frame.
        for name, (quantity, hits) in list(self.pending.items()):
            if hits >= self.confirm_frames:
                self.items[name] = (quantity, observed[name][1])
                del self.pending[name]
        return [(name, quantity, price) for name, (quantity, price) in self.items.items()]


def product_color(name):
    """Stable, dark product ink that stays readable on the white table."""
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.52)
    return tuple(round(channel * 255) for channel in (blue, green, red))


def draw_basket(annotated, rows):
    """Draw a readable item table beside the full detection image."""
    cv2, np = runtime.cv2, runtime.np
    height, width = annotated.shape[:2]
    panel_w = max(1, round(width * 0.40))
    image_w = width - panel_w
    canvas = np.full_like(annotated, 255)
    scaled_h = max(1, round(height * image_w / width))
    if image_w > 0:
        canvas[:, :image_w] = 24
        top = (height - scaled_h) // 2
        canvas[top:top + scaled_h, :image_w] = cv2.resize(annotated, (image_w, scaled_h))
        title = "Smart Grocery Shopping"
        title_scale = max(0.35, image_w / 900)
        title_thickness = max(2, round(title_scale * 2))
        (title_w, title_h), baseline = cv2.getTextSize(
            title, runtime.FONT, title_scale, title_thickness,
        )
        if top >= title_h + baseline + 8:
            cv2.putText(canvas, title, ((image_w - title_w) // 2,
                        (top + title_h - baseline) // 2), runtime.FONT,
                        title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
    font = max(0.46, min(panel_w / 600, height / 600))
    margin = max(6, round(panel_w * 0.035))
    left, right = image_w + margin, width - margin
    table_w = max(1, right - left)
    row_h = max(26, round(font * 55))
    ink, grid = (35, 42, 48), (215, 222, 229)
    edges = [left + round(table_w * f) for f in (0, .13, .51, .75, 1)]

    def text(value, x1, x2, top, size=font, bold=False, align="left", color=ink):
        value = str(value)
        thickness = max(2, round(size * 2)) if bold else 1
        padding = max(3, round(font * 7))
        available = max(1, x2 - x1 - 2 * padding)
        # Truncate long labels rather than shrinking the whole row's font.
        original = value
        while value and cv2.getTextSize(value, runtime.FONT, size, thickness)[0][0] > available:
            original = original[:-1]
            value = original + ".." if original else ""
        (tw, th), baseline = cv2.getTextSize(value, runtime.FONT, size, thickness)
        x = x2 - padding - tw if align == "right" else x1 + padding
        y = top + (row_h + th - baseline) // 2
        cv2.putText(canvas, value, (x, y), runtime.FONT, size, color,
                    thickness, cv2.LINE_AA)

    y = margin
    text("TROLLEY ITEMS", left, right, y, font * 1.15, True)
    y += row_h
    text("Estimated prices (USD)", left, right, y, font * .85, True)
    y += row_h
    table_top = y
    cv2.rectangle(canvas, (left, y), (right, y + row_h), (234, 240, 245), -1)
    for i, label in enumerate(("Qty", "Item", "Unit $", "Total $")):
        text(label, edges[i], edges[i + 1], y, font * .86, True)
    y += row_h
    footer_top = height - margin - 2 * row_h
    capacity = max(0, (footer_top - y - row_h) // row_h)
    shown = rows[:capacity]
    for index, (name, quantity, price) in enumerate(shown):
        color = product_color(name)
        if index % 2 == 0:
            cv2.rectangle(canvas, (left, y), (right, y + row_h), (248, 250, 252), -1)
        values = (quantity, name, "N/A" if price is None else f"{price:.2f}",
                  "N/A" if price is None else f"{quantity * price:.2f}")
        for i, value in enumerate(values):
            text(value, edges[i], edges[i + 1], y, bold=True, color=color,
                 align="right" if i >= 2 else "left")
        cv2.line(canvas, (left, y + row_h), (right, y + row_h), grid, 1)
        y += row_h
    if not rows:
        text("No items detected", left, right, y)
        y += row_h
    for x in edges:
        cv2.line(canvas, (x, table_top), (x, y), grid, 1)
    cv2.line(canvas, (left, table_top), (right, table_top), grid, 1)
    cv2.line(canvas, (left, table_top + row_h), (right, table_top + row_h), grid, 1)
    if len(rows) > len(shown):
        text(f"+ {len(rows) - len(shown)} more item types", left, right, y, font * .85)
    total = sum(quantity * price for _, quantity, price in rows if price is not None)
    partial = any(price is None for _, _, price in rows)
    cv2.rectangle(canvas, (left, footer_top), (right, footer_top + row_h), (234, 240, 245), -1)
    text(f"{'Subtotal' if partial else 'Est. total'}: ${total:.2f}",
         left, right, footer_top, font * 1.05, True)
    text("Confirmed items / estimates", left, right, footer_top + row_h, font * .8, True)
    return canvas


@dataclass(frozen=True)
class RetailAppConfig(BaseConfig):
    retail: RetailConfig = field(default_factory=RetailConfig)


class RetailRuntime(DetectRuntime):
    unit = "items"

    def __init__(self, cfg, pipeline):
        self.menu = BasketMenu()
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
        rows = self.menu.update(basket_rows(results, pipeline.labels, cfg.retail.prices))
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
