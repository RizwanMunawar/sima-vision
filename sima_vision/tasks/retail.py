"""Display detected trolley items using the shared detection workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True)
class RetailConfig:
    classes: tuple[str, ...] = ("bottle", "banana", "apple", "orange")


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
                            help="Item class names or ids; default bottle banana apple orange.")

    def extra_sections(self, raw):
        retail = _section(raw, "retail")
        return {"retail": RetailConfig(
            classes=_str_list(retail, "classes", RetailConfig().classes),
        )}

    def validate(self, cfg):
        super().validate(cfg)
        resolve_classes(cfg.retail.classes, load_labels(cfg.labels_path),
                        "retail.classes", cfg.labels_path)

    def describe(self, cfg):
        return [f"trolley items: classes={cfg.retail.classes or 'all'}"]

    def runtime(self, cfg, pipeline):
        return RetailRuntime(cfg, pipeline)
