"""Object detection: boxes, class names and confidence on every frame.

The thinnest of the three tasks. It reads the BBOX tensor, draws rectangles and
hands them to the sinks; everything else is inherited.
"""

from __future__ import annotations

from ..config import DrawConfig, TaskDefaults
from ..draw import draw_boxes, draw_fps
from ..runloop import TaskRuntime
from ..samples import (
    extract_bbox_payload,
    first_tensor,
    frame_to_bgr,
    joined_field,
    parse_boxes,
)
from ..sinks import Pipeline, box_metadata
from .base import Task

DETECT_DRAW = DrawConfig(box_thickness=3, centre_dot=True)


class DetectRuntime(TaskRuntime):
    output_label = "detector_output"
    stream = "object-detection"
    unit = "detections"

    def decode(self, pipeline: Pipeline, cfg, sample, index: int):
        # The result field, not the whole bundle: the bundle also holds the
        # decoded frame, and asking for "the BBOX tensor somewhere in here" is
        # a weaker question than the graph can already answer.
        payload, _ = extract_bbox_payload(joined_field(sample, "detections", 1))
        boxes = parse_boxes(payload, pipeline.frame_w, pipeline.frame_h, cfg.max_detections)
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        self.check_geometry(pipeline, frame)
        # `boxes` and `frame` are copies, so the decoder's buffer is free from
        # here on. See FrameStamp for why that matters.
        return frame, boxes, 0.0

    def render(self, cfg, pipeline: Pipeline, frame, results, fps: float):
        """Draw once per frame and share the result between the video and JPEG sinks."""
        annotated = frame.copy()
        # FPS first, so a detection in the top-left corner is never hidden by it.
        if cfg.video_hud:
            draw_fps(annotated, fps, cfg.draw)
        draw_boxes(annotated, results, pipeline.labels, cfg.draw)
        return annotated

    def metadata(self, pipeline: Pipeline, results) -> list[dict]:
        return box_metadata(results, pipeline.labels, pipeline.frame_w, pipeline.frame_h)


class DetectTask(Task):
    name = "detect"
    help = "Boxes, class names and confidence on every frame"
    graph_name = "yolo_detector"
    result_label = "detections"
    output_label = "detector_output"
    defaults = TaskDefaults(
        task="detect",
        family="yolo26",
        save_dir="frames",
        video_path="detections.mp4",
        insight_enable=False,
        draw=DETECT_DRAW,
    )

    def runtime(self, cfg, pipeline: Pipeline) -> TaskRuntime:
        return DetectRuntime()
