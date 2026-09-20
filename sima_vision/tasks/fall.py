"""Fall detection: track people and judge whether one has gone down.

Three signals, all available from a plain bounding box:

  aspect    a standing person is tall and narrow, a fallen one is wide and
            short. The box aspect ratio crossing 1.0 is the strongest single
            indicator there is without pose keypoints.
  collapse  the box height drops well below what this person's own upright
            height has been, which separates lying down from crouching.
  descent   the centre of the box moves down fast. This is what distinguishes
            a fall from someone lying down deliberately.

Any one of them can fire spuriously for a frame or two, so nothing is reported
until the condition has held for ``fall.confirm_seconds``. That delay is the
whole difference between a signal worth reading and one nobody trusts.

The frame this draws is the one ``detect`` draws -- the same palette, the same
boxes, the same captions. A fall changes the class a box is labelled with, and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import (
    BaseConfig,
    TaskDefaults,
    _flag,
    _float,
    _int,
    _section,
    _str_list,
)
from ..console import console
from ..draw import CLASS_COLORS, class_color, draw_boxes, draw_fps
from ..runloop import TaskRuntime
from ..samples import (
    extract_bbox_payload,
    first_tensor,
    frame_to_bgr,
    joined_field,
    parse_boxes,
    resolve_classes,
)
from ..sinks import Pipeline, load_labels
from .base import Task
from .detect import DETECT_DRAW

UPRIGHT, FALLING, FALLEN, RECOVERING = "upright", "falling", "fallen", "recovering"

#: The name a fallen track's box is captioned with. Not a colour: the box
#: takes a palette colour like every other class. See :func:`fall_class_id`.
FALL_CLASS = "FALL"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackConfig:
    """How detections are followed from frame to frame.

    Attributes:
        classes: Class names or ids that can fall. Empty means every class,
            which is almost never what you want in a warehouse.
        iou_threshold: Minimum overlap to call two boxes the same person.
            Lower it for a low frame rate, where boxes move further per frame.
        max_age: Frames a track survives with no detection before it is dropped.
            This is what carries someone through a brief occlusion.
        min_hits: Frames a track must be seen on before it is reported at all,
            which suppresses one-frame false positives.
        history_seconds: How much per-track history to keep. Must be at least
            ``fall.descent_window``.
        upright_aspect: Aspect ratio at or below which a person counts as
            upright for the purpose of learning their reference height.
    """

    classes: tuple[str, ...] = ("person",)
    iou_threshold: float = 0.30
    max_age: int = 30
    min_hits: int = 3
    history_seconds: float = 3.0
    upright_aspect: float = 0.70


@dataclass(frozen=True)
class FallConfig:
    """When a tracked person counts as fallen.

    Attributes:
        enable: Whether to evaluate falls at all. ``off`` leaves a plain
            person tracker, which is a useful way to tune the tracking first.
        aspect_ratio: Width over height at or above which the box reads as
            lying down. 1.0 is square; 1.2 gives a little margin against a
            crouch or a wide-armed gesture.
        height_drop: Fraction of this person's own learned upright height at or
            below which they count as collapsed. 0.55 means "less than 55% as
            tall as they were".
        descent_rate: Downward speed of the box centre that reads as a fall,
            as a fraction of frame height per second. 0.55 is roughly half the
            frame in one second.
        descent_window: How far back to measure that speed, in seconds.
        confirm_seconds: How long a signal must hold before a track is called
            fallen. The single most important knob here: too low and a bending
            forklift driver is reported, too high and the report is late.
        recover_seconds: How long someone must look upright again before the
            track is re-armed.
        min_box_height: Ignore boxes shorter than this fraction of the frame,
            which drops distant figures too small to judge.
    """

    enable: bool = True
    aspect_ratio: float = 1.20
    height_drop: float = 0.55
    descent_rate: float = 0.55
    descent_window: float = 0.7
    confirm_seconds: float = 1.5
    recover_seconds: float = 3.0
    min_box_height: float = 0.08


@dataclass(frozen=True)
class FallAppConfig(BaseConfig):
    """Base config plus the ``tracking`` and ``fall`` sections."""

    track: TrackConfig = TrackConfig()
    fall: FallConfig = FallConfig()


def load_track_config(raw: dict) -> TrackConfig:
    section = _section(raw, "tracking")
    d = TrackConfig()
    return TrackConfig(
        classes=_str_list(section, "classes", d.classes),
        iou_threshold=_float(section, "iou_threshold", d.iou_threshold),
        max_age=_int(section, "max_age", d.max_age),
        min_hits=_int(section, "min_hits", d.min_hits),
        history_seconds=_float(section, "history_seconds", d.history_seconds),
        upright_aspect=_float(section, "upright_aspect", d.upright_aspect),
    )


def load_fall_config(raw: dict) -> FallConfig:
    section = _section(raw, "fall")
    d = FallConfig()
    return FallConfig(
        enable=_flag(section, "enable", "on") == "on",
        aspect_ratio=_float(section, "aspect_ratio", d.aspect_ratio),
        height_drop=_float(section, "height_drop", d.height_drop),
        descent_rate=_float(section, "descent_rate", d.descent_rate),
        descent_window=_float(section, "descent_window", d.descent_window),
        confirm_seconds=_float(section, "confirm_seconds", d.confirm_seconds),
        recover_seconds=_float(section, "recover_seconds", d.recover_seconds),
        min_box_height=_float(section, "min_box_height", d.min_box_height),
    )


def validate_fall(cfg: FallAppConfig) -> None:
    if not 0.0 <= cfg.track.iou_threshold <= 1.0:
        raise ValueError("tracking.iou_threshold must be in [0.0, 1.0]")
    if cfg.track.max_age < 0:
        raise ValueError("tracking.max_age must be >= 0")
    if cfg.track.min_hits < 1:
        raise ValueError("tracking.min_hits must be >= 1")
    if cfg.track.history_seconds < cfg.fall.descent_window:
        raise ValueError(
            f"tracking.history_seconds ({cfg.track.history_seconds}) is shorter than "
            f"fall.descent_window ({cfg.fall.descent_window}), so the descent test "
            f"would never see far enough back to fire."
        )
    if cfg.fall.aspect_ratio <= 0:
        raise ValueError("fall.aspect_ratio must be > 0")
    if not 0.0 < cfg.fall.height_drop <= 1.0:
        raise ValueError("fall.height_drop must be in (0.0, 1.0]")
    if cfg.fall.descent_rate < 0:
        raise ValueError("fall.descent_rate must be >= 0")
    if cfg.fall.descent_window <= 0:
        raise ValueError("fall.descent_window must be > 0")
    if cfg.fall.confirm_seconds < 0:
        raise ValueError("fall.confirm_seconds must be >= 0")
    if cfg.fall.recover_seconds < 0:
        raise ValueError("fall.recover_seconds must be >= 0")
    if not 0.0 <= cfg.fall.min_box_height < 1.0:
        raise ValueError("fall.min_box_height must be in [0.0, 1.0)")


def describe_fall(cfg: FallAppConfig) -> str:
    watched = ", ".join(cfg.track.classes) if cfg.track.classes else "every class"
    if not cfg.fall.enable:
        return f"fall: off, tracking {watched} only"
    return (
        f"fall: watching {watched} | aspect>={cfg.fall.aspect_ratio} "
        f"height<={cfg.fall.height_drop:.0%} descent>={cfg.fall.descent_rate:.0%}/s "
        f"| confirm={cfg.fall.confirm_seconds}s recover={cfg.fall.recover_seconds}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tracking
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Track:
    """One person followed across frames, with just enough history to judge them.

    Attributes:
        track_id: Stable id, assigned once and never reused within a run.
        box: Most recent detection, in source-image pixels.
        hits: How many frames this track has been matched on.
        misses: Consecutive frames with no match. Drops the track past a limit.
        history: Recent ``(timestamp_s, centre_y, height, aspect)`` samples,
            oldest first, trimmed to the window the fall rules need.
        upright_height: Rolling reference height from frames where the person
            looked upright. A fallen box is short as well as wide, but only
            relative to how tall *that* person was, which is why this is
            per-track rather than a constant.
        state: One of ``upright``, ``falling``, ``fallen`` or ``recovering``.
        state_since: Timestamp the current state began.
        reported_at: When this track was last reported fallen, or 0.0.
    """

    track_id: int
    box: dict
    # 0, not 1: the frame that creates a track also runs it through _advance,
    # so counting it here too would make min_hits mean one frame fewer than it
    # says.
    hits: int = 0
    misses: int = 0
    history: list = field(default_factory=list)
    upright_height: float = 0.0
    state: str = UPRIGHT
    state_since: float = 0.0
    reported_at: float = 0.0

    @property
    def width(self) -> float:
        return max(1.0, self.box["x2"] - self.box["x1"])

    @property
    def height(self) -> float:
        return max(1.0, self.box["y2"] - self.box["y1"])

    @property
    def aspect(self) -> float:
        """Width over height. Standing is well under 1; lying down is over it."""
        return self.width / self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box["x1"] + self.box["x2"]) / 2.0,
                (self.box["y1"] + self.box["y2"]) / 2.0)


def box_iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU tracker with stable ids.

    Deliberately simple. It has no motion model, so it will swap ids when two
    people cross while overlapping heavily. That costs a duplicate report at
    worst, which is the right way round for a safety feature: a Kalman filter
    would be more correct and considerably more to get wrong.

    Attributes:
        cfg: Tracking settings.
        tracks: Live tracks.
    """

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[dict], now: float) -> list[Track]:
        """Associate this frame's boxes with existing tracks.

        Args:
            boxes: Detections for this frame, already filtered to the classes
                that count as people.
            now: Monotonic timestamp in seconds.

        Returns:
            Every live track seen often enough to be trusted.
        """
        pairs = sorted(
            (
                (box_iou(t.box, b), ti, bi)
                for ti, t in enumerate(self.tracks)
                for bi, b in enumerate(boxes)
            ),
            key=lambda p: p[0],
            reverse=True,
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.cfg.iou_threshold:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self._advance(self.tracks[ti], boxes[bi], now)

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.misses += 1

        for bi, box in enumerate(boxes):
            if bi not in used_b:
                track = Track(track_id=self._next_id, box=box, state_since=now)
                self._next_id += 1
                self._advance(track, box, now)
                self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_age]
        return [t for t in self.tracks if t.hits >= self.cfg.min_hits and t.misses == 0]

    def _advance(self, track: Track, box: dict, now: float) -> None:
        track.box = box
        track.misses = 0
        track.hits += 1
        _, cy = track.centre
        track.history.append((now, cy, track.height, track.aspect))
        # Keep only what the rules can still look at, so a long run does not
        # accumulate a per-track list the length of the video.
        cutoff = now - self.cfg.history_seconds
        while len(track.history) > 2 and track.history[0][0] < cutoff:
            track.history.pop(0)
        if track.aspect <= self.cfg.upright_aspect:
            # An exponential rather than a max: someone briefly clipped by the
            # frame edge should not raise the bar for the rest of the run.
            track.upright_height = (
                track.height
                if track.upright_height <= 0
                else 0.9 * track.upright_height + 0.1 * track.height
            )


# ─────────────────────────────────────────────────────────────────────────────
# Fall rules
# ─────────────────────────────────────────────────────────────────────────────


def descent_rate(track: Track, window: float) -> float:
    """Downward speed of the box centre, in pixels per second.

    Args:
        track: The track to measure.
        window: How far back to look, in seconds.

    Returns:
        Pixels per second, positive downwards. 0.0 when there is too little
        history to say.
    """
    if len(track.history) < 2:
        return 0.0
    now = track.history[-1][0]
    oldest = track.history[0]
    for sample in track.history:
        if now - sample[0] <= window:
            oldest = sample
            break
    dt = now - oldest[0]
    if dt <= 1e-3:
        return 0.0
    return (track.history[-1][1] - oldest[1]) / dt


def fall_signals(track: Track, fall: FallConfig, frame_h: int) -> dict:
    """Evaluate all three signals for one track, for judging and for reporting."""
    collapsed = (
        track.upright_height > 0
        and track.height <= track.upright_height * fall.height_drop
    )
    rate = descent_rate(track, fall.descent_window)
    return {
        "aspect": track.aspect >= fall.aspect_ratio,
        "collapse": bool(collapsed),
        "descent": rate >= fall.descent_rate * frame_h,
        "aspect_value": round(track.aspect, 2),
        "descent_value": round(rate, 1),
    }


def looks_fallen(track: Track, fall: FallConfig, frame_h: int) -> bool:
    """Whether this track currently satisfies any of the fall signals."""
    s = fall_signals(track, fall, frame_h)
    return s["aspect"] or s["collapse"] or s["descent"]


def update_fall_states(tracks: list[Track], fall: FallConfig, frame_h: int,
                       now: float) -> list[Track]:
    """Advance every track's state machine and return the ones that just fell.

    The machine is::

        upright ──looks_fallen──> falling ──held confirm_seconds──> fallen
           ^                         |                                 |
           |                     recovered                         recovered
           |                         v                                 v
           └──── held recover_seconds ──── recovering <────────────────┘

    Args:
        tracks: Live, confirmed tracks.
        fall: Fall rule settings.
        frame_h: Frame height, so descent thresholds stay resolution independent.
        now: Monotonic timestamp in seconds.

    Returns:
        Tracks that transitioned into ``fallen`` on this frame.
    """
    newly = []
    for track in tracks:
        down = looks_fallen(track, fall, frame_h)
        if track.state in (UPRIGHT, RECOVERING):
            if down:
                track.state, track.state_since = FALLING, now
            elif (
                track.state == RECOVERING
                and now - track.state_since >= fall.recover_seconds
            ):
                track.state, track.state_since = UPRIGHT, now
        elif track.state == FALLING:
            if not down:
                track.state, track.state_since = RECOVERING, now
            elif now - track.state_since >= fall.confirm_seconds:
                track.state, track.state_since = FALLEN, now
                newly.append(track)
        elif track.state == FALLEN:
            if not down:
                track.state, track.state_since = RECOVERING, now
    return newly


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────


def fall_class_id(labels: list[str], tracked: object) -> int:
    """The class id a fallen box is drawn as.

    FALL is not a class the model knows, so it is appended past the end of the
    model's own names. That alone would give it whichever palette entry the
    cycle happens to land on -- with 80 COCO classes and four colours, exactly
    the one `person` already uses, so a fallen box would come out the same
    colour as the person standing next to it.

    Nudged past that instead: the first id beyond the labels whose colour no
    tracked class is already using. There are four colours, so this finds one
    unless every single one is spoken for, and then it gives up and takes the
    first -- the caption still says FALL.
    """
    used = {class_color(int(class_id)) for class_id in (tracked or ())}
    base = len(labels)
    for offset in range(len(CLASS_COLORS)):
        if class_color(base + offset) not in used:
            return base + offset
    return base


def overlay_boxes(tracks: list[Track], labels: list[str],
                  tracked: object = None) -> tuple[list[dict], list[str]]:
    """Tracks as plain detection boxes, with a fallen one relabelled to FALL.

    This is the whole of what fall detection does to a frame. There is no
    fall-specific drawing: the boxes go through ``draw_boxes`` exactly as
    ``detect``'s do, so the palette, the captions, the centre dots and the
    ordering are the same code and cannot drift apart.

    Args:
        tracks: Live tracks with their fall state already resolved.
        labels: The model's class names.
        tracked: Class ids that can fall, used to keep FALL's colour off them.

    Returns:
        ``(boxes, labels)`` ready for :func:`sima_vision.draw.draw_boxes`. The
        labels are padded so FALL lands on the id that was chosen for it; the
        padding is never looked up.
    """
    fall_id = fall_class_id(labels, tracked)
    names = [*labels] + [""] * (fall_id - len(labels)) + [FALL_CLASS]
    boxes = []
    for track in tracks:
        box = dict(track.box)
        if track.state == FALLEN:
            box["class_id"] = fall_id
        boxes.append(box)
    return boxes, names


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline and run loop
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FallPipeline(Pipeline):
    """Pipeline plus the tracker and the fall tally.

    Attributes:
        tracker: The :class:`Tracker` following people across frames.
        fall_class_ids: Class ids that can fall, or None for every class.
        falls: Confirmed falls so far this run.
    """

    tracker: object = None
    fall_class_ids: object = None
    falls: int = 0

def person_boxes(cfg: FallAppConfig, pipeline: FallPipeline, boxes: list[dict],
                 frame_h: int) -> list[dict]:
    """Keep only the classes that can fall, and only boxes big enough to judge."""
    floor = cfg.fall.min_box_height * frame_h
    kept = []
    for box in boxes:
        if pipeline.fall_class_ids is not None:
            if int(box["class_id"]) not in pipeline.fall_class_ids:
                continue
        if (box["y2"] - box["y1"]) < floor:
            continue
        kept.append(box)
    return kept


class FallRuntime(TaskRuntime):
    output_label = "detector_output"
    unit = "people"

    def decode(self, pipeline: FallPipeline, cfg: FallAppConfig, sample, index: int):
        # The result field, not the whole bundle -- see DetectRuntime.decode.
        payload, _ = extract_bbox_payload(joined_field(sample, "detections", 1))
        boxes = parse_boxes(payload, pipeline.frame_w, pipeline.frame_h, cfg.max_detections)
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        stamp_pts = getattr(sample, "pts_ns", -1)
        # `boxes` and `frame` are copies, so give the decoder its buffer back
        # before anything else. See FrameStamp for why.
        sample = None

        # Track, then judge. Both need a clock, and the source's own timestamps
        # are the honest one: with overflow_policy block the run is slower than
        # realtime, so wall-clock seconds would make every descent look slow.
        # index is 1-based; the elapsed time before the first frame is 0.
        now = (
            stamp_pts / 1e9 if stamp_pts >= 0
            else (index - 1) / float(pipeline.fps or 25)
        )
        self.check_geometry(pipeline, frame)
        people = person_boxes(cfg, pipeline, boxes, pipeline.frame_h)
        tracks = pipeline.tracker.update(people, now)
        fallen_now = (
            update_fall_states(tracks, cfg.fall, pipeline.frame_h, now)
            if cfg.fall.enable else []
        )
        self.report_falls(pipeline, cfg, fallen_now, index, now)
        return frame, tracks, 0.0

    def report_falls(self, pipeline: FallPipeline, cfg: FallAppConfig,
                     fallen_now: list[Track], index: int, now: float) -> None:
        """Count and log each track that just crossed into FALLEN.

        A line on the console and a number in the run summary. There is no
        sending here any more: the frame says FALL, the recording keeps it, and
        anything that wants to act on it can watch this output.
        """
        for track in fallen_now:
            pipeline.falls += 1
            signals = fall_signals(track, cfg.fall, pipeline.frame_h)
            console.report(
                f"[FALL] track #{track.track_id} at frame {index} "
                f"aspect={signals['aspect_value']} "
                f"descent={signals['descent_value']}px/s"
            )
            track.reported_at = now

    def render(self, cfg: FallAppConfig, pipeline: FallPipeline, frame, results, fps: float):
        """Draw once per frame and share the result between the video and JPEG sinks.

        Line for line what ``DetectRuntime.render`` does, which is the point.
        """
        annotated = frame.copy()
        boxes, labels = overlay_boxes(results, pipeline.labels, pipeline.fall_class_ids)
        draw_boxes(annotated, boxes, labels, cfg.draw)
        # FPS last, so nothing is ever drawn over it. See DetectRuntime.render.
        if cfg.video_hud:
            draw_fps(annotated, fps, cfg.draw)
        return annotated

    def summarise(self, pipeline: FallPipeline, processed: int) -> list[str]:
        return [f"falls={pipeline.falls}"]


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

# `detect`'s own settings, the same object rather than a copy of the numbers.
# The two frames are meant to be indistinguishable apart from the word FALL,
# and a second DrawConfig here is how they would quietly stop being.


class FallTask(Task):
    name = "fall"
    help = "Detect people and relabel the box when one of them goes down"
    config_class = FallAppConfig
    graph_name = "yolo_detector"
    result_label = "detections"
    output_label = "detector_output"
    defaults = TaskDefaults(
        task="fall",
        family="yolo26",
        save_dir="frames",
        video_path="falls.mp4",
        draw=DETECT_DRAW,
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--classes", dest="tracking.classes", nargs="+", metavar="CLASS",
            help="Class names or ids that can fall. Default: person.",
        )
        parser.add_argument(
            "--confirm", dest="fall.confirm_seconds", type=float, metavar="S",
            help="How long a fall signal must hold before the box is relabelled. "
                 "Default 1.5.",
        )
        parser.add_argument(
            "--no-fall", dest="fall.enable", action="store_const", const=False,
            help="Track people without judging falls, which is how you tune tracking first.",
        )
    def extra_sections(self, raw: dict) -> dict:
        return {
            "track": load_track_config(raw),
            "fall": load_fall_config(raw),
        }

    def validate(self, cfg: FallAppConfig) -> None:
        super().validate(cfg)
        validate_fall(cfg)

    def describe(self, cfg: FallAppConfig) -> list[str]:
        # Resolving the classes here means a typo in tracking.classes is caught
        # off-board rather than on the DevKit.
        ids = resolve_classes(
            cfg.track.classes, load_labels(cfg.labels_path),
            "tracking.classes", cfg.labels_path,
        )
        lines = [describe_fall(cfg)]
        if ids is not None:
            lines.append(f"tracked class ids: {sorted(ids)}")
        return lines

    def make_pipeline(self, cfg: FallAppConfig, labels: list[str]) -> FallPipeline:
        return FallPipeline(
            labels=labels,
            tracker=Tracker(cfg.track),
            fall_class_ids=resolve_classes(
                cfg.track.classes, labels, "tracking.classes", cfg.labels_path
            ),
        )

    def prepare(self, cfg: FallAppConfig, pipeline: FallPipeline, step) -> None:
        step.detail(describe_fall(cfg))

    def runtime(self, cfg, pipeline) -> TaskRuntime:
        return FallRuntime()

