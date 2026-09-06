"""Config loading, validation and CLI overrides.

Everything here runs before pyneat is imported, so a bad config fails in under a
second on a laptop instead of part-way through a graph build on the board. That
is the whole reason the enum tables in :mod:`sima_vision.runtime` are strings.

The layering is:

1. dataclass defaults -- a complete, runnable configuration with no file at all
2. ``config.yaml`` -- whatever the file sets, on top of those
3. CLI flags -- ``--source``, ``--conf`` and friends, on top of that

Each task extends :class:`BaseConfig` with its own sections and its own extra
validation, and supplies a :class:`TaskDefaults` for the handful of base keys
whose sensible default differs per task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .assets import default_model_path, default_source_uri, is_url
from .runtime import (
    AUTO_FLAGS,
    COLOR_FORMATS,
    FAMILY_DECODE_TOKENS,
    NORMALIZE_PRESETS,
    RESIZE_MODES,
    SCALING_TYPES,
)

#: Everything shipped inside the wheel.
PACKAGE_ROOT = Path(__file__).resolve().parent
#: Labels used when nothing else resolves.
PACKAGED_LABELS = PACKAGE_ROOT / "data" / "coco_labels.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Scalar readers
# ─────────────────────────────────────────────────────────────────────────────


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config section `{key}` must be a mapping")
    return value


def _str(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    return default if value is None else str(value)


def _int(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config key `{key}` must be an integer")
    return int(value)


def _float(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config key `{key}` must be numeric")
    return float(value)


def _bool(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"config key `{key}` must be true or false")
    return value


def _flag(raw: dict, key: str, default: str) -> str:
    """
    Read a tri-state auto/on/off knob. YAML 1.1 resolves
    bare `on`/`off`/`yes`/`no` to booleans, so `enable: on`
    reaches us as True. Fold those back onto the token vocabulary.
    """
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    token = str(value).lower()
    if token in {"yes", "true"}:
        return "on"
    if token in {"no", "false"}:
        return "off"
    if token not in AUTO_FLAGS:
        raise ValueError(f"config key `{key}` must be auto, on or off (got {value!r})")
    return token


def _color(raw: dict, key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Read a ``[B, G, R]`` colour. OpenCV is BGR, so the order is not a typo."""
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"config key `{key}` must be a list of three numbers, [B, G, R]")
    channels = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ValueError(f"config key `{key}` must contain numbers")
        if not 0 <= channel <= 255:
            raise ValueError(f"config key `{key}` channels must be between 0 and 255")
        channels.append(int(channel))
    return (channels[0], channels[1], channels[2])


def _triple(raw: dict, key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"config key `{key}` must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _int_list(raw: dict, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config key `{key}` must be a list of integers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"config key `{key}` must contain integers")
        out.append(int(item))
    return tuple(out)


def _str_list(raw: dict, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a list of class names or ids. A bare scalar is accepted as one item."""
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (str(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config key `{key}` must be a list of class names or ids")
    return tuple(str(item) for item in value)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing intent, mirroring pyneat.ModelOptions.preprocess.

    This is what the application asks for. The route planner resolves it against
    the model archive's MPK contract and compiles the matching graph, so a value
    here is a request rather than a guarantee.

    Attributes:
        kind: Input kind. One of ``image``, ``tensor`` or ``auto``.
        enable: Master switch. One of ``on``, ``off`` or ``auto``.
        input_format: Colour format the source hands to Preproc, such as
            ``NV12`` for hardware-decoded video or ``BGR`` for cv2 images.
        output_format: Model input colour space, or ``auto`` to take it from the
            model contract.
        input_max_width: Preproc buffer width capacity. 0 uses the probed size.
        input_max_height: Preproc buffer height capacity. 0 uses the probed size.
        resize_enable: Whether to resize. ``auto``, ``on`` or ``off``.
        resize_width: Target width. 0 infers it from the model contract.
        resize_height: Target height. 0 infers it from the model contract.
        resize_mode: One of ``letterbox``, ``stretch`` or ``crop``.
        pad_value: Fill value for letterbox padding. 114 is the YOLO default.
        scaling_type: Interpolation token, for example ``BILINEAR``.
        normalize_enable: Whether to normalize. ``auto``, ``on`` or ``off``.
        normalize_preset: One of ``coco_yolo``, ``imagenet`` or ``none``.
        mean: Per-channel mean, read only when the preset is ``none``.
        stddev: Per-channel divisor, read only when the preset is ``none``.
        quantize_enable: Quantization control. Normally left on ``auto``.
        quantize_zero_point: Explicit zero point, applied only when enabled.
        quantize_scale: Explicit scale. 0.0 uses the model's calibration.
        tessellate_enable: MLA tile-layout control. Normally left on ``auto``.
        tessellate_slice_shape: Tile geometry override. Empty uses the contract.
    """

    kind: str = "image"
    enable: str = "on"
    input_format: str = "NV12"
    output_format: str = "auto"
    input_max_width: int = 0
    input_max_height: int = 0

    resize_enable: str = "on"
    resize_width: int = 0
    resize_height: int = 0
    resize_mode: str = "letterbox"
    pad_value: int = 114
    scaling_type: str = "BILINEAR"

    normalize_enable: str = "on"
    normalize_preset: str = "coco_yolo"
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stddev: tuple[float, float, float] = (1.0, 1.0, 1.0)

    quantize_enable: str = "auto"
    quantize_zero_point: int = 0
    quantize_scale: float = 0.0

    tessellate_enable: str = "auto"
    tessellate_slice_shape: tuple[int, ...] = ()


def load_preprocess_config(raw: dict) -> PreprocessConfig:
    resize = _section(raw, "resize")
    normalize = _section(raw, "normalize")
    quantize = _section(raw, "quantize")
    tessellate = _section(raw, "tessellate")
    slice_shape = tessellate.get("slice_shape") or []

    return PreprocessConfig(
        kind=_str(raw, "kind", "image").lower(),
        enable=_flag(raw, "enable", "on"),
        input_format=_str(raw, "input_format", "NV12").upper(),
        output_format=_str(raw, "output_format", "auto").upper(),
        input_max_width=_int(raw, "input_max_width", 0),
        input_max_height=_int(raw, "input_max_height", 0),
        resize_enable=_flag(resize, "enable", "on"),
        resize_width=_int(resize, "width", 0),
        resize_height=_int(resize, "height", 0),
        resize_mode=_str(resize, "mode", "letterbox").lower(),
        pad_value=_int(resize, "pad_value", 114),
        scaling_type=_str(resize, "scaling_type", "BILINEAR").upper(),
        normalize_enable=_flag(normalize, "enable", "on"),
        normalize_preset=_str(normalize, "preset", "coco_yolo").lower(),
        mean=_triple(normalize, "mean", (0.0, 0.0, 0.0)),
        stddev=_triple(normalize, "stddev", (1.0, 1.0, 1.0)),
        quantize_enable=_flag(quantize, "enable", "auto"),
        quantize_zero_point=_int(quantize, "zero_point", 0),
        quantize_scale=_float(quantize, "scale", 0.0),
        tessellate_enable=_flag(tessellate, "enable", "auto"),
        tessellate_slice_shape=tuple(int(v) for v in slice_shape),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DrawConfig:
    """Overlay appearance, straight from the ``visualization`` config section.

    This is the union of what all three tasks draw. A task simply ignores the
    fields it has no use for -- ``detect`` never reads ``mask_alpha``, ``segment``
    never reads ``banner`` -- which is cheaper than three near-identical
    dataclasses and means one ``visualization`` block documents them all. The
    few defaults that genuinely differ per task are supplied through
    :attr:`TaskDefaults.draw`.

    Every pixel size here is expressed for a 1080p frame. When ``auto_scale``
    is on they are multiplied by ``min(frame_height, frame_width) /
    reference_height``, so 4K does not get hairlines and 480p does not get
    slabs. Turn ``auto_scale`` off to use the numbers literally.

    Attributes:
        box_thickness: Detection rectangle outline weight, in pixels.
        text_scale: OpenCV font scale for captions.
        text_thickness: Caption stroke weight, in pixels.
        text_padding: Gap between caption text and the edge of its band.
        centre_dot: Whether to mark the centre of each box. ``detect``, ``fall``.
        centre_dot_radius: Radius of that marker, in pixels.
        show_labels: Whether the caption carries the class name.
        show_scores: Whether the caption carries the confidence.
        score_decimals: Decimal places for the confidence, so 2 gives ``0.57``.
        text_color: Caption text colour, BGR.
        mask_alpha: ``segment``: strength of the class-coloured tint painted
            over each instance, 0.0 to 1.0. 0 leaves the instance untouched,
            which is what you want when the blur alone should carry the effect.
        mask_outline: ``segment``: whether to trace the mask edge.
        outline_thickness: ``segment``: weight of that trace, in pixels.
        show_boxes: ``segment``: whether to draw the bounding rectangle as well.
            Off by default: the mask already shows the extent.
        show_track_ids: ``fall``: whether captions carry the track id.
        banner: ``fall``: whether to draw the full-width alert strip.
        banner_text_scale: Banner font scale. 0 follows ``text_scale``.
        banner_text_thickness: Banner stroke. 0 follows ``text_thickness``.
        banner_padding: Gap between banner text and the strip edge.
        banner_alpha: Strip opacity, 0.0 to 1.0.
        banner_bg_color: Strip fill, BGR.
        banner_text_color: Strip text, BGR.
        hud_text_color: Frame-rate badge text colour, BGR.
        hud_bg_color: Frame-rate badge fill colour, BGR. Purple by default,
            which reads as an overlay rather than as part of the footage the
            way a black block did.
        hud_text_scale: Badge font scale. 0 follows ``text_scale``. Set a
            little above it by default: the badge is glanced at while the
            video plays, not read, so it wants to be larger than a caption.
        hud_text_thickness: Badge stroke weight. 0 follows ``text_thickness``.
        hud_padding: Gap between badge text and badge edge, on every side. 0
            follows ``text_padding``. This is what sets the badge size when no
            minimum is given.
        hud_padding_x: Left/right gap. 0 follows ``hud_padding``.
        hud_padding_y: Top/bottom gap. 0 follows ``hud_padding``.
        hud_margin_x: Gap between the badge and the left frame edge. 0 follows
            the resolved horizontal padding.
        hud_margin_y: Gap between the badge and the top frame edge. 0 follows
            the resolved vertical padding.
        hud_fps_decimals: Decimal places on the frame rate. 0 gives ``FPS: 25``,
            1 gives ``FPS: 24.8``.
        hud_min_width: Floor on badge width in pixels. 0 fits the text.
        hud_min_height: Floor on badge height in pixels. 0 fits the text.
        auto_scale: Whether sizes scale with frame height.
        reference_height: The frame height the sizes above are tuned for.
    """

    box_thickness: int = 3
    text_scale: float = 1.0
    text_thickness: int = 2
    text_padding: int = 10
    centre_dot: bool = True
    centre_dot_radius: int = 7
    show_labels: bool = True
    show_scores: bool = True
    score_decimals: int = 2
    text_color: tuple[int, int, int] = (255, 255, 255)

    mask_alpha: float = 0.35
    mask_outline: bool = True
    outline_thickness: int = 3
    show_boxes: bool = False

    show_track_ids: bool = True
    banner: bool = True
    banner_text_scale: float = 0.0
    banner_text_thickness: int = 0
    banner_padding: int = 18
    banner_alpha: float = 0.75
    banner_bg_color: tuple[int, int, int] = (56, 56, 255)
    banner_text_color: tuple[int, int, int] = (255, 255, 255)

    hud_text_color: tuple[int, int, int] = (255, 255, 255)
    hud_bg_color: tuple[int, int, int] = (128, 0, 128)
    hud_text_scale: float = 1.3
    hud_text_thickness: int = 0
    hud_padding: int = 0
    hud_padding_x: int = 0
    hud_padding_y: int = 0
    hud_margin_x: int = 0
    hud_margin_y: int = 0
    hud_fps_decimals: int = 0
    hud_min_width: int = 0
    hud_min_height: int = 0

    auto_scale: bool = True
    reference_height: float = 1080.0


def load_draw_config(raw: dict, default: DrawConfig | None = None) -> DrawConfig:
    """Build a :class:`DrawConfig` from the ``visualization`` config section.

    Args:
        raw: The whole parsed config document.
        default: Per-task defaults. Every key is optional and falls back to
            this, so an absent section is valid.

    Returns:
        A populated :class:`DrawConfig`.
    """
    section = _section(raw, "visualization")
    hud = _section(section, "hud")
    banner = _section(section, "banner")
    default = default or DrawConfig()
    return DrawConfig(
        box_thickness=_int(section, "box_thickness", default.box_thickness),
        text_scale=_float(section, "text_scale", default.text_scale),
        text_thickness=_int(section, "text_thickness", default.text_thickness),
        text_padding=_int(section, "text_padding", default.text_padding),
        centre_dot=_flag(section, "centre_dot", "on" if default.centre_dot else "off") == "on",
        centre_dot_radius=_int(section, "centre_dot_radius", default.centre_dot_radius),
        show_labels=_flag(section, "show_labels", "on" if default.show_labels else "off") == "on",
        show_scores=_flag(section, "show_scores", "on" if default.show_scores else "off") == "on",
        score_decimals=_int(section, "score_decimals", default.score_decimals),
        text_color=_color(section, "text_color", default.text_color),
        mask_alpha=_float(section, "mask_alpha", default.mask_alpha),
        mask_outline=(
            _flag(section, "mask_outline", "on" if default.mask_outline else "off") == "on"
        ),
        outline_thickness=_int(section, "outline_thickness", default.outline_thickness),
        show_boxes=_flag(section, "show_boxes", "on" if default.show_boxes else "off") == "on",
        show_track_ids=(
            _flag(section, "show_track_ids", "on" if default.show_track_ids else "off") == "on"
        ),
        banner=_flag(banner, "enable", "on" if default.banner else "off") == "on",
        banner_text_scale=_float(banner, "text_scale", default.banner_text_scale),
        banner_text_thickness=_int(banner, "text_thickness", default.banner_text_thickness),
        banner_padding=_int(banner, "padding", default.banner_padding),
        banner_alpha=_float(banner, "alpha", default.banner_alpha),
        banner_bg_color=_color(banner, "bg_color", default.banner_bg_color),
        banner_text_color=_color(banner, "text_color", default.banner_text_color),
        hud_text_color=_color(hud, "text_color", default.hud_text_color),
        hud_bg_color=_color(hud, "bg_color", default.hud_bg_color),
        hud_text_scale=_float(hud, "text_scale", default.hud_text_scale),
        hud_text_thickness=_int(hud, "text_thickness", default.hud_text_thickness),
        hud_padding=_int(hud, "padding", default.hud_padding),
        hud_padding_x=_int(hud, "padding_x", default.hud_padding_x),
        hud_padding_y=_int(hud, "padding_y", default.hud_padding_y),
        hud_margin_x=_int(hud, "margin_x", default.hud_margin_x),
        hud_margin_y=_int(hud, "margin_y", default.hud_margin_y),
        hud_fps_decimals=_int(hud, "fps_decimals", default.hud_fps_decimals),
        hud_min_width=_int(hud, "min_width", default.hud_min_width),
        hud_min_height=_int(hud, "min_height", default.hud_min_height),
        auto_scale=_flag(section, "auto_scale", "on" if default.auto_scale else "off") == "on",
        reference_height=_float(section, "reference_height", default.reference_height),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Base application config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskDefaults:
    """The handful of base config defaults that differ per task.

    Attributes:
        family: Default ``model.family``. Detect heads for ``detect`` and
            ``fall``, a segment head for ``segment``.
        task: The task these belong to, used to look up its default model and
            clip in :data:`sima_vision.assets.CATALOGUE`. Empty leaves
            ``model.path`` and ``source.uri`` unset, which is what a task with
            no entry there wants.
        run_preset: Default ``runtime.preset``.
        overflow_policy: Default ``runtime.overflow_policy``.
        save_dir: Default ``output.save.dir``.
        video_path: Default ``output.video.path``.
        insight_enable: Default ``output.insight.enable``.
        draw: Per-task :class:`DrawConfig` defaults.
    """

    family: str = "yolo26"
    task: str = ""
    run_preset: str = "auto"
    overflow_policy: str = "auto"
    save_dir: str = "frames"
    video_path: str = "output.mp4"
    insight_enable: bool = False
    draw: DrawConfig = DrawConfig()


@dataclass(frozen=True)
class BaseConfig:
    """Everything the three tasks configure identically.

    Built from ``config.yaml`` and validated by :func:`validate_base` before
    anything touches the hardware. Each task subclasses this and adds its own
    sections; nothing here knows which task it belongs to.

    Attributes:
        model_path: Path to the compiled model archive on the DevKit. Unset
            means this task's default in ``assets/models/``, which
            :func:`sima_vision.assets.ensure_model` downloads on the first run.
        labels_path: Path to the newline-separated class label file.
        family: Head token, mapped by ``FAMILY_DECODE_TOKENS``.
        decode_type_option: Head packing override. Normally ``auto``.
        num_classes: Class count override. 0 reads it from the archive.
        source_type: One of ``video``, ``rtsp`` or ``usb``.
        source_uri: File path, ``http(s)`` URL or stream URL, as seen on the
            DevKit. Unset with ``source_type=video`` means this task's sample
            clip in ``assets/videos/``, downloaded on the first run; unset with
            ``usb`` means the DevKit camera.
        source_fps: Frame rate. 0 probes it from the source.
        source_width: Frame width. 0 probes it.
        source_height: Frame height. 0 probes it.
        rtsp_codec: ``h264`` or ``mjpeg``.
        rtsp_tcp: Whether to use TCP transport for RTSP.
        rtsp_latency_ms: Jitter buffer latency.
        usb_camera_name: libcamera device name, or empty for the default.
        usb_format: Pixel format requested from the camera.
        preprocess: Preprocessing intent. See :class:`PreprocessConfig`.
        score_threshold: Minimum detection confidence.
        nms_iou: Non-max suppression IoU threshold.
        max_detections: Top-K cap per frame.
        frames: Frame limit. 0 runs until interrupted.
        pull_timeout_ms: How long to wait for a frame before giving up.
        queue_depth: Depth of the Neat runtime's own queues. Every slot can
            park a decoded frame, so this counts against the decoder's pool
            along with ``output_buffers``: raise it and a stalling run stalls
            sooner. It does not change the ``max-buffers`` and ``num-buffers``
            in the printed pipeline, which pyneat fixes at 4.
        sink_queue_depth: How many finished frames may wait for the sink
            thread. These are numpy copies in host memory and hold no decoder
            buffer, so depth here is the cheap kind: it lets the pull loop keep
            draining the source instead of blocking on a slow recording. About
            6 MB per slot at 1080p.

            This is the floor. For a file source ``sink_queue_mb`` raises it
            towards holding the whole clip; see :func:`sink_depth_for`.
        segment_frames: Frames per piece when a clip is too long for one
            decode. The SiMa decoder stops part-way through a long clip -- 190
            to 202 frames of a 379 frame one, whatever the sinks, the pool or
            the container -- and building the decode again gets another run at
            it, so a long clip is cut at its keyframes and the pieces are
            decoded one after another into a single recording. 0 disables it
            and runs the clip whole. A clip whose keyframes are further apart
            than this cannot be cut usefully and is run whole regardless.
        decoder_buffers: Buffers to ask the hardware decoder for, through
            ``SimaDecodeOptions.num_buffers``. 0 sizes it from the stream: the
            reference frames its SPS declares, the picture being decoded, what
            the source appsink parks, and a little slack. A positive number
            pins it. A negative number leaves pyneat's own -1 in place, which
            is what the app did before and which lets the daemon pick 8 for
            1080p regardless of what the stream needs.
        decoder_pool: Decoded frames the hardware decoder's pool holds. The
            boot log prints the real number as ``BufferNum=`` when the decoder
            finds the stream's resolution, and it is per-resolution, so 8 is
            what a 1080p run reports rather than a constant. Used only to say
            whether a stream's own buffering will fit; set it to what your
            board prints if it differs.
        sink_queue_mb: Host memory the sink backlog may use, for a *file*
            source only. The recording is the slow part -- software-encoding
            1080p costs several times the frame interval on this board -- and
            the pull loop must not wait for it, because a loop that is not
            pulling is a decoder that starves and never comes back. So for a
            clip of known length the queue grows to hold the whole thing: the
            loop drains the source at full speed in seconds and the sink thread
            finishes the backlog afterwards, which the run already waits for.
            Costs about 6 MB a frame at 1080p, capped by the clip's own length.
            0 disables the growth and leaves ``sink_queue_depth`` alone.
        output_buffers: Buffers each public output may hold. Every one of them
            is a frame checked out of the hardware decoder's pool, that pool is
            small (the boot log prints ``BufferNum=8``), and there are two
            outputs. Counts against the same budget as the decoded path in
            front of the source appsink; see
            :func:`sima_vision.media.make_elementary_h264_source`.
        run_preset: One of ``auto``, ``realtime``, ``balanced`` or ``reliable``.
        overflow_policy: ``auto``, ``keep_latest``, ``block`` or ``drop_incoming``.
        profile: Whether to print per-stage timings.
        profile_interval: Frames per profiling window.
        save_enable: Whether to write annotated stills.
        save_dir: Directory for stills.
        save_every: Write every Nth frame. 0 disables.
        save_overlay: Whether stills carry the overlay.
        save_format: ``jpg`` or ``png``.
        video_enable: Whether to write an annotated video on the DevKit.
        video_path: Output video path.
        video_codec: Four-character FourCC, with an MJPG fallback.
        video_fps: Output frame rate. 0 matches the source.
        video_hud: Whether to draw the frame-rate badge.
        insight_enable: Whether to stream to Neat Insight.
        insight_annotated: Whether Insight receives the annotated frame. False
            sends the raw frame and lets Insight draw its own overlay.
        insight_host: Insight address as the DevKit sees it.
        insight_channel: Channel offset added to both port bases.
        video_port_base: First UDP video port.
        metadata_port_base: First UDP metadata port.
        bitrate_kbps: H.264 encoder bitrate for the Insight feed.
        draw: Overlay appearance. See :class:`DrawConfig`.
        config_path: The file this came from, or None when it is all defaults.
            Reported by ``--validate`` and used to resolve relative asset paths.
    """

    model_path: str = ""
    labels_path: Path = PACKAGED_LABELS
    family: str = "yolo26"
    decode_type_option: str = "auto"
    num_classes: int = 0

    source_type: str = "video"
    source_uri: str = ""
    source_fps: int = 0
    source_width: int = 0
    source_height: int = 0
    rtsp_codec: str = "h264"
    rtsp_tcp: bool = True
    rtsp_latency_ms: int = 100
    usb_camera_name: str = ""
    usb_format: str = "NV12"

    preprocess: PreprocessConfig = PreprocessConfig()

    score_threshold: float = 0.30
    nms_iou: float = 0.60
    max_detections: int = 50

    frames: int = 0
    pull_timeout_ms: int = 20000
    queue_depth: int = 1
    sink_queue_depth: int = 12
    sink_queue_mb: int = 1024
    decoder_pool: int = 8
    decoder_buffers: int = 0
    segment_frames: int = 150
    output_buffers: int = 1
    run_preset: str = "auto"
    overflow_policy: str = "auto"
    profile: bool = False
    profile_interval: int = 100

    save_enable: bool = True
    save_dir: str = "frames"
    save_every: int = 10
    save_overlay: bool = True
    save_format: str = "jpg"

    video_enable: bool = True
    video_path: str = "output.mp4"
    video_codec: str = "mp4v"
    video_fps: int = 0
    video_hud: bool = True

    insight_enable: bool = False
    insight_annotated: bool = True
    insight_host: str = "127.0.0.1"
    insight_channel: int = 0
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    bitrate_kbps: int = 2000

    draw: DrawConfig = DrawConfig()

    config_path: Path | None = None


def resolve_asset_path(value: str, config_path: Path | None) -> str:
    """Find a file the config points at, tolerating where the app was launched.

    Paths in ``config.yaml`` are written relative to the app directory, because
    that is where the DevKit launches from. A CLI is launched from anywhere, so
    try the literal path first -- which keeps every existing relative path
    working unchanged -- then the same path relative to the config file.

    A URL is not a path and is handed straight back; :mod:`sima_vision.assets`
    turns it into a local file at run time.

    Args:
        value: The raw config string.
        config_path: Path to the config file being loaded, or None.

    Returns:
        The first candidate that exists, or ``value`` unchanged so the caller
        still reports the name the user actually wrote.
    """
    if not value or config_path is None or is_url(value):
        return value
    given = Path(value)
    if given.is_absolute() or given.exists():
        return value
    beside = config_path.resolve().parent / value
    return str(beside) if beside.exists() else value


def resolve_labels_path(value: str, config_path: Path | None) -> Path:
    """Find the labels file, falling back to the copy inside the wheel.

    Validating a config off-board is normally done from the repo root rather
    than the app directory, and failing there would make ``--validate`` useless
    in exactly the place it is most useful. So try the literal path, then beside
    ``config.yaml``, then the packaged copy.
    """
    given = Path(value)
    candidates = [given]
    if config_path is not None:
        candidates.append(config_path.resolve().parent / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return PACKAGED_LABELS if PACKAGED_LABELS.is_file() else given


def load_base_config(raw: dict, path: Path | None, defaults: TaskDefaults) -> BaseConfig:
    """Read every shared section out of a parsed config document."""
    model = _section(raw, "model")
    source = _section(raw, "source")
    rtsp = _section(source, "rtsp")
    usb = _section(source, "usb")
    decode = _section(raw, "decode")
    runtime = _section(raw, "runtime")
    output = _section(raw, "output")
    save = _section(output, "save")
    video = _section(output, "video")
    insight = _section(output, "insight")
    source_type = _str(source, "type", "video").lower()

    # An unset path is not an error any more: it means "the default for this
    # task", which `sima_vision.assets` downloads on the first run. Resolving it
    # here rather than at run time keeps `--validate` honest about what a run
    # would actually open, without it fetching anything.
    model_default = default_model_path(defaults.task) if defaults.task else ""
    clip_default = default_source_uri(defaults.task) if defaults.task else ""

    return BaseConfig(
        model_path=resolve_asset_path(_str(model, "path") or model_default, path),
        labels_path=resolve_labels_path(_str(model, "labels", str(PACKAGED_LABELS)), path),
        family=_str(model, "family", defaults.family).lower(),
        decode_type_option=_str(model, "decode_type_option", "auto").lower(),
        num_classes=_int(model, "num_classes", 0),
        source_type=source_type,
        # Only a file source gets a default. An empty `uri` on `usb` means the
        # DevKit camera, and filling in a clip there would be wrong.
        source_uri=(
            resolve_asset_path(_str(source, "uri") or clip_default, path)
            if source_type == "video"
            else _str(source, "uri")
        ),
        source_fps=_int(source, "fps", 0),
        source_width=_int(source, "width", 0),
        source_height=_int(source, "height", 0),
        rtsp_codec=_str(rtsp, "codec", "h264").lower(),
        rtsp_tcp=_bool(rtsp, "tcp", True),
        rtsp_latency_ms=_int(rtsp, "latency_ms", 100),
        usb_camera_name=_str(usb, "camera_name"),
        usb_format=_str(usb, "format", "NV12").upper(),
        preprocess=load_preprocess_config(_section(raw, "preprocess")),
        score_threshold=_float(decode, "score_threshold", 0.30),
        nms_iou=_float(decode, "nms_iou", 0.60),
        max_detections=_int(decode, "max_detections", 50),
        frames=_int(runtime, "frames", 0),
        pull_timeout_ms=_int(runtime, "pull_timeout_ms", 20000),
        queue_depth=_int(runtime, "queue_depth", 1),
        sink_queue_depth=_int(runtime, "sink_queue_depth", 12),
        sink_queue_mb=_int(runtime, "sink_queue_mb", 1024),
        decoder_pool=_int(runtime, "decoder_pool", 8),
        decoder_buffers=_int(runtime, "decoder_buffers", 0),
        segment_frames=_int(runtime, "segment_frames", 150),
        output_buffers=_int(runtime, "output_buffers", 1),
        run_preset=_str(runtime, "preset", defaults.run_preset).lower(),
        overflow_policy=_str(runtime, "overflow_policy", defaults.overflow_policy).lower(),
        profile=_bool(runtime, "profile", False),
        profile_interval=_int(runtime, "profile_interval", 100),
        save_enable=_bool(save, "enable", True),
        save_dir=_str(save, "dir", defaults.save_dir),
        save_every=_int(save, "every", 10),
        save_overlay=_bool(save, "overlay", True),
        save_format=_str(save, "format", "jpg").lower().lstrip("."),
        video_enable=_bool(video, "enable", True),
        video_path=_str(video, "path", defaults.video_path),
        video_codec=_str(video, "codec", "mp4v"),
        video_fps=_int(video, "fps", 0),
        video_hud=_bool(video, "hud", True),
        insight_enable=_bool(insight, "enable", defaults.insight_enable),
        insight_annotated=_bool(insight, "annotated", True),
        insight_host=_str(insight, "host", "127.0.0.1"),
        insight_channel=_int(insight, "channel", 0),
        video_port_base=_int(insight, "video_port_base", 9000),
        metadata_port_base=_int(insight, "metadata_port_base", 9100),
        bitrate_kbps=_int(insight, "bitrate_kbps", 2000),
        draw=load_draw_config(raw, defaults.draw),
        config_path=path,
    )


def validate_base(cfg: BaseConfig) -> None:
    """Check every shared key. Tasks call this, then add their own rules."""
    if not cfg.model_path:
        raise ValueError(
            "model.path must be set. Pass --model /path/to/model.tar.gz, or set "
            "model.path in a config file and point --config at it."
        )
    if cfg.family not in FAMILY_DECODE_TOKENS:
        raise ValueError(
            f"model.family `{cfg.family}` is not supported. "
            f"Choose one of: {', '.join(sorted(FAMILY_DECODE_TOKENS))}"
        )
    if cfg.source_type not in {"video", "rtsp", "usb"}:
        raise ValueError("source.type must be video, rtsp or usb")
    if cfg.source_type in {"video", "rtsp"} and not cfg.source_uri:
        raise ValueError(
            f"source.uri must be set for source.type={cfg.source_type}. "
            f"Pass --source, or set source.uri in a config file."
        )
    if cfg.rtsp_codec not in {"h264", "mjpeg"}:
        raise ValueError("source.rtsp.codec must be h264 or mjpeg")
    if cfg.preprocess.kind not in {"image", "tensor", "auto"}:
        raise ValueError("preprocess.kind must be image, tensor or auto")
    if cfg.preprocess.resize_mode not in RESIZE_MODES:
        raise ValueError(f"preprocess.resize.mode must be one of: {', '.join(RESIZE_MODES)}")
    if cfg.preprocess.scaling_type not in SCALING_TYPES:
        raise ValueError(
            f"preprocess.resize.scaling_type must be one of: {', '.join(sorted(SCALING_TYPES))}"
        )
    if cfg.preprocess.input_format not in COLOR_FORMATS:
        raise ValueError(f"preprocess.input_format must be one of: {', '.join(COLOR_FORMATS)}")
    if cfg.preprocess.output_format not in COLOR_FORMATS:
        raise ValueError(f"preprocess.output_format must be one of: {', '.join(COLOR_FORMATS)}")
    if cfg.preprocess.normalize_preset not in NORMALIZE_PRESETS:
        raise ValueError(
            f"preprocess.normalize.preset must be one of: {', '.join(NORMALIZE_PRESETS)}"
        )
    if not 0.0 <= cfg.score_threshold <= 1.0:
        raise ValueError("decode.score_threshold must be in [0.0, 1.0]")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("decode.nms_iou must be in [0.0, 1.0]")
    if cfg.max_detections < 0:
        raise ValueError("decode.max_detections must be >= 0")
    if cfg.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
    if cfg.output_buffers < 1:
        raise ValueError("runtime.output_buffers must be >= 1")
    if cfg.queue_depth < 1:
        raise ValueError("runtime.queue_depth must be >= 1")
    if cfg.sink_queue_depth < 1:
        raise ValueError("runtime.sink_queue_depth must be >= 1")
    if cfg.sink_queue_mb < 0:
        raise ValueError("runtime.sink_queue_mb must be >= 0")
    if cfg.decoder_pool < 1:
        raise ValueError("runtime.decoder_pool must be >= 1")
    if cfg.segment_frames < 0:
        raise ValueError("runtime.segment_frames must be >= 0")
    if cfg.pull_timeout_ms <= 0:
        raise ValueError("runtime.pull_timeout_ms must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save.every must be >= 0")
    if cfg.save_format not in {"jpg", "jpeg", "png"}:
        raise ValueError("output.save.format must be jpg or png")
    if cfg.video_enable and not cfg.video_path:
        raise ValueError("output.video.path must be set when video output is enabled")
    if len(cfg.video_codec) != 4:
        raise ValueError(
            f"output.video.codec must be a 4-character FourCC such as mp4v or MJPG, "
            f"got {cfg.video_codec!r}"
        )
    if cfg.video_fps < 0:
        raise ValueError("output.video.fps must be >= 0")
    if cfg.insight_enable and not cfg.insight_host:
        raise ValueError("output.insight.host must be set when insight is enabled")
    # Two senders on one port is not a warning-level mistake: the H.264 encoder
    # fails to configure, and because it shares the codec daemon with the
    # decoder feeding the source, the whole pipeline stalls a few frames in.
    # That looks like "the output video is 12 frames long", which is a long way
    # from the actual cause.
    if cfg.insight_enable and cfg.video_port_base == cfg.metadata_port_base:
        raise ValueError(
            f"output.insight.video_port_base and metadata_port_base are both "
            f"{cfg.video_port_base}. They must differ; the defaults are 9000 and 9100.\n"
            f"  Sharing a port wedges the encoder, which stalls the source and "
            f"truncates the recording."
        )
    if cfg.insight_enable and 9900 in (cfg.video_port_base, cfg.metadata_port_base):
        raise ValueError(
            "output.insight port base 9900 is the Neat Insight web UI port, not a "
            "stream port.\n  Use video_port_base: 9000 and metadata_port_base: 9100."
        )
    if not 0.0 <= cfg.draw.mask_alpha <= 1.0:
        raise ValueError("visualization.mask_alpha must be in [0.0, 1.0]")


# ─────────────────────────────────────────────────────────────────────────────
# Loading and CLI overrides
# ─────────────────────────────────────────────────────────────────────────────


def read_config_file(path: Path | None) -> dict:
    """Parse a config file, or return an empty document when there is none."""
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config root must be a mapping")
    return raw


def apply_overrides(raw: dict, overrides: dict) -> dict:
    """Write dotted-path CLI values into a parsed config document, in place.

    ``{"source.uri": "clip.h264"}`` becomes ``raw["source"]["uri"]``. The result
    goes through the ordinary loaders and the ordinary validation, so a CLI flag
    cannot reach a state a config file could not.

    Args:
        raw: Parsed config document, modified in place.
        overrides: Dotted paths to values. Values of None are ignored, which is
            what lets argparse defaults mean "the user did not say".

    Returns:
        ``raw``, for chaining.
    """
    for path, value in overrides.items():
        if value is None:
            continue
        *parents, leaf = path.split(".")
        node = raw
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[leaf] = value
    return raw


#: Config file names looked for when ``--config`` is not given, in order.
DISCOVERY_NAMES = ("config.yaml", "config.yml")


def discover_config(explicit: Path | None) -> Path | None:
    """Find the config file to use.

    An explicit ``--config`` must exist. Otherwise take ``config.yaml`` from the
    working directory if there is one. Finding nothing is not an error: the
    dataclass defaults are a complete configuration, so ``--model`` and
    ``--source`` are enough to run without any file at all.

    Args:
        explicit: Whatever ``--config`` was given, or None.

    Returns:
        A path, or None to run on defaults and CLI flags alone.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"config file not found: {explicit}")
        return explicit
    for name in DISCOVERY_NAMES:
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    return None

