"""Instance segmentation, with an optional background blur.

Finds instances and draws their masks whatever the ``blur`` section says. The
blur is the headline effect -- ``keep_classes: [person]`` plus ``invert: on``
makes it an anonymiser -- but ``enable: off`` leaves a plain segmentation
overlay and the app is still a segmenter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from .. import runtime
from ..config import (
    BaseConfig,
    DrawConfig,
    TaskDefaults,
    _flag,
    _float,
    _int,
    _int_list,
    _section,
    _str,
    _str_list,
)
from ..draw import caption_text, class_color, draw_caption, draw_fps, draw_scale
from ..masks import (
    Instance,
    as_cv_mask,
    composite,
    detect_mask_space,
    extract_masks,
    foreground_mask,
    instance_plane,
    letterbox_transform,
    packed_mask_layout,
    plane_cutoff,
    refine_mask,
    resize_plane_to_box,
    warp_plane_to_box,
)
from ..runloop import TaskRuntime
from ..runtime import SEG_FAMILIES, time_ms
from ..samples import (
    BBOX_RECORD_SIZE,
    describe_tensors,
    extract_bbox_payload,
    first_tensor,
    frame_to_bgr,
    joined_field,
    parse_boxes,
    resolve_classes,
)
from ..sinks import Pipeline, load_labels
from .base import Task

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SegmentConfig:
    """How per-instance masks are recovered from the model output.

    A detect head emits one BBOX tensor and nothing else. A segment head emits
    mask data as well, and there are three shapes it comes in:

    * **packed** - what SiMa's ``neatobjectdecode`` actually produces: a single
      UInt8 tensor whose head is the ordinary BBOX array and whose tail is one
      finished mask per *box slot*. There are ``top_k`` slots, not ``count``, so
      the buffer is a fixed size every frame::

          [uint32 count][RawBox 24B * top_k][mask side*side uint8 * top_k]

      With ``top_k=50`` and a 640-input model that is
      ``4 + 50*24 + 50*160*160 = 1,281,204`` bytes.
    * **planes** - the same finished masks, but in a tensor of their own,
      ``N x mh x mw``.
    * **proto** - the Ultralytics YOLO-seg encoding. A bank of ``C`` prototype
      planes shared by the whole frame, ``C x mh x mw``, plus ``C`` coefficients
      per instance. An instance's mask is the coefficient-weighted sum of the
      prototypes, so it costs one matmul per frame.

    ``auto`` tries packed first, because it is the layout the shipped model
    packs use, then falls back to looking at any separate tensors. ``describe``
    prints what actually arrived on the first frame, so pinning ``source``
    afterwards is a one-line change.

    Attributes:
        masks: Master switch. ``off`` skips mask decoding entirely and blurs
            around plain bounding boxes, which needs no segment head at all.
        source: ``auto``, ``packed``, ``proto`` or ``planes``.
        space: Which coordinate space a mask lives in. ``net`` means the
            model's own letterboxed input space, which is the YOLO convention
            and needs the letterbox undone. ``box`` means the mask is already
            cropped to its own detection and only needs scaling. ``auto``
            decides on the first frame by checking whether the mask's occupied
            region lands where its box says it should, and prints the verdict.
        threshold: Mask cut-off as a probability, so 0.5 is the YOLO default.
            Lower it to grow instances, raise it to tighten them.
        stride: Model input pixels per mask pixel, used to infer the network
            input size when ``preprocess.resize`` is left at 0. 4 for YOLO-seg,
            whose 640x640 input gives 160x160 prototypes.
        net_width: Model input width. 0 derives it, see ``stride``.
        net_height: Model input height. 0 derives it.
        coeff_counts: Prototype bank sizes to recognise while auto-detecting.
        mask_sides: Mask edge lengths to try when solving a packed buffer whose
            slot count cannot be taken from ``decode.max_detections``.
        fallback_to_boxes: Whether to fall back to box-shaped regions when no
            mask data can be found, rather than failing the run. The app says so
            once, loudly, and keeps going.
        describe: Whether to print every tensor in the first frame's output,
            with its tag, dtype and shape. This is how you learn what your model
            pack emits without reading any SDK source.
        dilate: Grow the finished mask by this many pixels, expressed for a
            1080p frame. Small positive values cover the halo a tight mask
            leaves around hair and limbs.
        blur_mask: Smooth the mask over this many pixels before it is used, also
            expressed for a 1080p frame. Takes the staircase off the edge that a
            160x160 mask shows once it is scaled up. 0 disables.
    """

    masks: str = "on"
    source: str = "auto"
    space: str = "auto"
    threshold: float = 0.5
    stride: int = 4
    net_width: int = 0
    net_height: int = 0
    coeff_counts: tuple[int, ...] = (16, 32, 64)
    mask_sides: tuple[int, ...] = (
        64, 80, 88, 96, 104, 112, 120, 128, 144, 152, 160, 176, 192, 208, 224, 240, 256, 320
    )
    fallback_to_boxes: bool = True
    describe: bool = True
    dilate: int = 0
    blur_mask: int = 0


@dataclass(frozen=True)
class BlurConfig:
    """The background treatment, straight from the ``blur`` config section.

    Pixel sizes here are expressed for a 1080p frame and follow
    ``visualization.auto_scale``, so a kernel that looks right on a test clip
    still looks right at 4K.

    This whole block is **optional**. ``enable: off`` turns the background
    treatment off and leaves a plain segmentation overlay, and ``opacity`` is
    the dial between the two rather than an on/off switch.

    Attributes:
        enable: Whether to composite at all.
        opacity: How much of the treated background to keep, 0.0 to 1.0. 1.0 is
            a full-strength blur, 0.4 is a hint of one, 0.0 is indistinguishable
            from ``enable: off``. Applied after the method, so it dials
            ``grayscale`` and ``dim`` down too.
        method: ``gaussian``, ``pixelate`` or ``none``. ``none`` still applies
            ``dim`` and ``grayscale``, which is how you get a darkened rather
            than blurred background.
        kernel: Gaussian kernel width in pixels. Forced odd. Bigger is blurrier
            and slower, though ``downscale`` absorbs most of the cost.
        sigma: Gaussian sigma. 0 derives it from ``kernel``, which is what you
            want almost always.
        downscale: Blur at 1/N resolution and scale back up. The dominant speed
            knob: at 1080p, 2 roughly quarters the work and is visually free,
            because the result is a blur either way.
        pixel_size: Mosaic block size for ``pixelate``, in pixels.
        dim: Darken the background, 0.0 to 1.0. 0 leaves brightness alone.
        grayscale: Whether to desaturate the background.
        feather: Soften the mask edge over this many pixels. 0 gives a hard
            cut, which is faster but shows every jag in the mask.
        invert: Blur the instances instead of the background. This is the
            anonymiser: ``keep_classes: [person]`` plus ``invert: on`` blurs
            people and leaves the scene sharp.
        keep_classes: Class names or ids that count as foreground. Empty keeps
            every detected class.
    """

    enable: bool = True
    opacity: float = 1.0
    method: str = "gaussian"
    kernel: int = 41
    sigma: float = 0.0
    downscale: int = 2
    pixel_size: int = 24
    dim: float = 0.0
    grayscale: bool = False
    feather: int = 9
    invert: bool = False
    keep_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SegmentAppConfig(BaseConfig):
    """Base config plus the ``segmentation`` and ``blur`` sections."""

    segment: SegmentConfig = SegmentConfig()
    blur: BlurConfig = BlurConfig()


def load_segment_config(raw: dict) -> SegmentConfig:
    section = _section(raw, "segmentation")
    default = SegmentConfig()
    return SegmentConfig(
        masks=_flag(section, "masks", default.masks),
        source=_str(section, "source", default.source).lower(),
        space=_str(section, "space", default.space).lower(),
        threshold=_float(section, "threshold", default.threshold),
        stride=_int(section, "stride", default.stride),
        net_width=_int(section, "net_width", default.net_width),
        net_height=_int(section, "net_height", default.net_height),
        coeff_counts=_int_list(section, "coeff_counts", default.coeff_counts),
        mask_sides=_int_list(section, "mask_sides", default.mask_sides),
        fallback_to_boxes=_flag(section, "fallback_to_boxes", "on") == "on",
        describe=_flag(section, "describe", "on") == "on",
        dilate=_int(section, "dilate", default.dilate),
        blur_mask=_int(section, "blur_mask", default.blur_mask),
    )


def load_blur_config(raw: dict) -> BlurConfig:
    section = _section(raw, "blur")
    default = BlurConfig()
    return BlurConfig(
        enable=_flag(section, "enable", "on") == "on",
        opacity=_float(section, "opacity", default.opacity),
        method=_str(section, "method", default.method).lower(),
        kernel=_int(section, "kernel", default.kernel),
        sigma=_float(section, "sigma", default.sigma),
        downscale=_int(section, "downscale", default.downscale),
        pixel_size=_int(section, "pixel_size", default.pixel_size),
        dim=_float(section, "dim", default.dim),
        grayscale=_flag(section, "grayscale", "off") == "on",
        feather=_int(section, "feather", default.feather),
        invert=_flag(section, "invert", "off") == "on",
        keep_classes=_str_list(section, "keep_classes", default.keep_classes),
    )


def validate_segment(cfg: SegmentAppConfig) -> None:
    # A detect head emits boxes and nothing else, so asking it for masks gives a
    # run that starts cleanly and then blurs nothing. Catch it here rather than
    # three minutes into a clip.
    if cfg.segment.masks == "on" and cfg.family not in SEG_FAMILIES:
        raise ValueError(
            f"model.family `{cfg.family}` is a detect head, which emits no mask data.\n"
            f"  Either point model.family at a segment head "
            f"({', '.join(sorted(SEG_FAMILIES))}),\n"
            f"  or set segmentation.masks: off to blur around plain bounding boxes."
        )
    if cfg.segment.source not in {"auto", "packed", "proto", "planes"}:
        raise ValueError("segmentation.source must be auto, packed, proto or planes")
    if cfg.segment.space not in {"auto", "net", "box"}:
        raise ValueError("segmentation.space must be auto, net or box")
    if not 0.0 < cfg.segment.threshold < 1.0:
        raise ValueError("segmentation.threshold must be in (0.0, 1.0), exclusive")
    if cfg.segment.stride <= 0:
        raise ValueError("segmentation.stride must be > 0")
    if cfg.segment.net_width < 0 or cfg.segment.net_height < 0:
        raise ValueError("segmentation.net_width and net_height must be >= 0")
    if cfg.segment.dilate < 0:
        raise ValueError("segmentation.dilate must be >= 0")
    if cfg.segment.blur_mask < 0:
        raise ValueError("segmentation.blur_mask must be >= 0")
    if not 0.0 <= cfg.blur.opacity <= 1.0:
        raise ValueError("blur.opacity must be in [0.0, 1.0]")
    if cfg.blur.method not in {"gaussian", "pixelate", "none"}:
        raise ValueError("blur.method must be gaussian, pixelate or none")
    if cfg.blur.kernel < 1:
        raise ValueError("blur.kernel must be >= 1")
    if cfg.blur.sigma < 0:
        raise ValueError("blur.sigma must be >= 0")
    if cfg.blur.downscale < 1:
        raise ValueError("blur.downscale must be >= 1")
    if cfg.blur.pixel_size < 2:
        raise ValueError("blur.pixel_size must be >= 2")
    if not 0.0 <= cfg.blur.dim <= 1.0:
        raise ValueError("blur.dim must be in [0.0, 1.0]")
    if cfg.blur.feather < 0:
        raise ValueError("blur.feather must be >= 0")


def describe_blur(cfg: SegmentAppConfig) -> str:
    blur = cfg.blur
    if not blur.enable or blur.opacity <= 0.0:
        return "blur: off, drawing a plain segmentation overlay"
    parts = [f"method={blur.method}"]
    if blur.method == "gaussian":
        parts.append(f"kernel={blur.kernel} sigma={blur.sigma or 'auto'} down={blur.downscale}")
    elif blur.method == "pixelate":
        parts.append(f"block={blur.pixel_size}")
    if blur.grayscale:
        parts.append("grayscale")
    if blur.dim > 0:
        parts.append(f"dim={blur.dim}")
    if blur.opacity < 1.0:
        parts.append(f"opacity={blur.opacity}")
    parts.append(f"feather={blur.feather}")
    subject = "instances" if blur.invert else "background"
    keep = ", ".join(blur.keep_classes) if blur.keep_classes else "every detected class"
    return f"blur: {subject} | {' '.join(parts)} | foreground={keep}"


def resolve_net_size(cfg: SegmentAppConfig) -> tuple[int, int]:
    """Model input geometry, needed to map a mask back onto the frame.

    Boxes come back in original-image pixels because Preproc writes the resize
    and letterbox metadata onto the tensor and BoxDecode reads it. Masks do not
    get that treatment: they live in the network's own letterboxed space, so the
    app has to invert the letterbox itself, and that needs the network input
    size. ``preprocess.resize`` supplies it when set; otherwise the mask
    geometry does, once the first frame arrives.

    Returns:
        A ``(width, height)`` pair, or ``(0, 0)`` when it must wait for a mask.
    """
    if cfg.segment.net_width and cfg.segment.net_height:
        return cfg.segment.net_width, cfg.segment.net_height
    pre = cfg.preprocess
    if pre.resize_width and pre.resize_height:
        return pre.resize_width, pre.resize_height
    return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SegmentPipeline(Pipeline):
    """Pipeline plus the mask bookkeeping.

    Attributes:
        keep_ids: Class ids that count as foreground, or None for every class.
        net_w: Model input width, used to invert the letterbox for masks.
        net_h: Model input height.
        mask_kind: What the mask decoder settled on: ``packed``, ``tensors`` or
            empty. Filled in on the first frame that carries instances.
        mask_space: ``net`` or ``box``, decided once and then reused.
        described: Whether the first-frame tensor dump has already been printed.
        warned_no_masks: Whether the "no mask data" warning has been printed.
    """

    keep_ids: object = None
    net_w: int = 0
    net_h: int = 0
    mask_kind: str = ""
    mask_space: str = ""
    described: bool = False
    warned_no_masks: bool = False


def build_instances(pipeline: SegmentPipeline, cfg: SegmentAppConfig, boxes: list[dict],
                    bundle, frame_shape, scale: float) -> list[Instance]:
    """Turn boxes plus mask data into per-instance masks in frame coordinates.

    Args:
        pipeline: Live pipeline, for the frame geometry, the class filter and
            the one-shot fallback warning.
        cfg: Application configuration.
        boxes: Parsed detections, in source-image pixels.
        bundle: Mask data for this frame, possibly empty.
        frame_shape: The frame's ``(h, w, c)``.
        scale: Overlay scale factor for this frame.

    Returns:
        One :class:`~sima_vision.masks.Instance` per usable box, in input order.
    """
    height, width = frame_shape[:2]
    lb = None
    cutoff = 0.0
    if bundle.kind != "none":
        mask_h, mask_w = bundle.shape
        # A model pack that does not state its input size still tells us
        # indirectly: the mask is the input divided by the head's stride.
        net_w = pipeline.net_w or mask_w * cfg.segment.stride
        net_h = pipeline.net_h or mask_h * cfg.segment.stride
        pipeline.net_w, pipeline.net_h = net_w, net_h
        lb = letterbox_transform(width, height, net_w, net_h, cfg.preprocess.resize_mode)
        cutoff = plane_cutoff(bundle, cfg.segment.threshold)
    elif boxes and cfg.segment.masks == "on" and not pipeline.warned_no_masks:
        pipeline.warned_no_masks = True
        if cfg.segment.fallback_to_boxes:
            print(
                "[warn] no mask data in the model output, falling back to box-shaped\n"
                "       regions. The blur still works, the edges are just rectangles.\n"
                "       Set segmentation.describe: on and check the tensor dump above\n"
                "       against segmentation.source / coeff_counts.",
                file=sys.stderr, flush=True,
            )
        else:
            raise RuntimeError(
                "no mask data in the model output. Check model.family is a -seg head "
                "and segmentation.source, or set segmentation.fallback_to_boxes: on."
            )

    instances = []
    for index, box in enumerate(boxes):
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(width, int(round(box["x2"])))
        y2 = min(height, int(round(box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        mask = None
        if lb is not None and index < bundle.count:
            plane = instance_plane(bundle, index)
            if not pipeline.mask_space:
                # Decided once, from the first instance that has any ink in it.
                space = cfg.segment.space
                if space == "auto":
                    space = detect_mask_space(
                        plane, cutoff, lb, pipeline.net_w, pipeline.net_h, (x1, y1, x2, y2)
                    )
                if space:
                    pipeline.mask_space = space
                    print(f"masks: space={space}", flush=True)
            local = (
                resize_plane_to_box(plane, (x1, y1, x2, y2))
                if pipeline.mask_space == "box"
                else warp_plane_to_box(
                    plane, lb, pipeline.net_w, pipeline.net_h, (x1, y1, x2, y2)
                )
            )
            mask = refine_mask(local > cutoff, cfg.segment, scale)
            if not mask.any():
                # A mask that thresholds to nothing would silently remove the
                # instance from the composite. The box is a worse answer than a
                # mask but a much better one than a hole.
                mask = None

        class_id = int(box["class_id"])
        keep = pipeline.keep_ids is None or class_id in pipeline.keep_ids
        instances.append(Instance(box=box, x1=x1, y1=y1, x2=x2, y2=y2, mask=mask, keep=keep))
    return instances


def draw_instances(frame, instances: list[Instance], labels: list[str], draw) -> None:
    """Tint, outline and caption every instance in place.

    Each instance gets a translucent fill in its class colour, a traced edge and
    a filled caption sitting directly above its box.

    Args:
        frame: BGR image, modified in place.
        instances: Instances with masks already in frame coordinates.
        labels: Class names indexed by class id.
        draw: Visualization settings, from the ``visualization`` config section.
    """
    cv2, np = runtime.cv2, runtime.np
    scale = draw_scale(frame, draw)
    box_thickness = max(1, int(round(draw.box_thickness * scale)))
    outline_thickness = max(1, int(round(draw.outline_thickness * scale)))

    # Paint larger instances first, so small foreground objects stay legible.
    for inst in sorted(instances, key=lambda i: i.area, reverse=True):
        color = class_color(int(inst.box["class_id"]))
        region = frame[inst.y1 : inst.y2, inst.x1 : inst.x2]

        if draw.mask_alpha > 0 and region.size:
            alpha = draw.mask_alpha
            tinted = cv2.addWeighted(
                region, 1.0 - alpha, np.full_like(region, color, dtype=np.uint8), alpha, 0.0
            )
            if inst.mask is None:
                region[:] = tinted
            else:
                cv2.copyTo(tinted, as_cv_mask(inst.mask), region)

        if draw.mask_outline and inst.mask is not None:
            contours, _ = cv2.findContours(
                as_cv_mask(inst.mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                frame, contours, -1, color, outline_thickness, cv2.LINE_AA,
                offset=(inst.x1, inst.y1),
            )
        if draw.show_boxes or inst.mask is None:
            cv2.rectangle(frame, (inst.x1, inst.y1), (inst.x2, inst.y2), color, box_thickness)

        draw_caption(
            frame, caption_text(inst.box, labels, draw), (inst.x1, inst.y1), color, draw, scale
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run loop
# ─────────────────────────────────────────────────────────────────────────────


class SegmentRuntime(TaskRuntime):
    output_label = "segmenter_output"
    unit = "instances"
    stage = "masks"

    def decode(self, pipeline: SegmentPipeline, cfg: SegmentAppConfig, sample, index: int):
        instances_field = joined_field(sample, "instances", 1)
        payload, bbox_tensor = extract_bbox_payload(instances_field)
        boxes = parse_boxes(payload, pipeline.frame_w, pipeline.frame_h, cfg.max_detections)
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        self.check_geometry(pipeline, frame)
        decode_end = time_ms()

        if cfg.segment.describe and not pipeline.described and boxes:
            pipeline.described = True
            print(self._describe_output(cfg, instances_field, payload), flush=True)

        bundle = extract_masks(
            instances_field, bbox_tensor, payload, len(boxes), cfg.segment, cfg.max_detections
        )
        if bundle.kind != "none" and not pipeline.mask_kind:
            pipeline.mask_kind = bundle.origin or bundle.kind
            mask_h, mask_w = bundle.shape
            if bundle.probabilities:
                values = "0/1 binary" if bundle.peak <= 1 else f"0..{bundle.peak}"
            else:
                values = "logits"
            print(
                f"masks: source={pipeline.mask_kind} layout={bundle.kind} "
                f"{mask_w}x{mask_h} slots={bundle.count} values={values}",
                flush=True,
            )

        # The decoder's buffer is free from here on. `frame` and `payload` are
        # copies; a packed `bundle` is a view on `payload`, and a bundle built
        # from separate tensors is copied out in classify_mask_tensors. Nothing
        # below this line touches pyneat, so let go before doing any of it --
        # the mask warp alone is tens of milliseconds per frame.
        sample = instances_field = bbox_tensor = None

        instances = build_instances(
            pipeline, cfg, boxes, bundle, frame.shape, draw_scale(frame, cfg.draw)
        )
        bundle = None
        return frame, instances, time_ms() - decode_end

    @staticmethod
    def _describe_output(cfg: SegmentAppConfig, instances_field, payload: bytes) -> str:
        layout = packed_mask_layout(len(payload), cfg.max_detections, cfg.segment)
        if layout is None:
            solved = (
                f"  packed layout: {len(payload)} bytes does not decompose into "
                f"4 + slots*{BBOX_RECORD_SIZE} + slots*side*side"
            )
        else:
            slots, side = layout
            solved = (
                f"  packed layout: 4 + {slots}*{BBOX_RECORD_SIZE} + "
                f"{slots}*{side}*{side} = "
                f"{4 + slots * BBOX_RECORD_SIZE + slots * side * side} bytes"
            )
        return (
            "model output tensors (first frame with instances):\n"
            f"{describe_tensors(instances_field)}\n{solved}"
        )

    def render(self, cfg: SegmentAppConfig, pipeline: SegmentPipeline, frame, results, fps: float):
        """Composite the blur, then draw over it, once per frame for every sink."""
        scale = draw_scale(frame, cfg.draw)
        if cfg.blur.enable:
            annotated = composite(frame, foreground_mask(results, frame.shape), cfg.blur, scale)
        else:
            annotated = frame.copy()
        draw_instances(annotated, results, pipeline.labels, cfg.draw)
        # FPS last, so neither a caption nor a mask lands on top of it. See
        # DetectRuntime.render.
        if cfg.video_hud:
            draw_fps(annotated, fps, cfg.draw)
        return annotated

    def summarise(self, pipeline: SegmentPipeline, processed: int) -> list[str]:
        return [f"masks={pipeline.mask_kind or 'none'}"]


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

SEGMENT_DRAW = DrawConfig(box_thickness=2, centre_dot=False)


class SegmentTask(Task):
    name = "segment"
    help = "Per-pixel masks, and a background blur that keeps the subject sharp"
    config_class = SegmentAppConfig
    graph_name = "yolo_segmenter"
    result_label = "instances"
    output_label = "segmenter_output"
    defaults = TaskDefaults(
        task="segment",
        family="yolo26-seg",
        save_dir="frames",
        video_path="segmentation.mp4",
        draw=SEGMENT_DRAW,
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--blur", dest="blur.enable", action="store_const", const=True,
            help="Blur the background and keep the instances sharp. On by default, "
                 "so this is only needed to override a config file that turns it off.",
        )
        parser.add_argument(
            "--no-blur", dest="blur.enable", action="store_const", const=False,
            help="Draw a plain segmentation overlay with no background treatment. "
                 "The blur is on by default; this is how you turn it off.",
        )
        parser.add_argument(
            "--blur-method", dest="blur.method", choices=("gaussian", "pixelate", "none"),
            help="How the background is treated. Default gaussian.",
        )
        parser.add_argument(
            "--blur-strength", dest="blur.kernel", type=int, metavar="PX",
            help="Gaussian kernel width in pixels, for a 1080p frame. Default 41.",
        )
        parser.add_argument(
            "--keep-classes", dest="blur.keep_classes", nargs="+", metavar="CLASS",
            help="Class names or ids that stay sharp. Default: every detected class.",
        )
        parser.add_argument(
            "--anonymise", "--anonymize", dest="blur.invert", action="store_const", const=True,
            help="Blur the instances instead of the background. With "
                 "--keep-classes person this blurs people and leaves the scene sharp.",
        )
        parser.add_argument(
            "--mask-threshold", dest="segmentation.threshold", type=float, metavar="T",
            help="Mask cut-off as a probability. Default 0.5; lower grows instances.",
        )
        parser.add_argument(
            "--no-masks", dest="segmentation.masks", action="store_const", const="off",
            help="Skip mask decoding and blur around plain bounding boxes. Needs no "
                 "segment head.",
        )
        parser.add_argument(
            "--minimal", action="store_true",
            help="Pull frames and do nothing else: no masks, no blur, no overlay, no "
                 "video, no stills. If a run that stalls part-way through completes "
                 "with this, the cause is how much work the app does per frame; if it "
                 "stalls at the same frame, the cause is the graph.",
        )

    def post_process(self, cfg: SegmentAppConfig, args) -> SegmentAppConfig:
        if not getattr(args, "minimal", False):
            return cfg
        # Strip the consumer back to a bare pull loop. Nothing here changes the
        # graph, so it isolates "we are too slow" from "the graph is wrong" in a
        # single run.
        print(
            "[minimal] masks, blur, overlay, video and stills are all "
            "off.\n          Reaching the end of the clip means the graph is fine "
            "and the app was\n          simply holding buffers too long.",
            flush=True,
        )
        return replace(
            cfg,
            segment=replace(cfg.segment, masks="off", describe=False),
            blur=replace(cfg.blur, enable=False),
            save_enable=False, video_enable=False,
        )

    def extra_sections(self, raw: dict) -> dict:
        return {"segment": load_segment_config(raw), "blur": load_blur_config(raw)}

    def validate(self, cfg: SegmentAppConfig) -> None:
        super().validate(cfg)
        validate_segment(cfg)

    def describe(self, cfg: SegmentAppConfig) -> list[str]:
        net_w, net_h = resolve_net_size(cfg)
        # Needs the labels file, not the board, so a typo in blur.keep_classes
        # is caught here rather than on the DevKit.
        keep_ids = resolve_classes(
            cfg.blur.keep_classes, load_labels(cfg.labels_path),
            "blur.keep_classes", cfg.labels_path,
        )
        lines = [
            f"segmentation: masks={cfg.segment.masks} source={cfg.segment.source} "
            f"space={cfg.segment.space} threshold={cfg.segment.threshold} "
            f"net={f'{net_w}x{net_h}' if net_w else '<from the first mask>'}",
            describe_blur(cfg),
        ]
        if keep_ids is not None:
            lines.append(f"foreground class ids: {sorted(keep_ids)}")
        return lines

    def make_pipeline(self, cfg: SegmentAppConfig, labels: list[str]) -> SegmentPipeline:
        net_w, net_h = resolve_net_size(cfg)
        return SegmentPipeline(
            labels=labels,
            net_w=net_w,
            net_h=net_h,
            keep_ids=resolve_classes(
                cfg.blur.keep_classes, labels, "blur.keep_classes", cfg.labels_path
            ),
        )

    def prepare(self, cfg: SegmentAppConfig, pipeline: SegmentPipeline, step) -> None:
        step.detail(describe_blur(cfg))

    def runtime(self, cfg, pipeline) -> TaskRuntime:
        return SegmentRuntime()

