"""The per-frame code: box parsing, masks, compositing and the overlay.

All of this is plain numpy and OpenCV, so it runs anywhere. Between them these
cover the parts of the refactor that used to be copy-pasted three times.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from sima_vision.config import DrawConfig
from sima_vision.draw import (
    caption_text,
    class_color,
    draw_banner,
    draw_boxes,
    draw_caption,
    draw_fps,
    draw_scale,
    text_ink_extent,
)
from sima_vision.masks import (
    Instance,
    MaskBundle,
    composite,
    foreground_mask,
    instance_plane,
    letterbox_transform,
    masks_from_packed_payload,
    packed_mask_layout,
    plane_cutoff,
    warp_plane_to_box,
)
from sima_vision.samples import BBOX_RECORD, BBOX_RECORD_SIZE, parse_boxes
from sima_vision.tasks.segment import SegmentConfig


def frame(h=270, w=480):
    return np.full((h, w, 3), 40, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# BBOX payload
# ─────────────────────────────────────────────────────────────────────────────


def bbox_payload(records, slots=None):
    """Build the buffer BoxDecode emits: a count, then fixed-size box slots."""
    slots = len(records) if slots is None else slots
    out = bytearray(struct.pack("<I", len(records)))
    for x, y, w, h, score, cls in records:
        out += BBOX_RECORD.pack(x, y, w, h, score, cls)
    out += b"\x00" * ((slots - len(records)) * BBOX_RECORD_SIZE)
    return bytes(out)


def test_parse_boxes_converts_xywh_to_corners():
    payload = bbox_payload([(10, 20, 30, 40, 0.75, 3)])
    boxes = parse_boxes(payload, 480, 270, 50)
    assert len(boxes) == 1
    box = boxes[0]
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (10.0, 20.0, 40.0, 60.0)
    assert box["score"] == pytest.approx(0.75)
    assert box["class_id"] == 3


def test_parse_boxes_clips_to_the_frame():
    payload = bbox_payload([(-5, -5, 10000, 10000, 0.5, 0)])
    box = parse_boxes(payload, 480, 270, 50)[0]
    assert (box["x1"], box["y1"]) == (0.0, 0.0)
    assert (box["x2"], box["y2"]) == (480.0, 270.0)


def test_parse_boxes_rejects_a_count_beyond_the_payload():
    payload = struct.pack("<I", 9) + b"\x00" * BBOX_RECORD_SIZE
    with pytest.raises(RuntimeError, match="exceeds payload capacity"):
        parse_boxes(payload, 480, 270, 50)


def test_parse_boxes_rejects_a_count_beyond_top_k():
    payload = bbox_payload([(0, 0, 1, 1, 0.5, 0)] * 4)
    with pytest.raises(RuntimeError, match="exceeds top_k"):
        parse_boxes(payload, 480, 270, 2)


def test_parse_boxes_rejects_a_truncated_header():
    with pytest.raises(RuntimeError, match="too small"):
        parse_boxes(b"\x00\x00", 480, 270, 50)


# ─────────────────────────────────────────────────────────────────────────────
# Packed masks
# ─────────────────────────────────────────────────────────────────────────────


def test_packed_layout_solves_the_documented_example():
    """4 + 50*24 + 50*160*160 = 1281204, the yolo26m-seg case from the docs."""
    assert packed_mask_layout(1281204, 50, SegmentConfig()) == (50, 160)


def test_packed_layout_solves_without_a_known_top_k():
    """top_k 0 means the archive's own value won; the side table must find it."""
    assert packed_mask_layout(1281204, 0, SegmentConfig()) == (50, 160)


def test_packed_layout_returns_none_when_it_does_not_decompose():
    assert packed_mask_layout(12345, 0, SegmentConfig()) is None


def test_masks_from_packed_payload_reads_the_tail():
    slots, side, count = 4, 16, 2
    head = bbox_payload([(0, 0, 8, 8, 0.9, 0)] * count, slots=slots)
    planes = np.zeros((slots, side, side), dtype=np.uint8)
    planes[0, 2:10, 2:10] = 255
    planes[1, 4:6, 4:6] = 255
    bundle = masks_from_packed_payload(head + planes.tobytes(), count, slots, SegmentConfig())
    assert bundle.kind == "planes"
    assert bundle.origin == "packed"
    assert bundle.shape == (side, side)
    assert bundle.peak == 255
    assert bundle.probabilities is True
    np.testing.assert_array_equal(bundle.planes[0], planes[0])


def test_plane_cutoff_tells_binary_from_quantised():
    """A 0/1 mask and a 0..255 one need cut-offs 255x apart."""
    binary = MaskBundle(kind="planes", planes=np.zeros((1, 4, 4), np.uint8),
                        probabilities=True, peak=1)
    quantised = MaskBundle(kind="planes", planes=np.zeros((1, 4, 4), np.uint8),
                           probabilities=True, peak=255)
    assert plane_cutoff(binary, 0.5) == pytest.approx(0.5)
    assert plane_cutoff(quantised, 0.5) == pytest.approx(127.5)


def test_plane_cutoff_uses_a_logit_for_raw_scores():
    logits = MaskBundle(kind="proto", probabilities=False)
    assert plane_cutoff(logits, 0.5) == pytest.approx(0.0)


def test_instance_plane_from_prototypes():
    protos = np.zeros((2, 4, 4), np.float32)
    protos[0, 0, :] = 1.0
    protos[1, 1, :] = 1.0
    coeffs = np.array([[2.0, 3.0]], np.float32)
    plane = instance_plane(MaskBundle(kind="proto", protos=protos, coeffs=coeffs), 0)
    assert plane.shape == (4, 4)
    np.testing.assert_allclose(plane[0], 2.0)
    np.testing.assert_allclose(plane[1], 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox
# ─────────────────────────────────────────────────────────────────────────────


def test_letterbox_pads_the_short_axis():
    lb = letterbox_transform(1920, 1080, 640, 640, "letterbox")
    assert lb.sx == pytest.approx(640 / 1920)
    assert lb.sx == lb.sy
    assert lb.pad_x == pytest.approx(0.0)
    assert lb.pad_y == pytest.approx((640 - 1080 * 640 / 1920) / 2)


def test_stretch_scales_each_axis_independently():
    lb = letterbox_transform(1920, 1080, 640, 640, "stretch")
    assert lb.sx == pytest.approx(640 / 1920)
    assert lb.sy == pytest.approx(640 / 1080)
    assert (lb.pad_x, lb.pad_y) == (0.0, 0.0)


def test_crop_overfills_and_pads_negative():
    lb = letterbox_transform(1920, 1080, 640, 640, "crop")
    assert lb.sx == pytest.approx(640 / 1080)
    assert lb.pad_x < 0


def test_warp_lands_the_mask_where_the_box_is():
    """A blob drawn in network space must come back inside its own box."""
    net = 64
    lb = letterbox_transform(480, 270, net, net, "letterbox")
    box = (100, 60, 200, 160)
    plane = np.zeros((net, net), np.float32)
    # Paint exactly the region the box maps to.
    x1 = int(box[0] * lb.sx + lb.pad_x)
    y1 = int(box[1] * lb.sy + lb.pad_y)
    x2 = int(box[2] * lb.sx + lb.pad_x)
    y2 = int(box[3] * lb.sy + lb.pad_y)
    plane[y1:y2, x1:x2] = 1.0

    local = warp_plane_to_box(plane, lb, net, net, box)
    assert local.shape == (box[3] - box[1], box[2] - box[0])
    # Most of the box should be covered, and the centre certainly.
    assert (local > 0.5).mean() > 0.7
    assert local[local.shape[0] // 2, local.shape[1] // 2] > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Compositing
# ─────────────────────────────────────────────────────────────────────────────


def instance_at(x1, y1, x2, y2, keep=True, solid=True):
    mask = np.ones((y2 - y1, x2 - x1), dtype=bool) if solid else None
    return Instance(box={"class_id": 0, "score": 0.9}, x1=x1, y1=y1, x2=x2, y2=y2,
                    mask=mask, keep=keep)


def test_foreground_mask_unions_overlapping_instances():
    mask = foreground_mask(
        [instance_at(0, 0, 20, 20), instance_at(10, 10, 30, 30)], (40, 40, 3)
    )
    assert mask[5, 5] == 255
    assert mask[15, 15] == 255      # the overlap must not be punched back out
    assert mask[25, 25] == 255
    assert mask[35, 35] == 0


def test_foreground_mask_skips_instances_not_kept():
    mask = foreground_mask([instance_at(0, 0, 20, 20, keep=False)], (40, 40, 3))
    assert mask.max() == 0


def test_foreground_mask_uses_the_box_when_there_is_no_mask():
    mask = foreground_mask([instance_at(0, 0, 20, 20, solid=False)], (40, 40, 3))
    assert mask[10, 10] == 255


class Blur:
    """A minimal stand-in for BlurConfig, so these tests state their own inputs."""

    def __init__(self, **kw):
        self.enable = True
        self.opacity = 1.0
        self.method = "gaussian"
        self.kernel = 21
        self.sigma = 0.0
        self.downscale = 1
        self.pixel_size = 8
        self.dim = 0.0
        self.grayscale = False
        self.feather = 0
        self.invert = False
        self.__dict__.update(kw)


def noisy_frame(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_composite_keeps_the_foreground_and_blurs_the_rest():
    src = noisy_frame()
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 255
    out = composite(src, mask, Blur(), 1.0)
    # Inside the mask the pixels are untouched...
    np.testing.assert_array_equal(out[16:48, 16:48], src[16:48, 16:48])
    # ...and outside it they are not.
    assert not np.array_equal(out[0:8, 0:8], src[0:8, 0:8])


def test_invert_blurs_the_instances_instead():
    src = noisy_frame()
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 255
    out = composite(src, mask, Blur(invert=True), 1.0)
    np.testing.assert_array_equal(out[0:8, 0:8], src[0:8, 0:8])
    assert not np.array_equal(out[24:40, 24:40], src[24:40, 24:40])


def test_composite_never_writes_into_the_source():
    src = noisy_frame()
    original = src.copy()
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 255
    composite(src, mask, Blur(feather=9), 1.0)
    np.testing.assert_array_equal(src, original)


def test_pixelate_and_none_are_both_accepted():
    src = noisy_frame()
    mask = np.zeros((64, 64), np.uint8)
    for method in ("pixelate", "none", "gaussian"):
        out = composite(src, mask, Blur(method=method, dim=0.5), 1.0)
        assert out.shape == src.shape and out.dtype == np.uint8


def test_downscale_still_returns_a_full_size_frame():
    src = noisy_frame(128, 128)
    mask = np.zeros((128, 128), np.uint8)
    out = composite(src, mask, Blur(downscale=4), 1.0)
    assert out.shape == src.shape


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────


def test_draw_scale_tracks_frame_height():
    draw = DrawConfig(reference_height=1080.0)
    assert draw_scale(np.zeros((1080, 1920, 3), np.uint8), draw) == pytest.approx(1.0)
    assert draw_scale(np.zeros((2160, 3840, 3), np.uint8), draw) == pytest.approx(2.0)
    # Floored, so a tiny frame still gets readable strokes.
    assert draw_scale(np.zeros((100, 100, 3), np.uint8), draw) == pytest.approx(0.4)


def test_draw_scale_is_one_when_auto_scale_is_off():
    draw = DrawConfig(auto_scale=False)
    assert draw_scale(np.zeros((2160, 3840, 3), np.uint8), draw) == 1.0


def test_ink_extent_measures_descenders():
    """`person` drops below the baseline; `FPS: 24` does not. That is the point."""
    _, below_descender = text_ink_extent("person", 1.0, 2)
    _, below_plain = text_ink_extent("FPS 24", 1.0, 2)
    assert below_descender > below_plain


def test_ink_extent_is_cached_by_folded_digits():
    """Every confidence value must not get its own cache entry."""
    from sima_vision.draw import _ink_cache

    _ink_cache.clear()
    text_ink_extent("car 0.91", 1.0, 2)
    text_ink_extent("car 0.44", 1.0, 2)
    assert len(_ink_cache) == 1


def test_class_color_is_stable_and_wraps():
    assert class_color(0) == class_color(20)
    assert class_color(3) != class_color(4)


def test_caption_text_honours_the_switches():
    box = {"class_id": 1, "score": 0.5678}
    labels = ["person", "bicycle"]
    assert caption_text(box, labels, DrawConfig()) == "bicycle 0.57"
    assert caption_text(box, labels, DrawConfig(show_scores=False)) == "bicycle"
    assert caption_text(box, labels, DrawConfig(show_labels=False)) == "0.57"
    assert caption_text(box, labels, DrawConfig(show_labels=False, show_scores=False)) == ""


def test_caption_text_falls_back_to_the_class_id():
    box = {"class_id": 99, "score": 0.5}
    assert caption_text(box, ["person"], DrawConfig()) == "99 0.50"


def test_draw_boxes_marks_the_frame():
    img = frame()
    before = img.copy()
    draw_boxes(
        img,
        [{"x1": 50.0, "y1": 50.0, "x2": 150.0, "y2": 200.0, "score": 0.9, "class_id": 0}],
        ["person"],
        DrawConfig(),
    )
    assert not np.array_equal(img, before)


def test_draw_boxes_skips_degenerate_boxes():
    img = frame()
    before = img.copy()
    draw_boxes(
        img,
        [{"x1": 50.0, "y1": 50.0, "x2": 50.0, "y2": 50.0, "score": 0.9, "class_id": 0}],
        ["person"],
        DrawConfig(),
    )
    np.testing.assert_array_equal(img, before)


def test_draw_boxes_keeps_a_caption_on_a_box_at_the_top_edge():
    """The band flips inside the box rather than off the frame."""
    img = frame()
    draw_boxes(
        img,
        [{"x1": 5.0, "y1": 0.0, "x2": 120.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
        ["person"],
        DrawConfig(),
    )
    assert img[0:30, 5:100].std() > 0     # something was drawn in the top rows


def test_draw_fps_writes_a_badge_top_left():
    img = frame()
    draw_fps(img, 24.7, DrawConfig(hud_fps_decimals=1))
    assert not np.array_equal(img[0:40, 0:150], np.full((40, 150, 3), 40, np.uint8))


def fill_pixels(img, colour) -> int:
    """How many pixels of the top-left corner are exactly this colour."""
    corner = img[:300, :700]
    return int((corner == np.array(colour, np.uint8)).all(axis=2).sum())


def test_the_badge_is_magenta_and_larger_than_a_caption():
    """Both are deliberate, so both are pinned.

    The badge is glanced at while the video plays rather than read, so it is
    set above the caption scale. #C11C84 occurs in almost no real scene, which
    is what makes it read as an overlay rather than as part of the footage.
    """
    draw = DrawConfig()
    assert draw.hud_bg_color == (132, 28, 193)     # #C11C84 as BGR
    assert draw.hud_text_color == (255, 255, 255)
    assert draw.hud_text_scale > draw.text_scale

    img = np.full((1080, 1920, 3), 40, np.uint8)
    draw_fps(img, 24.0, draw)
    assert fill_pixels(img, (132, 28, 193)) > 1000, "no badge was painted"


def test_white_stays_readable_on_the_badge():
    """5.6:1. Large text needs 4.5:1, and the badge is deliberately large.

    Worth pinning rather than leaving to taste: the badge is the one thing on
    the frame that is not about the picture, so a fill that swallows its own
    text makes it decoration.
    """
    draw = DrawConfig()
    b, g, r = (c / 255 for c in draw.hud_bg_color)
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    assert 1.05 / (lum + 0.05) > 4.5


def test_the_badge_still_follows_the_caption_scale_when_asked_to():
    """0 has always meant "follow text_scale", and a real default must not
    quietly take that escape hatch away."""
    img = np.full((1080, 1920, 3), 40, np.uint8)
    draw_fps(img, 24.0, DrawConfig(hud_text_scale=0.0, hud_bg_color=(7, 8, 9)))
    assert fill_pixels(img, (7, 8, 9)) > 1000


def test_draw_banner_covers_the_bottom_strip():
    img = frame()
    before = img.copy()
    draw_banner(img, "FALL DETECTED - track #1", DrawConfig())
    assert np.array_equal(img[0:100], before[0:100])       # top untouched
    assert not np.array_equal(img[-30:], before[-30:])     # bottom strip painted


@pytest.mark.parametrize("size", [(120, 160), (1080, 1920)])
def test_overlay_survives_any_frame_size(size):
    img = np.full((*size, 3), 40, np.uint8)
    boxes = [{"x1": 10.0, "y1": 10.0, "x2": size[1] * 0.5, "y2": size[0] * 0.5,
              "score": 0.8, "class_id": 2}]
    draw_boxes(img, boxes, ["a", "b", "c"], DrawConfig())
    draw_fps(img, 30.0, DrawConfig())
    assert img.shape == (*size, 3)


# ─────────────────────────────────────────────────────────────────────────────
# The badge is the last thing drawn
# ─────────────────────────────────────────────────────────────────────────────

BADGE_FPS = 24.0


def render_case(app: str):
    """One app's runtime, config, pipeline stub and results, ready to render.

    Every box sits in the top-left corner, over the badge, because that is the
    one position where the drawing order shows.
    """
    import types

    from sima_vision.tasks import TASKS

    cfg = TASKS[app]().load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    pipeline = types.SimpleNamespace(labels=["person", "bike", "car"])
    box = {"x1": 0.0, "y1": 0.0, "x2": 420.0, "y2": 300.0,
           "score": 0.93, "class_id": 0}

    if app == "detect":
        from sima_vision.tasks.detect import DetectRuntime

        return DetectRuntime(), cfg, pipeline, [box]
    if app == "segment":
        from sima_vision.tasks.segment import SegmentRuntime

        instance = Instance(
            box=box, x1=0, y1=0, x2=420, y2=300,
            mask=np.ones((300, 420), dtype=bool), keep=True,
        )
        return SegmentRuntime(), cfg, pipeline, [instance]
    from sima_vision.tasks.fall import FALLEN, FallRuntime, Track

    # FALLEN, so the banner is drawn too: the badge has to come after that as
    # well, not merely after the boxes.
    return FallRuntime(), cfg, pipeline, [Track(track_id=1, box=box, state=FALLEN)]


@pytest.mark.parametrize("app", ["detect", "segment", "fall"])
def test_the_fps_badge_is_never_drawn_over(app):
    """Rendering with the HUD on must equal rendering without it, then stamping
    the badge on by hand.

    Which is only true if the badge is the very last thing the app draws. Every
    app used to draw it first, so a detection in the top-left corner put its
    caption straight through the frame rate -- and the badge is the one reading
    on the frame that is not about the picture.

    Asserted per app rather than by reading the source, because three separate
    render methods is three chances to put it back.
    """
    import dataclasses

    runtime, cfg, pipeline, results = render_case(app)
    frame = np.full((1080, 1920, 3), 40, np.uint8)

    with_hud = runtime.render(cfg, pipeline, frame, results, BADGE_FPS)

    without_hud = runtime.render(
        dataclasses.replace(cfg, video_hud=False), pipeline, frame, results, BADGE_FPS
    )
    draw_fps(without_hud, BADGE_FPS, cfg.draw)

    assert np.array_equal(with_hud, without_hud), (
        f"{app} draws something over the FPS badge"
    )


@pytest.mark.parametrize("app", ["detect", "segment", "fall"])
def test_every_pixel_of_the_badge_survives_a_box_on_top_of_it(app):
    """The same thing said in pixels, so a failure names what was lost.

    A caption over the badge changed a few hundred pixels out of forty
    thousand, which is easy to miss in a screenshot and easy to assert on.
    """
    runtime, cfg, pipeline, results = render_case(app)
    frame = np.full((1080, 1920, 3), 40, np.uint8)

    badge = frame.copy()
    draw_fps(badge, BADGE_FPS, cfg.draw)
    painted = (badge != frame).any(axis=2)
    assert painted.sum() > 1000, "no badge to test against"

    rendered = runtime.render(cfg, pipeline, frame, results, BADGE_FPS)
    lost = int((rendered[painted] != badge[painted]).any(axis=1).sum())
    assert lost == 0, f"{app} overwrote {lost} of {int(painted.sum())} badge pixels"


# ─────────────────────────────────────────────────────────────────────────────
# Text sized for the resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_captions_are_sized_for_1080p():
    """The numbers are the point, so they are written down.

    1.0 and 2 were sized for reading a still at 100%. Played in a window, or on
    a wall of camera tiles, a two-pixel stroke does not separate from the
    footage behind it.
    """
    draw = DrawConfig()
    assert (draw.text_scale, draw.text_thickness) == (1.6, 4)
    assert draw.reference_height == 1080.0
    assert draw.auto_scale is True


def test_the_badge_is_one_and_a_half_times_a_caption():
    """Asserted as a ratio, not as 2.4 and 6.

    The badge is derived from the caption precisely so that retuning the
    caption cannot leave the badge behind. A test against the literals would
    pass while that relationship quietly broke.
    """
    from sima_vision.config import HUD_MULTIPLE, TEXT_SCALE, TEXT_THICKNESS

    draw = DrawConfig()
    assert HUD_MULTIPLE == 1.5
    assert draw.hud_text_scale == round(TEXT_SCALE * HUD_MULTIPLE, 3) == 2.4
    assert draw.hud_text_thickness == round(TEXT_THICKNESS * HUD_MULTIPLE) == 6
    assert draw.hud_text_scale / draw.text_scale == pytest.approx(HUD_MULTIPLE)
    assert draw.hud_text_thickness / draw.text_thickness == pytest.approx(HUD_MULTIPLE)


@pytest.mark.parametrize(
    ("size", "multiplier"),
    [
        ((480, 640), pytest.approx(4 / 9, abs=0.01)),
        ((720, 1280), pytest.approx(2 / 3, abs=0.01)),
        ((1080, 1920), 1.0),
        ((1440, 2560), pytest.approx(4 / 3, abs=0.01)),
        ((2160, 3840), 2.0),
    ],
)
def test_text_grows_with_the_resolution(size, multiplier):
    """1080p is 1.0 by definition; everything else follows the short side."""
    assert draw_scale(np.zeros((*size, 3), np.uint8), DrawConfig()) == multiplier


def caption_ink_height(frame_h: int, frame_w: int) -> int:
    """How tall the caption's ink actually is on a frame of this size."""
    draw = DrawConfig()
    scale = draw_scale(np.zeros((frame_h, frame_w, 3), np.uint8), draw)
    above, below = text_ink_extent(
        "person 0.90",
        draw.text_scale * scale,
        max(1, int(round(draw.text_thickness * scale))),
    )
    return above + below


def test_a_4k_caption_is_twice_the_height_of_a_1080p_one():
    """The whole point of scaling by resolution, measured on real glyphs.

    Not a restatement of `draw_scale`: this goes through the font metrics, which
    is where a scale that is computed and then dropped on the floor would show.
    """
    hd = caption_ink_height(1080, 1920)
    uhd = caption_ink_height(2160, 3840)
    assert hd > 20, "a 1080p caption should be substantial"
    assert uhd == pytest.approx(hd * 2, rel=0.08)


def test_a_1080p_caption_is_visibly_bigger_than_the_old_default():
    """The change is worth having, so its size is asserted rather than assumed."""
    draw = DrawConfig()
    old = text_ink_extent("person 0.90", 1.0, 2)
    new = text_ink_extent("person 0.90", draw.text_scale, draw.text_thickness)
    assert sum(new) > sum(old) * 1.4


# ─────────────────────────────────────────────────────────────────────────────
# The badge's own box
# ─────────────────────────────────────────────────────────────────────────────


def badge_box(draw, fps: float = 28.0, size=(1080, 1920)):
    """The painted badge's bounding box as ``(left, top, width, height)``."""
    img = np.full((*size, 3), 40, np.uint8)
    draw_fps(img, fps, draw)
    filled = (img == np.array(draw.hud_bg_color, np.uint8)).all(axis=2)
    rows, cols = np.where(filled)
    assert rows.size, "no badge was painted"
    return (int(cols.min()), int(rows.min()),
            int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1))


def test_the_badge_sits_off_the_corner_by_its_margin():
    """Margin is its own number now.

    It used to fall through to the padding, so the one value both sized the
    badge and placed it: tightening the box also shoved it into the corner.
    """
    from sima_vision.config import HUD_MARGIN

    draw = DrawConfig()
    assert draw.hud_margin_x == draw.hud_margin_y == HUD_MARGIN == 28
    left, top, _, _ = badge_box(draw)
    assert (left, top) == (HUD_MARGIN, HUD_MARGIN)


def test_the_padding_is_what_sizes_the_badge_around_its_text():
    """260x51 of text in a 280x71 box read as a fill left on by accident."""
    import cv2

    import sima_vision.runtime as rt
    from sima_vision.config import HUD_PADDING

    draw = DrawConfig()
    assert draw.hud_padding == HUD_PADDING == 22
    (text_w, _), _ = cv2.getTextSize(
        "FPS: 28", rt.FONT, draw.hud_text_scale, draw.hud_text_thickness
    )
    above, below = text_ink_extent("FPS: 28", draw.hud_text_scale, draw.hud_text_thickness)

    # cv2.rectangle paints both endpoints, so the filled span is one pixel
    # wider than the box it was asked for.
    _, _, width, height = badge_box(draw)
    assert width - 1 == text_w + HUD_PADDING * 2
    assert height - 1 == above + below + HUD_PADDING * 2


def test_padding_and_margin_scale_with_the_frame():
    """Both are 1080p numbers, like everything else in DrawConfig."""
    draw = DrawConfig()
    left_hd, top_hd, w_hd, h_hd = badge_box(draw, size=(1080, 1920))
    left_4k, top_4k, w_4k, h_4k = badge_box(draw, size=(2160, 3840))
    assert (left_4k, top_4k) == (left_hd * 2, top_hd * 2)
    assert w_4k == pytest.approx(w_hd * 2, rel=0.03)
    assert h_4k == pytest.approx(h_hd * 2, rel=0.03)


def test_zero_still_means_follow_the_caption():
    """The escape hatch survives the defaults moving off 0."""
    draw = DrawConfig(hud_padding=0, hud_margin_x=0, hud_margin_y=0)
    left, top, _, _ = badge_box(draw)
    # With no margin of its own, the badge falls back to its resolved padding,
    # which with hud_padding at 0 is the caption's.
    assert (left, top) == (draw.text_padding, draw.text_padding)


# ─────────────────────────────────────────────────────────────────────────────
# The class palette
# ─────────────────────────────────────────────────────────────────────────────

#: The palette as it was specified, in the order it was given.
PALETTE_HEX = ["F28C28", "05299E", "46237A", "080708"]


def bgr_of(hex_rgb: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_rgb[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def test_the_class_palette_is_the_one_that_was_specified():
    """Written out in hex here, because that is how it will be checked again.

    A BGR tuple in a test is unreadable next to a design; the conversion is the
    part worth asserting, since reversing it silently swaps every box colour.
    """
    from sima_vision.draw import CLASS_COLORS

    assert CLASS_COLORS == [bgr_of(h) for h in PALETTE_HEX]
    assert len(set(CLASS_COLORS)) == len(CLASS_COLORS), "a repeat makes two classes look alike"


def test_class_colours_cycle_and_are_stable():
    from sima_vision.draw import CLASS_COLORS, class_color

    assert class_color(0) == bgr_of("F28C28")
    assert class_color(1) == bgr_of("05299E")
    assert class_color(len(CLASS_COLORS)) == class_color(0)
    assert class_color(79) == CLASS_COLORS[79 % len(CLASS_COLORS)]


def test_every_class_colour_gets_a_readable_caption():
    """The palette is mixed, so the ink is picked per band rather than fixed.

    Checked as a contrast ratio rather than by eye, because a colour added
    later will not be looked at as carefully as these four were.
    """
    from sima_vision.draw import (
        CLASS_COLORS,
        MIN_CONTRAST,
        contrast_ratio,
        readable_text_color,
    )

    preferred = DrawConfig().text_color
    assert preferred == (255, 255, 255)
    for color in CLASS_COLORS:
        ink = readable_text_color(color, preferred)
        ratio = contrast_ratio(color, ink)
        assert ratio >= MIN_CONTRAST, f"{ink} on {color} is only {ratio:.1f}:1"


def test_the_ink_only_moves_where_white_would_fail():
    """The override is a repair, not a restyle.

    White on #F28C28 is 2.5:1 -- text you can see is there and cannot read --
    and black on it is 8.6:1. Every other band keeps the white it already had;
    a caption turning black over a colour that was never a problem would be a
    worse surprise than the one this fixes.
    """
    from sima_vision.draw import BLACK, WHITE, contrast_ratio, readable_text_color

    assert contrast_ratio(class_color(0), WHITE) < 3.0
    assert readable_text_color(class_color(0), WHITE) == BLACK
    for class_id in (1, 2, 3):
        assert readable_text_color(class_color(class_id), WHITE) == WHITE


def test_a_configured_text_colour_is_honoured_while_it_is_readable():
    """`text_color` is a setting, not a suggestion -- until it is unreadable."""
    from sima_vision.draw import BLACK, readable_text_color

    amber = (0, 194, 255)
    # Dark bands take the configured amber, which clears the bar on them.
    assert readable_text_color((8, 7, 8), amber) == amber
    # The dark yellow does not, so it is overridden rather than left illegible.
    assert readable_text_color((11, 134, 184), amber) == BLACK


def test_the_threshold_is_the_large_text_one():
    """4.5 is for body text. These captions are 1.6 scale with 4px strokes."""
    from sima_vision.draw import MIN_CONTRAST

    assert MIN_CONTRAST == 3.0


def test_boxes_and_masks_share_one_palette():
    """detect draws boxes and segment draws masks, both off `class_color`.

    Two lookups that happened to agree would drift; this pins that they are the
    same function.
    """
    from sima_vision.draw import class_color as boxes_use
    from sima_vision.tasks.segment import class_color as masks_use

    assert boxes_use is masks_use


# ─────────────────────────────────────────────────────────────────────────────
# Captions stay inside their own box
# ─────────────────────────────────────────────────────────────────────────────


def band_width(frame, color, above_row: int) -> int:
    """Widest run of the band's fill colour in the rows above ``above_row``.

    Restricted to those rows on purpose: the box outline is the same colour,
    and its top and bottom edges are as wide as the box, so measuring the whole
    frame measures the box rather than the caption.
    """
    region = frame[: max(0, above_row - 2)]
    filled = (region == np.array(color, np.uint8)).all(axis=2)
    return int(filled.sum(axis=1).max()) if filled.size else 0


def one_box(x1, y1, x2, y2, class_id=0, score=0.93):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "score": score, "class_id": class_id}


def test_every_caption_is_the_same_size_whatever_it_labels():
    """One label size per frame, full stop.

    A caption that shrank to fit its box made the same class look like two
    different things across one frame, and gave the smallest detections the
    smallest text -- which is backwards, since those are the ones worth reading
    carefully. The size comes from `text_scale` and the frame, never from the
    box.
    """
    widths = []
    for box_w in (30, 120, 190, 400, 900):
        frame = np.full((1080, 1920, 3), 40, np.uint8)
        draw_boxes(frame, [one_box(400, 400, 400 + box_w, 900)], ["person"], DrawConfig())
        widths.append(band_width(frame, class_color(0), 400))
    assert len(set(widths)) == 1, f"caption size varied with the box: {widths}"


def test_the_caption_size_is_the_configured_one():
    """And that one size is the setting, not something derived behind its back."""
    frame = np.full((1080, 1920, 3), 40, np.uint8)
    draw_boxes(frame, [one_box(400, 400, 500, 900)], ["person"], DrawConfig())

    reference = np.full((1080, 1920, 3), 40, np.uint8)
    draw_caption(reference, "person 0.93", (400, 400), class_color(0), DrawConfig(), 1.0)

    assert band_width(frame, class_color(0), 400) == band_width(
        reference, class_color(0), 400
    )


# ─────────────────────────────────────────────────────────────────────────────
# The background blur never reaches the overlay
# ─────────────────────────────────────────────────────────────────────────────


def segment_render(blur: bool):
    """One segment frame, rendered with the blur on or off."""
    import types
    from dataclasses import replace

    from sima_vision.tasks import TASKS
    from sima_vision.tasks.segment import SegmentRuntime

    cfg = TASKS["segment"]().load(
        None, {"model.path": "m", "source.uri": "c"}, use_file=False
    )
    cfg = replace(cfg, blur=replace(cfg.blur, enable=blur))

    frame = np.zeros((1080, 1920, 3), np.uint8)
    for y in range(0, 1080, 24):                     # fine checks: blur is obvious
        for x in range(0, 1920, 24):
            frame[y:y + 24, x:x + 24] = (
                (210, 205, 195) if (x // 24 + y // 24) % 2 else (55, 60, 70)
            )

    mask = np.zeros((600, 400), bool)
    mask[50:550, 40:360] = True
    inst = Instance(box={"class_id": 0, "score": 0.9}, x1=300, y1=300, x2=700, y2=900,
                    mask=mask, keep=True)
    pipeline = types.SimpleNamespace(labels=["person"])
    return SegmentRuntime().render(cfg, pipeline, frame, [inst], 27.0)


def test_the_blur_is_on_by_default():
    """`sima-vision segment` blurs. The README said "optional" for a while."""
    from sima_vision.tasks import TASKS

    cfg = TASKS["segment"]().load(
        None, {"model.path": "m", "source.uri": "c"}, use_file=False
    )
    assert cfg.blur.enable is True


def test_the_blur_softens_the_background_and_leaves_the_subject_sharp():
    """Measured as local variance: a blurred checkerboard has almost none."""
    blurred, plain = segment_render(True), segment_render(False)

    def detail(img, y, x):
        patch = img[y:y + 120, x:x + 120].astype(np.float32)
        return float(patch.std())

    assert detail(blurred, 60, 1500) < detail(plain, 60, 1500) * 0.5, "background not blurred"
    # Inside the mask, well clear of the feathered edge.
    assert detail(blurred, 500, 420) > detail(plain, 500, 420) * 0.9, "subject was blurred too"


def test_the_overlay_is_identical_whether_the_blur_ran_or_not():
    """The blur composites the image; the overlay is drawn after it.

    So the badge, the captions and the outlines are the same pixels either way.
    The mask tint is deliberately excluded -- it is translucent, so it takes the
    colour of whatever it sits on, which is the whole point of it.
    """
    blurred, plain = segment_render(True), segment_render(False)
    draw = DrawConfig()

    # Every pixel the overlay paints opaquely: the badge fill, the caption
    # band, the box outline and the ink on them. Located by colour rather than
    # by slice, so the test does not have to know where they landed.
    opaque = np.zeros(plain.shape[:2], bool)
    for color in (draw.hud_bg_color, draw.hud_text_color, class_color(0), (255, 255, 255)):
        opaque |= (plain == np.array(color, np.uint8)).all(axis=2)
    assert opaque.sum() > 20_000, "found no overlay to compare"

    differing = (blurred[opaque] != plain[opaque]).any(axis=1).sum()
    assert differing == 0, f"the blur reached {differing} overlay pixels"
