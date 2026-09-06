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


def test_the_badge_is_purple_and_larger_than_a_caption():
    """Both are deliberate, so both are pinned.

    The badge is glanced at while the video plays rather than read, so it is
    set a little above the caption scale. Purple because a black block reads as
    part of the footage -- as a blown-out shadow or a letterbox bar -- while a
    colour that does not occur in the scene reads as an overlay.
    """
    draw = DrawConfig()
    assert draw.hud_bg_color == (128, 0, 128)
    assert draw.hud_text_scale > draw.text_scale

    img = np.full((1080, 1920, 3), 40, np.uint8)
    draw_fps(img, 24.0, draw)
    assert fill_pixels(img, (128, 0, 128)) > 1000, "no purple badge was painted"


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
