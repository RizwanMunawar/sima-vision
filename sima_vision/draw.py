"""Overlay drawing: palette, text metrics, boxes, the HUD badge and the banner.

Sizes in :class:`~sima_vision.config.DrawConfig` are expressed for a 1080p
frame and multiplied by :func:`draw_scale`, so a config tuned on a test clip
still looks right at 4K.
"""

from __future__ import annotations

from . import runtime

# The class palette, BGR because that is what OpenCV expects. Written as hex
# beside each entry because that is how it was chosen and how it will be
# checked against a design again.
#
# Every one of these is dark -- the brightest sits at a relative luminance of
# 0.038, the darkest at 0.0004 -- so the caption band carries the reading and
# the box outline is a marker rather than the thing you read. Captions are
# white on the same colour, which clears 19:1 on all four.
CLASS_COLORS = [
    (21, 1, 36),      # #240115
    (158, 41, 5),     # #05299E
    (122, 35, 70),    # #46237A
    (8, 7, 8),        # #080708
]

# All digits share one vertical extent in this font, so folding them to a single
# digit keeps the cache to one entry per distinct caption rather than one per
# confidence value.
DIGIT_FOLD = str.maketrans("123456789", "000000000")

_ink_cache: dict[tuple[str, int, int], tuple[int, int]] = {}


def class_color(class_id: int) -> tuple[int, int, int]:
    """Pick a stable colour for a class.

    Args:
        class_id: Model class id.

    Returns:
        A BGR tuple, repeating every :data:`CLASS_COLORS` entries.
    """
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def text_ink_extent(text: str, text_scale: float, text_thickness: int) -> tuple[int, int]:
    """Measure how far this string's ink reaches above and below the baseline.

    ``cv2.getTextSize`` reports a height that includes internal leading and a
    baseline drop sized for descenders, so a band sized from it holds the glyphs
    visibly off centre in whichever direction the string happens to lack. The
    correction has to be per string: ``FPS: 24.7`` has no descender, ``person``
    does. Rendering once and reading back the occupied rows measures exactly
    what will be drawn.

    Args:
        text: The string that will be drawn.
        text_scale: OpenCV font scale.
        text_thickness: Stroke weight in pixels.

    Returns:
        An ``(above, below)`` pair of pixel counts relative to the baseline.
    """
    cv2, np = runtime.cv2, runtime.np
    key = (text.translate(DIGIT_FOLD), int(round(text_scale * 1000)), int(text_thickness))
    cached = _ink_cache.get(key)
    if cached is not None:
        return cached

    (width, height), baseline = cv2.getTextSize(text, runtime.FONT, text_scale, text_thickness)
    margin = max(4, text_thickness * 3)
    canvas = np.zeros((height + baseline + margin * 2, width + margin * 2), np.uint8)
    origin_y = margin + height
    cv2.putText(
        canvas, text, (margin, origin_y), runtime.FONT, text_scale, 255,
        text_thickness, cv2.LINE_AA,
    )
    rows = np.where(canvas.max(axis=1) > 0)[0]
    if rows.size == 0:                          # blank or degenerate, fall back
        extent = (height, baseline)
    else:
        extent = (origin_y - int(rows.min()), max(0, int(rows.max()) - origin_y + 1))
    _ink_cache[key] = extent
    return extent


def draw_scale(frame, draw) -> float:
    """Scale factor that keeps strokes and text proportional to the frame.

    Args:
        frame: The image being annotated.
        draw: Visualization settings.

    Returns:
        A multiplier, 1.0 at the reference height, floored so small frames stay
        readable. Always 1.0 when ``auto_scale`` is off.
    """
    if not draw.auto_scale or draw.reference_height <= 0:
        return 1.0
    return max(0.4, min(frame.shape[:2]) / draw.reference_height)


def caption_text(box: dict, labels: list[str], draw) -> str:
    """Build the caption for one detection.

    Args:
        box: A single detection.
        labels: Class names indexed by class id.
        draw: Visualization settings.

    Returns:
        The caption, which is empty when both the label and the score are off.
    """
    class_id = int(box["class_id"])
    parts = []
    if draw.show_labels:
        parts.append(labels[class_id] if 0 <= class_id < len(labels) else str(class_id))
    if draw.show_scores:
        parts.append(f"{box['score']:.{max(0, draw.score_decimals)}f}")
    return " ".join(parts)


def draw_caption(frame, text: str, anchor: tuple[int, int], color, draw, scale: float) -> None:
    """Draw one filled caption band sitting directly above ``anchor``.

    The band flips to sit inside the box when it would otherwise clip off the
    top of the frame, and is nudged left when it would run off the right edge.

    Args:
        frame: BGR image, modified in place.
        text: Caption string. Nothing is drawn when it is empty.
        anchor: ``(x1, y1)`` top-left of the thing being captioned.
        color: Band fill colour, BGR.
        draw: Visualization settings.
        scale: Frame scale factor from :func:`draw_scale`.
    """
    if not text:
        return
    cv2 = runtime.cv2
    width = frame.shape[1]
    text_scale = draw.text_scale * scale
    text_thickness = max(1, int(round(draw.text_thickness * scale)))
    pad = max(2, int(round(draw.text_padding * scale)))

    (text_w, _), _ = cv2.getTextSize(text, runtime.FONT, text_scale, text_thickness)
    above, below = text_ink_extent(text, text_scale, text_thickness)
    band_w = text_w + pad * 2
    band_h = above + below + pad * 2

    x1, y1 = anchor
    top = y1 - band_h
    if top < 0:                      # would clip off the top, so sit inside
        top = y1
    left = max(0, min(x1, width - band_w))

    cv2.rectangle(frame, (left, top), (left + band_w, top + band_h), color, -1)
    cv2.putText(
        frame, text, (left + pad, top + pad + above), runtime.FONT, text_scale,
        draw.text_color, text_thickness, cv2.LINE_AA,
    )


def draw_boxes(frame, boxes: list[dict], labels: list[str], draw) -> None:
    """Draw detection boxes, centre markers and labelled captions in place.

    Each box is a plain rectangle in the class colour, with a centre dot and a
    filled caption sitting directly above it.

    Args:
        frame: BGR image, modified in place.
        boxes: Detections with ``x1``, ``y1``, ``x2``, ``y2``, ``score`` and
            ``class_id`` keys, in original-image pixels.
        labels: Class names indexed by class id.
        draw: Visualization settings, from the ``visualization`` config section.
    """
    cv2 = runtime.cv2
    h, w = frame.shape[:2]
    scale = draw_scale(frame, draw)
    thickness = max(1, int(round(draw.box_thickness * scale)))
    radius = max(2, int(round(draw.centre_dot_radius * scale)))

    # Paint larger boxes first, so small foreground objects stay legible.
    ordered = sorted(
        boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True
    )
    for box in ordered:
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(w - 1, int(round(box["x2"])))
        y2 = min(h - 1, int(round(box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        color = class_color(int(box["class_id"]))

        if draw.centre_dot:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius, color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        draw_caption(frame, caption_text(box, labels, draw), (x1, y1), color, draw, scale)


def draw_fps(frame, fps: float, draw) -> None:
    """Draw the frame rate centred in a filled badge at the top left.

    Font size, stroke weight and padding fall back to the shared caption
    settings when their ``hud`` counterparts are 0, so the badge matches the
    captions unless it is deliberately given its own look.

    Args:
        frame: BGR image, modified in place.
        fps: Frames per second to display. 0 means nothing has been measured
            yet, and the badge is left off rather than showing a guess.
        draw: Visualization settings.
    """
    if fps <= 0:
        return
    cv2 = runtime.cv2
    scale = draw_scale(frame, draw)
    text_scale = (draw.hud_text_scale or draw.text_scale) * scale
    raw_thickness = draw.hud_text_thickness or draw.text_thickness
    text_thickness = max(1, int(round(raw_thickness * scale)))

    # padding_x / padding_y fall back to the shared hud padding, which itself
    # falls back to the caption padding, so one number still styles everything
    # and two numbers give independent control of the horizontal and vertical
    # breathing room.
    base_pad = draw.hud_padding or draw.text_padding
    pad_x = max(0, int(round((draw.hud_padding_x or base_pad) * scale)))
    pad_y = max(0, int(round((draw.hud_padding_y or base_pad) * scale)))

    text = f"FPS: {fps:.{max(0, draw.hud_fps_decimals)}f}"
    (text_w, _), _ = cv2.getTextSize(text, runtime.FONT, text_scale, text_thickness)
    above, below = text_ink_extent(text, text_scale, text_thickness)

    # Padding sets the badge size; a minimum can only grow it. The text is then
    # centred in whatever box results, so a forced size never pushes it off.
    box_w = max(text_w + pad_x * 2, int(round(draw.hud_min_width * scale)))
    box_h = max(above + below + pad_y * 2, int(round(draw.hud_min_height * scale)))
    left = max(0, int(round((draw.hud_margin_x or (draw.hud_padding_x or base_pad)) * scale))) or 1
    top = max(0, int(round((draw.hud_margin_y or (draw.hud_padding_y or base_pad)) * scale))) or 1

    cv2.rectangle(
        frame, (left, top), (left + box_w, top + box_h), draw.hud_bg_color, -1
    )
    cv2.putText(
        frame,
        text,
        (
            left + (box_w - text_w) // 2,
            top + (box_h - (above + below)) // 2 + above,
        ),
        runtime.FONT,
        text_scale,
        draw.hud_text_color,
        text_thickness,
        cv2.LINE_AA,
    )


def draw_banner(frame, text: str, draw) -> None:
    """Draw a full-width alert strip across the bottom of the frame.

    A red box around one person is easy to miss on a wall of camera tiles. A
    band across the whole frame is not, which is the point of it.

    Args:
        frame: BGR image, modified in place.
        text: Banner text.
        draw: Visualization settings.
    """
    cv2, np = runtime.cv2, runtime.np
    height, width = frame.shape[:2]
    scale = draw_scale(frame, draw)
    text_scale = (draw.banner_text_scale or draw.text_scale) * scale
    text_thickness = max(1, int(round((draw.banner_text_thickness or draw.text_thickness) * scale)))
    pad = max(4, int(round(draw.banner_padding * scale)))

    (text_w, _), _ = cv2.getTextSize(text, runtime.FONT, text_scale, text_thickness)
    above, below = text_ink_extent(text, text_scale, text_thickness)
    band_h = above + below + pad * 2
    top = height - band_h

    strip = frame[top:height, 0:width]
    tint = np.empty_like(strip)
    tint[:] = draw.banner_bg_color
    # Translucent rather than solid, so the band never hides the thing it is
    # drawing attention to.
    cv2.addWeighted(tint, draw.banner_alpha, strip, 1.0 - draw.banner_alpha, 0.0, dst=strip)
    cv2.putText(
        frame, text, (max(pad, (width - text_w) // 2), top + pad + above),
        runtime.FONT, text_scale, draw.banner_text_color, text_thickness, cv2.LINE_AA,
    )
