"""Reading frames and detections out of a pyneat sample.

BoxDecode emits one UInt8 tensor tagged BBOX per frame::

    [uint32 N][RawBox 24B] * N ... trailing padding
    RawBox = <iiiifi  ->  x, y, w, h, score, class_id   (source-image pixels)

A segment head packs its masks into the tail of that same buffer, which is why
:func:`extract_bbox_payload` hands the tensor back as well as the bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from . import runtime

BBOX_RECORD = struct.Struct("<iiiifi")
BBOX_RECORD_SIZE = BBOX_RECORD.size  # 24

#: Payload tags that mean "this is a box array". A segment head tags its packed
#: buffer differently from a detect head, but the box records are identical.
BBOX_TAGS = {"BBOX", "BBOXSEG", "BBOX_SEG", "BBOXSEGMENT", "DETECTION", "DETECTIONS"}


@dataclass(frozen=True)
class FrameStamp:
    """The timing fields a sink needs, copied out of a sample as plain ints.

    The point is to stop holding the sample itself. A decoded frame is a buffer
    from the hardware decoder's pool, and that pool is small: this board reports
    ``BufferNum=8``. Anything still referencing a sample is still holding one of
    those eight, so keeping one alive across the next ``pull()`` -- while
    compositing, encoding a JPEG and writing a video frame -- starves the
    decoder and deadlocks the pipeline. The run stops with a pull timeout after
    almost exactly as many frames as the pool has buffers.
    """

    pts_ns: int = -1
    dts_ns: int = -1
    duration_ns: int = 0
    frame_id: int = -1
    stream_id: int = 0

    @classmethod
    def of(cls, sample) -> FrameStamp:
        return cls(
            pts_ns=getattr(sample, "pts_ns", -1),
            dts_ns=getattr(sample, "dts_ns", -1),
            duration_ns=getattr(sample, "duration_ns", 0),
            frame_id=getattr(sample, "frame_id", -1),
            stream_id=getattr(sample, "stream_id", 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sample tree navigation
# ─────────────────────────────────────────────────────────────────────────────


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def first_tensor(sample):
    pyneat = runtime.pyneat
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for candidate in sample.fields:
        tensor = first_tensor(candidate)
        if tensor is not None:
            return tensor
    return None


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for candidate in getattr(sample, "fields", []):
        found = find_field(candidate, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    pyneat = runtime.pyneat
    field_sample = find_field(sample, label)
    if field_sample is not None:
        return field_sample
    fields = list(getattr(sample, "fields", []))
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output is missing the `{label}` field")


def iter_tensors(sample):
    """Yield every ``(sample, tensor)`` pair in a sample tree, depth first."""
    pyneat = runtime.pyneat
    kind = getattr(sample, "kind", None)
    if kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample, sample.tensor
    elif kind == pyneat.SampleKind.TensorSet:
        for tensor in sample.tensors or []:
            yield sample, tensor
    for candidate in getattr(sample, "fields", []) or []:
        yield from iter_tensors(candidate)


def sample_payload_tag(sample, tensor) -> str:
    """Best-effort payload tag for one tensor, upper-cased. Empty when unknown."""
    tag = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    if not tag:
        semantic = getattr(tensor, "semantic", None)
        tess = getattr(semantic, "tess", None)
        if tess is not None:
            tag = getattr(tess, "format", "")
    return str(tag or "").upper()


def frame_to_bgr(tensor):
    """Decoded frames arrive as NV12 from the hardware decoder / libcamera."""
    cv2, np = runtime.cv2, runtime.np
    if tensor.is_nv12() or tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"YUV payload too small: {payload.size} < {expected}")
        planar = payload[:expected].reshape((height * 3 // 2, width))
        code = cv2.COLOR_YUV2BGR_NV12 if tensor.is_nv12() else cv2.COLOR_YUV2BGR_I420
        return np.ascontiguousarray(cv2.cvtColor(planar, code))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


# ─────────────────────────────────────────────────────────────────────────────
# BBOX payload
# ─────────────────────────────────────────────────────────────────────────────


def tensor_bbox_payload(sample, tensor=None, require_tag: bool = False) -> bytes:
    """The bytes of one BBOX tensor.

    Args:
        sample: The sample owning the tensor, for its payload tag.
        tensor: The tensor, or None to take ``sample.tensor``.
        require_tag: When True, an *untagged* tensor is rejected too. A decoded
            frame is untagged, and accepting one here would parse an image as
            box records; see :func:`extract_bbox_payload`.
    """
    tensor = tensor if tensor is not None else getattr(sample, "tensor", None)
    if tensor is None:
        raise RuntimeError("detection sample carries no tensor")
    fmt = sample_payload_tag(sample, tensor)
    if require_tag and fmt not in BBOX_TAGS:
        raise RuntimeError(f"not a tagged BBOX tensor (tag {fmt or '<none>'})")
    if fmt and fmt not in BBOX_TAGS:
        raise RuntimeError(
            f"expected a BBOX tensor but got {fmt}. If this is `raw_output_heads`, "
            "the route did not include BoxDecode. Check model.family and the model archive."
        )
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("empty BBOX payload")
    return payload


def extract_bbox_payload(sample, require_tag: bool = False) -> tuple[bytes, object]:
    """Find the BBOX tensor in a sample tree.

    A tensor that carries no payload tag at all is accepted, because not every
    head tags its output -- but only once nothing *explicitly* tagged BBOX has
    been found. A joined sample carries the decoded frame as well, and that
    frame is untagged: taking the first untagged tensor would parse an image as
    box records and produce coordinates in the millions.

    Args:
        sample: A tensor, tensor set or bundle.
        require_tag: Internal. Restricts a pass to explicitly tagged tensors.

    Returns:
        A ``(payload, tensor)`` pair. The tensor is handed back so the mask
        decoder can exclude it when it goes looking for mask data. Detection
        callers ignore the second element.
    """
    pyneat = runtime.pyneat
    if sample.kind == pyneat.SampleKind.Bundle:
        # Tagged first, untagged only as a fallback.
        for strict in ((True, False) if not require_tag else (True,)):
            for candidate in sample.fields:
                try:
                    return extract_bbox_payload(candidate, require_tag=strict)
                except RuntimeError:
                    continue
        raise RuntimeError("bundle has no BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        # A segment head packs several tensors into one set. The box tensor is
        # normally first, but scan rather than assume: an unexpected order would
        # otherwise be parsed as boxes and produce garbage coordinates.
        for strict in ((True, False) if not require_tag else (True,)):
            for tensor in sample.tensors:
                try:
                    return tensor_bbox_payload(sample, tensor, require_tag=strict), tensor
                except RuntimeError:
                    continue
        raise RuntimeError("tensor set has no BBOX tensor")
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError(f"unexpected sample kind {sample.kind}")
    return tensor_bbox_payload(sample, require_tag=require_tag), sample.tensor


def parse_boxes(payload: bytes, img_w: int, img_h: int, expected_topk: int) -> list[dict]:
    if len(payload) < 4:
        raise RuntimeError("BBOX buffer too small to hold a count header")
    count = struct.unpack_from("<I", payload, 0)[0]
    capacity = (len(payload) - 4) // BBOX_RECORD_SIZE
    if count > capacity:
        raise RuntimeError(f"BBOX header count {count} exceeds payload capacity {capacity}")
    if expected_topk > 0 and count > expected_topk:
        raise RuntimeError(f"BBOX header count {count} exceeds top_k {expected_topk}")

    boxes = []
    offset = 4
    for _ in range(count):
        x, y, w, h, score, class_id = BBOX_RECORD.unpack_from(payload, offset)
        offset += BBOX_RECORD_SIZE
        boxes.append(
            {
                "x1": max(0.0, min(float(x), float(img_w))),
                "y1": max(0.0, min(float(y), float(img_h))),
                "x2": max(0.0, min(float(x + w), float(img_w))),
                "y2": max(0.0, min(float(y + h), float(img_h))),
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes


def frame_geometry_warning(frame, frame_w: int, frame_h: int) -> str:
    """A warning when the decoded frame is not the geometry the graph was told.

    The whole box pipeline is built on these two numbers agreeing. They are
    probed off the stream's SPS, handed to Neat as the preprocess capacity and
    the letterbox target, and the box decoder maps its output back into *that*
    rectangle -- while the overlay is drawn on whatever the decoder actually
    handed back. When the two differ, every box is placed against a frame that
    is not the frame being drawn on, and the boxes sit near the people instead
    of on them, by a margin that grows down and across the picture.

    They can differ honestly. H.264 codes 1080 rows as 1088 and carries the
    crop in the SPS, so a path that does not apply the crop yields a frame
    eight rows taller than the geometry every other part of the run agreed on.

    Nothing is corrected here, deliberately. Whether the right fix is a scale
    or a crop depends on *why* they differ, and guessing wrong moves boxes that
    were already right. Saying so precisely, once, is worth more than a
    plausible silent adjustment.

    Args:
        frame: The decoded BGR frame, or None.
        frame_w: Width the graph was built for.
        frame_h: Height the graph was built for.

    Returns:
        The warning, or "" when the geometry agrees or cannot be read.
    """
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return ""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if (width, height) == (frame_w, frame_h) or not frame_w or not frame_h:
        return ""
    return (
        f"the decoded frame is {width}x{height}, and the graph was built for "
        f"{frame_w}x{frame_h}.\n"
        "Boxes are decoded against the second and drawn on the first, so every "
        "one of them\n"
        "will sit off its subject. Pin the real geometry and the two agree:\n"
        f"  --width {width} --height {height}"
    )


def describe_tensors(sample) -> str:
    """One line per tensor: tag, dtype, shape and byte length.

    Which tensors a head emits, and in what shape, is the one thing the app
    cannot know ahead of time, so it reports what actually arrived instead of
    guessing silently.
    """
    np = runtime.np
    lines = []
    for index, (owner, tensor) in enumerate(iter_tensors(sample)):
        tag = sample_payload_tag(owner, tensor) or "<untagged>"
        label = getattr(owner, "stream_label", "") or "<unlabelled>"
        try:
            array = np.asarray(tensor.to_numpy(copy=False))
            shape, dtype = tuple(int(d) for d in array.shape), str(array.dtype)
        except Exception as exc:
            shape, dtype = f"<to_numpy failed: {exc}>", "?"
        try:
            nbytes = len(tensor.copy_payload_bytes())
        except Exception:
            nbytes = -1
        lines.append(
            f"  [{index}] stream={label} tag={tag} dtype={dtype} shape={shape} bytes={nbytes}"
        )
    return "\n".join(lines) or "  <no tensors>"


def resolve_classes(tokens, labels: list[str], what: str, labels_path) -> set[int] | None:
    """Turn a list of class names or ids into a set of class ids.

    Accepts either names as they appear in the labels file or bare numeric ids,
    so ``[person, 2]`` is valid. A name that is not in the labels file is a
    config error, not a silently empty filter: the alternative is a run that
    does the wrong thing and gives no clue why.

    Args:
        tokens: Class names or ids from the config.
        labels: Class names indexed by class id.
        what: The config key being resolved, for the error message.
        labels_path: Where the labels came from, for the error message.

    Returns:
        A set of class ids, or None when the list is empty and every class
        counts.
    """
    if not tokens:
        return None
    lookup = {name.lower(): index for index, name in enumerate(labels)}
    ids: set[int] = set()
    for token in tokens:
        text = str(token).strip()
        if text.isdigit():
            index = int(text)
            if not 0 <= index < len(labels):
                raise ValueError(
                    f"{what} has class id {index}, but the labels file "
                    f"only has {len(labels)} classes (0 to {len(labels) - 1})"
                )
            ids.add(index)
            continue
        index = lookup.get(text.lower())
        if index is None:
            near = [name for name in labels if text.lower() in name.lower()][:5]
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise ValueError(
                f"{what} has unknown class {text!r}, which is not in {labels_path}.{hint}"
            )
        ids.add(index)
    return ids
