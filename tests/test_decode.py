"""Each task's ``decode``: sample tree in, frame and results out.

This is where the model's output becomes boxes, and where the decoder's buffer
is handed back. It only ever ran on the board because ``pyneat`` supplies the
sample types -- but the only part of pyneat it touches is ``SampleKind``, so a
stub of that is enough to drive the whole path here.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import sima_vision.runtime as rt
from sima_vision.samples import BBOX_RECORD, extract_bbox_payload, joined_field, parse_boxes
from sima_vision.sinks import Pipeline
from sima_vision.tasks import TASKS
from sima_vision.tasks.fall import FALLEN


class SampleKind:
    Tensor = "tensor"
    TensorSet = "tensorset"
    Bundle = "bundle"


class FakePyneat:
    """Only what the decode path actually reaches for."""

    SampleKind = SampleKind


class FakeTensor:
    def __init__(self, payload: bytes = b"", array=None, nv12=None) -> None:
        self._payload = payload
        self._array = array
        self._nv12 = nv12

    def copy_payload_bytes(self) -> bytes:
        return self._payload

    def to_numpy(self, copy: bool = True):
        if self._array is None:
            raise RuntimeError("not an array tensor")
        return self._array.copy() if copy else self._array

    def is_nv12(self) -> bool:
        return self._nv12 is not None

    def is_i420(self) -> bool:
        return False

    @property
    def width(self):
        return self._nv12[1]

    @property
    def height(self):
        return self._nv12[0]

    semantic = None


class FakeSample:
    def __init__(self, kind, *, tensor=None, tensors=None, fields=(),
                 stream_label="", payload_tag="") -> None:
        self.kind = kind
        self.tensor = tensor
        self.tensors = tensors or []
        self.fields = list(fields)
        self.stream_label = stream_label
        self.payload_tag = payload_tag
        self.pts_ns = -1
        self.dts_ns = -1
        self.duration_ns = 0
        self.frame_id = 0
        self.stream_id = 0


@pytest.fixture(autouse=True)
def fake_pyneat(monkeypatch):
    monkeypatch.setattr(rt, "pyneat", FakePyneat)
    yield


def bbox_payload(boxes) -> bytes:
    out = bytearray(struct.pack("<I", len(boxes)))
    for x, y, w, h, score, cls in boxes:
        out += BBOX_RECORD.pack(x, y, w, h, score, cls)
    return bytes(out)


def nv12_frame(width: int, height: int, luma: int = 128) -> FakeTensor:
    """An NV12 buffer the way the hardware decoder hands one over."""
    plane = np.full((height * 3 // 2, width), luma, np.uint8)
    plane[height:] = 128                       # neutral chroma
    return FakeTensor(payload=plane.tobytes(), nv12=(height, width))


def joined(boxes, width=64, height=48, result_label="detections"):
    """The combined sample the run loop pulls: a frame field and a result field."""
    frame_field = FakeSample(
        SampleKind.Tensor, tensor=nv12_frame(width, height), stream_label="frame"
    )
    result_field = FakeSample(
        SampleKind.Tensor,
        tensor=FakeTensor(payload=bbox_payload(boxes)),
        stream_label=result_label,
        payload_tag="BBOX",
    )
    return FakeSample(SampleKind.Bundle, fields=[frame_field, result_field])


BOXES = [(10, 5, 20, 30, 0.9, 0), (40, 12, 15, 25, 0.6, 2)]


# ── the sample tree ──


def test_joined_field_finds_by_label():
    sample = joined(BOXES)
    assert joined_field(sample, "frame", 0).stream_label == "frame"
    assert joined_field(sample, "detections", 1).stream_label == "detections"


def test_joined_field_falls_back_to_position():
    """An unlabelled bundle still has to be readable."""
    sample = joined(BOXES)
    for field in sample.fields:
        field.stream_label = ""
    assert joined_field(sample, "frame", 0) is sample.fields[0]
    assert joined_field(sample, "detections", 1) is sample.fields[1]


def test_a_missing_field_is_an_error():
    sample = FakeSample(SampleKind.Bundle, fields=[])
    with pytest.raises(RuntimeError, match="missing the `frame` field"):
        joined_field(sample, "frame", 0)


def test_extract_walks_a_bundle_to_the_bbox_tensor():
    payload, tensor = extract_bbox_payload(joined(BOXES))
    assert parse_boxes(payload, 64, 48, 50)[0]["class_id"] == 0
    assert tensor is not None


def test_a_wrongly_tagged_tensor_is_rejected():
    """`raw_output_heads` means the route skipped BoxDecode; say so."""
    bad = FakeSample(
        SampleKind.Tensor, tensor=FakeTensor(payload=b"\x00" * 8),
        payload_tag="RAW_OUTPUT_HEADS",
    )
    with pytest.raises(RuntimeError, match="expected a BBOX tensor"):
        extract_bbox_payload(bad)


def test_a_tensor_set_is_scanned_not_assumed():
    """A segment head packs several tensors; the box one is not always first."""
    junk = FakeTensor(payload=b"")            # empty -> rejected
    good = FakeTensor(payload=bbox_payload(BOXES))
    sample = FakeSample(
        SampleKind.TensorSet, tensors=[junk, good], payload_tag="BBOX"
    )
    payload, tensor = extract_bbox_payload(sample)
    assert tensor is good
    assert len(parse_boxes(payload, 64, 48, 50)) == 2


# ── detect ──


def detect_pipeline(width=64, height=48):
    pipeline = Pipeline(labels=["person", "bike", "car"], frame_w=width, frame_h=height)
    pipeline.fps = 25
    return pipeline


def test_detect_decode_returns_a_bgr_frame_and_boxes():
    task = TASKS["detect"]()
    cfg = task.load(None, {"model.path": "m", "source.uri": "c"}, use_file=False)
    pipeline = detect_pipeline()
    frame, boxes, stage = task.runtime(cfg, pipeline).decode(
        pipeline, cfg, joined(BOXES), 1
    )
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert [b["class_id"] for b in boxes] == [0, 2]
    assert boxes[0]["x2"] == 30.0          # x + w
    assert stage == 0.0


# ── fall ──


def fall_setup(**settings):
    task = TASKS["fall"]()
    cfg = task.load(
        None,
        {"model.path": "m", "source.uri": "c", "tracking.min_hits": 1, **settings},
        use_file=False,
    )
    pipeline = task.make_pipeline(cfg, ["person"] + [f"c{i}" for i in range(1, 10)])
    pipeline.frame_w, pipeline.frame_h, pipeline.fps = 200, 400, 25
    return task, cfg, pipeline


def person(x, y, w, h, score=0.9):
    return (x, y, w, h, score, 0)


def test_fall_decode_tracks_across_frames():
    task, cfg, pipeline = fall_setup()
    runtime = task.runtime(cfg, pipeline)
    ids = []
    for index in range(1, 5):
        _, tracks, _ = runtime.decode(
            pipeline, cfg, joined([person(50, 40 + index, 30, 90)], 200, 400), index
        )
        ids.append([t.track_id for t in tracks])
    assert ids[-1] == [1], "one person should stay one track"


def test_fall_decode_drops_classes_that_cannot_fall():
    task, cfg, pipeline = fall_setup()
    runtime = task.runtime(cfg, pipeline)
    # class 2 is not `person`, so it must never become a track.
    _, tracks, _ = runtime.decode(
        pipeline, cfg,
        joined([(50, 40, 30, 90, 0.9, 2)], 200, 400), 1,
    )
    assert tracks == []


def test_fall_decode_drops_boxes_too_small_to_judge():
    task, cfg, pipeline = fall_setup(**{"fall.min_box_height": 0.5})
    runtime = task.runtime(cfg, pipeline)
    _, tracks, _ = runtime.decode(
        pipeline, cfg, joined([person(50, 40, 30, 90)], 200, 400), 1
    )
    assert tracks == [], "a 90px box in a 400px frame is under the 50% floor"


def test_a_fall_is_counted_and_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task, cfg, pipeline = fall_setup(**{"fall.confirm_seconds": 0.0})
    runtime = task.runtime(cfg, pipeline)
    # Upright first, so the track learns a reference height...
    for index in range(1, 4):
        runtime.decode(pipeline, cfg, joined([person(50, 40, 30, 200)], 200, 400), index)
    # ...then wide and short, which is what lying down looks like.
    for index in range(4, 8):
        _, tracks, _ = runtime.decode(
            pipeline, cfg, joined([person(40, 300, 160, 60)], 200, 400), index
        )
    assert pipeline.falls >= 1
    assert any(t.state == FALLEN for t in tracks)


def test_a_fall_only_changes_the_class_on_the_frame(tmp_path, monkeypatch):
    """No separate overlay: the fallen box is captioned FALL and that is all."""
    from sima_vision.tasks.fall import FALL_CLASS, overlay_boxes

    monkeypatch.chdir(tmp_path)
    task, cfg, pipeline = fall_setup(**{"fall.confirm_seconds": 0.0})
    runtime = task.runtime(cfg, pipeline)
    for index in range(1, 4):
        runtime.decode(pipeline, cfg, joined([person(50, 40, 30, 200)], 200, 400), index)
    for index in range(4, 8):
        _, tracks, _ = runtime.decode(
            pipeline, cfg, joined([person(40, 300, 160, 60)], 200, 400), index
        )
    boxes, labels = overlay_boxes(tracks, pipeline.labels, pipeline.fall_class_ids)
    assert FALL_CLASS in [labels[b["class_id"]] for b in boxes]


# ── segment ──


def packed_payload(boxes, slots: int, side: int, count: int) -> bytes:
    """What neatobjectdecode emits for a segment head.

    ``[uint32 count][RawBox 24B * slots][mask side*side uint8 * slots]`` -- the
    masks live in the tail of the *same* buffer as the boxes.
    """
    out = bytearray(struct.pack("<I", len(boxes)))
    for x, y, w, h, score, cls in boxes:
        out += BBOX_RECORD.pack(x, y, w, h, score, cls)
    out += b"\x00" * ((slots - len(boxes)) * BBOX_RECORD.size)
    planes = np.zeros((slots, side, side), np.uint8)
    for i in range(count):
        planes[i, side // 4 : 3 * side // 4, side // 4 : 3 * side // 4] = 255
    return bytes(out) + planes.tobytes()


def segment_sample(boxes, slots=8, side=32, count=2, width=64, height=48):
    frame_field = FakeSample(
        SampleKind.Tensor, tensor=nv12_frame(width, height), stream_label="frame"
    )
    instances = FakeSample(
        SampleKind.Tensor,
        tensor=FakeTensor(payload=packed_payload(boxes, slots, side, count)),
        stream_label="instances",
        payload_tag="BBOX",
    )
    return FakeSample(SampleKind.Bundle, fields=[frame_field, instances])


def segment_setup(**settings):
    task = TASKS["segment"]()
    cfg = task.load(
        None,
        {"model.path": "m", "source.uri": "c", "decode.max_detections": 8,
         "segmentation.mask_sides": [32], **settings},
        use_file=False,
    )
    pipeline = task.make_pipeline(cfg, ["person", "bike", "car"])
    pipeline.frame_w, pipeline.frame_h, pipeline.fps = 64, 48, 25
    return task, cfg, pipeline


def test_segment_decode_recovers_packed_masks():
    task, cfg, pipeline = segment_setup()
    runtime = task.runtime(cfg, pipeline)
    frame, instances, stage_ms = runtime.decode(
        pipeline, cfg, segment_sample(BOXES, slots=8, side=32, count=2), 1
    )
    assert frame.shape == (48, 64, 3)
    assert len(instances) == 2
    assert pipeline.mask_kind == "packed", "the packed layout should have solved"
    assert all(i.mask is not None for i in instances), "every instance needs a mask"
    assert stage_ms >= 0.0


def test_segment_masks_are_shaped_to_their_own_box():
    """foreground_mask slices the frame by the box, so the two must agree."""
    task, cfg, pipeline = segment_setup()
    runtime = task.runtime(cfg, pipeline)
    _, instances, _ = runtime.decode(pipeline, cfg, segment_sample(BOXES), 1)
    for inst in instances:
        assert inst.mask.shape == (inst.y2 - inst.y1, inst.x2 - inst.x1)


def test_segment_composites_without_a_shape_error():
    """The end-to-end check that a decoded mask can actually be blurred with."""
    from sima_vision.masks import composite, foreground_mask

    task, cfg, pipeline = segment_setup()
    runtime = task.runtime(cfg, pipeline)
    frame, instances, _ = runtime.decode(pipeline, cfg, segment_sample(BOXES), 1)
    out = composite(frame, foreground_mask(instances, frame.shape), cfg.blur, 1.0)
    assert out.shape == frame.shape


def test_segment_falls_back_to_boxes_when_there_is_no_mask_data(capsys):
    task, cfg, pipeline = segment_setup(**{"segmentation.source": "planes"})
    runtime = task.runtime(cfg, pipeline)
    # `planes` never matches a packed buffer, so no masks are found.
    _, instances, _ = runtime.decode(pipeline, cfg, segment_sample(BOXES), 1)
    assert instances and all(i.mask is None for i in instances)
    assert "falling back to box-shaped" in capsys.readouterr().err


def test_segment_can_refuse_to_fall_back():
    task, cfg, pipeline = segment_setup(**{
        "segmentation.source": "planes", "segmentation.fallback_to_boxes": False,
    })
    runtime = task.runtime(cfg, pipeline)
    with pytest.raises(RuntimeError, match="no mask data"):
        runtime.decode(pipeline, cfg, segment_sample(BOXES), 1)


def test_segment_describes_the_output_once(capsys):
    task, cfg, pipeline = segment_setup(**{"segmentation.describe": True})
    runtime = task.runtime(cfg, pipeline)
    for index in (1, 2, 3):
        runtime.decode(pipeline, cfg, segment_sample(BOXES), index)
    printed = capsys.readouterr().out
    assert printed.count("model output tensors") == 1, "the dump is a one-off"
    assert "packed layout" in printed


def test_keep_classes_marks_only_those_as_foreground():
    task, cfg, pipeline = segment_setup(**{"blur.keep_classes": ["person"]})
    runtime = task.runtime(cfg, pipeline)
    _, instances, _ = runtime.decode(pipeline, cfg, segment_sample(BOXES), 1)
    kept = {i.box["class_id"]: i.keep for i in instances}
    assert kept[0] is True, "person is kept sharp"
    assert kept[2] is False, "car is not"
