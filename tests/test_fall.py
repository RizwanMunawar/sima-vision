"""Tracking and the fall state machine.

Pure Python, no board and no OpenCV. These are the rules a safety feature turns
on, so they are worth stating as tests rather than trusting to a clip.
"""

from __future__ import annotations

import pytest

from sima_vision.tasks import fall
from sima_vision.tasks.fall import (
    FALLEN,
    FALLING,
    RECOVERING,
    UPRIGHT,
    FallConfig,
    Track,
    TrackConfig,
    Tracker,
    box_iou,
    descent_rate,
    fall_signals,
    update_fall_states,
)

FRAME_H = 1080


def box(x1, y1, x2, y2, cls=0, score=0.9):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "score": score, "class_id": cls}


def standing(cx=100, cy=500, h=300):
    """A tall, narrow box: aspect well under 1."""
    w = h * 0.4
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def lying(cx=100, cy=900, w=300):
    """A wide, short box: aspect well over 1."""
    h = w * 0.4
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ── IoU ──


def test_box_iou_identical_is_one():
    assert box_iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_box_iou_disjoint_is_zero():
    assert box_iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_box_iou_half_overlap():
    # 10x10 and 10x10 sharing a 5x10 strip: 50 / (100 + 100 - 50)
    assert box_iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(50 / 150)


# ── tracker ──


def test_min_hits_suppresses_a_one_frame_detection():
    tracker = Tracker(TrackConfig(min_hits=3))
    assert tracker.update([standing()], 0.0) == []
    assert tracker.update([standing()], 0.1) == []
    assert len(tracker.update([standing()], 0.2)) == 1


def test_track_ids_are_stable_across_frames():
    tracker = Tracker(TrackConfig(min_hits=1))
    first = tracker.update([standing(cx=100)], 0.0)[0].track_id
    second = tracker.update([standing(cx=104)], 0.1)[0].track_id
    assert first == second


def test_two_people_get_two_ids():
    tracker = Tracker(TrackConfig(min_hits=1))
    tracks = tracker.update([standing(cx=100), standing(cx=900)], 0.0)
    assert len({t.track_id for t in tracks}) == 2


def test_a_track_survives_a_brief_occlusion():
    tracker = Tracker(TrackConfig(min_hits=1, max_age=5))
    first = tracker.update([standing(cx=100)], 0.0)[0].track_id
    for step in range(3):
        assert tracker.update([], 0.1 * (step + 1)) == []      # missed, not reported
    back = tracker.update([standing(cx=100)], 0.4)
    assert back and back[0].track_id == first


def test_a_track_is_dropped_past_max_age():
    tracker = Tracker(TrackConfig(min_hits=1, max_age=2))
    tracker.update([standing(cx=100)], 0.0)
    for step in range(4):
        tracker.update([], 0.1 * (step + 1))
    assert tracker.tracks == []


def test_history_is_trimmed_to_the_window():
    tracker = Tracker(TrackConfig(min_hits=1, history_seconds=0.5))
    for step in range(40):
        tracker.update([standing(cx=100)], step * 0.1)
    # 0.5s at 10 fps is about 5 samples; it must not grow to 40.
    assert len(tracker.tracks[0].history) < 12


def test_upright_height_is_learned_only_from_upright_frames():
    tracker = Tracker(TrackConfig(min_hits=1, upright_aspect=0.7))
    tracker.update([standing(h=300)], 0.0)
    learned = tracker.tracks[0].upright_height
    assert learned == pytest.approx(300, abs=1)
    # A lying box is wider than upright_aspect, so it must not move the bar.
    tracker.update([lying(w=300)], 0.1)
    assert tracker.tracks[0].upright_height == pytest.approx(learned, abs=1)


# ── signals ──


def track_with_history(samples, box_now):
    track = Track(track_id=1, box=box_now)
    track.history = list(samples)
    return track


def test_descent_rate_measures_downward_speed():
    # centre_y goes 100 -> 200 over 0.5s == 200 px/s
    track = track_with_history(
        [(0.0, 100.0, 300.0, 0.4), (0.5, 200.0, 300.0, 0.4)], standing()
    )
    assert descent_rate(track, 1.0) == pytest.approx(200.0)


def test_descent_rate_needs_history():
    assert descent_rate(track_with_history([], standing()), 1.0) == 0.0


def test_aspect_signal_fires_when_lying_down():
    track = Track(track_id=1, box=lying())
    assert fall_signals(track, FallConfig(), FRAME_H)["aspect"] is True


def test_aspect_signal_is_quiet_when_standing():
    track = Track(track_id=1, box=standing())
    assert fall_signals(track, FallConfig(), FRAME_H)["aspect"] is False


def test_collapse_signal_needs_a_learned_upright_height():
    track = Track(track_id=1, box=standing(h=100))
    assert fall_signals(track, FallConfig(), FRAME_H)["collapse"] is False
    track.upright_height = 300.0                 # now 100 is a third of upright
    assert fall_signals(track, FallConfig(), FRAME_H)["collapse"] is True


# ── state machine ──


def test_a_fall_must_be_held_before_it_is_confirmed():
    fall = FallConfig(confirm_seconds=1.0)
    track = Track(track_id=1, box=lying(), state=UPRIGHT, state_since=0.0)

    assert update_fall_states([track], fall, FRAME_H, 0.0) == []
    assert track.state == FALLING

    assert update_fall_states([track], fall, FRAME_H, 0.5) == []
    assert track.state == FALLING                # not long enough yet

    fired = update_fall_states([track], fall, FRAME_H, 1.1)
    assert fired == [track]
    assert track.state == FALLEN


def test_a_fall_fires_once_not_every_frame():
    fall = FallConfig(confirm_seconds=0.5)
    track = Track(track_id=1, box=lying(), state=UPRIGHT, state_since=0.0)
    update_fall_states([track], fall, FRAME_H, 0.0)
    assert update_fall_states([track], fall, FRAME_H, 1.0) == [track]
    for t in (1.1, 1.2, 2.0):
        assert update_fall_states([track], fall, FRAME_H, t) == []


def test_brief_recovery_does_not_report_the_same_fall_again():
    cfg = FallConfig(confirm_seconds=0.1, recover_seconds=1.0)
    track = Track(track_id=1, box=lying())
    update_fall_states([track], cfg, FRAME_H, 0.0)
    assert update_fall_states([track], cfg, FRAME_H, 0.2) == [track]
    track.box = standing()
    update_fall_states([track], cfg, FRAME_H, 0.3)
    track.box = lying()
    for now in (0.4, 0.6, 1.0):
        assert update_fall_states([track], cfg, FRAME_H, now) == []
    assert track.state == FALLEN
    track.box = standing()
    update_fall_states([track], cfg, FRAME_H, 2.0)
    update_fall_states([track], cfg, FRAME_H, 3.1)
    track.box = lying()
    update_fall_states([track], cfg, FRAME_H, 3.2)
    assert update_fall_states([track], cfg, FRAME_H, 3.4) == [track]


def test_standing_back_up_moves_to_recovering_then_upright():
    fall = FallConfig(confirm_seconds=0.1, recover_seconds=1.0)
    track = Track(track_id=1, box=lying(), state=UPRIGHT, state_since=0.0)
    update_fall_states([track], fall, FRAME_H, 0.0)
    update_fall_states([track], fall, FRAME_H, 0.5)
    assert track.state == FALLEN

    track.box = standing()
    update_fall_states([track], fall, FRAME_H, 1.0)
    assert track.state == RECOVERING
    update_fall_states([track], fall, FRAME_H, 3.0)
    assert track.state == UPRIGHT


def test_a_brief_crouch_never_reaches_fallen():
    fall = FallConfig(confirm_seconds=2.0)
    track = Track(track_id=1, box=lying(), state=UPRIGHT, state_since=0.0)
    update_fall_states([track], fall, FRAME_H, 0.0)
    assert track.state == FALLING
    track.box = standing()
    assert update_fall_states([track], fall, FRAME_H, 0.5) == []
    assert track.state == RECOVERING


# -- what a box says --

def _track(state, track_id=3, class_id=0, score=0.873):
    box = {"x1": 10.0, "y1": 10.0, "x2": 100.0, "y2": 300.0,
           "score": score, "class_id": class_id}
    return fall.Track(track_id=track_id, box=box, state=state)


def test_a_fall_is_a_change_of_class_and_nothing_else():
    """The whole visual difference between this app and `detect`.

    A fallen track's box is relabelled; every other field stays exactly as the
    detector reported it, so the same `draw_boxes` draws both.
    """
    tracks = [_track(fall.UPRIGHT, class_id=0), _track(fall.FALLEN, class_id=0)]
    boxes, labels = fall.overlay_boxes(tracks, ["person"], [0])

    assert labels[boxes[0]["class_id"]] == "person"
    assert labels[boxes[1]["class_id"]] == fall.FALL_CLASS
    for box, track in zip(boxes, tracks, strict=True):
        assert box["score"] == track.box["score"]
        assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (10.0, 10.0, 100.0, 300.0)


def test_the_state_machines_own_words_never_reach_the_frame():
    """`upright` and `recovering` are internal vocabulary.

    Someone watching a corridor reads the class name, and only ever sees FALL
    when that person is on the floor.
    """
    for state in (fall.UPRIGHT, fall.FALLING, fall.RECOVERING):
        boxes, labels = fall.overlay_boxes([_track(state)], ["person"], [0])
        assert labels[boxes[0]["class_id"]] == "person", state


def test_a_fallen_box_is_relabelled_whatever_it_was_detected_as():
    """FALL replaces the class; it does not depend on the class being person."""
    boxes, labels = fall.overlay_boxes(
        [_track(fall.FALLEN, class_id=1)], ["worker", "forklift"], [1]
    )
    assert labels[boxes[0]["class_id"]] == fall.FALL_CLASS


def test_fall_is_coloured_from_the_same_palette_as_everything_else():
    """No fall-specific colour. It takes a palette entry like any class."""
    from sima_vision.draw import CLASS_COLORS, class_color

    boxes, _ = fall.overlay_boxes([_track(fall.FALLEN)], ["person"], [0])
    assert class_color(boxes[0]["class_id"]) in CLASS_COLORS


def test_fall_does_not_come_out_the_same_colour_as_the_class_it_replaces():
    """80 COCO classes and four colours put FALL straight back on person's.

    Which would paint a fallen person the same colour as the one standing next
    to them -- the one case the colour has to distinguish.
    """
    from sima_vision.draw import class_color

    labels = [f"class{i}" for i in range(80)]
    boxes, _ = fall.overlay_boxes([_track(fall.FALLEN, class_id=0)], labels, [0])
    assert class_color(boxes[0]["class_id"]) != class_color(0)


def test_fall_avoids_every_tracked_class_colour_it_can():
    """Not just the first one: any class that can fall is a class to avoid."""
    from sima_vision.draw import class_color

    labels = [f"class{i}" for i in range(80)]
    tracked = [0, 1, 2]
    boxes, _ = fall.overlay_boxes([_track(fall.FALLEN, class_id=0)], labels, tracked)
    used = {class_color(i) for i in tracked}
    assert class_color(boxes[0]["class_id"]) not in used


def test_the_fall_app_draws_with_the_detect_settings():
    """The same object, not a copy of the numbers, so they cannot drift."""
    from sima_vision.tasks.detect import DETECT_DRAW
    from sima_vision.tasks.fall import FallTask

    assert FallTask.defaults.draw is DETECT_DRAW
