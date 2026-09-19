"""Tracking, the fall state machine and the alert queue.

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
    Alert,
    AlertConfig,
    AlertSender,
    FallConfig,
    Track,
    TrackConfig,
    Tracker,
    alert_password,
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


# ── alerts ──


def test_alerts_off_queue_nothing():
    sender = AlertSender(AlertConfig(enable=False))
    assert sender.offer(make_alert(), 0.0) is False


def make_alert():
    return Alert(track_id=1, label="person", box=(0, 0, 10, 10),
                 signals={}, frame_index=1)


def test_cooldown_suppresses_a_burst():
    sender = AlertSender(AlertConfig(enable=True, dry_run=True, cooldown_seconds=60.0))
    try:
        assert sender.offer(make_alert(), 100.0) is True
        assert sender.offer(make_alert(), 110.0) is False
        assert sender.suppressed == 1
        assert sender.offer(make_alert(), 200.0) is True
    finally:
        sender.close(timeout=2.0)


def test_a_full_queue_drops_rather_than_blocking():
    """A stalled pipeline misses the next fall, which is worse than one lost email."""
    cfg = AlertConfig(enable=True, dry_run=True, cooldown_seconds=0.0, queue_depth=1)
    sender = AlertSender(cfg)
    sender._thread = None                 # stop the worker draining, to fill the queue
    try:
        sender.offer(make_alert(), 0.0)
        sender.offer(make_alert(), 1.0)
        assert sender.dropped >= 1
    finally:
        sender._thread = None


def test_password_comes_from_the_environment(monkeypatch):
    cfg = AlertConfig(username="bot@x.com", password_env="TEST_SMTP_PW")
    monkeypatch.setenv("TEST_SMTP_PW", "hunter2")
    assert alert_password(cfg) == "hunter2"


def test_a_missing_password_is_an_error_not_a_silent_failure(monkeypatch):
    cfg = AlertConfig(username="bot@x.com", password_env="TEST_SMTP_PW")
    monkeypatch.delenv("TEST_SMTP_PW", raising=False)
    with pytest.raises(RuntimeError, match="TEST_SMTP_PW"):
        alert_password(cfg)


def test_no_username_needs_no_password():
    assert alert_password(AlertConfig(username="")) == ""


def test_the_message_carries_the_signals_that_fired():
    cfg = AlertConfig(enable=True, dry_run=True, sender="a@x.com",
                      recipients=("b@x.com",), site="Aisle 4")
    sender = AlertSender(cfg)
    try:
        alert = Alert(
            track_id=7, label="person", box=(10, 20, 110, 60),
            signals={"aspect": True, "aspect_value": 1.9, "collapse": False,
                     "descent": True, "descent_value": 812.5},
            frame_index=42,
        )
        msg = sender.build_message(alert)
        assert "Aisle 4" in msg["Subject"]
        assert "#7" in msg["Subject"]
        body = msg.get_content()
        assert "track     : #7 (person)" in body
        assert "1.9" in body and "812.5" in body
    finally:
        sender.close(timeout=2.0)


# -- what a box says --

def _track(state, track_id=3, class_id=0, score=0.873):
    box = {"x1": 10.0, "y1": 10.0, "x2": 100.0, "y2": 300.0,
           "score": score, "class_id": class_id}
    return fall.Track(track_id=track_id, box=box, state=state)


def test_a_fallen_box_says_one_word():
    """It is the only thing anyone is scanning the frame for.

    The caption used to read `#3 fallen 0.87`: an id nobody asked for, the
    state machine's own vocabulary, and a score -- with the one fact that
    matters third in line and in lower case.
    """
    assert fall.track_caption(_track(fall.FALLEN), fall.FALL_DRAW, ["person"]) == "FALL"


def test_an_upright_box_says_what_was_detected():
    """The class name off the detection, not the state machine's word.

    `upright` is this program's internal vocabulary. Someone watching a
    corridor reads `person`.
    """
    caption = fall.track_caption(_track(fall.UPRIGHT), fall.FALL_DRAW, ["person"])
    assert caption == "person"
    assert fall.UPRIGHT not in caption


def test_a_pending_fall_does_not_start_counting_in_the_caption():
    """`person 0.8/1.5s` is four facts where one is wanted.

    The state is already in the box colour, which is where it belongs.
    """
    caption = fall.track_caption(_track(fall.FALLING), fall.FALL_DRAW, ["person"])
    assert caption == "person"
    assert "/" not in caption


def test_the_class_name_is_read_off_the_detection():
    """A model trained on other classes must not be captioned `person`."""
    labels = ["worker", "forklift", "pallet"]
    caption = fall.track_caption(_track(fall.UPRIGHT, class_id=1), fall.FALL_DRAW, labels)
    assert caption == "forklift"


def test_an_unknown_class_id_does_not_crash_the_overlay():
    track = _track(fall.UPRIGHT, class_id=99)
    assert fall.track_caption(track, fall.FALL_DRAW, ["person"]) == "person"


def test_ids_and_scores_are_off_by_default_but_still_available():
    """Off for reading a frame, on for tuning a tracker."""
    from dataclasses import replace

    assert fall.FALL_DRAW.show_track_ids is False
    assert fall.FALL_DRAW.show_scores is False
    verbose = replace(fall.FALL_DRAW, show_track_ids=True, show_scores=True)
    assert fall.track_caption(_track(fall.UPRIGHT), verbose, ["person"]) == "#3 person 0.87"
    # A fall still says FALL: the flags add detail, they do not bury the word.
    assert fall.track_caption(_track(fall.FALLEN), verbose, ["person"]) == "FALL"
