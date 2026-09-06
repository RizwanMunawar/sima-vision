"""The pull loop and the sink thread.

This is the part that only ran on the board, and the part most likely to be
wrong: it is threaded, it owns buffer lifetime, and its failure mode is a
deadlock rather than an exception. Faking ``pipeline.run`` is enough to drive
all of it off the board -- ``pull`` returning objects is the entire contract
the loop depends on.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from sima_vision.runloop import (
    FrameStamp,
    Stopper,
    TaskRuntime,
    run_pipeline,
    source_stopped_message,
    timing_report,
)
from sima_vision.sinks import Pipeline
from sima_vision.tasks import TASKS


class FakeSample:
    """What ``pull`` hands back. Only the timing fields are ever read."""

    def __init__(self, index: int) -> None:
        self.pts_ns = index * 40_000_000       # 25 fps
        self.dts_ns = self.pts_ns
        self.duration_ns = 40_000_000
        self.frame_id = index
        self.stream_id = 0


class FakeRun:
    """A ``Run`` that yields ``frames`` samples, then times out forever.

    ``timeouts_before`` inserts a run of empty pulls at a given frame, which is
    how a starved decoder looks from here.
    """

    def __init__(self, frames: int, timeouts_at: dict[int, int] | None = None) -> None:
        self.frames = frames
        self.timeouts_at = dict(timeouts_at or {})
        self.pulled = 0
        self.closed = False
        self.labels: list[str] = []

    def pull(self, label: str, timeout_ms: int):
        self.labels.append(label)
        pending = self.timeouts_at.get(self.pulled, 0)
        if pending:
            self.timeouts_at[self.pulled] = pending - 1
            return None
        if self.pulled >= self.frames:
            return None
        self.pulled += 1
        return FakeSample(self.pulled)

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    """An OpenCV VideoWriter stand-in that records what it was given."""

    def __init__(self) -> None:
        self.frames: list = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class CountingRuntime(TaskRuntime):
    """A task that records the order it saw frames in, on both threads."""

    output_label = "detector_output"
    stream = "test"
    unit = "things"

    def __init__(self, per_frame: int = 2, fail_render_on: int | None = None) -> None:
        self.per_frame = per_frame
        self.fail_render_on = fail_render_on
        self.decoded: list[int] = []
        self.rendered: list[int] = []
        self.decode_thread: set[str] = set()
        self.render_thread: set[str] = set()

    def decode(self, pipeline, cfg, sample, index: int):
        self.decoded.append(index)
        self.decode_thread.add(threading.current_thread().name)
        frame = np.full((16, 24, 3), index % 256, np.uint8)
        return frame, [{"i": index}] * self.per_frame, 0.0

    def render(self, cfg, pipeline, frame, results, fps: float):
        index = int(frame[0, 0, 0])
        self.rendered.append(index)
        self.render_thread.add(threading.current_thread().name)
        if self.fail_render_on is not None and index == self.fail_render_on:
            raise RuntimeError("sink exploded")
        return frame.copy()

    def metadata(self, pipeline, results) -> list[dict]:
        return [{"id": str(i)} for i, _ in enumerate(results)]


def make(frames: int = 5, source_frames: int = 0, writer: bool = True, **settings):
    """A config and a Pipeline wired to a FakeRun."""
    cfg = TASKS["detect"]().load(
        None,
        {"model.path": "m", "source.uri": "c.h264", "output.save.enable": False,
         **settings},
        use_file=False,
    )
    pipeline = Pipeline(labels=["thing"], frame_w=24, frame_h=16, fps=25)
    pipeline.run = FakeRun(frames)
    pipeline.source_frames = source_frames
    if writer:
        pipeline.writer = FakeWriter()
        pipeline.writer_path = "out.mp4"
    return cfg, pipeline


# ── the loop ──


def test_every_frame_is_processed_and_written():
    cfg, pipeline = make(frames=7)
    task = CountingRuntime()
    processed = run_pipeline(pipeline, cfg, Stopper(), task)
    assert processed == 7
    assert task.decoded == list(range(1, 8))
    assert pipeline.writer_frames == 7


def test_the_sinks_keep_source_order():
    """One worker draining a FIFO is what guarantees the recording is in order."""
    cfg, pipeline = make(frames=25)
    task = CountingRuntime()
    run_pipeline(pipeline, cfg, Stopper(), task)
    assert task.rendered == list(range(1, 26))


def test_rendering_happens_off_the_pull_thread():
    """If drawing ran on the pull loop it would hold decoder buffers."""
    cfg, pipeline = make(frames=6)
    task = CountingRuntime()
    run_pipeline(pipeline, cfg, Stopper(), task)
    assert task.decode_thread and task.render_thread
    assert task.decode_thread.isdisjoint(task.render_thread)


def test_frames_caps_the_run():
    cfg, pipeline = make(frames=50, **{"runtime.frames": 4})
    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert processed == 4


def test_the_label_pulled_is_the_tasks_own():
    cfg, pipeline = make(frames=2)
    task = CountingRuntime()
    task.output_label = "segmenter_output"
    run_pipeline(pipeline, cfg, Stopper(), task)
    assert set(pipeline.run.labels) == {"segmenter_output"}


def test_a_stopper_ends_the_run():
    cfg, pipeline = make(frames=1000)
    stopper = Stopper()
    stopper.stop = True
    assert run_pipeline(pipeline, cfg, stopper, CountingRuntime()) == 0


# ── timeouts ──


def test_a_single_timeout_is_retried_not_fatal():
    """A starved pool answers on the retry; a finished clip does not."""
    cfg, pipeline = make(frames=6)
    pipeline.run.timeouts_at = {3: 1}          # one empty pull after 3 frames
    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert processed == 6, "the retry should have recovered the run"


def test_a_finished_clip_is_not_retried_or_warned_about(capsys):
    """A healthy run used to end with a warning and a `pull_timeout_ms` wait.

    Every frame of the clip had arrived, and the loop still drained, waited the
    full timeout again and printed "timed out waiting for results" before
    stopping. Twenty idle seconds and a scare, on the successful path.
    """
    cfg, pipeline = make(frames=3, source_frames=3)
    task = CountingRuntime()
    processed = run_pipeline(pipeline, cfg, Stopper(), task)

    assert processed == 3
    out = capsys.readouterr().out
    assert "the run is complete" in out, "a complete run must not read as a stall"
    assert "timed out" not in out
    assert "timeouts=0" in out, "the end of a file is not a timeout"
    # Four pulls: three frames and the one empty pull that ends the clip. A
    # fifth would be the retry this test exists to prevent.
    assert pipeline.run.pulled == 3
    assert len(pipeline.run.labels) == 4


def test_a_clip_with_frames_left_is_fought_for_not_abandoned(capsys):
    """The bug behind a 28-of-379 frame recording.

    One retry, then the run ended -- on a clip whose length the app had already
    counted and printed. Draining the sink queue is what releases the stall, so
    the frames the app knows are still coming are worth more than one attempt.
    """
    cfg, pipeline = make(frames=4, source_frames=100)
    # Silent once, then the backlog clears and the source comes back.
    pipeline.run.timeouts_at = {4: 1}
    pipeline.run.frames = 9

    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())

    assert processed == 9, "the run gave up on a clip that still had frames"
    assert pipeline.writer_frames == 9
    out = capsys.readouterr().out
    assert "recovered once the backlog was flushed" in out


def test_the_recovery_advice_points_the_knob_the_right_way(capsys):
    """It said *lower* sink_queue_depth, which causes the stall it follows.

    A shallower sink queue means `submit` blocks sooner, and a blocked pull loop
    is exactly what lets decoded frames pile up against the decoder's pool. The
    stall advice in `stall_causes` and the `--sink-queue-depth` help both say
    raise it; this message was the odd one out, and it is the one a stalling run
    actually prints.
    """
    cfg, pipeline = make(frames=2, source_frames=100)
    pipeline.run.timeouts_at = {1: 1}
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())

    out = capsys.readouterr().out
    assert "raise runtime.sink_queue_depth" in out
    assert "lower runtime.sink_queue_depth" not in out


def test_a_finished_clip_is_the_only_silence_not_worth_retrying():
    """Two different questions wearing the same silence.

    A larger retry budget for a clip with frames left was tried and removed:
    four DevKit runs reported `recovered=0`, so draining and asking again never
    once helped, and each attempt cost a full `pull_timeout_ms`.
    """
    from sima_vision.runloop import stall_attempts

    # Every frame arrived: this is the end of the file, not a stall.
    assert stall_attempts(stalled_pipeline(total=379), 379) == 0
    assert stall_attempts(stalled_pipeline(total=379), 400) == 0
    # Anything else is worth exactly one retry.
    assert stall_attempts(stalled_pipeline(total=379), 28) == 1
    assert stall_attempts(stalled_pipeline(total=0), 28) == 1


def test_a_short_run_against_a_known_clip_length_is_called_out(capsys):
    cfg, pipeline = make(frames=4, source_frames=100)
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    # Warnings share stdout with the steps, deliberately: a warning about the
    # recording only makes sense read in place, next to the run it belongs to.
    # Only errors go to stderr.
    combined = capsys.readouterr()
    assert "stalled" in combined.out
    assert "incomplete" in combined.out
    assert combined.err == ""


# ── failures ──


def test_a_failing_sink_is_raised_not_swallowed():
    cfg, pipeline = make(frames=10)
    task = CountingRuntime(fail_render_on=3)
    with pytest.raises(RuntimeError, match="sink exploded"):
        run_pipeline(pipeline, cfg, Stopper(), task)


def test_a_failing_sink_does_not_deadlock_the_pull_loop():
    """The worker must keep draining after an error, or submit() blocks forever."""
    cfg, pipeline = make(frames=200, **{"runtime.queue_depth": 1})
    task = CountingRuntime(fail_render_on=2)
    done = threading.Event()

    def go():
        try:
            run_pipeline(pipeline, cfg, Stopper(), task)
        except RuntimeError:
            pass
        finally:
            done.set()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    assert done.wait(timeout=30), "the run loop deadlocked after a sink failure"


def test_a_failing_decode_still_closes_the_sinks():
    cfg, pipeline = make(frames=10)

    class Exploding(CountingRuntime):
        def decode(self, pipeline, cfg, sample, index):
            if index == 3:
                raise RuntimeError("decode exploded")
            return super().decode(pipeline, cfg, sample, index)

    task = Exploding()
    with pytest.raises(RuntimeError, match="decode exploded"):
        run_pipeline(pipeline, cfg, Stopper(), task)
    # Two frames made it through and were written before the failure.
    assert pipeline.writer_frames == 2


# ── reporting ──


def test_the_summary_reports_what_the_task_adds(capsys):
    class Summarising(CountingRuntime):
        def summarise(self, pipeline, processed):
            return ["masks=packed"]

    cfg, pipeline = make(frames=3)
    run_pipeline(pipeline, cfg, Stopper(), Summarising())
    assert "masks=packed" in capsys.readouterr().out


def test_profiling_prints_a_line(capsys):
    cfg, pipeline = make(
        frames=4, **{"runtime.profile": True, "runtime.profile_interval": 2}
    )
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert "profile: frames=" in capsys.readouterr().out


def test_the_heartbeat_counts_what_the_task_returns(capsys):
    from sima_vision import runloop

    cfg, pipeline = make(frames=4)
    monkey = runloop.HEARTBEAT_EVERY
    runloop.HEARTBEAT_EVERY = 2
    try:
        run_pipeline(pipeline, cfg, Stopper(), CountingRuntime(per_frame=3))
    finally:
        runloop.HEARTBEAT_EVERY = monkey
    out = capsys.readouterr().out
    assert "3.0 things/frame" in out


def test_a_cut_up_clip_runs_every_piece_into_one_recording():
    """The board stops around 195 frames, so a longer clip is decoded in pieces.

    The model, the writer and the sink thread carry across; only the Run is
    rebuilt. What comes out has to be one continuous recording with every
    frame in it, not one file per piece.
    """
    cfg, pipeline = make(frames=4, source_frames=12)
    task = CountingRuntime()
    remaining = [4, 4]

    def rebuild():
        if not remaining:
            return False
        pipeline.run = FakeRun(remaining.pop(0))
        return True

    processed = run_pipeline(pipeline, cfg, Stopper(), task, rebuild=rebuild)

    assert processed == 12, "every piece has to be pulled"
    assert pipeline.writer_frames == 12, "and land in the one recording"
    assert task.rendered == list(range(1, 13)), "in order, across the joins"
    assert task.decoded == list(range(1, 13)), (
        "frame numbers must carry on: they name the stills, and two pieces "
        "both numbering from one would overwrite each other's"
    )


def test_a_stopped_run_does_not_start_the_next_piece():
    """Ctrl-C means stop, not stop after the remaining pieces."""
    cfg, pipeline = make(frames=1000, source_frames=2000)
    stopper = Stopper()
    stopper.stop = True
    asked = []

    run_pipeline(pipeline, cfg, stopper, CountingRuntime(),
                 rebuild=lambda: asked.append(1) or True)
    assert not asked


def test_frame_ids_restarting_at_each_piece_are_not_read_as_losses():
    """Each piece numbers from zero, and that is not the clip going missing."""
    from sima_vision.runloop import SourceTiming

    timing = SourceTiming()
    for n in range(4):
        timing.add(FrameStamp(frame_id=n))
    timing.restart()
    for n in range(4):
        timing.add(FrameStamp(frame_id=n))

    assert timing.pulled == 8
    assert timing.missing() == 0, "a restart is not a gap"


def test_losses_inside_a_piece_still_count_after_a_restart():
    from sima_vision.runloop import SourceTiming

    timing = SourceTiming()
    for n in (0, 2, 4):                      # two missing in piece one
        timing.add(FrameStamp(frame_id=n))
    timing.restart()
    for n in (0, 1, 2):                      # piece two is clean
        timing.add(FrameStamp(frame_id=n))

    assert timing.missing() == 2


# -- what the app says about playback --


def stamps(*pts_ms):
    """A SourceTiming fed the given presentation times, in milliseconds."""
    from sima_vision.runloop import SourceTiming

    timing = SourceTiming()
    for ms in pts_ms:
        timing.add(FrameStamp(pts_ns=int(ms * 1_000_000)))
    return timing


def test_the_rate_comes_from_the_timestamps_not_the_gaps():
    """Rate is span over intervals, so one long gap does not rewrite it."""
    assert stamps(0, 40, 80, 120).fps() == pytest.approx(25.0)
    assert stamps(0, 20, 40, 60, 80).fps() == pytest.approx(50.0)
    # Fewer than two stamps, or no span, cannot say anything.
    assert stamps(0).fps() == 0.0
    assert stamps().fps() == 0.0
    # A source that stamps nothing is not a source running at zero fps.
    assert stamps(-1, -1).frames == 0


def ids(*frame_ids):
    """A SourceTiming fed frames carrying these ids and no timestamps."""
    from sima_vision.runloop import SourceTiming

    timing = SourceTiming()
    for n in frame_ids:
        timing.add(FrameStamp(frame_id=n))
    return timing


def test_frames_that_never_arrived_are_counted_from_their_ids():
    """Timestamps cannot see this: a frame that was dropped has none."""
    assert ids(0, 1, 2, 3).missing() == 0
    assert ids(0, 2, 4, 6, 8).missing() == 4        # every other one
    assert ids(0).missing() == 0                     # too little to say
    assert ids().missing() == 0
    # A source that numbers nothing cannot be checked this way.
    from sima_vision.runloop import SourceTiming
    blind = SourceTiming()
    blind.add(FrameStamp(pts_ns=0))
    assert blind.missing() == 0


def test_a_recording_missing_half_the_clip_says_so_and_says_why_it_plays_fast():
    """The failure a rate setting cannot fix, and kept being mistaken for one.

    Handed every other frame, the app writes every frame it was given at the
    source's own rate. The result is half as long as the clip and plays twice
    as fast, and nothing about `--video-fps` helps: the frames are not there
    to slow down.
    """
    cfg, pipeline = make(frames=4)
    pipeline.fps, pipeline.writer_frames = 25, 171
    lines = timing_report(cfg, pipeline, ids(*range(0, 341, 2)))

    assert any("were dropped below this app" in line for line in lines)
    joined = chr(10).join(lines)
    assert "1.99x" in joined, "341 numbered against 171 arrived"
    assert "50% of the clip" in joined
    assert "not there to slow down" in joined


def test_a_complete_run_is_not_accused_of_dropping_anything():
    cfg, pipeline = make(frames=4)
    pipeline.fps, pipeline.writer_frames = 25, 341
    assert timing_report(cfg, pipeline, ids(*range(341))) == []


def test_a_source_that_says_nothing_at_all_is_admitted_to(capsys):
    """Silence used to mean "fine". It meant "no idea", which is not the same.

    With neither ids nor timestamps there is no way to tell a complete
    recording from one holding every other frame, and a report that stays
    quiet reads as a clean bill of health.
    """
    from sima_vision.runloop import SourceTiming

    cfg, pipeline = make(frames=4)
    pipeline.fps, pipeline.writer_frames = 25, 100
    blind = SourceTiming()
    for _ in range(100):
        blind.add(FrameStamp())              # pts_ns -1, frame_id -1

    lines = timing_report(cfg, pipeline, blind)
    assert any("neither timestamps nor frame ids" in line for line in lines)


def test_a_recording_written_at_the_wrong_rate_says_so_and_says_what_to_use():
    """Written at one rate, sourced at another: the whole clip plays wrong.

    Nothing checked the rate guessed before the run against the frames that
    actually turned up, so the only symptom was a video that played too fast.
    """
    cfg, pipeline = make(frames=4, **{"output.video.fps": 30})
    pipeline.fps, pipeline.writer_frames = 30, 4
    lines = timing_report(cfg, pipeline, stamps(0, 40, 80, 120))   # a 25 fps source

    assert len(lines) == 1
    assert "written at 30 fps" in lines[0]
    assert "source is 25.00 fps" in lines[0]
    assert "1.20x too fast" in lines[0]
    assert "--video-fps 25" in lines[0]


def test_a_rate_that_only_looks_wrong_to_rounding_is_left_alone():
    """24000/1001 is 23.976, and an SPS that rounds it to 24 is not a bug."""
    cfg, pipeline = make(frames=4)
    pipeline.fps, pipeline.writer_frames = 24, 4
    assert timing_report(cfg, pipeline, stamps(0, 41.7083, 83.4166, 125.125)) == []


def test_timestamps_that_go_backwards_are_named_as_decode_order():
    """B-frames arriving unreordered, which is what jerky motion looks like."""
    cfg, pipeline = make(frames=4)
    pipeline.fps, pipeline.writer_frames = 25, 6
    lines = timing_report(cfg, pipeline, stamps(0, 80, 40, 120, 200, 160))

    assert any("decode order, not presentation order" in line for line in lines)
    assert any("2 frame(s) arrived" in line for line in lines)


def test_playback_is_silent_when_there_is_no_recording_to_play():
    """Nothing was written, so there is nothing to be wrong about."""
    cfg, pipeline = make(frames=4, writer=False)
    pipeline.fps = 30
    assert timing_report(cfg, pipeline, stamps(0, 40, 80)) == []


def test_the_run_measures_the_source_it_actually_pulled(capsys):
    """End to end: the loop feeds the stamps and the closing report uses them."""
    cfg, pipeline = make(frames=6, **{"output.video.fps": 30})
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())

    out = capsys.readouterr().out
    assert "written at 30 fps" in out, "the run should have measured 25 fps"
    assert "--video-fps 25" in out


# -- what the app says when the source stalls --


def stall_config(**settings):
    base = {"model.path": "m.tar.gz", "source.uri": "c.h264"}
    return TASKS["detect"]().load(None, {**base, **settings}, use_file=False)


def stalled_pipeline(fps: int = 24, total: int = 379) -> Pipeline:
    pipeline = Pipeline(labels=["person"])
    pipeline.fps = fps
    pipeline.source_frames = total
    return pipeline


def test_the_two_queue_depths_are_separate_knobs():
    """One setting drove both, and they pull opposite ways.

    `RunOptions.queue_depth` parks decoded frames from the hardware decoder's
    eight-buffer pool. The sink queue holds numpy copies in host memory and no
    decoder buffer at all, so depth there is what lets the pull loop keep
    draining. Raising the single old knob for slack deepened the runtime queues
    too, making the buffer exhaustion it was meant to relieve slightly worse.
    """
    cfg = stall_config(**{"runtime.queue_depth": 1, "runtime.sink_queue_depth": 8})
    assert cfg.queue_depth == 1
    assert cfg.sink_queue_depth == 8


def test_the_sink_queue_is_the_one_the_worker_gets():
    """A regression here is invisible: the run works, just with less slack."""
    import sima_vision.runloop as loop

    seen = {}
    real = loop.SinkWorker

    class Spy(real):
        def __init__(self, cfg, pipeline, depth, *args):
            seen["depth"] = depth
            super().__init__(cfg, pipeline, depth, *args)

    cfg, pipeline = make(frames=2, **{"runtime.sink_queue_depth": 6})
    loop.SinkWorker = Spy
    try:
        run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    finally:
        loop.SinkWorker = real
    assert seen["depth"] == 6, "the sink worker must get the sink depth"


def test_a_file_backlog_is_sized_to_the_clip_not_to_a_fixed_depth():
    """The stall scaled with the queue: depth 4 died at 28, depth 12 at 36.

    The sinks are slower than the source and always will be -- software
    encoding 1080p costs several times the frame interval -- so a fixed depth
    only moves the stall, it never removes it. What removes it is never making
    the pull loop wait, and for a clip of known length that backlog is bounded.
    """
    from sima_vision.runloop import sink_depth_for

    cfg = stall_config()
    pipeline = stalled_pipeline(total=379)
    pipeline.frame_w, pipeline.frame_h = 1920, 1080

    # 1 GB at ~6 MB a frame, and the clip is shorter than the budget allows.
    assert sink_depth_for(cfg, pipeline) == (1024 << 20) // (1920 * 1080 * 3)

    # A short clip is capped by its own length, not by the budget.
    pipeline.source_frames = 40
    assert sink_depth_for(cfg, pipeline) == 40

    # A live source has no length, so no bound: it keeps the floor.
    pipeline.source_frames = 0
    assert sink_depth_for(cfg, pipeline) == cfg.sink_queue_depth

    # And the floor is never lowered by a stingy budget.
    tiny = stall_config(**{"runtime.sink_queue_mb": 1})
    pipeline.source_frames = 379
    assert sink_depth_for(tiny, pipeline) == tiny.sink_queue_depth


def test_the_worker_gets_the_grown_depth_not_the_floor():
    """A regression here is silent: the run works, it just stalls again."""
    import sima_vision.runloop as loop

    seen = {}
    real = loop.SinkWorker

    class Spy(real):
        def __init__(self, cfg, pipeline, depth, *args):
            seen["depth"] = depth
            super().__init__(cfg, pipeline, depth, *args)

    cfg, pipeline = make(frames=2, source_frames=300)
    loop.SinkWorker = Spy
    try:
        run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    finally:
        loop.SinkWorker = real
    assert seen["depth"] == 300, "the backlog should have been sized to the clip"


def test_a_complete_run_is_not_called_a_stall():
    message = source_stopped_message(stall_config(), stalled_pipeline(total=23), 23)
    assert "the run is complete" in message
    assert "In order of likelihood" not in message
