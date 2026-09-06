"""The pull loop, shared by all three tasks.

Everything expensive happens on the :class:`~sima_vision.sinks.SinkWorker`, so
the only work between two ``pull`` calls is whatever the task's ``decode`` hook
does: copying the frame out, parsing boxes and -- for segmentation -- rebuilding
masks. That is what keeps the decoder's pool turning over.

A task plugs into this by implementing :class:`TaskRuntime`.
"""

from __future__ import annotations

import signal

from .console import console
from .runtime import time_ms
from .samples import FrameStamp
from .sinks import Pipeline, SinkJob, SinkWorker

HEARTBEAT_EVERY = 50


class Stopper:
    """Cooperative stop flag driven by SIGINT, SIGTERM and SIGHUP.

    ``dk`` and ``devkit-run`` invoke over SSH without a pty, so a terminal
    Ctrl-C is never forwarded and the app can be orphaned on the DevKit while
    still holding the MLA. The next run then fails with a busy device. Launch
    interactive runs with ``ssh -tt`` and let these handlers close the Run.

    Attributes:
        stop: Set to True once a signal has been received.
    """

    #: Looked up by name, not by attribute. ``signal.SIGHUP`` does not exist on
    #: Windows, and naming it directly raised AttributeError while building the
    #: tuple -- before the try block that was meant to tolerate exactly that.
    SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")

    def __init__(self) -> None:
        self.stop = False
        for name in self.SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                signal.signal(number, self._handle)
            except (ValueError, OSError):
                # ValueError: not the main thread, which is fine -- the run
                # loop still checks `stop`, it just cannot be set by a signal.
                pass

    def _handle(self, signum, _frame) -> None:
        if not self.stop:
            console.report(f"[signal {signum}] stopping, closing Run...")
        self.stop = True


class ProfileWindow:
    """Rolling per-stage timing accumulator.

    Sums pull, decode, an optional task stage and sink latencies over a fixed
    number of frames, then prints one averaged line and resets. Averaging avoids
    the noise of per-frame timings without needing to keep every sample.

    Attributes:
        enabled: Whether profiling output is on at all.
        interval: Frames per window before a flush.
        stage: Name of the task-specific stage, such as ``masks``. Empty when
            the task has no third stage, and then it is left out of the line.
        unit: What the counter counts, such as ``detections``.
    """

    def __init__(self, enabled: bool, interval: int, stage: str = "",
                 unit: str = "objects") -> None:
        self.enabled = enabled
        self.interval = interval
        self.stage = stage
        self.unit = unit
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.objects = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.stage_ms = 0.0
        self.sink_ms = 0.0

    def add(self, pull_ms: float, decode_ms: float, stage_ms: float, sink_ms: float,
            count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.objects += count
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.stage_ms += stage_ms
        self.sink_ms += sink_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        fps = self.frames * 1000.0 / elapsed if elapsed > 0 else 0.0
        stage = (
            f"{self.stage}={self.stage_ms / self.frames:.1f}ms " if self.stage else ""
        )
        console.write(
            f"  profile: frames={self.frames} fps={fps:.1f} "
            f"pull={self.pull_ms / self.frames:.1f}ms "
            f"decode={self.decode_ms / self.frames:.1f}ms "
            f"{stage}"
            f"sinks={self.sink_ms / self.frames:.1f}ms "
            f"{self.unit}={self.objects / self.frames:.1f}"
        )
        self.reset()


class SourceTiming:
    """What the frame timestamps say, as opposed to what the SPS claims.

    The recording is written at one constant rate, chosen before a single frame
    has arrived: ``video_fps``, or the SPS rate, or 25. Nothing afterwards ever
    checks that guess against the frames themselves, and both ways of being
    wrong look the same in a player -- like a bad recording.

    * A rate that does not match the source plays the whole clip at the wrong
      speed. The motion is smooth, it is just too fast or too slow.
    * Timestamps that go backwards mean frames are arriving in decode order
      rather than presentation order, which is what a stream with B-frames does
      when nothing reorders it. Written in arrival order, motion jerks back and
      forth a frame at a time.

    Attributes:
        frames: Frames that carried a usable timestamp.
        out_of_order: Frames whose timestamp went backwards.
    """

    def __init__(self) -> None:
        self.pulled = 0
        self.frames = 0
        self.out_of_order = 0
        self.first_ns = -1
        self.last_ns = -1
        self.previous_ns = -1
        self.first_id = -1
        self.last_id = -1
        self._missing = 0
        self._pulled_before = 0
        self._span_ns = 0
        self._intervals = 0
        self._frames_before = 0

    def add(self, stamp: FrameStamp) -> None:
        self.pulled += 1

        # Frame ids, when the source sets them, catch the failure timestamps
        # cannot: frames that never arrived. Ids running 1, 3, 5 mean half the
        # clip was dropped somewhere below this app, and a recording written
        # from what did arrive holds every other frame -- so it is half as long
        # as the clip and plays twice as fast, which is not a rate problem and
        # no rate setting fixes it.
        if stamp.frame_id >= 0:
            if self.first_id < 0:
                self.first_id = stamp.frame_id
            self.last_id = max(self.last_id, stamp.frame_id)

        pts = stamp.pts_ns
        if pts < 0:                      # a source that stamps nothing
            return
        self.frames += 1
        if self.first_ns < 0:
            self.first_ns = pts
        if self.previous_ns >= 0 and pts < self.previous_ns:
            self.out_of_order += 1
        self.previous_ns = pts
        self.last_ns = max(self.last_ns, pts)

    def restart(self) -> None:
        """Begin a new piece of a cut-up clip.

        Frame ids and timestamps both start again at each piece, so the gap
        check has to forget the last one or it reads the restart as the whole
        clip going missing. The counts carry on: they describe the recording,
        which spans every piece.
        """
        self._missing = self.missing()
        self._pulled_before = self.pulled
        self._span_ns += max(0, self.last_ns - self.first_ns)
        self._intervals += max(0, self.frames - self._frames_before - 1)
        self._frames_before = self.frames
        self.first_id = -1
        self.last_id = -1
        self.first_ns = -1
        self.last_ns = -1
        self.previous_ns = -1

    def missing(self) -> int:
        """Frames the ids say existed but that never reached the pull loop."""
        if self.first_id < 0 or self.last_id <= self.first_id:
            return self._missing
        seen = self.pulled - self._pulled_before
        return self._missing + max(0, (self.last_id - self.first_id + 1) - seen)

    def fps(self) -> float:
        """Frame rate implied by the timestamps, or 0.0 when they cannot say.

        Summed over the pieces rather than measured end to end: each piece
        stamps from zero again, so first-to-last across a cut-up clip spans
        nothing meaningful.
        """
        span = self._span_ns + max(0, self.last_ns - self.first_ns)
        intervals = self._intervals + max(0, self.frames - self._frames_before - 1)
        if intervals < 1 or span <= 0:
            return 0.0
        return intervals * 1_000_000_000.0 / span


def timing_report(cfg, pipeline: Pipeline, timing: SourceTiming) -> list[str]:
    """Lines about playback, and only when there is something wrong with it."""
    lines: list[str] = []
    if pipeline.writer is None or not pipeline.writer_frames:
        return lines

    missing = timing.missing()
    if missing:
        span = timing.last_id - timing.first_id + 1
        lines.append(
            f"playback: the source numbered {span} frames but only {timing.pulled}"
            f" arrived, so {missing} were dropped below this app.\n"
            f"  The recording holds every frame it was given, so it covers"
            f" {timing.pulled / span:.0%} of the clip and plays {span / timing.pulled:.2f}x"
            " too fast.\n"
            "  No frame rate setting fixes that: the frames are not there to slow down."
        )

    if timing.out_of_order:
        lines.append(
            f"playback: {timing.out_of_order} frame(s) arrived with a timestamp"
            " earlier than the one before, so the source is handing over\n"
            "  decode order, not presentation order. This clip has B-frames\n"
            "  and nothing is reordering them. The recording is written in\n"
            "  arrival order, so motion will jerk back and forth a frame at a time."
        )

    measured = timing.fps()
    written = cfg.video_fps or pipeline.fps or 25
    # 2% covers rounding an SPS rate like 24000/1001 to 24, which is right.
    if measured and abs(measured - written) / written > 0.02:
        lines.append(
            f"playback: the recording is written at {written} fps but the frame"
            f" timestamps say the source is {measured:.2f} fps, so it plays"
            f" {written / measured:.2f}x too fast.\n"
            f"  Re-run with --video-fps {round(measured)} to match the source."
        )

    # Never silently blind. With neither ids nor timestamps there is no way to
    # tell a complete recording from one holding every other frame, and saying
    # so beats saying nothing at all.
    if not timing.frames and timing.first_id < 0:
        lines.append(
            "playback: the source set neither timestamps nor frame ids, so\n"
            "  nothing here can confirm the recording runs at the right speed\n"
            "  or holds every frame. Compare its length against the clip by hand."
        )
    return lines

    if timing.out_of_order:
        lines.append(
            f"playback: {timing.out_of_order} frame(s) arrived with a timestamp"
            " earlier than the one before, so the source is handing over\n"
            "  decode order, not presentation order. This clip has B-frames\n"
            "  and nothing is reordering them. The recording is written in\n"
            "  arrival order, so motion will jerk back and forth a frame at a time."
        )

    measured = timing.fps()
    written = cfg.video_fps or pipeline.fps or 25
    # 2% covers rounding an SPS rate like 24000/1001 to 24, which is right.
    if measured and abs(measured - written) / written > 0.02:
        lines.append(
            f"playback: the recording is written at {written} fps but the frame"
            f" timestamps say the source is {measured:.2f} fps, so it plays"
            f" {written / measured:.2f}x too fast.\n"
            f"  Re-run with --video-fps {round(measured)} to match the source."
        )
    return lines


class TaskRuntime:
    """What the pull loop needs from a task.

    Attributes:
        output_label: Public output the loop pulls, such as ``detector_output``.
        stream: Insight stream name, such as ``object-detection``.
        unit: Plural noun for the heartbeat and the profile line.
        stage: Name of the task's own profiling stage, or "" for none.
    """

    output_label = "detector_output"
    stream = "objects"
    unit = "detections"
    stage = ""

    def decode(self, pipeline: Pipeline, cfg, sample, index: int):
        """Turn one pulled sample into a frame and this task's results.

        Owns the sample's whole lifetime. It must drop every reference to it
        before returning, because the buffer it holds belongs to the hardware
        decoder's small pool -- see
        :class:`~sima_vision.samples.FrameStamp` for what happens otherwise.

        Args:
            pipeline: Live pipeline.
            cfg: Application configuration.
            sample: The joined sample from ``pull``.
            index: 1-based frame number.

        Returns:
            A ``(frame, results, stage_ms)`` triple.
        """
        raise NotImplementedError

    def render(self, cfg, pipeline: Pipeline, frame, results, fps: float):
        """Draw one frame's overlay. Runs on the sink thread."""
        raise NotImplementedError

    def metadata(self, pipeline: Pipeline, results) -> list[dict]:
        """The Insight JSON payload for one frame's results."""
        raise NotImplementedError

    def summarise(self, pipeline: Pipeline, processed: int) -> list[str]:
        """Extra lines printed once the run is over."""
        return []


def sink_depth_for(cfg, pipeline: Pipeline) -> int:
    """How many finished frames may pile up waiting for the sinks.

    The sinks do not have to keep up. They only have to stay out of the pull
    loop's way, and those are very different requirements.

    Software-encoding 1080p costs several times the frame interval on this
    board -- a measured 117 ms against 42 ms -- so a bounded queue fills and
    then ``submit`` parks the loop. That pause is the whole problem: a loop
    that is not pulling lets decoded frames pile up against a pool of eight,
    the decoder starves, and it does not come back. Runs died after
    ``24 + sink_queue_depth`` frames, which is how this was traced: a depth of
    4 stopped at 28 and a depth of 12 stopped at 36, on the same clip.

    A sink job is plain numpy in host memory and holds no decoder buffer, so
    the fix is to let the backlog grow rather than let the loop wait. For a
    clip of known length that is bounded work: the loop drains the source in
    seconds, the sink thread finishes afterwards, and ``run_pipeline`` already
    joins it before anything reads ``writer_frames``. Memory is the only cost,
    and ``sink_queue_mb`` is the budget for it.

    A live source has no length, so no bound, and keeps the floor.
    """
    depth = cfg.sink_queue_depth
    frame_bytes = pipeline.frame_w * pipeline.frame_h * 3
    if not (cfg.sink_queue_mb and pipeline.source_frames and frame_bytes):
        return depth
    affordable = (cfg.sink_queue_mb << 20) // frame_bytes
    return max(depth, min(pipeline.source_frames, affordable))


def stall_attempts(pipeline: Pipeline, processed: int) -> int:
    """How hard to fight a silent source, given what is known about the clip.

    The clip's length is counted before the run starts, so silence is not
    always the same question:

    * **Every frame arrived.** This is the end of the file, not a stall. The
      old code still drained and waited a full ``pull_timeout_ms`` here, so a
      perfectly healthy run ended with a scary warning and twenty idle seconds.
    * **Anything else.** One retry, which is what tells a starved pool from
      a finished clip.

    There was briefly a larger budget for a clip with frames demonstrably
    left, on the theory that draining the sink queue releases the stall.
    Four runs on a DevKit said otherwise -- ``recovered=0`` every time -- so
    the extra attempts bought nothing and cost a ``pull_timeout_ms`` each.
    The stall that was really happening is fixed in :func:`sink_depth_for`
    instead, by not letting the pull loop park in the first place.
    """
    total = pipeline.source_frames
    if total and processed >= total:
        return 0
    return 1


def pull_frame(pipeline: Pipeline, cfg, sinks: SinkWorker, label: str, processed: int):
    """Pull one joined sample, flushing our own backlog before giving up.

    A starved decoder and a finished clip look identical from here: both are
    silence. So on a timeout, hand back everything the app is still holding --
    the sink queue is several decoded frames deep -- and ask again. A pool that
    refills answers straight away; a clip that ended stays quiet.

    Draining is the whole point of the retry, not politeness. The stall is the
    pull loop parked in ``submit`` while decoded frames piled up between the
    decoder and this app; emptying the sink queue is what lets the pool turn
    over again. How many times that is worth trying is :func:`stall_attempts`.

    Args:
        pipeline: Live pipeline.
        cfg: Application configuration, for ``pull_timeout_ms``.
        sinks: Sink worker to drain before each retry.
        label: Public output to pull.
        processed: Frames processed so far.

    Returns:
        A ``(sample, timed_out, recovered)`` triple. ``sample`` is None only
        once every attempt has come back empty.
    """
    sample = pipeline.run.pull(label, cfg.pull_timeout_ms)
    if sample is not None:
        return sample, False, False

    attempts = stall_attempts(pipeline, processed)
    if not attempts:
        # The clip delivered every frame it has. Nothing is wrong, so nothing
        # is warned about and nothing is waited for.
        return None, False, False

    for attempt in range(1, attempts + 1):
        console.warn(
            f"timed out waiting for results after {processed} frames; flushing "
            f"the sink queue and retrying ({attempt} of {attempts})"
        )
        sinks.drain()
        sample = pipeline.run.pull(label, cfg.pull_timeout_ms)
        if sample is not None:
            console.warn(
                "the source recovered once the backlog was flushed. That was "
                "back-pressure from this app rather than the end of the clip; "
                "raise runtime.sink_queue_depth if it keeps happening, so the "
                "pull loop keeps draining the decoder instead of parking."
            )
            return sample, True, True
    return None, True, False


def source_stopped_message(cfg, pipeline: Pipeline, processed: int) -> str:
    """Explain a source that went quiet, ruling out what the frame count rules out.

    The clip's length is known before the run starts, so "it just ended" is
    either the whole answer or not on the list at all. Saying which turns a
    short recording from something to be interpreted into something decided.

    There used to be a ranked list of causes under this, written while the
    stall was still a mystery. It is not one any more: the decoder stops around
    195 frames whatever the sinks, the pool or the GOP are doing, and
    :mod:`sima_vision.segments` handles it by cutting the clip. Guesses that
    outlive the thing they were guessing at only send people down the wrong
    path, so they are gone.
    """
    total = pipeline.source_frames
    if total and processed >= total:
        # Not a timeout report at all: every frame arrived, so the silence that
        # brought us here is just the end of the file.
        return (
            f"the source ended after {processed} frames, which is the whole clip "
            f"({total} frames). Nothing is wrong: the run is complete."
        )

    tries = stall_attempts(pipeline, processed) + 1
    head = (
        f"source produced nothing for {cfg.pull_timeout_ms} ms "
        f"{tries} times in a row after {processed} frames"
    )
    if total:
        return (
            f"{head}, {processed / total:.0%} of the way through a {total} frame "
            "clip. The source stalled; it did not end."
        )
    return (
        f"{head}. If that is far short of the clip, the source stalled rather "
        "than ended."
    )


def consume_frames(pipeline: Pipeline, cfg, stopper: Stopper, sinks: SinkWorker,
                   profile: ProfileWindow, task: TaskRuntime,
                   timing: SourceTiming, already: int = 0) -> tuple[int, int, int]:
    """The pull loop. Returns ``(processed, timeouts, recovered)``.

    Args:
        already: Frames processed by earlier pieces of a cut-up clip. Frame
            numbers carry on from there, because they name the stills: two
            pieces both numbering from one would write frame_000001 twice and
            the second would overwrite the first.
    """
    processed = 0
    timeouts = 0
    recovered = 0
    heartbeat_start = time_ms()
    heartbeat_count = 0
    live_fps = float(pipeline.fps or 25)   # HUD value, refreshed each heartbeat

    while not stopper.stop and (cfg.frames <= 0 or already + processed < cfg.frames):
        pull_start = time_ms()
        sample, timed_out, came_back = pull_frame(
            pipeline, cfg, sinks, task.output_label, processed
        )
        pull_end = time_ms()
        timeouts += int(timed_out)
        recovered += int(came_back)

        if sample is None:
            if cfg.source_type == "video":
                console.report(source_stopped_message(cfg, pipeline, processed))
                break
            continue

        stamp = FrameStamp.of(sample)
        timing.add(stamp)
        frame, results, stage_ms = task.decode(
            pipeline, cfg, sample, already + processed + 1
        )
        sample = None
        decode_end = time_ms()

        processed += 1
        sinks.submit(
            SinkJob(already + processed, stamp, frame, results, live_fps)
        )
        sink_end = time_ms()

        count = len(results)
        profile.add(
            pull_end - pull_start,
            (decode_end - pull_end) - stage_ms,
            stage_ms,
            sink_end - decode_end,
            count,
        )

        # Heartbeat, so a healthy run does not look identical to a stalled one.
        heartbeat_count += count
        if processed % HEARTBEAT_EVERY == 0:
            elapsed = time_ms() - heartbeat_start
            rate = HEARTBEAT_EVERY * 1000.0 / elapsed if elapsed > 0 else 0.0
            live_fps = rate or live_fps
            console.write(
                f"  {already + processed:>6}  {rate:.1f} fps, "
                f"{heartbeat_count / HEARTBEAT_EVERY:.1f} {task.unit}/frame avg"
            )
            heartbeat_start = time_ms()
            heartbeat_count = 0

    return processed, timeouts, recovered


def report_recording(cfg, pipeline: Pipeline, timeouts: int) -> None:
    """Say whether the recording is complete, and rank the reasons when it is not.

    An incomplete recording is the most commonly reported symptom, and it has
    several quite different causes. "Incomplete" is measured against the clip's
    own length where that is known, not against an arbitrary few seconds: a 15
    second clip cut off at 3 seconds used to pass this check in silence.
    """
    if pipeline.writer is None or not pipeline.writer_frames:
        return

    total = pipeline.source_frames
    out_fps = cfg.video_fps or pipeline.fps or 25
    seconds = pipeline.writer_frames / out_fps if out_fps else 0.0
    short = pipeline.writer_frames < total if total else seconds < 2.0

    if not short:
        if total and pipeline.writer_frames >= total:
            console.report(f"video: complete, all {total} frames of the clip.")
        return

    causes = []
    if cfg.frames:
        causes.append(f"runtime.frames is {cfg.frames}, which capped the run.")
    if cfg.insight_enable:
        causes.append(
            "output.insight.enable is true. Its H.264 encoder shares the codec "
            "daemon with the decoder feeding the source, so a failing encoder "
            "stalls the run. Set it to false; the recording does not need it."
        )
    if timeouts:
        causes.append(
            f"the source stopped producing frames ({timeouts} timeout(s)), so "
            "the run ended before the clip did."
        )
    if not causes:
        causes.append(
            "frames were dropped rather than blocked on. Check that "
            "runtime.overflow_policy resolved to block, as printed at startup."
        )
    listed = "\n".join(f"       {i}. {c}" for i, c in enumerate(causes, 1))
    missing = (
        f"{pipeline.writer_frames} of {total} frames, {seconds:.1f}s of "
        f"{total / out_fps:.1f}s" if total
        else f"only {seconds:.1f}s ({pipeline.writer_frames} frames at {out_fps} fps)"
    )
    console.warn(f"the recording is incomplete: {missing}.\n{listed}")


def run_pipeline(pipeline: Pipeline, cfg, stopper: Stopper, task: TaskRuntime,
                 rebuild=None) -> int:
    """Run one task to completion and print the closing report.

    Args:
        pipeline: Live pipeline, already built for the first piece.
        cfg: Application configuration.
        stopper: Cooperative stop flag.
        task: The pull-loop implementation.
        rebuild: Optional ``() -> bool`` that points ``pipeline.run`` at the
            next piece of a cut-up clip and reports whether there was one. The
            recording, the sinks and the model are untouched across the call,
            so the pieces land in one continuous video. None runs the source
            once, which is every case but a clip too long for one decode.

    Returns:
        Frames processed across every piece.
    """
    profile = ProfileWindow(cfg.profile, cfg.profile_interval, task.stage, task.unit)
    sinks = SinkWorker(
        cfg, pipeline, sink_depth_for(cfg, pipeline), task.render, task.stream,
        task.metadata,
    )
    timing = SourceTiming()
    processed = timeouts = recovered = 0
    try:
        while True:
            done, timed_out, came_back = consume_frames(
                pipeline, cfg, stopper, sinks, profile, task, timing, processed
            )
            processed += done
            timeouts += timed_out
            recovered += came_back
            if stopper.stop or rebuild is None or not rebuild():
                break
            # Each piece numbers its frames from zero again, so the gap check
            # would read the restart as thousands of missing frames.
            timing.restart()
    finally:
        # Ordered before anything that reads writer_frames: frames may still be
        # queued, and they belong in the recording. close() re-raises whatever
        # the worker hit, so a failing sink is not swallowed.
        sinks.close()

    profile.flush()
    total = pipeline.source_frames
    summary = " ".join(task.summarise(pipeline, processed))
    console.write()
    console.report(
        f"processed={processed}{f' of {total}' if total else ''} timeouts={timeouts} "
        f"recovered={recovered}{f' {summary}' if summary else ''}"
    )
    if sinks.blocked_ms > 1000.0 and processed:
        console.report(
            f"sinks: the pull loop waited {sinks.blocked_ms / 1000.0:.1f}s in total for "
            f"the sink thread ({sinks.blocked_ms / processed:.0f} ms/frame). "
            "Cheaper settings are in the README under \"It runs slower than the detector\"."
        )

    report_recording(cfg, pipeline, timeouts)
    for line in timing_report(cfg, pipeline, timing):
        console.warn(line)

    if pipeline.metadata_sender is not None:
        stats = pipeline.metadata_sender.stats()
        console.report(
            f"metadata: sent={stats.datagrams_sent} failures={stats.send_failures} "
            f"would_block={stats.would_block}"
        )
    if pipeline.video_dropped:
        console.report(
            f"insight: dropped {pipeline.video_dropped} preview frames because the "
            f"feed was busy. The recording is unaffected."
        )
    return processed
