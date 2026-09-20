"""What happens to a frame once the MLA is done with it.

Two destinations, both optional and independent: an annotated video on the
DevKit and annotated stills. :class:`Pipeline` owns every handle so teardown has
one place to look, and :class:`SinkWorker` runs the expensive part on a thread of
its own.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import runtime
from .console import console, human_bytes
from .recorder import NeatVideoWriter
from .runtime import time_ms
from .samples import FrameStamp


@dataclass
class Pipeline:
    """Everything a running pipeline owns, so teardown has one place to look.

    Tasks subclass this to add their own state -- the tracker
    for ``fall``, the mask bookkeeping for ``segment`` -- and inherit
    :meth:`close`, which is what guarantees the MLA is released even when a run
    ends on an exception.

    Attributes:
        model: The loaded ``pyneat.Model``.
        graph: The task ``Graph``. Never read after it is set, and it still
            has to be here: ``graph.build()`` hands back a ``Run`` that keeps
            using the C++ object behind the graph, so dropping the last Python
            reference would let it be collected out from under a live run. A
            dead-code pass will offer to delete it; do not.
        run: Live ``Run`` handle for the task graph.
        labels: Class names, indexed by class id.
        frame_w: Source frame width in pixels.
        frame_h: Source frame height in pixels.
        fps: Source frame rate.
        source_frames: Coded pictures in the source file, counted before the
            run. 0 for a live source, where there is no such number.
        writer: OpenCV ``VideoWriter`` for the on-device recording.
        writer_path: Path the writer actually opened, after any fallback.
        writer_frames: Frames written so far.
    """

    model: object = None
    graph: object = None
    run: object = None
    labels: list[str] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    fps: int = 0
    source_frames: int = 0
    writer: object = None
    writer_path: str = ""
    writer_frames: int = 0

    def close_extras(self) -> None:
        """Hook for subclasses. Runs before the writer and the Run are closed."""

    def close(self) -> None:
        self.close_extras()
        if self.writer is not None:
            try:
                self.writer.release()
                size = Path(self.writer_path).stat().st_size
                console.report(
                    f"video: wrote {self.writer_frames} frames to {self.writer_path} "
                    f"({human_bytes(size)})"
                )
            except Exception as exc:
                console.warn(f"closing the video writer failed: {exc}")
            self.writer = None
        if self.run is not None:
            try:
                self.run.close()
            except Exception as exc:  # pragma: no cover - teardown must not mask errors
                console.warn(f"close failed: {exc}")


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {path}")
    return labels


def open_video_writer(cfg, width: int, height: int, fps: int):
    """Annotated MP4 written on the DevKit. Returns (writer, path) or (None, '')."""
    cv2 = runtime.cv2
    if not cfg.video_enable:
        return None, ""

    path = Path(cfg.video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = cfg.video_fps or fps or 25

    if cfg.video_encoder == "sima":
        try:
            writer = NeatVideoWriter(
                str(path.with_suffix(".mp4")), width, height, out_fps, cfg.video_bitrate_kbps
            )
            return writer, str(path.with_suffix(".mp4"))
        except Exception as exc:
            console.warn(
                f"the hardware encoder could not be started ({exc}); "
                "recording with OpenCV instead, which is much slower"
            )

    fourcc = cv2.VideoWriter_fourcc(*cfg.video_codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(out_fps), (width, height))
    if writer.isOpened():
        return writer, str(path)

    # mp4v is not always built into the DevKit's OpenCV. MJPG in an AVI works
    # essentially everywhere, at the cost of a much larger file.
    writer.release()
    fallback = path.with_suffix(".avi")
    console.warn(
        f"codec {cfg.video_codec!r} unavailable for {path.name}, "
        f"falling back to MJPG/{fallback.name}"
    )
    writer = cv2.VideoWriter(
        str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), float(out_fps), (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"could not open a video writer for {path} with codec {cfg.video_codec} "
            f"or MJPG fallback. Set output.video.enable: false to skip video output."
        )
    return writer, str(fallback)


def wants_jpeg(cfg, index: int) -> bool:
    return cfg.save_enable and cfg.save_every > 0 and index % cfg.save_every == 0


def wants_annotated(cfg, pipeline: Pipeline, need_jpeg: bool) -> bool:
    """Whether any sink on this frame needs the overlay rendered."""
    return bool(
        pipeline.writer is not None
        or (need_jpeg and cfg.save_overlay)
    )


def save_frame(cfg, index: int, frame) -> None:
    out_path = Path(cfg.save_dir) / f"frame_{index:06d}.{cfg.save_format}"
    if not runtime.cv2.imwrite(str(out_path), frame):
        console.warn(f"failed to write {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sink thread
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SinkJob:
    """One finished frame, handed to the sink thread.

    Every field is plain numpy or plain Python. Nothing here references a
    ``pyneat`` sample, so the decoder's buffer is already back in its pool by
    the time a job is queued.

    Attributes:
        index: 1-based frame number, used for the stills filename and ``every``.
        stamp: Timing fields copied out of the source sample.
        frame: Untouched BGR frame.
        results: Whatever the task detected on it -- boxes, instances or tracks.
        fps: Pipeline frame rate for the HUD badge, 0 until measured.
    """

    index: int
    stamp: FrameStamp
    frame: object
    results: list
    fps: float


class SinkWorker:
    """Runs compositing, stills and the recording on a thread of its own.

    The pull loop used to blur, draw, JPEG-encode and write the video before
    asking for the next frame. All of that is pure numpy and needs no buffer
    from the decoder, but it still gated ``pull()``, and a consumer that pauses
    for a few hundred milliseconds per frame is what lets decoded frames pile up
    in the queues between the decoder and the app. The pool is eight buffers on
    this board, so a big enough pile starves the decoder, which then cannot
    produce the frame that would release the pile: the run stops with a pull
    timeout part-way through the clip.

    Moving that work here means the loop pulls again immediately, so buffers go
    back at the rate the decoder can reuse them. Ordering is preserved because
    there is exactly one worker draining a FIFO, so the recording still has the
    source's frame order.

    The queue is bounded. When the sinks fall behind, ``submit`` blocks, which is
    the backpressure that keeps memory flat -- but it blocks after ``depth``
    frames of slack rather than on every single one.

    Attributes:
        error: First exception raised on the worker, re-raised by ``close``.
        blocked_ms: Total time ``submit`` spent waiting for a free slot.
    """

    def __init__(self, cfg, pipeline: Pipeline, depth: int, render) -> None:
        """
        Args:
            cfg: Application configuration.
            pipeline: Live pipeline.
            depth: Queue depth, from ``runtime.queue_depth``.
            render: ``(cfg, pipeline, frame, results, fps) -> annotated frame``.
        """
        self.cfg = cfg
        self.pipeline = pipeline
        self.render = render
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.error: BaseException | None = None
        self.blocked_ms = 0.0
        self.thread = threading.Thread(target=self._run, name="sinks", daemon=True)
        self.thread.start()

    def submit(self, job: SinkJob) -> None:
        start = time_ms()
        self.queue.put(job)
        self.blocked_ms += time_ms() - start

    def drain(self) -> None:
        """Block until every queued frame has been written."""
        self.queue.join()

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                self._handle(job)
            except BaseException as exc:  # noqa: BLE001 - reported by close()
                if self.error is None:
                    self.error = exc
            finally:
                self.queue.task_done()

    def _handle(self, job: SinkJob) -> None:
        cfg, pipeline = self.cfg, self.pipeline
        need_jpeg = wants_jpeg(cfg, job.index)
        need_annotated = wants_annotated(cfg, pipeline, need_jpeg)

        # Render once, then share the result across every sink that wants it.
        annotated = (
            self.render(cfg, pipeline, job.frame, job.results, job.fps)
            if need_annotated
            else None
        )

        if pipeline.writer is not None:
            pipeline.writer.write(annotated)
            pipeline.writer_frames += 1
        if need_jpeg:
            save_frame(cfg, job.index, annotated if cfg.save_overlay else job.frame)
