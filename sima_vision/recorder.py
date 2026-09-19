"""The annotated recording, encoded on the DevKit's hardware H.264 encoder.

OpenCV's ``mp4v`` writer encodes in software on the A65 cores, at roughly
100 ms a 1080p frame: a recording ran at about ten frames a second while the
pipeline feeding it managed sixty, and the sink backlog that built up behind it
was what eventually held up the pull loop. The board has an H.264 encoder in
hardware, and Neat exposes it as ``nodes.h264_encode_sima``.

So the recorder is a small Neat graph of its own::

    input(NV12) -> h264_encode_sima -> h264_parse -> output("encoded")

frames are pushed into it from the sink thread, and a second thread pulls the
encoded access units and hands them to :class:`~sima_vision.mp4.Mp4Writer`.
Converting a frame to NV12 and pushing it costs about 11 ms, so the recorder
keeps up with the pipeline instead of setting its pace.
"""

from __future__ import annotations

import threading

from . import runtime
from .mp4 import Mp4Writer

#: How long the drain thread waits for one encoded frame before checking
#: whether it should stop.
PULL_TIMEOUT_MS = 1000

#: Empty pulls, after the input is closed, before the stream counts as flushed.
FLUSH_PULLS = 3

#: H.264 level for the encoder. 4.1 covers 1080p at 60 fps.
LEVEL = "4.1"


def bgr_to_nv12(bgr, np, cv2):
    """One BGR frame as a contiguous NV12 buffer: the Y plane, then interleaved UV."""
    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    nv12 = np.empty((h * 3 // 2, w), np.uint8)
    nv12[:h] = i420[:h]
    uv = nv12[h:].reshape(-1)
    uv[0::2] = i420[h:h + h // 4].reshape(-1)
    uv[1::2] = i420[h + h // 4:].reshape(-1)
    return nv12


class NeatVideoWriter:
    """``cv2.VideoWriter``'s interface over the hardware encoder.

    Only what the sinks use: :meth:`write`, :meth:`release` and
    :meth:`isOpened`. Frames must be BGR at the size the writer was opened
    with, which is what every task renders.

    Attributes:
        error: First exception raised by the drain thread, re-raised by
            :meth:`release`.
    """

    def __init__(self, path: str, width: int, height: int, fps: int,
                 bitrate_kbps: int, profile: str = "high") -> None:
        if width % 2 or height % 2:
            raise ValueError(f"NV12 needs even dimensions, got {width}x{height}")
        pyneat = runtime.pyneat
        self.width, self.height = width, height
        self.error: BaseException | None = None

        options = pyneat.InputOptions()
        options.width, options.height = width, height
        options.fps_n, options.fps_d = fps, 1
        options.block = True
        # A raw byte tensor with the caps spelled out, because from_numpy cannot
        # build a two-plane NV12 image and the encoder takes nothing else.
        options.caps_override = (
            f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
        )
        graph = pyneat.Graph("recorder")
        graph.add(pyneat.nodes.input(options))
        graph.add(pyneat.nodes.h264_encode_sima(width, height, fps, bitrate_kbps, profile, LEVEL))
        graph.add(pyneat.nodes.h264_parse(config_interval=-1))
        graph.add(pyneat.nodes.output("encoded", pyneat.OutputOptions.every_frame(8)))
        # Held for the same reason as Pipeline.graph: the Run uses what it owns.
        self.graph = graph

        np = runtime.np
        seed = self._tensor(np.zeros((height * 3 // 2, width), np.uint8))
        self.run = graph.build([seed])
        self.mp4 = Mp4Writer(path, width, height, fps)
        self.closing = False
        self.thread = threading.Thread(target=self._drain, name="recorder", daemon=True)
        self.thread.start()

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoWriter's name
        return True

    def write(self, frame) -> None:
        if self.error is not None:
            raise self.error
        np, pyneat = runtime.np, runtime.pyneat
        sample = pyneat.make_tensor_sample(
            "", self._tensor(bgr_to_nv12(frame, np, runtime.cv2))
        )
        if not self.run.push([sample]):
            raise RuntimeError("the hardware encoder refused a frame")

    def release(self) -> None:
        """Flush the encoder, finish the MP4 and stop the graph."""
        if self.closing:
            return
        self.closing = True
        try:
            self.run.close_input()
            self.thread.join()
        finally:
            self.mp4.close()
            self.run.close()
        if self.error is not None:
            raise self.error

    @property
    def frames(self) -> int:
        """Frames that have reached the file."""
        return self.mp4.frames

    def _tensor(self, nv12):
        pyneat = runtime.pyneat
        return pyneat.Tensor.from_numpy(
            nv12.reshape(-1), copy=True, byte_format=pyneat.ByteFormat.Raw
        )

    def _drain(self) -> None:
        from .samples import first_tensor

        quiet = 0
        try:
            while True:
                sample = self.run.pull("encoded", PULL_TIMEOUT_MS)
                if sample is None:
                    # Quiet is normal while the pipeline catches up. Once the
                    # input is closed, it means the encoder has flushed -- but
                    # one empty pull can still beat the last frames out, so
                    # give it a few before calling the stream finished.
                    if self.closing:
                        quiet += 1
                        if quiet >= FLUSH_PULLS or not self.run.running():
                            return
                    continue
                quiet = 0
                self.mp4.add(bytes(first_tensor(sample).copy_payload_bytes()))
        except BaseException as exc:  # noqa: BLE001 - reported by release()
            self.error = exc
