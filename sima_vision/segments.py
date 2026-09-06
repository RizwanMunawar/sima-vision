"""Cutting a long clip into pieces a single decode can finish.

The SiMa decoder stops part-way through a long clip. Measured on a Modalix
DevKit across two clips, three resolutions and every lever this app has: 190,
195 and 202 frames of a 379 frame clip, 173 and 181 of a 341 frame one. It is
not the file (both decode end to end under OpenCV), not the container, not the
sinks (190 frames with the sinks costing 0.1 ms), not the pool size (13 buffers
at 720p, same behaviour), and not the GOP layout (the two clips are laid out
completely differently). Frame ids confirm the source simply stops numbering.

A fresh process always gets another ~195 frames, so whatever wedges is reset by
building the decode again. That is the whole idea here: hand the runtime one
piece at a time, each short enough to finish, and let the recording span them.

A decoder can only start at an IDR, so the cuts go there. A clip whose
keyframes are further apart than a piece can be is left alone and said so --
splitting it would produce pieces that stall exactly as before. ``ffmpeg -g 50``
is the fix for that, and it is what the stall advice already recommends.
"""

from __future__ import annotations

from pathlib import Path

from .media import BitReader, unescape_rbsp

#: NAL unit types that begin a coded picture.
SLICE_TYPES = frozenset({1, 5})

#: Bytes of slice header to read. Only ``first_mb_in_slice`` is wanted, and it
#: is the first field, but Exp-Golomb needs room to be sure of a long value.
SLICE_HEADER_BYTES = 24

START_CODE = b"\x00\x00\x00\x01"


class Picture:
    """One coded picture's place in the stream.

    Attributes:
        start: Byte offset of the start code that begins it.
        idr: Whether it is an IDR, and so somewhere a decode may begin.
    """

    __slots__ = ("start", "idr")

    def __init__(self, start: int, idr: bool) -> None:
        self.start = start
        self.idr = idr


def scan_pictures(data: bytes) -> tuple[list[Picture], bytes]:
    """Every coded picture in an Annex-B stream, and its parameter sets.

    Returns:
        A ``(pictures, header)`` pair. ``header`` is the SPS and PPS as Annex-B
        bytes, to be written at the top of every piece: an elementary stream
        has no container to carry them, so a decoder starting at piece three
        has nowhere else to get them.
    """
    pictures: list[Picture] = []
    header = bytearray()
    seen: set[bytes] = set()
    pos = 0
    while True:
        index = data.find(b"\x00\x00\x01", pos)
        if index < 0:
            break
        body = index + 3
        if body >= len(data):
            break
        kind = data[body] & 0x1F
        # Back up over the leading zero of a four byte start code, so a cut
        # here keeps the whole start code with the picture that follows it.
        begin = index - 1 if index and data[index - 1] == 0 else index

        if kind in (7, 8):                       # SPS, PPS
            end = data.find(b"\x00\x00\x01", body)
            nal = data[body:end - 1 if end > 0 and data[end - 1] == 0 else end] \
                if end > 0 else data[body:]
            if nal not in seen:
                seen.add(bytes(nal))
                header += START_CODE + nal
        elif kind in SLICE_TYPES:
            try:
                first_mb = BitReader(
                    unescape_rbsp(data[body + 1:body + 1 + SLICE_HEADER_BYTES])
                ).ue()
            except (ValueError, IndexError):
                first_mb = -1
            if first_mb == 0:
                pictures.append(Picture(begin, kind == 5))
        pos = body
    return pictures, bytes(header)


def plan_cuts(pictures: list[Picture], max_frames: int) -> list[int]:
    """Which pictures to begin a piece at, by index into ``pictures``.

    A cut may only land on an IDR. Within that constraint the aim is pieces as
    close to ``max_frames`` as the keyframes allow, so the last IDR that still
    fits wins rather than the first one past the limit.

    Returns:
        Picture indices, always starting with 0. One entry means no useful cut
        exists and the clip should be run whole.
    """
    if max_frames < 1 or not pictures:
        return [0]
    cuts = [0]
    for index, picture in enumerate(pictures):
        if picture.idr and index - cuts[-1] >= max_frames:
            cuts.append(index)
    return cuts


def longest_piece(pictures: list[Picture], cuts: list[int]) -> int:
    """Frames in the longest piece these cuts produce."""
    bounds = [*cuts, len(pictures)]
    return max(b - a for a, b in zip(bounds, bounds[1:], strict=False)) if pictures else 0


def split(source: Path, out_dir: Path, max_frames: int) -> list[tuple[Path, int]]:
    """Write ``source`` out as pieces of at most ``max_frames`` frames each.

    Args:
        source: Raw Annex-B H.264 to cut.
        out_dir: Directory for the pieces, created if it is not there.
        max_frames: Frames to aim for per piece.

    Returns:
        A list of ``(path, frames)``, in order. A single entry means the clip
        was not worth cutting -- either it is short enough already, or its
        keyframes are too far apart for cutting to help.
    """
    data = source.read_bytes()
    pictures, header = scan_pictures(data)
    if len(pictures) <= max_frames:
        return [(source, len(pictures))]

    cuts = plan_cuts(pictures, max_frames)
    # Cutting is only worth doing if every piece ends up short enough to
    # finish. A clip with keyframes 250 apart cannot be cut into 150s: the
    # first piece is still 250 and still stalls, and all the cut achieved was
    # to make the failure harder to read.
    if len(cuts) < 2 or longest_piece(pictures, cuts) > max_frames:
        return [(source, len(pictures))]

    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = [*cuts, len(pictures)]
    pieces: list[tuple[Path, int]] = []
    for number, (first, stop) in enumerate(zip(bounds, bounds[1:], strict=False), start=1):
        begin = pictures[first].start
        end = pictures[stop].start if stop < len(pictures) else len(data)
        piece = out_dir / f"{source.stem}-part{number:03d}.h264"
        body = data[begin:end]
        # The parameter sets lead every piece. Only piece one is guaranteed to
        # have them already, and a decoder handed piece two without them
        # negotiates nothing and produces no frames at all.
        piece.write_bytes(body if body.startswith(header) else header + body)
        pieces.append((piece, stop - first))
    return pieces


def describe(pieces: list[tuple[Path, int]], total: int) -> str:
    """One line for the startup step."""
    if len(pieces) < 2:
        return ""
    counts = ", ".join(str(frames) for _, frames in pieces)
    return f"decoding in {len(pieces)} pieces of {counts} frames ({total} total)"
