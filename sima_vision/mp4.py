"""MP4 to Annex-B, so a container can be run on a board without ffmpeg.

A container goes through ``groups.video_input``, which builds its own
decoder, and that decoder cannot be asked for ``decoder-tuning=default``. Left
on ``auto`` it discards pictures: 57 of 78 frames of the short sample clip came
through it on Neat 0.4.0. See
:func:`sima_vision.media.default_tuned_decoder`. (Neat 0.3.0 could not build
a container source at all, because of a demuxer naming bug.)

The way past it is to stop handing Neat a container at all. An MP4 holds the
very H.264 the raw path already runs; only the framing differs. In a container
the NAL units are length-prefixed and the parameter sets live in ``avcC``; in
an elementary stream they are separated by start codes and the parameter sets
travel in-band. Rewriting one into the other is a remux, not a re-encode: every
coded bit survives, so there is no quality to lose and no MLA time to spend.

This is ``ffmpeg -c:v copy -bsf:v h264_mp4toannexb``, which is what the app used
to tell people to run by hand -- on a DevKit that does not have ffmpeg.

Only the plain ``moov``/``stbl`` layout is read. Fragmented MP4s keep their
sample tables in ``moof`` boxes instead, and are refused by name rather than
misparsed into silence.
"""

from __future__ import annotations

import struct
from pathlib import Path

#: Suffixes routed through here. ``.mov`` is the same box structure.
CONTAINER_SUFFIXES = {".mp4", ".m4v", ".mov", ".m4a"}

#: Sample entries whose payload is H.264. ``avc3`` keeps its parameter sets
#: in-band rather than in ``avcC``, which changes nothing here: in-band sets
#: survive the copy, and any in ``avcC`` are written out as well.
H264_SAMPLE_ENTRIES = (b"avc1", b"avc3")

#: Boxes that hold other boxes, and how many payload bytes to skip first.
#: ``stsd`` carries a version, flags and an entry count; a visual sample entry
#: carries the 8 byte SampleEntry plus the 70 byte VisualSampleEntry before its
#: children begin.
CONTAINERS: dict[bytes, int] = {
    b"moov": 0,
    b"trak": 0,
    b"mdia": 0,
    b"minf": 0,
    b"stbl": 0,
    b"stsd": 8,
    b"avc1": 78,
    b"avc3": 78,
}

START_CODE = b"\x00\x00\x00\x01"


def is_container(path: str) -> bool:
    """Whether this path is one this module should be asked about."""
    return Path(path).suffix.lower() in CONTAINER_SUFFIXES


def looks_like_mp4(head: bytes) -> bool:
    """Whether these opening bytes are an ISO base media file.

    Every ISO BMFF file opens with a box, and in practice that box is ``ftyp``,
    whose type sits at offset 4. Checked on content rather than on the suffix
    because a renamed file is the failure this is here to catch.
    """
    return len(head) >= 8 and head[4:8] in (b"ftyp", b"styp")


# ─────────────────────────────────────────────────────────────────────────────
# Box walking
# ─────────────────────────────────────────────────────────────────────────────


def iter_boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield ``(type, payload_start, payload_end)`` for boxes in one range.

    Args:
        data: Buffer holding the boxes.
        start: Offset of the first box header.
        end: Offset just past the last box, or None for the whole buffer.

    Yields:
        One triple per box, in file order.
    """
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        header = 8
        if size == 1:                       # 64 bit largesize follows the type
            if pos + 16 > end:
                return
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header = 16
        elif size == 0:                     # runs to the end of its container
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def find_box(data: bytes, path: tuple[bytes, ...], start: int = 0,
             end: int | None = None) -> tuple[int, int] | None:
    """Follow a box path, descending only into boxes known to hold others.

    Args:
        data: Buffer to search.
        path: Box types to follow, outermost first.
        start: Where to begin.
        end: Where to stop, or None for the whole buffer.

    Returns:
        The final box's ``(payload_start, payload_end)``, or None if any step
        of the path is missing.
    """
    if not path:
        return (start, len(data) if end is None else end)
    head, rest = path[0], path[1:]
    for kind, body, stop in iter_boxes(data, start, end):
        if kind != head:
            continue
        if not rest:
            return body, stop
        found = find_box(data, rest, body + CONTAINERS.get(kind, 0), stop)
        if found is not None:
            return found
    return None


def video_track(moov: bytes) -> tuple[int, int]:
    """The sample table of the first video track.

    A file can carry audio, subtitles and several video tracks. Picking the
    first ``trak`` outright would hand back an audio sample table on most real
    files, so the handler type decides.

    Raises:
        RuntimeError: When no track has a video handler with a sample table.
    """
    for kind, body, stop in iter_boxes(moov):
        if kind != b"trak":
            continue
        handler = find_box(moov, (b"mdia", b"hdlr"), body, stop)
        if handler is None:
            continue
        # hdlr: version and flags, pre_defined, then the four byte handler type.
        start, _ = handler
        if moov[start + 8:start + 12] != b"vide":
            continue
        stbl = find_box(moov, (b"mdia", b"minf", b"stbl"), body, stop)
        if stbl is not None:
            return stbl
    raise RuntimeError("no video track with a sample table in this MP4")


# ─────────────────────────────────────────────────────────────────────────────
# Sample table
# ─────────────────────────────────────────────────────────────────────────────


def parse_avcc(payload: bytes) -> tuple[int, list[bytes]]:
    """Read an AVCDecoderConfigurationRecord.

    Returns:
        A ``(length_size, parameter_sets)`` pair. ``length_size`` is how many
        bytes prefix each NAL unit in ``mdat``, and the parameter sets are the
        SPS and PPS payloads in the order they should be written out.
    """
    if len(payload) < 7:
        raise RuntimeError("avcC box is too short to be a configuration record")
    length_size = (payload[4] & 0x03) + 1
    sets: list[bytes] = []
    pos = 5
    for count_mask in (0x1F, 0xFF):          # SPS count is 5 bits, PPS count 8
        if pos >= len(payload):
            break
        count = payload[pos] & count_mask
        pos += 1
        for _ in range(count):
            if pos + 2 > len(payload):
                raise RuntimeError("avcC ended inside a parameter set length")
            size = int.from_bytes(payload[pos:pos + 2], "big")
            pos += 2
            if pos + size > len(payload):
                raise RuntimeError("avcC ended inside a parameter set")
            sets.append(payload[pos:pos + size])
            pos += size
    if not sets:
        raise RuntimeError("avcC carries no SPS or PPS")
    return length_size, sets


def parse_stsz(payload: bytes) -> list[int]:
    """Sample sizes, expanding the constant-size form."""
    if len(payload) < 12:
        raise RuntimeError("stsz box is too short")
    uniform, count = struct.unpack_from(">II", payload, 4)
    if uniform:
        return [uniform] * count
    if len(payload) < 12 + 4 * count:
        raise RuntimeError("stsz is shorter than its own sample count")
    return list(struct.unpack_from(f">{count}I", payload, 12))


def parse_chunk_offsets(stbl: bytes, start: int, end: int) -> list[int]:
    """Chunk offsets from ``stco``, or ``co64`` on a file over 4 GB."""
    box = find_box(stbl, (b"stco",), start, end)
    if box is not None:
        body, _ = box
        count = struct.unpack_from(">I", stbl, body + 4)[0]
        return list(struct.unpack_from(f">{count}I", stbl, body + 8))
    box = find_box(stbl, (b"co64",), start, end)
    if box is None:
        raise RuntimeError("sample table has neither stco nor co64")
    body, _ = box
    count = struct.unpack_from(">I", stbl, body + 4)[0]
    return list(struct.unpack_from(f">{count}Q", stbl, body + 8))


def parse_stsc(payload: bytes) -> list[tuple[int, int]]:
    """Sample-to-chunk runs as ``(first_chunk, samples_per_chunk)`` pairs."""
    count = struct.unpack_from(">I", payload, 4)[0]
    runs = []
    for i in range(count):
        first, per_chunk, _ = struct.unpack_from(">III", payload, 8 + 12 * i)
        runs.append((first, per_chunk))
    return runs


def sample_offsets(sizes: list[int], chunks: list[int],
                   runs: list[tuple[int, int]]) -> list[int]:
    """Absolute file offset of every sample, in decode order.

    The sample table stores this by chunk to keep itself small: ``stsc`` says
    how many samples each run of chunks holds, ``stco`` says where each chunk
    starts, and sizes accumulate within a chunk. Flattening it here means the
    remux is one ordered pass afterwards.
    """
    if not runs:
        raise RuntimeError("sample table has no sample-to-chunk entries")
    offsets: list[int] = []
    sample = 0
    for index, chunk_start in enumerate(chunks):
        chunk_number = index + 1
        # The last run whose first_chunk has been reached governs this chunk.
        per_chunk = runs[0][1]
        for first, count in runs:
            if first <= chunk_number:
                per_chunk = count
            else:
                break
        position = chunk_start
        for _ in range(per_chunk):
            if sample >= len(sizes):
                return offsets
            offsets.append(position)
            position += sizes[sample]
            sample += 1
    return offsets


# ─────────────────────────────────────────────────────────────────────────────
# Remux
# ─────────────────────────────────────────────────────────────────────────────


def to_annex_b(sample: bytes, length_size: int) -> tuple[bytes, bool]:
    """Rewrite one length-prefixed sample as Annex-B NAL units.

    Returns:
        A ``(bytes, has_idr)`` pair. ``has_idr`` reports whether the sample
        holds an IDR slice, which is where parameter sets have to be repeated.
    """
    out = bytearray()
    idr = False
    pos = 0
    while pos + length_size <= len(sample):
        size = int.from_bytes(sample[pos:pos + length_size], "big")
        pos += length_size
        if size <= 0 or pos + size > len(sample):
            break
        out += START_CODE
        out += sample[pos:pos + size]
        if sample[pos] & 0x1F == 5:          # nal_unit_type 5 == IDR slice
            idr = True
        pos += size
    return bytes(out), idr


def top_level(fh, size: int):
    """Yield ``(type, payload_start, payload_end)`` for the outermost boxes.

    By seeking rather than reading, so a file larger than memory can still be
    indexed. Only ``moov`` is ever loaded whole; ``mdat`` is read a sample at a
    time, which matters on a board where a careless allocation is what started
    all of this.
    """
    pos = 0
    while pos + 8 <= size:
        fh.seek(pos)
        raw = fh.read(8)
        if len(raw) < 8:
            return
        box_size = int.from_bytes(raw[:4], "big")
        kind = raw[4:8]
        header = 8
        if box_size == 1:
            box_size = int.from_bytes(fh.read(8), "big")
            header = 16
        elif box_size == 0:
            box_size = size - pos
        if box_size < header or pos + box_size > size:
            return
        yield kind, pos + header, pos + box_size
        pos += box_size


def remux(src: Path, dst: Path) -> int:
    """Write ``src`` out as a raw Annex-B H.264 stream at ``dst``.

    Parameter sets are written ahead of every IDR, not only once at the top.
    An elementary stream has no container to hold them, and a decoder joining
    at any IDR -- which is what the SiMa decoder does after a mid-stream
    reconfigure -- needs them in band.

    Args:
        src: MP4 to read.
        dst: Where to write the elementary stream.

    Returns:
        The number of samples written, which is the frame count.

    Raises:
        RuntimeError: When the file is fragmented, has no H.264 video track, or
            its sample table cannot be read.
    """
    size = src.stat().st_size
    with src.open("rb") as fh:
        boxes = {kind: (start, stop) for kind, start, stop in top_level(fh, size)}
        if b"moov" not in boxes:
            kind = "fragmented" if b"moof" in boxes else "unreadable"
            raise RuntimeError(
                f"{src} has no moov box, so its sample table cannot be read "
                f"({kind} MP4).\n"
                "  Convert it with ffmpeg on a machine that has one:\n"
                "  ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb "
                "-f h264 clip.h264"
            )
        moov_start, moov_end = boxes[b"moov"]
        fh.seek(moov_start)
        moov = fh.read(moov_end - moov_start)
        return _write_annex_b(src, dst, fh, moov, size)


def _write_annex_b(src: Path, dst: Path, fh, moov: bytes, size: int) -> int:
    """The second half of :func:`remux`, once ``moov`` is in hand."""
    stbl_start, stbl_end = video_track(moov)

    entry = None
    for name in H264_SAMPLE_ENTRIES:
        entry = find_box(moov, (b"stsd", name, b"avcC"), stbl_start, stbl_end)
        if entry is not None:
            break
    if entry is None:
        raise RuntimeError(
            f"{src} has a video track that is not H.264 (no avc1/avc3 with an "
            "avcC box). This app decodes H.264 only."
        )
    length_size, parameter_sets = parse_avcc(moov[entry[0]:entry[1]])

    stsz = find_box(moov, (b"stsz",), stbl_start, stbl_end)
    stsc = find_box(moov, (b"stsc",), stbl_start, stbl_end)
    if stsz is None or stsc is None:
        raise RuntimeError(f"{src} sample table is missing stsz or stsc")

    sizes = parse_stsz(moov[stsz[0]:stsz[1]])
    runs = parse_stsc(moov[stsc[0]:stsc[1]])
    chunks = parse_chunk_offsets(moov, stbl_start, stbl_end)
    offsets = sample_offsets(sizes, chunks, runs)

    header = b"".join(START_CODE + one for one in parameter_sets)
    written = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as out:
        for index, offset in enumerate(offsets):
            length = sizes[index]
            if offset + length > size:
                break
            fh.seek(offset)
            body, idr = to_annex_b(fh.read(length), length_size)
            if not body:
                continue
            if written == 0 or idr:
                out.write(header)
            out.write(body)
            written += 1
    if not written:
        raise RuntimeError(f"{src} yielded no frames; its sample table may be wrong")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────


def box(kind: bytes, *payload: bytes) -> bytes:
    """One box: size, type, payload."""
    body = b"".join(payload)
    return struct.pack(">I", 8 + len(body)) + kind + body


def full_box(kind: bytes, version: int, flags: int, *payload: bytes) -> bytes:
    """A box whose payload opens with a version byte and 24 bits of flags."""
    return box(kind, bytes([version]) + flags.to_bytes(3, "big"), *payload)


def split_annex_b(data: bytes) -> list[bytes]:
    """The NAL units of an Annex-B buffer, start codes removed."""
    starts = []
    pos = data.find(b"\x00\x00\x01")
    while pos >= 0:
        starts.append(pos + 3)
        pos = data.find(b"\x00\x00\x01", pos + 3)
    nals = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 3 if index + 1 < len(starts) else len(data)
        # A four byte start code leaves its leading zero on the previous unit.
        if index + 1 < len(starts) and end > start and data[end - 1] == 0:
            end -= 1
        if end > start:
            nals.append(data[start:end])
    return nals


#: SPS, PPS and access unit delimiters live in ``avcC`` or nowhere, not in a
#: sample.
OUT_OF_BAND = frozenset({7, 8, 9})

#: A unity transformation matrix, as ``mvhd`` and ``tkhd`` both want it.
UNITY_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


class Mp4Writer:
    """Constant frame rate H.264 into an MP4, one access unit at a time.

    The inverse of :func:`remux`, and deliberately as plain: one video track,
    one chunk, every sample the same duration. The duration is the point. A
    recording's frames are all one source interval apart however unevenly the
    app produced them, and a muxer fed wall-clock timestamps would write that
    unevenness into playback -- which is the choppiness this exists to avoid.

    Samples stream into ``mdat`` as they arrive, so memory stays flat; the
    sample table is written on :meth:`close`.

    Attributes:
        path: Where the file is being written.
        frames: Samples written so far.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: int) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = max(1, int(fps))
        self.frames = 0
        self.sizes: list[int] = []
        self.sync: list[int] = []
        self.sps = b""
        self.pps = b""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("wb")
        self.fh.write(box(b"ftyp", b"isom", struct.pack(">I", 0x200), b"isomiso2avc1mp41"))
        # A 64 bit mdat header, patched on close, so size never has to be known
        # up front or fit in 32 bits.
        self.mdat_at = self.fh.tell()
        self.fh.write(struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 0))

    def add(self, access_unit: bytes) -> None:
        """Append one Annex-B access unit as the next frame."""
        sample = bytearray()
        idr = False
        for nal in split_annex_b(access_unit):
            kind = nal[0] & 0x1F
            if kind == 7 and not self.sps:
                self.sps = bytes(nal)
            elif kind == 8 and not self.pps:
                self.pps = bytes(nal)
            if kind in OUT_OF_BAND:
                continue
            idr = idr or kind == 5
            sample += struct.pack(">I", len(nal)) + nal
        if not sample:
            return
        self.fh.write(sample)
        self.sizes.append(len(sample))
        self.frames += 1
        if idr:
            self.sync.append(self.frames)

    def close(self) -> None:
        """Write the sample table and finish the file."""
        if self.fh.closed:
            return
        end = self.fh.tell()
        self.fh.seek(self.mdat_at + 8)
        self.fh.write(struct.pack(">Q", end - self.mdat_at))
        self.fh.seek(end)
        if self.frames and self.sps and self.pps:
            self.fh.write(self._moov(self.mdat_at + 16))
        self.fh.close()

    def _avcc(self) -> bytes:
        sps = self.sps
        record = bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1])
        record += struct.pack(">H", len(sps)) + sps
        record += b"\x01" + struct.pack(">H", len(self.pps)) + self.pps
        if sps[1] in (100, 110, 122, 144):
            # High profiles carry chroma format and bit depth: 4:2:0, 8 bit,
            # which is all the hardware encoder produces.
            record += bytes([0xFD, 0xF8, 0xF8, 0])
        return box(b"avcC", record)

    def _moov(self, data_offset: int) -> bytes:
        timescale = self.fps * 1000
        duration = self.frames * 1000
        avc1 = box(
            b"avc1",
            bytes(6), struct.pack(">H", 1),                       # data_reference_index
            bytes(16),
            struct.pack(">HH", self.width, self.height),
            struct.pack(">II", 0x480000, 0x480000), bytes(4),    # 72 dpi
            struct.pack(">H", 1), bytes(32),                      # frame count, name
            struct.pack(">Hh", 0x18, -1),
            self._avcc(),
        )
        offsets = (
            full_box(b"co64", 0, 0, struct.pack(">IQ", 1, data_offset))
            if data_offset > 0xFFFFFFFF
            else full_box(b"stco", 0, 0, struct.pack(">II", 1, data_offset))
        )
        stbl = box(
            b"stbl",
            full_box(b"stsd", 0, 0, struct.pack(">I", 1), avc1),
            full_box(b"stts", 0, 0, struct.pack(">III", 1, self.frames, 1000)),
            full_box(b"stss", 0, 0, struct.pack(f">I{len(self.sync)}I",
                                                len(self.sync), *self.sync)),
            full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, self.frames, 1)),
            full_box(b"stsz", 0, 0, struct.pack(f">II{self.frames}I", 0, self.frames,
                                                *self.sizes)),
            offsets,
        )
        minf = box(
            b"minf",
            full_box(b"vmhd", 0, 1, bytes(8)),
            box(b"dinf", full_box(b"dref", 0, 0, struct.pack(">I", 1),
                                  full_box(b"url ", 0, 1))),
            stbl,
        )
        mdia = box(
            b"mdia",
            full_box(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, timescale, duration,
                                                0x55C4, 0)),        # language "und"
            full_box(b"hdlr", 0, 0, bytes(4), b"vide", bytes(12), b"VideoHandler\x00"),
            minf,
        )
        tkhd = full_box(
            b"tkhd", 0, 3,
            struct.pack(">IIIII", 0, 0, 1, 0, duration), bytes(8),
            struct.pack(">hhhH", 0, 0, 0, 0), UNITY_MATRIX,
            struct.pack(">II", self.width << 16, self.height << 16),
        )
        mvhd = full_box(
            b"mvhd", 0, 0,
            struct.pack(">IIIIIH", 0, 0, timescale, duration, 0x10000, 0x100), bytes(10),
            UNITY_MATRIX, bytes(24), struct.pack(">I", 2),
        )
        return box(b"moov", mvhd, box(b"trak", tkhd, mdia))
