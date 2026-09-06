"""MP4 to Annex-B.

The remux has to be exact: every coded bit of the H.264 survives, only the
framing changes. So these build an MP4 byte by byte, remux it, and check the
elementary stream that comes out NAL by NAL. Nothing here needs ffmpeg, a
board, or a sample file.
"""

from __future__ import annotations

import struct

import pytest

from sima_vision.mp4 import (
    is_container,
    looks_like_mp4,
    parse_avcc,
    remux,
    sample_offsets,
    to_annex_b,
)

START = b"\x00\x00\x00\x01"

SPS = bytes([0x67, 0x42, 0xC0, 0x1E, 0xAA, 0xBB])
PPS = bytes([0x68, 0xCE, 0x3C, 0x80])


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + kind + payload


def avcc(length_size: int = 4) -> bytes:
    return box(b"avcC", bytes([1, 0x42, 0xC0, 0x1E, 0xFC | (length_size - 1), 0xE1])
               + struct.pack(">H", len(SPS)) + SPS
               + bytes([1]) + struct.pack(">H", len(PPS)) + PPS)


def visual_entry(kind: bytes = b"avc1", length_size: int = 4) -> bytes:
    # 8 bytes of SampleEntry plus 70 of VisualSampleEntry, then the children.
    return box(kind, b"\x00" * 78 + avcc(length_size))


def sample_table(sizes: list[int], chunk_offset: int, entry: bytes | None = None,
                 per_chunk: int | None = None, length_size: int = 4) -> bytes:
    entry = visual_entry(length_size=length_size) if entry is None else entry
    stsd = box(b"stsd", b"\x00" * 4 + struct.pack(">I", 1) + entry)
    stsz = box(b"stsz", b"\x00" * 4 + struct.pack(">II", 0, len(sizes))
               + b"".join(struct.pack(">I", n) for n in sizes))
    count = len(sizes) if per_chunk is None else per_chunk
    stsc = box(b"stsc", b"\x00" * 4 + struct.pack(">I", 1)
               + struct.pack(">III", 1, count, 1))
    stco = box(b"stco", b"\x00" * 4 + struct.pack(">I", 1)
               + struct.pack(">I", chunk_offset))
    return box(b"stbl", stsd + stsz + stsc + stco)


def track(handler: bytes, stbl: bytes) -> bytes:
    hdlr = box(b"hdlr", b"\x00" * 8 + handler + b"\x00" * 12 + b"track\x00")
    return box(b"trak", box(b"mdia", hdlr + box(b"minf", stbl)))


def build_mp4(samples: list[bytes], length_size: int = 4, audio_first: bool = True,
              entry: bytes | None = None) -> bytes:
    """A minimal but real MP4: ftyp, mdat, then moov with absolute offsets.

    ``mdat`` goes before ``moov`` so the chunk offsets in ``stco`` have to be
    genuine file positions, which is what a real muxer writes and what the
    parser has to cope with.
    """
    payload = b"".join(
        b"".join(struct.pack(">I", len(n))[4 - length_size:] + n for n in nals)
        for nals in samples
    )
    sizes = [
        sum(length_size + len(n) for n in nals) for nals in samples
    ]
    ftyp = box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomavc1")
    mdat = box(b"mdat", payload)
    data_at = len(ftyp) + 8                       # past ftyp and the mdat header

    video = track(b"vide", sample_table(sizes, data_at, entry, length_size=length_size))
    if audio_first:
        # A real file usually has audio first; picking trak[0] would take it.
        silent = track(b"soun", sample_table([1], data_at))
        moov = box(b"moov", silent + video)
    else:
        moov = box(b"moov", video)
    return ftyp + mdat + moov


def nals_in(stream: bytes) -> list[bytes]:
    """Split an Annex-B stream on its four byte start codes."""
    assert stream.startswith(START)
    return [part for part in stream.split(START) if part]


# ── the shape of the thing ──


def test_a_container_is_recognised_by_suffix_and_by_content():
    assert is_container("clip.mp4") and is_container("CLIP.MOV")
    assert not is_container("clip.h264")
    # Content, because a renamed file is the failure worth catching.
    assert looks_like_mp4(b"\x00\x00\x00\x18ftypisom")
    assert not looks_like_mp4(START + b"\x67\x42")


def test_the_avcc_record_gives_up_its_length_size_and_parameter_sets():
    length_size, sets = parse_avcc(avcc(4)[8:])
    assert length_size == 4
    assert sets == [SPS, PPS]
    assert parse_avcc(avcc(2)[8:])[0] == 2


def test_a_sample_becomes_start_code_separated_nal_units():
    sample = struct.pack(">I", 3) + b"\x41ab" + struct.pack(">I", 2) + b"\x01c"
    body, idr = to_annex_b(sample, 4)
    assert body == START + b"\x41ab" + START + b"\x01c"
    assert not idr


def test_an_idr_sample_is_reported_so_the_parameter_sets_can_be_repeated():
    sample = struct.pack(">I", 2) + b"\x65X"      # nal_unit_type 5
    _, idr = to_annex_b(sample, 4)
    assert idr


def test_a_truncated_sample_stops_rather_than_reading_past_its_end():
    """A length prefix claiming more than is there must not run off the buffer."""
    body, _ = to_annex_b(struct.pack(">I", 2) + b"\x41a" + struct.pack(">I", 99), 4)
    assert body == START + b"\x41a"


# ── the sample table ──


def test_samples_are_placed_by_walking_the_chunk_runs():
    """Two chunks of two, so the run table is actually consulted."""
    offsets = sample_offsets([10, 20, 30, 40], chunks=[1000, 5000],
                             runs=[(1, 2)])
    assert offsets == [1000, 1010, 5000, 5030]


def test_a_later_run_takes_over_from_the_chunk_it_names():
    offsets = sample_offsets([1, 1, 1, 1, 1], chunks=[100, 200, 300],
                             runs=[(1, 1), (2, 2)])
    assert offsets == [100, 200, 201, 300, 301]


def test_a_sample_table_promising_more_than_it_has_stops_cleanly():
    assert sample_offsets([5], chunks=[100, 200], runs=[(1, 4)]) == [100]


# ── end to end ──


def test_a_remuxed_clip_carries_every_nal_through_unchanged(tmp_path):
    """The bits are the point: a remux may reframe, never re-encode."""
    frames = [
        [bytes([0x65]) + b"idr-payload"],
        [bytes([0x41]) + b"inter-one"],
        [bytes([0x41]) + b"inter-two", bytes([0x41]) + b"second-slice"],
    ]
    src = tmp_path / "clip.mp4"
    src.write_bytes(build_mp4(frames))
    out = tmp_path / "clip.h264"

    assert remux(src, out) == 3

    got = nals_in(out.read_bytes())
    # Parameter sets lead, because an elementary stream carries them in band.
    assert got[0] == SPS and got[1] == PPS
    assert [n for n in got if n not in (SPS, PPS)] == [n for f in frames for n in f]


def test_the_parameter_sets_are_repeated_at_every_idr(tmp_path):
    """A decoder joining at a later IDR has no container to ask."""
    frames = [
        [bytes([0x65]) + b"first"],
        [bytes([0x41]) + b"middle"],
        [bytes([0x65]) + b"second-idr"],
    ]
    src = tmp_path / "clip.mp4"
    src.write_bytes(build_mp4(frames))
    out = tmp_path / "clip.h264"
    remux(src, out)

    assert nals_in(out.read_bytes()).count(SPS) == 2


def test_the_video_track_is_chosen_by_handler_not_by_position(tmp_path):
    """trak[0] is audio here, and taking it would remux silence."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(build_mp4([[bytes([0x65]) + b"video"]], audio_first=True))
    out = tmp_path / "clip.h264"
    remux(src, out)
    assert bytes([0x65]) + b"video" in out.read_bytes()


@pytest.mark.parametrize("length_size", [1, 2, 4])
def test_every_nal_length_size_an_avcc_can_declare_is_handled(tmp_path, length_size):
    frames = [[bytes([0x65]) + b"x" * 20]]
    src = tmp_path / "clip.mp4"
    src.write_bytes(build_mp4(frames, length_size=length_size))
    out = tmp_path / "clip.h264"
    assert remux(src, out) == 1
    assert bytes([0x65]) + b"x" * 20 in out.read_bytes()


def test_the_output_is_something_the_raw_path_will_accept(tmp_path):
    """It is fed straight to the h264 source, whose own check is strict."""
    from sima_vision.media import count_h264_pictures

    src = tmp_path / "clip.mp4"
    src.write_bytes(build_mp4([[bytes([0x65]) + b"a"], [bytes([0x41]) + b"b"]]))
    out = tmp_path / "clip.h264"
    remux(src, out)

    head = out.read_bytes()[:4]
    assert head == START, "the raw path rejects anything not starting Annex-B"
    assert count_h264_pictures(str(out)) >= 0


# ── refusing what it cannot do, by name ──


def test_a_fragmented_mp4_is_named_not_misparsed(tmp_path):
    """No moov means no sample table, and silence would look like a stall."""
    src = tmp_path / "frag.mp4"
    src.write_bytes(box(b"ftyp", b"iso5") + box(b"moof", b"\x00" * 8))
    with pytest.raises(RuntimeError, match="fragmented"):
        remux(src, tmp_path / "out.h264")


def test_a_video_track_that_is_not_h264_says_so(tmp_path):
    src = tmp_path / "hevc.mp4"
    src.write_bytes(build_mp4([[b"\x65x"]], entry=box(b"hvc1", b"\x00" * 78)))
    with pytest.raises(RuntimeError, match="not H.264"):
        remux(src, tmp_path / "out.h264")


def test_a_file_with_no_video_track_at_all_says_so(tmp_path):
    audio = track(b"soun", sample_table([1], 100))
    src = tmp_path / "audio.mp4"
    src.write_bytes(box(b"ftyp", b"isom") + box(b"moov", audio))
    with pytest.raises(RuntimeError, match="no video track"):
        remux(src, tmp_path / "out.h264")
