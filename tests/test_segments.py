"""Cutting a clip into pieces one decode can finish.

The board stops around 195 frames however it is asked, so a longer clip is cut
at its keyframes and decoded piece by piece into one recording. What matters is
that the cuts land only where a decoder can start, that every frame survives the
cut, and that a clip which cannot be cut usefully is left alone rather than
carved into pieces that stall exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from sima_vision import segments
from sima_vision.media import count_h264_pictures

START = b"\x00\x00\x00\x01"

SPS = bytes([0x67, 0x42, 0xC0, 0x1E, 0xAA])
PPS = bytes([0x68, 0xCE, 0x3C, 0x80])

#: first_mb_in_slice = 0 as one Exp-Golomb bit, so every slice here starts a
#: picture. 0x80 is `1` followed by padding, which reads as ue() == 0.
FIRST_MB_ZERO = bytes([0x80] + [0] * 24)


def stream(kinds: str) -> bytes:
    """An Annex-B stream from a shorthand: `i` for an IDR, `p` for anything else.

    Parameter sets lead, as a real stream's do, and are not repeated: a piece
    cut from the middle has to be given them, which is the point.
    """
    out = bytearray(START + SPS + START + PPS)
    for kind in kinds:
        header = 0x65 if kind == "i" else 0x41
        out += START + bytes([header]) + FIRST_MB_ZERO
    return bytes(out)


def write(tmp_path: Path, kinds: str, name: str = "clip.h264") -> Path:
    path = tmp_path / name
    path.write_bytes(stream(kinds))
    return path


# ── reading the stream ──


def test_pictures_and_parameter_sets_are_found():
    pictures, header = segments.scan_pictures(stream("ippipp"))
    assert len(pictures) == 6
    assert [p.idr for p in pictures] == [True, False, False, True, False, False]
    assert header == START + SPS + START + PPS


def test_the_parameter_sets_are_collected_once_however_often_they_repeat():
    """config-interval=1 streams carry them before every keyframe."""
    doubled = START + SPS + START + PPS + stream("ip")
    _, header = segments.scan_pictures(doubled)
    assert header == START + SPS + START + PPS


# ── planning ──


def test_cuts_land_only_on_keyframes():
    pictures, _ = segments.scan_pictures(stream("i" + "p" * 9 + "i" + "p" * 9))
    cuts = segments.plan_cuts(pictures, 5)
    assert cuts == [0, 10], "10 is the only other IDR, so it is the only cut"
    assert all(pictures[i].idr for i in cuts)


def test_the_last_keyframe_that_still_fits_wins():
    """Cutting at the first one past the limit would make pieces too small."""
    pictures, _ = segments.scan_pictures(stream(("i" + "p" * 4) * 6))
    cuts = segments.plan_cuts(pictures, 10)
    assert cuts == [0, 10, 20]


def test_a_clip_with_no_second_keyframe_cannot_be_cut():
    pictures, _ = segments.scan_pictures(stream("i" + "p" * 20))
    assert segments.plan_cuts(pictures, 5) == [0]


# ── writing the pieces ──


def test_every_frame_survives_the_cut(tmp_path):
    source = write(tmp_path, ("i" + "p" * 4) * 8)          # 40 frames, IDR every 5
    pieces = segments.split(source, tmp_path / "parts", 10)

    assert len(pieces) == 4
    assert [n for _, n in pieces] == [10, 10, 10, 10]
    assert sum(count_h264_pictures(str(p)) for p, _ in pieces) == 40


def test_each_piece_starts_at_a_keyframe_and_carries_the_parameter_sets(tmp_path):
    """A decoder handed a piece without them negotiates nothing at all."""
    source = write(tmp_path, ("i" + "p" * 4) * 8)
    pieces = segments.split(source, tmp_path / "parts", 10)

    for path, _ in pieces[1:]:
        body = path.read_bytes()
        assert body.startswith(START + SPS + START + PPS)
        after = body[len(START + SPS + START + PPS):]
        assert after.startswith(START + bytes([0x65])), "a piece must open on an IDR"


def test_a_clip_short_enough_already_is_handed_back_untouched(tmp_path):
    source = write(tmp_path, "i" + "p" * 20)
    assert segments.split(source, tmp_path / "parts", 150) == [(source, 21)]
    assert not (tmp_path / "parts").exists(), "nothing to write, nothing written"


def test_a_clip_whose_keyframes_are_too_far_apart_is_not_carved_up(tmp_path):
    """The mall clip: 379 frames with keyframes only at 0 and 250.

    Cutting it into 150s gives a first piece of 250, which stalls exactly as
    the whole clip did. All the cut would achieve is to make the failure
    harder to read, so it is refused and the caller says why.
    """
    source = write(tmp_path, "i" + "p" * 249 + "i" + "p" * 128)
    pieces = segments.split(source, tmp_path / "parts", 150)

    assert len(pieces) == 1 and pieces[0][0] == source
    assert segments.describe(pieces, 379) == ""


def test_segmenting_can_be_turned_off(tmp_path):
    source = write(tmp_path, ("i" + "p" * 4) * 8)
    assert segments.split(source, tmp_path / "parts", 0) == [(source, 40)]


def test_the_startup_line_names_the_pieces(tmp_path):
    source = write(tmp_path, ("i" + "p" * 4) * 8)
    pieces = segments.split(source, tmp_path / "parts", 10)
    told = segments.describe(pieces, 40)
    assert told == "decoding in 4 pieces of 10, 10, 10, 10 frames (40 total)"
