"""H.264 Annex-B inspection.

These are the functions that make "the clip ended" distinguishable from "the
source stalled", so they are worth pinning down. All of them read bytes and
need no board.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sima_vision.media import (
    BitReader,
    count_h264_pictures,
    count_pictures_in,
    fps_from_rate,
    is_elementary_h264,
    parse_sps,
    probe_h264_sps,
    unescape_rbsp,
)
from sima_vision.tasks import TASKS

# Real SPS NAL payloads, lifted byte for byte out of the two DevKit sample
# clips. Both are 1920x1080 High profile with VUI timing, and the 24 fps one
# carries an emulation-prevention byte (`00 00 03 00 20`), so parsing it end to
# end exercises unescape_rbsp against something a decoder actually accepts
# rather than against a fixture built by the same code that reads it.
SPS_1080P_24 = bytes.fromhex(
    "640028acd940780227e59a808080a0000003002000000601e30632c000"
)
SPS_1080P_30 = bytes.fromhex(
    "640028acd940780227e59a808080a000007d20001d4c01e30632c000"
)
REAL_SPS = [(SPS_1080P_24, 24), (SPS_1080P_30, 30)]

#: The same footage re-encoded with `-refs 1 -bf 0`, which is the advice the
#: decoder-budget warning gives. Real ffmpeg output, so the pair of streams
#: differ in exactly the field the warning turns on.
SPS_1080P_ONE_REF = bytes.fromhex(
    "4d4028da01e0089f97016a020202800000030080000019078c1950"
)


def test_bitreader_unsigned():
    r = BitReader(b"\xa0")           # 1010 0000
    assert r.u(1) == 1
    assert r.u(1) == 0
    assert r.u(2) == 0b10


def test_bitreader_exp_golomb():
    # 1 -> 0, 010 -> 1, 011 -> 2
    assert BitReader(b"\x80").ue() == 0
    assert BitReader(b"\x40").ue() == 1
    assert BitReader(b"\x60").ue() == 2


def test_bitreader_signed_exp_golomb():
    assert BitReader(b"\x80").se() == 0
    assert BitReader(b"\x40").se() == 1
    assert BitReader(b"\x60").se() == -1


def test_bitreader_rejects_a_truncated_sps():
    r = BitReader(b"\x00")
    try:
        r.u(64)
    except ValueError as exc:
        assert "truncated" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")


def test_unescape_strips_emulation_prevention():
    assert unescape_rbsp(bytes.fromhex("000003010203")) == bytes.fromhex("0000010203")
    # Only after two zeros, and only for 0x03.
    assert unescape_rbsp(bytes.fromhex("000103")) == bytes.fromhex("000103")


def test_unescape_leaves_clean_data_alone():
    payload = bytes(range(1, 32))
    assert unescape_rbsp(payload) == payload


@pytest.mark.parametrize("payload,expected_fps", REAL_SPS, ids=["24fps", "30fps"])
def test_parse_sps_reads_geometry_and_rate(payload, expected_fps):
    """The bytes on disk are authoritative; this is what reads them."""
    assert parse_sps(unescape_rbsp(payload)) == (1920, 1080, expected_fps)


def test_fps_from_rate():
    assert fps_from_rate("25/1") == 25
    assert fps_from_rate("30000/1001") == 30
    assert fps_from_rate("0/0") == 0
    assert fps_from_rate("") == 0
    assert fps_from_rate("nonsense") == 0


def test_is_elementary_h264():
    assert is_elementary_h264("clip.h264")
    assert is_elementary_h264("CLIP.264")
    assert not is_elementary_h264("clip.mp4")
    assert not is_elementary_h264("clip")


def _annexb(*nals: bytes) -> bytes:
    return b"".join(b"\x00\x00\x00\x01" + n for n in nals)


def _slice_nal(first_mb_zero: bool) -> bytes:
    """One IDR slice NAL. first_mb_in_slice = 0 starts a new picture."""
    # nal_unit_type 5 (IDR), then first_mb_in_slice as Exp-Golomb: 0 -> 0b1,
    # 1 -> 0b010. Padding after that is never read for the count.
    header = b"\x65"
    return header + (b"\x80" + b"\x00" * 30 if first_mb_zero else b"\x40" + b"\x00" * 30)


def test_count_pictures_counts_first_mb_zero_only():
    stream = _annexb(_slice_nal(True), _slice_nal(False), _slice_nal(True))
    count, _ = count_pictures_in(stream, final=True)
    assert count == 2


def test_count_pictures_ignores_non_slice_nals():
    stream = _annexb(b"\x67" + b"\x00" * 20, _slice_nal(True))   # SPS then a slice
    count, _ = count_pictures_in(stream, final=True)
    assert count == 1


def test_count_pictures_defers_a_split_start_code():
    """A buffer ending mid-start-code must hand those bytes to the next chunk."""
    stream = _annexb(_slice_nal(True)) + b"\x00\x00"
    count, consumed = count_pictures_in(stream, final=False)
    assert count == 1
    assert consumed <= len(stream) - 2


def test_count_h264_pictures_on_a_file(tmp_path):
    path = tmp_path / "clip.h264"
    path.write_bytes(_annexb(*[_slice_nal(True) for _ in range(7)]))
    assert count_h264_pictures(str(path)) == 7


def test_count_h264_pictures_on_a_missing_file(tmp_path):
    assert count_h264_pictures(str(tmp_path / "nope.h264")) == 0


def test_probe_sps_on_a_file(tmp_path):
    path = tmp_path / "clip.h264"
    path.write_bytes(_annexb(b"\x67" + SPS_1080P_24, _slice_nal(True)))
    assert probe_h264_sps(str(path)) == (1920, 1080, 24)


def test_probe_sps_returns_zeros_without_an_sps(tmp_path):
    path = tmp_path / "clip.h264"
    path.write_bytes(_annexb(_slice_nal(True)))
    assert probe_h264_sps(str(path)) == (0, 0, 0)


# -- refusing a Neat build that cannot do the job --


def test_an_old_neat_build_is_refused_while_probing(tmp_path, monkeypatch):
    """A DevKit paired with an older SDK has no `SimaDecodeOptions`.

    It surfaced as `AttributeError: module 'pyneat' has no attribute
    'SimaDecodeOptions'` from inside graph construction -- after a model load
    that takes the better part of a minute, and naming a symbol rather than the
    problem. It has to be caught while probing the source, which is the last
    cheap moment before that load.
    """
    from sima_vision import bootstrap, media, runtime

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67]))
    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )

    old = type(runtime)("pyneat")
    old.__version__ = "0.2.2"
    monkeypatch.setattr(runtime, "pyneat", old)

    with pytest.raises(RuntimeError) as caught:
        media.check_source_support(cfg)
    message = str(caught.value)
    assert "0.2.2" in message, "say which build is installed"
    assert "pyneat.SimaDecodeOptions" in message, "and exactly what it is missing"
    # The fix runs on the board. Sending someone to their PC to re-pair, which
    # is what this said first, is a much longer way round for a core that one
    # command installs in place.
    assert bootstrap.NEAT_INSTALL in message, "and the command that fixes it"
    assert bootstrap.NEAT_VERSION in message, "and which version it wants"


def test_a_capable_build_passes(tmp_path, monkeypatch):
    from sima_vision import media, runtime

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67]))
    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )

    current = type(runtime)("pyneat")
    current.SimaDecodeOptions = current.SimaDecodeType = object
    monkeypatch.setattr(runtime, "pyneat", current)
    media.check_source_support(cfg)          # must not raise


def test_the_check_is_skipped_when_nothing_is_bound(tmp_path, monkeypatch):
    """--validate never binds pyneat, and must not be made to."""
    from sima_vision import media, runtime

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67]))
    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )
    monkeypatch.setattr(runtime, "pyneat", None)
    media.check_source_support(cfg)


def test_every_source_kind_has_requirements_listed():
    """A source path with no entry would silently skip the check entirely."""
    from sima_vision.media import SOURCE_REQUIREMENTS

    assert set(SOURCE_REQUIREMENTS) == {"h264", "container", "rtsp", "usb"}


# ── containers, reframed rather than refused ──


def detect_cfg(**settings):
    return TASKS["detect"]().load(
        None, {"model.path": "m.tar.gz", **settings}, use_file=False
    )


def write_mp4(path, frames=None):
    """A real little MP4 on disk, built by the mp4 tests' own fixture."""
    from test_mp4 import build_mp4

    frames = frames or [[bytes([0x65]) + b"idr"], [bytes([0x41]) + b"inter"]]
    path.write_bytes(build_mp4(frames))
    return path


def test_an_mp4_source_is_reframed_and_the_config_points_at_the_result(tmp_path):
    """Neat 0.3.0 cannot demux, so the app stopped and asked for ffmpeg.

    A DevKit has no ffmpeg, which made "use your own footage" mean "go and find
    another machine first". The container holds the same H.264 the raw path
    already runs, so it is reframed here instead.
    """
    from sima_vision.media import ensure_annex_b

    clip = write_mp4(tmp_path / "clip.mp4")
    cfg = ensure_annex_b(detect_cfg(**{"source.uri": str(clip)}))

    out = tmp_path / "clip-annexb.h264"
    assert cfg.source_uri == str(out)
    assert out.read_bytes().startswith(b"\x00\x00\x00\x01")
    # And the result is something the raw path will take.
    from sima_vision.media import is_elementary_h264
    assert is_elementary_h264(cfg.source_uri)


def test_a_raw_stream_is_left_exactly_as_it_was(tmp_path):
    from sima_vision.media import ensure_annex_b

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67, 0x42]))
    cfg = detect_cfg(**{"source.uri": str(clip)})
    assert ensure_annex_b(cfg) is cfg
    assert not (tmp_path / "clip-annexb.h264").exists()


def test_an_mp4_renamed_to_h264_is_reframed_instead_of_rejected(tmp_path):
    """This used to be a hard error telling you to go and run ffmpeg.

    It is the same bytes as any other MP4, so the suffix is not worth failing
    over. Detected on content, which is how the old error spotted it too.
    """
    from sima_vision.media import ensure_annex_b

    clip = write_mp4(tmp_path / "clip.h264")
    cfg = ensure_annex_b(detect_cfg(**{"source.uri": str(clip)}))

    assert cfg.source_uri == str(tmp_path / "clip-annexb.h264")
    assert Path(cfg.source_uri).read_bytes().startswith(b"\x00\x00\x00\x01")


def test_the_reframed_copy_is_reused_rather_than_rebuilt(tmp_path):
    """One pass over the file, on the first run only."""
    from sima_vision.media import ensure_annex_b

    clip = write_mp4(tmp_path / "clip.mp4")
    first = ensure_annex_b(detect_cfg(**{"source.uri": str(clip)}))
    out = Path(first.source_uri)
    out.write_bytes(b"\x00\x00\x00\x01sentinel")   # would be overwritten if rebuilt

    second = ensure_annex_b(detect_cfg(**{"source.uri": str(clip)}))
    assert second.source_uri == first.source_uri
    assert out.read_bytes() == b"\x00\x00\x00\x01sentinel"


def test_a_stale_copy_is_rebuilt_when_the_source_moves_on(tmp_path):
    from sima_vision.media import ensure_annex_b

    clip = write_mp4(tmp_path / "clip.mp4")
    out = tmp_path / "clip-annexb.h264"
    out.write_bytes(b"old")
    import os
    os.utime(out, (0, 0))                          # older than the source

    ensure_annex_b(detect_cfg(**{"source.uri": str(clip)}))
    assert out.read_bytes().startswith(b"\x00\x00\x00\x01")


def test_a_camera_is_never_mistaken_for_a_file(tmp_path):
    from sima_vision.media import ensure_annex_b

    cfg = detect_cfg(**{"source.type": "usb", "source.uri": "/dev/video0"})
    assert ensure_annex_b(cfg) is cfg


# ── the decoder's buffer budget, said before the run rather than after ──


def test_the_dpb_size_comes_from_the_level_and_the_frame_size():
    """Table A-1 arithmetic. 1080p at level 4.0 is the case that bites."""
    from sima_vision.media import dpb_frames

    assert dpb_frames(40, 1920, 1080) == 4       # 32768 / 8160 macroblocks
    assert dpb_frames(31, 1920, 1080) == 2       # a shallower level fits easily
    assert dpb_frames(40, 640, 480) == 16        # capped at 16, not unbounded
    assert dpb_frames(40, 0, 0) == 0             # nothing to divide by


def test_the_real_sps_says_four_reference_frames_at_level_four():
    """Both DevKit clips, and both stalled part-way through a run."""
    from sima_vision.media import parse_sps_dpb, unescape_rbsp

    for payload, _ in REAL_SPS:
        level, refs = parse_sps_dpb(unescape_rbsp(payload))
        assert (level, refs) == (40, 4)


def annexb(tmp_path, sps: bytes, name="clip.h264"):
    path = tmp_path / name
    path.write_bytes(b"\x00\x00\x00\x01\x67" + sps + b"\x00\x00\x00\x01\x65ab")
    return path


def test_a_stream_whose_own_references_do_not_fit_is_named_before_the_run(tmp_path):
    """This is the stall, stated as arithmetic instead of as a timeout.

    The pool is 8 and the appsink pyneat generates declares 4, leaving 4. Both
    DevKit clips declare max_num_ref_frames=4, so they need 5. Five into four
    does not go, the decoder starves part-way through, and the run reports a
    pull timeout that reads like a bug in the app or a damaged file. Neither.
    """
    from sima_vision.media import decoder_budget_warning

    warning = decoder_budget_warning(str(annexb(tmp_path, SPS_1080P_24)), 1920, 1080)

    assert "max_num_ref_frames=4, so it needs 5" in warning
    assert "does not fit" in warning
    assert "not your\n  file" in warning
    assert "-bf 0 -refs 1" in warning, "the way out has to be in the message"


def test_a_shallow_stream_is_flagged_only_as_decoder_dependent(tmp_path):
    """The bug this pair exists to prevent.

    Sizing from the level alone reported a 4 frame DPB for a stream re-encoded
    down to a single reference frame, which would have sent someone off to
    re-encode a file already as shallow as it goes. What a stream needs and
    what its level permits are different numbers, and only the first is the
    stream's own doing.
    """
    from sima_vision.media import decoder_budget_warning

    warning = decoder_budget_warning(
        str(annexb(tmp_path, SPS_1080P_ONE_REF)), 1920, 1080
    )

    assert "needs only 2 (max_num_ref_frames=1), which fits" in warning
    assert "pool from the level" in warning
    assert "does not fit" not in warning


def test_a_source_with_no_readable_sps_says_nothing(tmp_path):
    """Silence beats a guess: not every source is a raw stream on disk."""
    from sima_vision.media import decoder_budget_warning

    path = tmp_path / "empty.h264"
    path.write_bytes(b"\x00\x00\x00\x01\x41no-sps-here")
    assert decoder_budget_warning(str(path), 1920, 1080) == ""
    assert decoder_budget_warning(str(tmp_path / "missing.h264"), 1920, 1080) == ""


# ── asking the decoder for what the stream needs ──


def test_the_pool_is_sized_from_the_streams_own_reference_frames(tmp_path):
    """pyneat leaves num_buffers at -1 and the daemon picks 8 for 1080p.

    Eight is not a property of the stream. A clip keeping four reference
    frames needs five of those before the source appsink's four are counted,
    so the pool is oversubscribed from the first frame and the run stops
    part-way through. The SPS says how many it keeps, and it has been read by
    the time the graph is built, so the number can simply be asked for.
    """
    from sima_vision.media import decoder_buffers_for

    clip = annexb(tmp_path, SPS_1080P_24)                 # max_num_ref_frames=4
    cfg = detect_cfg(**{"source.uri": str(clip)})
    # 4 references + 1 decoding + 4 appsink + 2 slack.
    assert decoder_buffers_for(cfg, 1920, 1080) == 11


def test_the_pool_is_sized_for_the_worse_of_the_two_readings(tmp_path):
    """Sizing from the stream alone got a re-encoded clip exactly backwards.

    A file re-encoded down to max_num_ref_frames=1 asked for 8 -- the number
    pyneat would have picked unaided -- while the very same run warned that a
    decoder sizing its pool from the level would want 5 of those 8 and starve.
    The run advised against itself.

    Which kind of decoder this is cannot be settled from here, so the larger
    reading wins. It costs a few megabytes and removes the question.
    """
    from sima_vision.media import decoder_buffers_for, dpb_frames

    clip = annexb(tmp_path, SPS_1080P_ONE_REF)
    cfg = detect_cfg(**{"source.uri": str(clip)})

    assert dpb_frames(40, 1920, 1080) == 4, "the level permits four, the stream keeps one"
    # 4 permitted + 1 decoding + 4 appsink + 2 slack, not 1 + 1 + 4 + 2.
    assert decoder_buffers_for(cfg, 1920, 1080) == 11


def test_the_sizing_never_drops_below_what_the_daemon_would_pick(tmp_path):
    """Asking for less than the default would make a working run worse."""
    from sima_vision.media import decoder_buffers_for

    clip = annexb(tmp_path, SPS_1080P_ONE_REF)
    cfg = detect_cfg(**{"source.uri": str(clip), "runtime.decoder_pool": 40})
    assert decoder_buffers_for(cfg, 1920, 1080) == 40


def test_an_explicit_count_wins_and_a_negative_one_stands_aside(tmp_path):
    """Negative is the way back to what the app did before, byte for byte."""
    from sima_vision.media import decoder_buffers_for

    clip = str(annexb(tmp_path, SPS_1080P_24))
    pinned = detect_cfg(**{"source.uri": clip, "runtime.decoder_buffers": 20})
    assert decoder_buffers_for(pinned, 1920, 1080) == 20

    off = detect_cfg(**{"source.uri": clip, "runtime.decoder_buffers": -1})
    assert decoder_buffers_for(off, 1920, 1080) == 0, "0 leaves num_buffers unset"


def test_a_source_that_is_not_a_raw_stream_is_left_to_pyneat(tmp_path):
    """No SPS to read means no better guess than the daemon's own."""
    from sima_vision.media import decoder_buffers_for

    camera = detect_cfg(**{"source.type": "usb", "source.uri": "/dev/video0"})
    assert decoder_buffers_for(camera, 1920, 1080) == 0

    unreadable = tmp_path / "empty.h264"
    unreadable.write_bytes(b"\x00\x00\x00\x01\x41no-sps")
    cfg = detect_cfg(**{"source.uri": str(unreadable)})
    assert decoder_buffers_for(cfg, 1920, 1080) == 0


def test_the_sized_pool_silences_the_warning_it_was_added_for(tmp_path):
    """The two halves have to agree, or the run advises against itself."""
    from sima_vision.media import decoder_budget_warning, decoder_buffers_for

    clip = annexb(tmp_path, SPS_1080P_24)
    cfg = detect_cfg(**{"source.uri": str(clip)})
    asked = decoder_buffers_for(cfg, 1920, 1080)

    assert decoder_budget_warning(str(clip), 1920, 1080, cfg.decoder_pool) != ""
    assert decoder_budget_warning(str(clip), 1920, 1080, asked) == ""
