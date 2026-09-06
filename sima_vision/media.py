"""Source geometry, H.264 inspection and the source head of the Neat graph.

``ffprobe`` and OpenCV both routinely fail on raw Annex-B elementary streams,
which is what pushes people into hand-writing ``source.width`` and
``source.height`` into a config. A wrong value there is silent and fatal: the
caps filter stops negotiating and the run reports zero frames after a pull
timeout. The bytes on disk are authoritative, so this module reads them.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from . import mp4, runtime
from .bootstrap import NEAT_INSTALL, NEAT_VERSION
from .console import console, human_bytes

# ─────────────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────────────


def fps_from_rate(value: str) -> int:
    if not value or value in {"0/0", "0/1"}:
        return 0
    try:
        fps = float(Fraction(value)) if "/" in value else float(value)
    except (ValueError, ZeroDivisionError):
        return 0
    return int(round(fps)) if fps > 0 else 0


def probe_ffprobe(uri: str) -> tuple[int, int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-rw_timeout", "5000000",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of", "default=nw=1", uri,
    ]
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0, 0, 0
    if result.returncode != 0:
        return 0, 0, 0
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )

    def as_int(v: str | None) -> int:
        try:
            return int(v or 0)
        except ValueError:
            return 0

    return as_int(values.get("width")), as_int(values.get("height")), fps


def probe_opencv(uri: str) -> tuple[int, int, int]:
    """Best-effort probe. Returns zeros rather than raising.

    Raw H.264 elementary streams frequently cannot be opened by OpenCV, and that is
    not a fatal condition: the caller falls back to the configured geometry and
    produces a clearer message than "failed to open source".
    """
    cv2 = runtime.cv2
    try:
        cap = cv2.VideoCapture(uri)
    except Exception:
        return 0, 0, 0
    if not cap.isOpened():
        cap.release()
        return 0, 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    return width, height, fps


# ─────────────────────────────────────────────────────────────────────────────
# H.264 Annex-B parsing
# ─────────────────────────────────────────────────────────────────────────────


class BitReader:
    """Minimal MSB-first bit reader with Exp-Golomb support, for SPS parsing."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def u(self, n: int) -> int:
        value = 0
        for _ in range(n):
            byte = self.pos >> 3
            if byte >= len(self.data):
                raise ValueError("SPS truncated")
            bit = (self.data[byte] >> (7 - (self.pos & 7))) & 1
            value = (value << 1) | bit
            self.pos += 1
        return value

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 32:
                raise ValueError("SPS Exp-Golomb overrun")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)


def unescape_rbsp(data: bytes) -> bytes:
    """Strip H.264 emulation prevention bytes (00 00 03 -> 00 00)."""
    out = bytearray()
    zeros = 0
    for byte in data:
        if zeros >= 2 and byte == 0x03:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0x00 else 0
    return bytes(out)


# Profiles whose SPS carries the chroma_format_idc block.
HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}


def skip_scaling_list(reader: BitReader, size: int) -> None:
    last = next_scale = 8
    for _ in range(size):
        if next_scale:
            next_scale = (last + reader.se() + 256) % 256
        last = last if next_scale == 0 else next_scale


def parse_sps(rbsp: bytes) -> tuple[int, int, int]:
    """Decode width, height and frame rate from one SPS payload.

    Args:
        rbsp: SPS NAL payload with the header byte removed and emulation
            prevention bytes already stripped.

    Returns:
        A ``(width, height, fps)`` triple. ``fps`` is 0 when the stream carries
        no VUI timing information, which is common for camera captures and for
        anything remuxed with ``-c:v copy``.
    """
    r = BitReader(rbsp)
    profile_idc = r.u(8)
    r.u(8)  # constraint flags and reserved bits
    r.u(8)  # level_idc
    r.ue()  # seq_parameter_set_id

    chroma_format_idc = 1
    separate_colour_plane = 0
    if profile_idc in HIGH_PROFILES:
        chroma_format_idc = r.ue()
        if chroma_format_idc == 3:
            separate_colour_plane = r.u(1)
        r.ue()  # bit_depth_luma_minus8
        r.ue()  # bit_depth_chroma_minus8
        r.u(1)  # qpprime_y_zero_transform_bypass_flag
        if r.u(1):  # seq_scaling_matrix_present_flag
            for i in range(8 if chroma_format_idc != 3 else 12):
                if r.u(1):
                    skip_scaling_list(r, 16 if i < 6 else 64)

    r.ue()  # log2_max_frame_num_minus4
    pic_order_cnt_type = r.ue()
    if pic_order_cnt_type == 0:
        r.ue()  # log2_max_pic_order_cnt_lsb_minus4
    elif pic_order_cnt_type == 1:
        r.u(1)  # delta_pic_order_always_zero_flag
        r.se()  # offset_for_non_ref_pic
        r.se()  # offset_for_top_to_bottom_field
        for _ in range(r.ue()):
            r.se()  # offset_for_ref_frame[i]

    r.ue()  # max_num_ref_frames
    r.u(1)  # gaps_in_frame_num_value_allowed_flag
    width_mbs = r.ue() + 1
    height_map_units = r.ue() + 1
    frame_mbs_only = r.u(1)
    if not frame_mbs_only:
        r.u(1)  # mb_adaptive_frame_field_flag
    r.u(1)  # direct_8x8_inference_flag

    crop_left = crop_right = crop_top = crop_bottom = 0
    if r.u(1):  # frame_cropping_flag
        crop_left, crop_right = r.ue(), r.ue()
        crop_top, crop_bottom = r.ue(), r.ue()

    width = width_mbs * 16
    height = (2 - frame_mbs_only) * height_map_units * 16
    if chroma_format_idc == 0 or separate_colour_plane:
        unit_x, unit_y = 1, 2 - frame_mbs_only
    else:
        sub_w, sub_h = {1: (2, 2), 2: (2, 1), 3: (1, 1)}.get(chroma_format_idc, (2, 2))
        unit_x, unit_y = sub_w, sub_h * (2 - frame_mbs_only)
    width -= unit_x * (crop_left + crop_right)
    height -= unit_y * (crop_top + crop_bottom)

    fps = 0
    if r.u(1):  # vui_parameters_present_flag
        try:
            fps = parse_vui_fps(r)
        except ValueError:
            fps = 0
    return width, height, fps


#: Decoded frames the hardware decoder's pool holds, as the boot log reports it
#: (``BufferNum=8``). Every one is 1920x1080 NV12 on this board.
DECODER_POOL = 8

#: What the source appsink alone declares in the pipeline pyneat generates:
#: ``appsink ... max-buffers=4 drop=false``. Not settable from here.
SOURCE_APPSINK_BUFFERS = 4

#: Table A-1 of the H.264 spec: MaxDpbMbs per level, which is what bounds how
#: many decoded frames a conforming decoder has to keep.
MAX_DPB_MBS = {
    10: 396, 11: 900, 12: 2376, 13: 2376, 20: 2376, 21: 4752, 22: 8100,
    30: 8100, 31: 18000, 32: 20480, 40: 32768, 41: 32768, 42: 34816,
    50: 110400, 51: 184320, 52: 184320, 60: 696320, 61: 696320, 62: 696320,
}


def dpb_frames(level_idc: int, width: int, height: int) -> int:
    """How many frames this stream's decoded picture buffer has to hold.

    Reference frames and frames waiting to be output in presentation order
    both live in the DPB, so a stream with B-frames keeps several pictures
    alive at once. Those are frames out of the same small pool the app is
    trying to pull from, which is what makes the number worth knowing before
    the run rather than after it stalls.
    """
    macroblocks = ((width + 15) // 16) * ((height + 15) // 16)
    if macroblocks <= 0:
        return 0
    return min(MAX_DPB_MBS.get(level_idc, 32768) // macroblocks, 16)


def probe_h264_dpb(path: str, scan_bytes: int = 4 << 20) -> tuple[int, int]:
    """``(level_idc, max_num_ref_frames)`` from the first SPS, or ``(0, 0)``."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(scan_bytes)
    except OSError:
        return 0, 0

    pos = 0
    while True:
        idx = head.find(b"\x00\x00\x01", pos)
        if idx < 0:
            return 0, 0
        start = idx + 3
        if start >= len(head):
            return 0, 0
        if head[start] & 0x1F == 7:
            end = head.find(b"\x00\x00\x01", start)
            try:
                return parse_sps_dpb(unescape_rbsp(head[start + 1:end if end > 0 else len(head)]))
            except (ValueError, IndexError):
                return 0, 0
        pos = start


def parse_sps_dpb(rbsp: bytes) -> tuple[int, int]:
    """Read only as far as ``max_num_ref_frames``, which is all this needs."""
    r = BitReader(rbsp)
    profile_idc = r.u(8)
    r.u(8)
    level_idc = r.u(8)
    r.ue()

    if profile_idc in HIGH_PROFILES:
        chroma_format_idc = r.ue()
        if chroma_format_idc == 3:
            r.u(1)
        r.ue()
        r.ue()
        r.u(1)
        if r.u(1):
            for i in range(8 if chroma_format_idc != 3 else 12):
                if r.u(1):
                    skip_scaling_list(r, 16 if i < 6 else 64)

    r.ue()
    pic_order_cnt_type = r.ue()
    if pic_order_cnt_type == 0:
        r.ue()
    elif pic_order_cnt_type == 1:
        r.u(1)
        r.se()
        r.se()
        for _ in range(r.ue()):
            r.se()
    return level_idc, r.ue()


#: Spare frames left over the strict requirement when sizing the pool. Two,
#: because the strict sum has no room for a consumer that pauses even briefly,
#: and buffers are cheap next to a run that stops half way.
DECODER_SLACK = 2


def decoder_buffers_for(cfg, width: int, height: int) -> int:
    """How many buffers to ask the decoder for, or 0 to leave pyneat alone.

    ``SimaDecodeOptions.num_buffers`` defaults to -1, which lets the daemon
    pick, and what it picks is 8 for 1080p. The stream does not get a say, and
    it should: a clip keeping five reference frames needs six of those eight
    before the source appsink's four are counted, so the pool is oversubscribed
    from the first frame and the run dies part-way through.

    The SPS says how many reference frames the stream keeps, and it has already
    been read by the time the graph is built, so the number can simply be
    asked for.

    Args:
        cfg: Application configuration, for ``decoder_buffers`` and the source.
        width: Source frame width.
        height: Source frame height.

    Returns:
        A buffer count, or 0 to leave ``num_buffers`` unset.
    """
    if cfg.decoder_buffers > 0:
        return cfg.decoder_buffers
    if cfg.decoder_buffers < 0:                # explicitly "leave pyneat alone"
        return 0
    if not is_elementary_h264(cfg.source_uri):
        return 0
    level_idc, refs = probe_h264_dpb(cfg.source_uri)
    if not level_idc:
        return 0
    # The worst of the two readings, not the stream's own. A decoder that sizes
    # its pool from the level takes what the level permits whether the stream
    # uses it or not, and which kind this one is cannot be settled from here.
    #
    # Sizing from the stream alone got this exactly backwards on a re-encoded
    # clip: max_num_ref_frames=1 asked for 8, the number pyneat would have
    # picked anyway, while the very same run warned that a level-sized decoder
    # would want 5 and starve. Asking for the larger number costs a few
    # megabytes and settles it.
    held = max(refs, dpb_frames(level_idc, width, height)) + 1
    needed = held + SOURCE_APPSINK_BUFFERS + DECODER_SLACK
    return max(needed, cfg.decoder_pool)


def decoder_budget_warning(path: str, width: int, height: int,
                           pool: int = DECODER_POOL) -> str:
    """Warn when the stream needs more of the pool than the pipeline leaves it.

    The board's decoder pool is eight frames and it is shared. The stream's DPB
    takes its share first -- five frames for High profile 1080p with B-frames,
    which is what a phone or an editor produces by default -- and the source
    appsink pyneat generates declares four more. Nine into eight does not go,
    so the decoder is starved before the app has done anything wrong, and the
    run dies part-way through with the pull timeout that looks like a stall.

    Two numbers, because two kinds of decoder read the same stream
    differently, and they disagree about exactly the streams worth re-encoding:

    * ``max_num_ref_frames + 1`` is what the stream itself says it needs -- the
      references it will reach back for, plus the picture being decoded.
    * The level's DPB is the most a *conforming* decoder may hold at this frame
      size, and a hardware decoder that sizes its pool from the level alone
      takes that whether the stream uses it or not.

    Getting this wrong is not academic. Sizing by level alone reported a 4
    frame DPB for a stream re-encoded down to a single reference frame, which
    would have sent someone off to re-encode a file that was already as
    shallow as it goes.

    Returns:
        The warning, or "" when the stream fits either way.
    """
    level_idc, refs = probe_h264_dpb(path)
    if not level_idc:
        return ""
    needed = refs + 1
    capacity = dpb_frames(level_idc, width, height) + 1
    spare = pool - SOURCE_APPSINK_BUFFERS
    head = (
        f"the decoder's pool is {pool} frames and the source appsink "
        f"pyneat generates\n  declares max-buffers="
        f"{SOURCE_APPSINK_BUFFERS} of them, leaving {spare} for the decoder itself."
    )
    ask = needed + SOURCE_APPSINK_BUFFERS + DECODER_SLACK
    reencode = (
        "  Ask the decoder for more, which is what --decoder-buffers does:\n"
        f"    sima-vision detect --source clip.mp4 --decoder-buffers {ask}\n"
        "  That is now the default, so this warning means it was turned off.\n"
        "  Failing that, re-encode with fewer references:\n"
        "    ffmpeg -i clip.mp4 -c:v libx264 -profile:v main -bf 0 -refs 1 \\\n"
        "      -g 50 -keyint_min 50 -sc_threshold 0 -c:a copy shallow.mp4\n"
        "  -bf 0 also puts the frames in presentation order, which is a separate\n"
        "  win: the recording is written in arrival order."
    )

    if needed > spare:
        return (
            f"{head}\n"
            f"  This stream declares max_num_ref_frames={refs}, so it needs "
            f"{needed}. That does not fit,\n"
            "  and the run will stop part-way through with a pull timeout. It is "
            "not your\n  file -- the pool is shared, and this stream's own "
            "buffering fills it.\n"
            f"{reencode}"
        )
    if capacity > spare:
        return (
            f"{head}\n"
            f"  This stream needs only {needed} (max_num_ref_frames={refs}), which "
            f"fits. But level\n  {level_idc / 10:.1f} at {width}x{height} permits a "
            f"DPB of {capacity - 1}, and a decoder that sizes its\n"
            f"  pool from the level rather than from the stream would take "
            f"{capacity} and starve.\n"
            "  If this run stops part-way through with a pull timeout, that is why.\n"
            f"{reencode}"
        )
    return ""


def parse_vui_fps(r: BitReader) -> int:
    """Read timing_info out of a VUI block. Returns 0 when it is absent."""
    if r.u(1):  # aspect_ratio_info_present_flag
        if r.u(8) == 255:  # Extended_SAR
            r.u(16)
            r.u(16)
    if r.u(1):  # overscan_info_present_flag
        r.u(1)
    if r.u(1):  # video_signal_type_present_flag
        r.u(3)
        r.u(1)
        if r.u(1):  # colour_description_present_flag
            r.u(24)
    if r.u(1):  # chroma_loc_info_present_flag
        r.ue()
        r.ue()
    if not r.u(1):  # timing_info_present_flag
        return 0
    num_units_in_tick = r.u(32)
    time_scale = r.u(32)
    if num_units_in_tick <= 0 or time_scale <= 0:
        return 0
    return int(round(time_scale / (2.0 * num_units_in_tick)))


def probe_h264_sps(path: str, scan_bytes: int = 4 << 20) -> tuple[int, int, int]:
    """Read geometry straight out of the first SPS in an Annex-B stream.

    Args:
        path: Path to a raw Annex-B H.264 file.
        scan_bytes: How much of the head of the file to search.

    Returns:
        A ``(width, height, fps)`` triple, or zeros if no SPS was found.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(scan_bytes)
    except OSError:
        return 0, 0, 0

    pos = 0
    while True:
        idx = head.find(b"\x00\x00\x01", pos)
        if idx < 0:
            return 0, 0, 0
        start = idx + 3
        if start >= len(head):
            return 0, 0, 0
        if head[start] & 0x1F == 7:  # nal_unit_type 7 == SPS
            end = head.find(b"\x00\x00\x01", start)
            payload = head[start + 1 : end if end > 0 else len(head)]
            try:
                return parse_sps(unescape_rbsp(payload))
            except (ValueError, IndexError):
                return 0, 0, 0
        pos = start


SLICE_NAL_TYPES = frozenset({1, 5})
SLICE_HEADER_BYTES = 24
PICTURE_SCAN_LIMIT = 512 << 20


def count_pictures_in(buf: bytes, final: bool) -> tuple[int, int]:
    """Count picture starts in one buffer.

    Args:
        buf: Annex-B bytes, starting on a start-code boundary or earlier.
        final: True when no more bytes follow, so a NAL near the end can be
            parsed from what is there rather than deferred.

    Returns:
        A ``(count, consumed)`` pair. ``consumed`` is how many leading bytes are
        finished with; the caller carries the remainder into the next chunk.
    """
    count = 0
    pos = 0
    while True:
        idx = buf.find(b"\x00\x00\x01", pos)
        if idx < 0:
            # Two trailing bytes could still be the head of a split start code.
            return count, len(buf) if final else max(0, len(buf) - 2)
        start = idx + 3
        if start + SLICE_HEADER_BYTES > len(buf):
            if not final:
                return count, idx
            if start >= len(buf):
                return count, len(buf)
        if buf[start] & 0x1F in SLICE_NAL_TYPES:
            rbsp = unescape_rbsp(buf[start + 1 : start + 1 + SLICE_HEADER_BYTES])
            try:
                if BitReader(rbsp).ue() == 0:
                    count += 1
            except (ValueError, IndexError):
                pass
        pos = start


def count_h264_pictures(path: str, limit_bytes: int = PICTURE_SCAN_LIMIT) -> int:
    """Count the coded pictures in a raw Annex-B stream.

    Without this number, "the clip ended" and "the source stalled" are the same
    event from the pull loop: both are silence. That is what let a run stop at 83
    frames of a 379 frame clip and still write a plausible-looking recording. The
    bytes on disk settle it before the run even starts, and they need neither a
    container nor a decoder to do it.

    ``ffprobe -count_frames`` would answer the same question, but it is not on
    the DevKit and it is unreliable on elementary streams, which is the same
    reason :func:`probe_h264_sps` exists.

    Args:
        path: Path to a raw Annex-B H.264 file.
        limit_bytes: Stop scanning after this many bytes, so a very large file
            cannot hold up startup. The result is then a lower bound.

    Returns:
        The number of coded pictures, or 0 if the file could not be read.
    """
    count = 0
    carry = b""
    scanned = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20) if scanned < limit_bytes else b""
                scanned += len(chunk)
                buf = carry + chunk
                if not buf:
                    break
                found, consumed = count_pictures_in(buf, final=not chunk)
                count += found
                if not chunk:
                    break
                carry = buf[consumed:]
    except OSError:
        return 0
    return count


ELEMENTARY_H264_SUFFIXES = {".h264", ".264", ".bin", ".avc"}


def is_elementary_h264(path: str) -> bool:
    return Path(path).suffix.lower() in ELEMENTARY_H264_SUFFIXES


def annex_b_path(source: Path) -> Path:
    """Where a remuxed container is cached: beside it, and obviously derived."""
    return source.with_name(f"{source.stem}-annexb.h264")


def needs_remux(path: Path) -> bool:
    """Whether this file is a container that has to be reframed first.

    Decided on content as well as on suffix. An MP4 renamed to ``.h264`` used
    to be a hard error telling you to go and run ffmpeg; it is the same bytes
    as any other MP4, so it can simply be remuxed like one.
    """
    if not path.is_file():
        return False                      # check_source_file has better words
    if mp4.is_container(str(path)):
        return True
    with path.open("rb") as handle:
        return mp4.looks_like_mp4(handle.read(12))


def ensure_annex_b(cfg, step=None):
    """Reframe a container source into a raw stream, and point cfg at it.

    Neat 0.3.0 cannot build a container source at all -- see
    :func:`make_elementary_h264_source` for the demuxer naming bug -- so the
    app used to stop and ask for ``ffmpeg``, on a board that does not have it.
    The container holds the same H.264 the raw path already runs, so reframing
    it here costs one pass over the file and no quality at all.

    The result is cached beside the source and reused while it is newer, so
    the cost lands on the first run only.

    Args:
        cfg: Application configuration.
        step: Console step to report on, or None to stay quiet.

    Returns:
        ``cfg``, or a copy of it pointing at the remuxed stream.
    """
    if cfg.source_type != "video" or not needs_remux(Path(cfg.source_uri)):
        return cfg

    source = Path(cfg.source_uri)
    out = annex_b_path(source)
    fresh = out.is_file() and out.stat().st_mtime >= source.stat().st_mtime
    if fresh:
        if step is not None:
            step.detail(f"have  {out.name}  (remuxed from {source.name} earlier)")
    else:
        frames = mp4.remux(source, out)
        if step is not None:
            step.detail(
                f"remuxed {source.name} -> {out.name}  "
                f"({frames} frames, {human_bytes(out.stat().st_size)})"
            )
    return replace(cfg, source_uri=str(out))


# ─────────────────────────────────────────────────────────────────────────────
# Source geometry
# ─────────────────────────────────────────────────────────────────────────────


def resolve_source_geometry(cfg) -> tuple[int, int, int]:
    """Return (width, height, fps). Config values win; anything left at 0 is probed."""
    width, height, fps = cfg.source_width, cfg.source_height, cfg.source_fps

    if cfg.source_type == "video" and is_elementary_h264(cfg.source_uri):
        sps_w, sps_h, sps_fps = probe_h264_sps(cfg.source_uri)
        if sps_w > 0 and sps_h > 0:
            if (width > 0 and width != sps_w) or (height > 0 and height != sps_h):
                console.warn(
                    f"config says {width}x{height} but the stream's SPS says "
                    f"{sps_w}x{sps_h}. Using the stream.\n"
                    f"Fix source.width and source.height in config.yaml, or set "
                    f"them to 0 to always read the stream."
                )
            width, height = sps_w, sps_h
        if sps_fps > 0:
            if fps > 0 and fps != sps_fps:
                console.warn(
                    f"config says {fps} fps but the stream is {sps_fps} fps. "
                    f"Using the stream.\n"
                    f"Set source.fps to 0 in config.yaml to always read the stream."
                )
            fps = sps_fps
        if fps <= 0:
            fps = 25
            console.warn(
                "the stream carries no frame rate, assuming 25. Set source.fps to override."
            )
        return width, height, fps

    if cfg.source_type == "usb":
        # libcamera is queried at build time, not probeable here, so fall back to
        # the CameraInputOptions defaults.
        return width or 1920, height or 1080, fps or 30

    if width <= 0 or height <= 0 or fps <= 0:
        probed_w, probed_h, probed_fps = probe_ffprobe(cfg.source_uri)
        width = width if width > 0 else probed_w
        height = height if height > 0 else probed_h
        fps = fps if fps > 0 else probed_fps

    if width <= 0 or height <= 0 or fps <= 0:
        cv_w, cv_h, cv_fps = probe_opencv(cfg.source_uri)
        width = width if width > 0 else cv_w
        height = height if height > 0 else cv_h
        fps = fps if fps > 0 else cv_fps

    if width <= 0 or height <= 0 or fps <= 0:
        # Elementary H.264 never reaches here: that path reads the SPS and
        # returns above.
        hint = (
            "\nNeither ffprobe nor OpenCV could read this source. Set the values "
            "explicitly:\n  source:\n    width: 1920\n    height: 1080\n    fps: 25"
        )
        missing = []
        if width <= 0 or height <= 0:
            missing.append("source.width and source.height")
        if fps <= 0:
            missing.append("source.fps")
        raise RuntimeError(f"could not resolve {', '.join(missing)}{hint}")
    return width, height, fps


def check_source_file(cfg) -> int:
    """Fail fast when a file source is missing, empty or a container.

    ``filesrc`` reports a missing file on the GStreamer bus rather than raising,
    so without this the run looks healthy right up to a 20 second pull timeout
    reporting zero frames, which is indistinguishable from a stall.

    Args:
        cfg: Application configuration.

    Returns:
        The file's size in bytes, so the caller can say how big it is without
        stat-ing it a second time. Zero for a source that is not a file.

    Raises:
        RuntimeError: If the file is missing or empty.
    """
    if cfg.source_type != "video":
        return 0

    path = Path(cfg.source_uri)
    if not path.exists():
        listing = ""
        parent = path.parent
        for candidate in (parent, Path("assets/video"), Path("assets/videos")):
            if candidate.is_dir():
                names = sorted(p.name for p in candidate.iterdir() if p.is_file())
                if names:
                    listing += f"\n  {candidate}/ contains: {', '.join(names[:8])}"
        raise RuntimeError(
            f"source file not found: {path}\n"
            f"  looked in: {path.resolve().parent}\n"
            f"  launched from: {Path.cwd()}"
            f"{listing}\n"
            "source.uri is relative to the directory you launch from, or to the "
            "config file. Pass --source with a path that exists from here."
        )

    size = path.stat().st_size
    if size == 0:
        raise RuntimeError(f"source file is empty: {path}")

    if is_elementary_h264(cfg.source_uri):
        with path.open("rb") as handle:
            head = handle.read(12)
        # Annex-B streams open with a 3 or 4 byte start code. An MP4 carries
        # "ftyp" at offset 4, which is what a rename rather than a convert looks
        # like, and h264parse would simply never produce a frame.
        annex_b = head.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01"))
        if not annex_b:
            hint = (
                " That looks like an MP4 container renamed to .h264."
                if b"ftyp" in head
                else ""
            )
            raise RuntimeError(
                f"{path} is not a raw H.264 elementary stream.{hint}\n"
                f"  first bytes: {head[:8].hex(' ')}\n"
                "Convert rather than rename:\n"
                "  ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 clip.h264"
            )

    return size


# ─────────────────────────────────────────────────────────────────────────────
# Source graph
# ─────────────────────────────────────────────────────────────────────────────


def set_output_caps(caps, fps: int, width: int, height: int) -> None:
    pyneat = runtime.pyneat
    if width <= 0 or height <= 0 or fps <= 0:
        return
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any


#: pyneat names each source path needs, beyond what every run needs. Checked
#: while probing the source, which is the last cheap moment before the model
#: load: an AttributeError from inside graph construction costs the better part
#: of a minute to learn something knowable at the start.
SOURCE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "h264": ("SimaDecodeOptions", "SimaDecodeType"),
    "container": ("VideoInputGroupOptions",),
    "rtsp": ("RtspDecodedInputOptions", "RtspCodec"),
    "usb": ("CameraInputOptions",),
}

#: How to name each of those in a sentence.
SOURCE_NAMES = {
    "h264": "a raw H.264 file",
    "container": "a container file",
    "rtsp": "an RTSP stream",
    "usb": "a camera",
}


def source_kind(cfg) -> str:
    """Which entry of :data:`SOURCE_REQUIREMENTS` this config will take."""
    if cfg.source_type != "video":
        return cfg.source_type
    return "h264" if is_elementary_h264(cfg.source_uri) else "container"


def check_source_support(cfg) -> None:
    """Refuse a Neat build that cannot construct this source.

    The Neat Library is a compiled extension whose API moves between SDK
    releases, and a missing name surfaces as ``AttributeError`` from somewhere
    deep in graph construction -- true, but it names a symbol rather than the
    problem. This says which build is installed and what it is missing.

    Raises:
        RuntimeError: When this pyneat lacks something the source path needs.
    """
    pyneat = runtime.pyneat
    if pyneat is None:                        # --validate never gets this far
        return
    kind = source_kind(cfg)
    missing = [
        name for name in SOURCE_REQUIREMENTS.get(kind, ()) if not hasattr(pyneat, name)
    ]
    if not missing:
        return
    version = getattr(pyneat, "__version__", "unknown")
    raise RuntimeError(
        f"this Neat Library build cannot read {SOURCE_NAMES.get(kind, kind)}.\n"
        f"  installed: pyneat {version}\n"
        f"  wanted:    pyneat {NEAT_VERSION}\n"
        f"  missing:   {', '.join('pyneat.' + name for name in missing)}\n"
        "Install the core this is written against, here on the board:\n"
        f"  sima-cli login\n  {NEAT_INSTALL}"
    )


def make_elementary_h264_source(cfg, width: int, height: int, fps: int):
    """Build a file source chain without a demuxer.

    This is ``VideoInputGroup`` rebuilt by hand to work around a Neat 0.3.0 bug.
    ``VideoTrackSelect`` emits ``qtdemux name=<base> <base>.video_0``, which is
    internally consistent, but the graph then appends an instance suffix to
    element *names* only. The declaration becomes ``name=n1_demux_8`` while the
    pad reference stays ``n1_demux.video_0``, so gst_parse_launch fails with
    ``No src-element named "n1_demux"``. ``element_names()`` reports only the one
    name, so the renamer never learns to rewrite the pad reference, and any
    non-empty suffix triggers it. Reordering graph construction does not help.

    Dropping the container removes the demuxer, and with it the bug::

        ffmpeg -i input.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 output.h264

    Args:
        cfg: Application configuration, for ``source_uri``.
        width: Unused. Geometry comes from the stream, see the caps note below.
        height: Unused.
        fps: Unused.

    Returns:
        A ``pyneat.Graph`` of FileInput, H264Parse, Queue and SimaDecode,
        producing NV12 frames.
    """
    pyneat = runtime.pyneat
    graph = pyneat.Graph("file_source")
    graph.add(pyneat.nodes.file_input(cfg.source_uri))
    graph.add(pyneat.nodes.h264_parse(config_interval=1))
    graph.add(pyneat.nodes.queue())

    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = 2
    dec.out_format = pyneat.Format.NV12
    dec.raw_output = False
    # Left at pyneat's -1, the daemon sizes its own pool and reports it as
    # BufferNum=8 for 1080p. Eight is not enough for a stream that keeps four
    # or five reference frames: the DPB takes those plus the picture being
    # decoded, the source appsink declares four more, and the sum is over
    # eight before the app has done anything. That is the stall this whole
    # branch chased, and the pool being askable for is the fix -- see
    # :func:`decoder_buffers_for`.
    requested = decoder_buffers_for(cfg, width, height)
    if requested > 0:
        dec.num_buffers = requested
    graph.add(pyneat.nodes.sima_decode(dec))

    # No CapsRaw node here, deliberately, and this is the difference between a
    # run that finishes the clip and one that dies on a pull timeout part-way
    # through.
    #
    # `raw_output = False` already appends `videoconvert ! capsfilter
    # caps="video/x-raw(memory:SystemMemory),format=NV12"` to the decoder, so a
    # CapsRaw("NV12") after it constrains nothing that is not already fixed. It
    # is not free, though: the Graph inserts `queue max-size-buffers=5` between
    # adjacent nodes, and never before a terminal appsink. With the extra node
    # the decoded path was
    #
    #   neatdecoder ! videoconvert ! capsfilter ! queue(5) ! capsfilter ! appsink(4)
    #
    # so 5 + 4 = 9 decoded frames could sit downstream at once. The hardware
    # decoder's pool is 8 (`BufferNum=8` in the boot log) and it needs several
    # of those for its own reference frames, so the path could swallow the
    # entire pool. The decoder then cannot produce, the app cannot consume, and
    # nothing is ever released. Without the node the path is
    #
    #   neatdecoder ! videoconvert ! capsfilter ! appsink(4)
    #
    # which caps it at 4 and leaves the rest of the pool to the decoder.
    #
    # If negotiation ever does fail here, the node comes back as
    # `graph.add(pyneat.nodes.caps_raw("NV12", -1, -1, -1, pyneat.CapsMemory.Any))`
    # -- format only. Passing width, height or fps instead would add
    # `framerate=F/1`, and a raw elementary stream has no container to state its
    # rate, so h264parse publishes `framerate=0/1`, which intersects with
    # nothing. That fails silently: zero frames and a pull timeout, with nothing
    # on the bus to explain it.
    return graph


def make_source_graph(cfg, width: int, height: int, fps: int):
    """File / RTSP / camera head of the Graph. All three produce NV12 frames."""
    pyneat = runtime.pyneat
    if cfg.source_type == "video":
        if is_elementary_h264(cfg.source_uri):
            console.note("raw H.264 elementary stream, demuxer bypassed")
            return make_elementary_h264_source(cfg, width, height, fps)

        console.warn(
            "container input uses groups.video_input, which hits a demuxer\n"
            "naming bug in Neat 0.3.0. If the pipeline fails to start with\n"
            "'No src-element named \"nN_demux\"', convert to a raw stream:\n"
            f"  ffmpeg -i {cfg.source_uri} -c:v copy -bsf:v h264_mp4toannexb \\\n"
            f"    -f h264 {Path(cfg.source_uri).with_suffix('.h264')}\n"
            "then point source.uri at the .h264 file."
        )
        opt = pyneat.VideoInputGroupOptions()
        opt.path = cfg.source_uri
        opt.insert_queue = True
        opt.sync_mode = False
        opt.out_format = pyneat.Format.NV12
        set_output_caps(opt.output_caps, fps, width, height)
        return pyneat.groups.video_input(opt)

    if cfg.source_type == "rtsp":
        opt = pyneat.RtspDecodedInputOptions()
        opt.url = cfg.source_uri
        opt.latency_ms = cfg.rtsp_latency_ms
        opt.tcp = cfg.rtsp_tcp
        opt.insert_queue = True
        opt.decoder_name = "decoder"
        opt.decoder_raw_output = True
        opt.source_fps = fps
        opt.codec = (
            pyneat.RtspCodec.H264 if cfg.rtsp_codec == "h264" else pyneat.RtspCodec.MJPEG
        )
        if cfg.rtsp_codec == "h264":
            opt.payload_type = 96
            opt.auto_caps_from_stream = True
            opt.fallback_h264_width = width
            opt.fallback_h264_height = height
        else:
            opt.mjpeg_payload_type = 26
            opt.dec_width = width
            opt.dec_height = height
        set_output_caps(opt.output_caps, fps, width, height)
        return pyneat.groups.rtsp_decoded_input(opt)

    # usb / on-board camera, libcamera-backed. Confirm the device is visible
    # with `cam -l` on the DevKit and put its name in source.usb.camera_name.
    opt = pyneat.CameraInputOptions()
    opt.camera_name = cfg.usb_camera_name or None
    opt.width = width
    opt.height = height
    opt.framerate_num = fps
    opt.framerate_den = 1
    opt.format = cfg.usb_format
    opt.insert_queue = True
    opt.leaky_queue = True
    return pyneat.nodes.camera_input(opt)


def source_frame_count(cfg) -> int:
    """Coded pictures in the source, or 0 when there is no such number."""
    if cfg.source_type == "video" and is_elementary_h264(cfg.source_uri):
        return count_h264_pictures(cfg.source_uri)
    return 0
