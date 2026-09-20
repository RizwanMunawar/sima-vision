"""Model options, run options and Graph assembly.

The pipeline is a Graph rather than a single ``Model.run`` call, because it has
multiple stages, named public endpoints and a branch with a fan-in::

    source --> branch --> frame ---------------+
                    |                          +--> combine("<task>_output")
                    +---> model --> results ---+

All three tasks use exactly that shape; only the names differ, which is why
:func:`build_task_graph` takes them as arguments rather than existing three
times.
"""

from __future__ import annotations

from pathlib import Path

from . import runtime
from .media import make_source_graph
from .runtime import (
    AUTO_FLAGS,
    COLOR_FORMATS,
    DECODE_TYPE_OPTIONS,
    FAMILY_DECODE_TOKENS,
    INPUT_KINDS,
    NORMALIZE_PRESETS,
    OVERFLOW_POLICIES,
    RESIZE_MODES,
    RUN_PRESETS,
    enum_value,
)


def apply_preprocess_options(opt, pre, frame_w: int, frame_h: int) -> None:
    """
    Translate the config's preprocess block onto pyneat ModelOptions.

    ``Model`` resolves this against the archive's MPK contract and builds the
    matching Preproc, Quant, Tess or QuantTess graph family. Anything left on
    ``auto`` is decided by the planner rather than here.

    Args:
        opt: A ``pyneat.ModelOptions`` to populate in place.
        pre: Preprocessing intent from the config file.
        frame_w: Probed source width, used when no capacity is configured.
        frame_h: Probed source height, used when no capacity is configured.
    """
    pyneat = runtime.pyneat
    p = opt.preprocess

    p.kind = enum_value(pyneat.InputKind, pre.kind, INPUT_KINDS, "preprocess.kind")
    p.enable = enum_value(pyneat.AutoFlag, pre.enable, AUTO_FLAGS, "preprocess.enable")

    # Buffer capacity for the Preproc node. Defaults to 1920x1080 when left at 0;
    # pin it to the real stream geometry so buffers are sized correctly.
    p.input_max_width = pre.input_max_width or frame_w
    p.input_max_height = pre.input_max_height or frame_h

    # ── Resize / letterbox ──
    p.resize.enable = enum_value(
        pyneat.AutoFlag, pre.resize_enable, AUTO_FLAGS, "preprocess.resize.enable"
    )
    # 0 means "infer the target from the model input contract".
    p.resize.width = pre.resize_width
    p.resize.height = pre.resize_height
    p.resize.mode = enum_value(
        pyneat.ResizeMode, pre.resize_mode, RESIZE_MODES, "preprocess.resize.mode"
    )
    p.resize.pad_value = pre.pad_value
    p.resize.scaling_type = pre.scaling_type

    # ── Colour conversion ──
    # input_format must match what the source hands to Preproc: NV12 for the
    # hardware H.264 decoder and libcamera, BGR for cv2 images.
    p.color_convert.input_format = enum_value(
        pyneat.PreprocessColorFormat,
        pre.input_format,
        COLOR_FORMATS,
        "preprocess.input_format",
    )
    p.color_convert.output_format = enum_value(
        pyneat.PreprocessColorFormat,
        pre.output_format,
        COLOR_FORMATS,
        "preprocess.output_format",
    )
    if pre.input_format != "AUTO" or pre.output_format != "AUTO":
        p.color_convert.enable = pyneat.AutoFlag.On

    # ── Normalisation ──
    p.normalize.enable = enum_value(
        pyneat.AutoFlag, pre.normalize_enable, AUTO_FLAGS, "preprocess.normalize.enable"
    )
    p.preset = enum_value(
        pyneat.NormalizePreset,
        pre.normalize_preset,
        NORMALIZE_PRESETS,
        "preprocess.normalize.preset",
    )
    if pre.normalize_preset == "none":
        # Explicit stats are only read when no preset supplies them.
        p.normalize.mean = list(pre.mean)
        p.normalize.stddev = list(pre.stddev)

    # ── Quantise / tessellate ──
    # Normally left on auto: the model pack's dtype contract decides whether
    # these run in the Preproc graph or inside the MLA.
    p.quantize.enable = enum_value(
        pyneat.AutoFlag, pre.quantize_enable, AUTO_FLAGS, "preprocess.quantize.enable"
    )
    if pre.quantize_enable == "on":
        p.quantize.zero_point = pre.quantize_zero_point
        p.quantize.scale = pre.quantize_scale

    p.tessellate.enable = enum_value(
        pyneat.AutoFlag, pre.tessellate_enable, AUTO_FLAGS, "preprocess.tessellate.enable"
    )
    if pre.tessellate_slice_shape:
        p.tessellate.set_slice_shape(list(pre.tessellate_slice_shape))


def make_model(cfg, frame_w: int, frame_h: int):
    pyneat = runtime.pyneat
    opt = pyneat.ModelOptions()
    apply_preprocess_options(opt, cfg.preprocess, frame_w, frame_h)

    opt.decode_type = enum_value(
        pyneat.BoxDecodeType, cfg.family, FAMILY_DECODE_TOKENS, "model.family"
    )
    opt.decode_type_option = enum_value(
        pyneat.BoxDecodeTypeOption,
        cfg.decode_type_option,
        DECODE_TYPE_OPTIONS,
        "model.decode_type_option",
    )
    # 0 on any of these preserves the value packaged in the model archive.
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    if cfg.num_classes > 0:
        opt.num_classes = cfg.num_classes

    if not Path(cfg.model_path).exists():
        raise RuntimeError(f"model archive not found: {cfg.model_path}")
    return pyneat.Model(cfg.model_path, opt)


def describe_preprocess(cfg, frame_w: int, frame_h: int) -> str:
    pre = cfg.preprocess
    target = (
        f"{pre.resize_width}x{pre.resize_height}"
        if pre.resize_width and pre.resize_height
        else "<from model contract>"
    )
    norm = (
        pre.normalize_preset
        if pre.normalize_preset != "none"
        else f"mean={list(pre.mean)} stddev={list(pre.stddev)}"
    )
    return (
        f"preprocess: kind={pre.kind} enable={pre.enable} "
        f"in={pre.input_format} out={pre.output_format} "
        f"capacity={pre.input_max_width or frame_w}x{pre.input_max_height or frame_h} | "
        f"resize={pre.resize_mode} target={target} pad={pre.pad_value} "
        f"scaler={pre.scaling_type} | normalize={norm} | "
        f"quantize={pre.quantize_enable} tessellate={pre.tessellate_enable}"
    )


def resolve_flow_control(cfg) -> tuple[str, str]:
    """Resolve ``auto`` flow-control settings to concrete tokens.

    A file and a live camera want opposite things, so ``auto`` splits on source
    type:

    * **File** -> ``reliable`` + ``block``. A file has no deadline. Every frame
      matters, and the source can be made to wait.
    * **RTSP or USB** -> ``realtime`` + ``keep_latest``. A camera has no pause
      button, so blocking only buys unbounded latency. Staying current is worth
      more than completeness.

    Getting this wrong on a file is what shortens the recording. ``keep_latest``
    makes the runtime discard buffers whenever inference falls behind the
    decoder, and inference is much slower than decode. Only the survivors reach
    the writer, which then stamps them at the source rate, so the output plays
    fast and ends early: a 15 second clip processed at a quarter of realtime
    becomes a 4 second video. ``block`` instead lets backpressure reach
    ``filesrc``, so decoding slows to the speed of inference and every frame
    survives. The run takes longer than the clip. That is correct, not a stall.

    Args:
        cfg: Application configuration.

    Returns:
        A ``(preset, overflow_policy)`` pair of config tokens.
    """
    live = cfg.source_type in {"rtsp", "usb"}
    preset = cfg.run_preset
    policy = cfg.overflow_policy
    if preset == "auto":
        preset = "realtime" if live else "reliable"
    if policy == "auto":
        policy = "keep_latest" if live else "block"
    return preset, policy


def make_run_options(cfg):
    """Build RunOptions, resolving any ``auto`` flow-control settings.

    Args:
        cfg: Application configuration.

    Returns:
        A populated ``pyneat.RunOptions``.
    """
    pyneat = runtime.pyneat
    preset, policy = resolve_flow_control(cfg)
    run_options = pyneat.RunOptions()
    run_options.preset = enum_value(pyneat.RunPreset, preset, RUN_PRESETS, "runtime.preset")
    run_options.queue_depth = cfg.queue_depth
    run_options.overflow_policy = enum_value(
        pyneat.OverflowPolicy, policy, OVERFLOW_POLICIES, "runtime.overflow_policy"
    )
    # Block is not honoured unconditionally. The runtime quietly rewrites it to
    # KeepLatest when the public output is zero-copy and carries no explicit
    # OutputOptions, so asking for every frame while also asking for zero-copy
    # gets you neither. Auto lets the preset choose, and `reliable` chooses
    # owned buffers, which keeps Block meaning Block. Live sources drop by
    # design, so they keep zero-copy and its lower latency.
    run_options.output_memory = (
        pyneat.OutputMemory.Auto if policy == "block" else pyneat.OutputMemory.ZeroCopy
    )
    return run_options


def build_task_graph(cfg, model, width: int, height: int, fps: int,
                     graph_name: str, result_label: str, output_label: str):
    """``source -> branch -> {frame, model->results} -> combine(output_label)``.

    Args:
        cfg: Application configuration, for ``output_buffers``.
        model: The loaded ``pyneat.Model``.
        width: Source frame width.
        height: Source frame height.
        fps: Source frame rate.
        graph_name: Name for the whole graph, such as ``yolo_detector``.
        result_label: Name of the model's public output, such as ``detections``
            or ``instances``. Callers read it back with ``joined_field``.
        output_label: Name of the combined output the run loop pulls.

    Returns:
        A ``pyneat.Graph`` ready to build.
    """
    pyneat = runtime.pyneat
    source = make_source_graph(cfg, width, height, fps)
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(
        pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(cfg.output_buffers))
    )

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    result_graph = pyneat.Graph(result_label)
    result_graph.add(
        pyneat.nodes.output(result_label, pyneat.OutputOptions.every_frame(cfg.output_buffers))
    )

    joined = pyneat.graphs.combine(
        ["frame", result_label], output_label, pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph(graph_name)
    graph.connect(source, branch)
    graph.connect(branch, frame_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, result_graph)
    graph.connect(frame_graph, joined)
    graph.connect(result_graph, joined)
    return graph
