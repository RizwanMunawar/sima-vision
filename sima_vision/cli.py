"""The ``sima-vision`` command.

    pip install sima-vision
    sima-vision detect

That is the whole of it. There is no setup command, no init, no fetch and no
doctor, because a run does all of it: it finds the Neat runtime, puts the
board's numpy and OpenCV on the path, downloads the model pack and the sample
clip, and says what it is doing at each step. See
:mod:`sima_vision.bootstrap`.

The only other commands are ``push`` and ``pull``, which move files to and from
the board. See :mod:`sima_vision.devkit`.

One subcommand per task, and a task is a plugin -- the built-in three are
registered exactly the way a fourth one from another package would be. See
:mod:`sima_vision.tasks`.

Every flag that corresponds to a config key declares its dotted path as its
argparse ``dest``, so the whole override mechanism is this::

    parser.add_argument("--source", dest="source.uri")
    ...
    {"source.uri": "clip.h264"}  ->  raw["source"]["uri"] = "clip.h264"

Overrides are written into the parsed YAML *before* the loaders run, so a CLI
flag goes through exactly the same defaulting and validation a config file does,
and cannot reach a state a config file could not. Config is optional and so are
the flags: the dataclass defaults are a complete configuration down to a model
and a clip, which is why ``sima-vision detect`` runs with no arguments at all.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

from . import __version__
from .assets import models_dir
from .bootstrap import detect_environment, ensure_runtime
from .console import console, human_bytes
from .devkit import DEVKIT_ENV, run_pull, run_push
from .export import (
    DEFAULT_IMGSZ,
    DEFAULT_OPSET,
    compile_recipe,
    export_onnx,
    model_sdk_present,
    next_steps,
)
from .neat import describe_preprocess
from .runloop import Stopper
from .runtime import FAMILY_DECODE_TOKENS
from .tasks import TASKS

EPILOG = """\
examples:
  sima-vision detect                       the sample clip and model, fetched for you
  sima-vision detect  --source clip.h264 --model yolo26m-det.tar.gz
  sima-vision detect  --source clip.mp4                    reframed for you, once
  sima-vision detect  --source https://example.com/clip.h264
  sima-vision segment --blur --keep-classes person
  sima-vision fall    --source rtsp://cam/live --alert-to ops@example.com

without a board:
  sima-vision detect --validate            check the settings, no hardware at all

moving files between your PC and the board:
  sima-vision push clip.h264               copy files over
  sima-vision pull                         bring the results back

Everything a run needs is found or downloaded on the way in, once, into
./assets. A config.yaml in the working directory is picked up automatically if
there is one, and flags win over it.
"""


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags every task understands. The dest is the config key it writes."""
    source = parser.add_argument_group("source")
    source.add_argument(
        "--source", "-s", dest="source.uri", metavar="URI",
        help="Video file, https URL, RTSP URL, or empty for this task's sample "
             "clip. An https URL is downloaded into assets/videos/ once and "
             "reused. Raw .h264 only for files; see the README on converting.",
    )
    source.add_argument(
        "--source-type", dest="source.type", choices=("video", "rtsp", "usb"),
        help="Where frames come from. Default video.",
    )
    source.add_argument(
        "--fps", dest="source.fps", type=int, metavar="N",
        help="Source frame rate. Default 0, which reads it from the stream.",
    )
    source.add_argument(
        "--width", dest="source.width", type=int, metavar="PX",
        help="Source width. Default 0, which reads it from the stream's SPS.",
    )
    source.add_argument(
        "--height", dest="source.height", type=int, metavar="PX",
        help="Source height. Default 0, which reads it from the stream's SPS.",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model", "-m", dest="model.path", metavar="PATH",
        help="Compiled model archive (.tar.gz), or an https URL to one. Empty "
             "uses this task's default in assets/models/, fetched with sima-cli "
             "on the first run.",
    )
    model.add_argument(
        "--labels", dest="model.labels", metavar="PATH",
        help="Newline-separated class names. Defaults to the packaged COCO list.",
    )
    model.add_argument(
        "--family", dest="model.family", metavar="NAME",
        choices=sorted(FAMILY_DECODE_TOKENS),
        help="Detection head. Must match the model or you get no detections.",
    )
    model.add_argument(
        "--conf", dest="decode.score_threshold", type=float, metavar="T",
        help="Minimum detection confidence. Default 0.30.",
    )
    model.add_argument(
        "--iou", dest="decode.nms_iou", type=float, metavar="T",
        help="Non-max suppression IoU threshold. Default 0.60.",
    )
    model.add_argument(
        "--max-det", dest="decode.max_detections", type=int, metavar="N",
        help="Top-K cap per frame. Default 50.",
    )

    run = parser.add_argument_group("runtime")
    run.add_argument(
        "--frames", "-n", dest="runtime.frames", type=int, metavar="N",
        help="Stop after N frames. Default 0, which runs until interrupted.",
    )
    run.add_argument(
        "--timeout", dest="runtime.pull_timeout_ms", type=int, metavar="MS",
        help="How long to wait for a frame before giving up. Default 20000.",
    )
    run.add_argument(
        "--queue-depth", dest="runtime.queue_depth", type=int, metavar="N",
        help="Depth of the Neat runtime's own queues. Every slot can hold a "
             "decoded frame, so raising this makes a buffer-starved run worse, "
             "not better. Default 1.",
    )
    run.add_argument(
        "--sink-queue-depth", dest="runtime.sink_queue_depth", type=int, metavar="N",
        help="How many finished frames may wait for the recorder. Costs host "
             "memory only, about 6 MB a slot at 1080p, and lets the pull loop "
             "keep draining the decoder. Raise it if a run stalls. Default 12.",
    )
    run.add_argument(
        "--segment-frames", dest="runtime.segment_frames", type=int, metavar="N",
        help="Frames per piece when a clip is too long for one decode. The "
             "decoder stops around 195 frames, so a longer clip is cut at its "
             "keyframes and decoded piece by piece into one recording. "
             "0 runs the clip whole. Default 150.",
    )
    run.add_argument(
        "--output-buffers", dest="runtime.output_buffers", type=int, metavar="N",
        help="Buffers each public output may hold. Default 1. Every one is a "
             "frame checked out of the decoder's pool, so raising it used to "
             "make a starved run worse -- but --decoder-buffers can now pay "
             "for it. Worth a 2 if frames are being dropped at the join.",
    )
    run.add_argument(
        "--decoder-buffers", dest="runtime.decoder_buffers", type=int, metavar="N",
        help="Buffers to ask the hardware decoder for. Default 0, which sizes "
             "it from the stream's own reference frames -- the fix for a run "
             "that stops part-way through. Negative leaves pyneat to pick.",
    )
    run.add_argument(
        "--sink-queue-mb", dest="runtime.sink_queue_mb", type=int, metavar="MB",
        help="Host memory the sink backlog may use on a file source. The queue "
             "grows towards holding the whole clip so the pull loop never waits "
             "for the recorder, which is what starves the decoder. 0 disables "
             "the growth. Default 1024.",
    )
    run.add_argument(
        "--profile", dest="runtime.profile", action="store_const", const=True,
        help="Print per-stage timings every runtime.profile_interval frames.",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--video-path", dest="output.video.path", metavar="PATH",
        help="Where to write the annotated recording on the DevKit.",
    )
    out.add_argument(
        "--no-video", dest="output.video.enable", action="store_const", const=False,
        help="Do not record.",
    )
    out.add_argument(
        "--save-dir", dest="output.save.dir", metavar="DIR",
        help="Where to write annotated stills.",
    )
    out.add_argument(
        "--save-every", dest="output.save.every", type=int, metavar="N",
        help="Write every Nth still. Default 10; 0 disables.",
    )
    out.add_argument(
        "--no-save", dest="output.save.enable", action="store_const", const=False,
        help="Do not write stills.",
    )
    out.add_argument(
        "--no-hud", dest="output.video.hud", action="store_const", const=False,
        help="Leave the frame-rate badge off the overlay.",
    )
    out.add_argument(
        "--insight", dest="output.insight.enable", action="store_const", const=True,
        help="Stream to Neat Insight over UDP. Off by default: its encoder shares "
             "the codec daemon with the decoder, so it can stall a file run.",
    )
    out.add_argument(
        "--insight-host", dest="output.insight.host", metavar="HOST",
        help="Insight address as the DevKit sees it. Default 127.0.0.1.",
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Which config file to read, or none at all."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config", "-c", type=Path, metavar="PATH",
        help="Config file. Defaults to ./config.yaml in the working directory.",
    )
    group.add_argument(
        "--no-config", action="store_true",
        help="Ignore any config file and use the built-in defaults plus these "
             "flags, even when a config.yaml is sitting right there.",
    )


def add_task_arguments(parser: argparse.ArgumentParser, task) -> None:
    """Everything one task understands: the shared flags plus its own."""
    add_shared_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument(
        "--validate", action="store_true",
        help="Resolve and check the settings, print what they came to, and exit. "
             "Needs neither the Neat runtime nor the board, so it works on a laptop.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Only warnings, errors and the closing report. Steps are silent.",
    )
    task.add_arguments(parser.add_argument_group(f"{task.name} options"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sima-vision",
        description="Live YOLO computer vision on a SiMa Modalix DevKit 3.0. "
                    "Install it and run it; there is no setup step.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sima-vision {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name, task_cls in TASKS.items():
        task = task_cls()
        sub = subparsers.add_parser(
            name,
            help=task.help,
            description=task.help + ".",
            epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_task_arguments(sub, task)
        sub.set_defaults(_task=task_cls)

    add_compile_parser(subparsers)
    add_push_parser(subparsers)
    add_pull_parser(subparsers)
    return parser


def add_host_argument(parser: argparse.ArgumentParser) -> None:
    """Which board. The same flag on both transfer commands."""
    parser.add_argument(
        "--host", "-H", metavar="USER@ADDR",
        help=f"The DevKit, as ssh takes it. Defaults to ${DEVKIT_ENV} so you "
             f"only say it once.",
    )


def add_compile_parser(subparsers) -> None:
    """``compile`` -- a trained .pt towards a pack the board can run."""
    parser = subparsers.add_parser(
        "compile",
        help="Turn a trained YOLO26 .pt into a DevKit model pack",
        description=(
            "Export a trained YOLO26 .pt to the raw-head ONNX the board's box "
            "decoder reads, then compile it with the SiMa Model SDK if this "
            "machine has one. Run it on your PC: exporting needs torch, and "
            "compiling needs the Palette container."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision compile best.pt\n"
            "  sima-vision compile best.pt --imgsz 512 --out build/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("weights", help="Trained YOLO26 detection .pt.")
    parser.add_argument(
        "--out", metavar="DIR", default="build",
        help="Where the ONNX and the recipe are written. Default build/.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ, metavar="N",
        help=f"Square input side. Default {DEFAULT_IMGSZ}, which is what the "
             "published packs use.",
    )
    parser.add_argument(
        "--opset", type=int, default=DEFAULT_OPSET, metavar="N",
        help=f"ONNX opset. Default {DEFAULT_OPSET}.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Warnings and errors only.",
    )


def add_push_parser(subparsers) -> None:
    """``push`` -- copy files to the board."""
    sub = subparsers.add_parser(
        "push",
        help="Copy files or folders to the DevKit",
        description=(
            "Copy local files to the DevKit's home directory with scp. Folders "
            "are copied whole. On Windows this is also the way to avoid scp "
            "reading a drive letter as a hostname."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision push config.yaml\n"
            "  sima-vision push my-clip.h264 my-model.tar.gz\n"
            "  sima-vision push assets --dest '~/'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("paths", nargs="+", type=Path, metavar="PATH",
                     help="Files or folders to copy.")
    sub.add_argument("--dest", default="~/", metavar="DIR",
                     help="Where to put them on the board. Default ~/.")
    add_host_argument(sub)


def add_pull_parser(subparsers) -> None:
    """``pull`` -- copy results back."""
    sub = subparsers.add_parser(
        "pull",
        help="Copy results back from the DevKit",
        description=(
            "Copy a run's output back to this machine. With no names it asks "
            "for everything any task could have written -- the annotated video, "
            "frames/, alerts/ and config.yaml -- and takes whatever is there, "
            "so it does not need to be told which task ran."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision pull\n"
            "  sima-vision pull detections.mp4\n"
            "  sima-vision pull --into results/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("names", nargs="*", metavar="NAME",
                     help="Names on the board, relative to its home directory.")
    sub.add_argument("--into", type=Path, default=Path("."), metavar="DIR",
                     help="Where to put them here. Default the current directory.")
    add_host_argument(sub)


def collect_overrides(args: argparse.Namespace) -> dict:
    """Every dotted-dest flag the user actually gave, as config paths.

    ``None`` means the flag was not given, which is how an unset flag defers to
    the config file rather than overwriting it with an argparse default.
    """
    return {
        key: value
        for key, value in vars(args).items()
        if "." in key and value is not None
    }


def run_compile(args) -> int:
    """``compile`` -- export, then compile if the Model SDK is here."""
    console.banner(f"sima-vision {__version__}", "compile")
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"no such file: {weights}")

    out_dir = Path(args.out)
    onnx_path = out_dir / f"{weights.stem}-raw.onnx"
    with console.step(f"Exporting {weights.name} to ONNX", "export") as step:
        step.note(
            "the board decodes boxes itself, so the head's six raw tensors are exported\n"
            "rather than ultralytics' assembled [1, 84, 8400] output"
        )
        shapes = export_onnx(weights, onnx_path, args.imgsz, args.opset)
        for name, shape in shapes.items():
            step.detail(f"{name:<16} {tuple(shape)}")
        step.done(f"{onnx_path} ({human_bytes(onnx_path.stat().st_size)})")

    with console.step("Compiling the DevKit pack", "compile") as step:
        recipe_path = write_recipe(out_dir, step)
        if not model_sdk_present():
            console.warn(next_steps(onnx_path, recipe_path))
            step.done("ONNX ready, compile it in Palette")
            return 0
        step.done("the Model SDK is here; run the recipe above on the ONNX")
    return 0


def write_recipe(out_dir: Path, step) -> Path | None:
    """Copy a published pack's own compile script next to the ONNX.

    Taken from a pack rather than written here, because the settings that
    matter -- bfloat16, MSE calibration, the MLA tessellation layouts -- are
    the ones SiMa actually shipped, and a paraphrase of them would drift.
    """
    packs = sorted(models_dir().glob("*.tar.gz"))
    if not packs:
        step.note(
            "no model pack here to copy a recipe from. Any real run fetches one,\n"
            "and the recipe comes inside it."
        )
        return None
    try:
        script = compile_recipe(packs[0])
    except (RuntimeError, OSError, tarfile.TarError) as exc:
        step.note(f"could not read a recipe from {packs[0].name}: {exc}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compile_modelsdk.py"
    path.write_text(script, encoding="utf-8")
    step.detail(f"recipe from {packs[0].name} -> {path}")
    return path


def print_validation(task, cfg) -> None:
    """What ``--validate`` prints. Deliberately the same shape for every task."""
    console.banner(f"sima-vision {__version__}", f"{task.name} --validate")
    console.success(f"config OK: {cfg.config_path or '<defaults and flags only>'}")
    lines = [
        f"model:   {cfg.model_path or '<unset>'}",
        f"labels:  {cfg.labels_path}",
        f"family:  {cfg.family} -> BoxDecodeType.{FAMILY_DECODE_TOKENS[cfg.family]}",
        f"source:  type={cfg.source_type} uri={cfg.source_uri or '<default camera>'}",
        f"decode:  conf={cfg.score_threshold} iou={cfg.nms_iou} "
        f"max_det={cfg.max_detections}",
        describe_preprocess(cfg, cfg.source_width, cfg.source_height),
        *task.describe(cfg),
    ]
    outputs = []
    if cfg.video_enable:
        outputs.append(f"video={cfg.video_path}")
    if cfg.save_enable:
        outputs.append(f"stills={cfg.save_dir}/ every={cfg.save_every}")
    if cfg.insight_enable:
        outputs.append(f"insight={cfg.insight_host}:{cfg.video_port_base}")
    lines.append(f"output:  {' '.join(outputs) or '<nothing written>'}")
    for line in lines:
        console.info(f"  {line}")
    console.write()
    console.note("  nothing was downloaded and no hardware was touched.")


def run_task(args) -> int:
    """Resolve the config, set the machine up, and run. The whole of a run."""
    task = args._task()
    cfg = task.post_process(
        task.load(args.config, collect_overrides(args), use_file=not args.no_config),
        args,
    )

    if args.validate:
        print_validation(task, cfg)
        return 0

    early = task.early_exit(cfg, args)
    if early is not None:
        return early

    console.banner(f"sima-vision {__version__}", task.name)
    with console.step("Checking the environment", "check") as step:
        env = detect_environment()
        step.done(env.summary())
    ensure_runtime(env)

    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
    if cfg.save_enable:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    task.run(cfg, Stopper())
    return 0


def run_devkit_command(args) -> int:
    """push and pull: the two that talk to a board."""
    if args.command == "push":
        return run_push(args.paths, args.host, args.dest)
    return run_pull(args.names, args.host, args.into)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 2

    console.configure(quiet=getattr(args, "quiet", False))

    try:
        if args.command in TASKS:
            return run_task(args)
        if args.command == "compile":
            return run_compile(args)
        return run_devkit_command(args)
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        # These carry a message, not a status: `raise SystemExit("...")` is how
        # devkit.py refuses. An int code is argparse's, and is already the answer.
        if isinstance(exc.code, int):
            return exc.code
        console.error(str(exc))
        return 1
    except ImportError as exc:
        # bootstrap has already worked out which case this is -- wrong machine,
        # or right machine and nothing to install from -- and said so.
        console.error(str(exc))
        return 1
    except Exception as exc:
        console.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
