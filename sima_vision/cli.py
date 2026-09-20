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
import time
from pathlib import Path

from . import __version__
from .assets import default_model_path, ensure_model, models_dir
from .bootstrap import detect_environment, ensure_runtime
from .config import DECODER_TUNINGS, VIDEO_ENCODERS
from .console import console, human_bytes, human_time
from .devkit import DEVKIT_ENV, run_pull, run_push
from .export import (
    COMPILE_LOG,
    DEFAULT_IMGSZ,
    DEFAULT_OPSET,
    choose_sdk_python,
    compile_recipe,
    export_onnx,
    needs_fixed_batch,
    next_steps,
    pin_batch_size,
    requirements_help,
    run_recipe,
    sdk_candidates,
)
from .neat import describe_preprocess
from .pack import complete_pack
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
        help="Depth of the Neat runtime's own queues. Below 4 the graph drops "
             "frames whenever the recorder slows the pull loop, which makes "
             "the recording choppy. Default 4.",
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
        "--decoder-tuning", dest="runtime.decoder_tuning",
        choices=DECODER_TUNINGS,
        help="Hardware decoder tuning preset. Default 'default', which hands "
             "over every picture. 'auto' drops pictures in bursts on a file "
             "and makes the recording choppy.",
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
        "--video-encoder", dest="output.video.encoder", choices=VIDEO_ENCODERS,
        help="'sima' encodes H.264 on the DevKit's hardware encoder; 'opencv' "
             "uses OpenCV's software writer, about ten times slower. Default sima.",
    )
    out.add_argument(
        "--video-bitrate", dest="output.video.bitrate_kbps", type=int, metavar="KBPS",
        help="Target bitrate for the hardware encoder. Default 12000.",
    )
    out.add_argument(
        "--save", dest="output.save.enable", action="store_const", const=True,
        help="Also write annotated stills. Off by default: the video is the "
             "output, and stills every 10 frames left hundreds of JPEGs beside "
             "it that nobody asked for.",
    )
    out.add_argument(
        "--save-dir", dest="output.save.dir", metavar="DIR",
        help="Where to write annotated stills. Implies --save.",
    )
    out.add_argument(
        "--save-every", dest="output.save.every", type=int, metavar="N",
        help="Write every Nth still. Default 10 once stills are on; implies "
             "--save. 0 disables.",
    )
    out.add_argument(
        "--no-save", dest="output.save.enable", action="store_const", const=False,
        help="Do not write stills. The default, so this is only needed to "
             "override a config file that turns them on.",
    )
    out.add_argument(
        "--no-hud", dest="output.video.hud", action="store_const", const=False,
        help="Leave the frame-rate badge off the overlay.",
    )
    # The badge's look was configurable from the first version and reachable
    # only through a config file, which meant nobody found it. These four are
    # the ones people actually want; the other eight stay in
    # `visualization.hud`.
    out.add_argument(
        "--hud-scale", dest="visualization.hud.text_scale", type=float,
        metavar="N",
        help="Frame-rate badge font size. Default 2.4, which is 1.5x the caption "
             "scale; 0 follows the caption scale exactly.",
    )
    out.add_argument(
        "--hud-thickness", dest="visualization.hud.text_thickness", type=int,
        metavar="N",
        help="Badge stroke weight. Default 6, which is 1.5x the caption thickness; "
             "0 follows the caption thickness exactly.",
    )
    out.add_argument(
        "--hud-bg", dest="visualization.hud.bg_color", type=bgr_colour,
        metavar="B,G,R",
        help="Badge fill colour, as B,G,R. Default 128,0,128.",
    )
    out.add_argument(
        "--hud-color", dest="visualization.hud.text_color", type=bgr_colour,
        metavar="B,G,R",
        help="Badge text colour, as B,G,R. Default 255,255,255.",
    )
    out.add_argument(
        "--hud-padding", dest="visualization.hud.padding", type=int, metavar="PX",
        help="Gap between badge text and its edge, which is what sizes the badge.",
    )


def bgr_colour(value: str) -> list[int]:
    """Parse ``B,G,R`` for a colour flag.

    BGR rather than RGB because the whole overlay is OpenCV's, and one
    convention throughout beats a flag that reverses what the config file next
    to it means. Named colours are deliberately not accepted: `red` would have
    to be `0,0,255` here, and a flag that reads correctly and paints the wrong
    colour is worse than one that only takes numbers.
    """
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected three numbers as B,G,R -- got {value!r}"
        )
    channels = []
    for part in parts:
        if not part.isdigit() or not 0 <= int(part) <= 255:
            raise argparse.ArgumentTypeError(
                f"{part!r} is not a channel value: each of B, G and R is 0-255"
            )
        channels.append(int(part))
    return channels


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
    parser.add_argument(
        "weights",
        help="Trained YOLO26 .pt. Detection and segmentation both compile; "
             "a segmentation model keeps its mask coefficients.",
    )
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

    Asking where the stills go, or how often, is taken as asking for stills.
    They are off by default, so on its own `--save-every 5` would be accepted
    and write nothing -- and `--save-every 0` already carries that same
    enable/disable sense in the other direction. An explicit `--no-save`
    alongside either still wins, because it lands on the same key first.
    """
    overrides = {
        key: value
        for key, value in vars(args).items()
        if "." in key and value is not None
    }
    asks_for_stills = overrides.get("output.save.every") or overrides.get(
        "output.save.dir"
    )
    if asks_for_stills and "output.save.enable" not in overrides:
        overrides["output.save.enable"] = True
    return overrides


class Narration:
    """One long-running thing's own output, line by line, under a step.

    Every line is stamped with how long that thing has been going. A compile
    takes ten to fifteen minutes, and the stamps are what separate a slow phase
    from a stuck one while it runs -- and afterwards, what says which phase to
    blame. Dimmed, because it is the SDK talking and not this program.
    """

    def __init__(self, step) -> None:
        self.step = step
        self.started = time.perf_counter()
        self.lines = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def line(self, text: str) -> None:
        """One line of output. Counted always; shown unless it is blank.

        Counted before the blank check, so the total this reports is the log's
        own length rather than the number of lines that happened to be worth
        printing.
        """
        self.lines += 1
        if not text.strip():
            return
        self.step.note(f"{human_time(self.elapsed):>6}  {text}")

    def silence(self, elapsed: float) -> None:
        """Nothing said for a while. Says so, rather than looking hung."""
        self.step.note(f"{human_time(elapsed):>6}  still working")


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
            "the board decodes boxes itself, so the head's raw tensors are exported\n"
            "rather than ultralytics' assembled [1, 84, 8400] output"
        )
        narration = Narration(step)
        shapes = export_onnx(
            weights, onnx_path, args.imgsz, args.opset, on_line=narration.line,
        )
        for name, shape in shapes.items():
            step.detail(f"{name:<16} {tuple(shape)}")
        step.done(f"{onnx_path} ({human_bytes(onnx_path.stat().st_size)})", timed=True)

    with console.step("Compiling the DevKit pack", "compile") as step:
        # The two halves fail for different reasons and want different answers,
        # so they are asked separately. Collapsed into one branch, a machine
        # that was simply missing a recipe read as one that could never compile.
        # Which python, not whether this one. The recipe runs as a subprocess,
        # so the SDK has to be importable to *it* -- and `pip install
        # sima-vision` and `activate-model-compiler` land in different
        # virtualenvs often enough that asking only about this interpreter
        # stopped compiles on machines that could have finished them.
        sdk_python, absent = choose_sdk_python()
        if sdk_python is None:
            recipe_path = write_recipe(out_dir, step)
            step.done("stopped at the ONNX: no python here can import `afe`")
            console.warn(next_steps(onnx_path, recipe_path, sdk_candidates()))
            return 0
        step.detail(f"Model SDK: {sdk_python}")

        # Reported before the pack download, not after: 21 MB spent to
        # discover that the interpreter cannot run what is inside it is 21 MB
        # wasted, and the answer was known without spending any of it.
        if absent:
            step.done("stopped at the ONNX: the compile's own imports are not all here")
            console.warn(requirements_help(absent, sdk_python))
            return 0

        recipe_path = write_recipe(out_dir, step)
        if recipe_path is None:
            step.done("stopped at the ONNX: no pack to take a compile recipe from")
            console.warn(next_steps(onnx_path, None))
            return 0

        log_path = out_dir / COMPILE_LOG
        step.note(
            "quantizing to bfloat16, calibrating, tessellating for the MLA and\n"
            "emitting the ELF. Ten to fifteen minutes is normal."
        )
        step.note(f"every line below is the recipe's own, and all of it lands in {log_path}")
        pack = None
        # Two attempts at most, and the second only for the one failure afe
        # names a fix for. See `pin_batch_size`.
        for attempt in (1, 2):
            narration = Narration(step)
            try:
                pack = run_recipe(
                    recipe_path, onnx_path, out_dir,
                    on_line=narration.line, on_silence=narration.silence,
                    python=sdk_python,
                )
                break
            except RuntimeError:
                retry = (
                    attempt == 1
                    and needs_fixed_batch(log_path)
                    and pin_batch_size(recipe_path)
                )
                if not retry:
                    raise
                step.note(
                    "afe will not load this graph with the batch size left flexible:\n"
                    "its attention blocks reshape across the batch axis. Pinning it to 1\n"
                    "and running the compile again, which is what afe asked for."
                )
        step.detail(f"{narration.lines} lines of compiler output -> {log_path}")
        finish_pack(pack, step)
        step.done(f"{pack} ({human_bytes(pack.stat().st_size)})", timed=True)

    console.report(f"run it with:  sima-vision detect --model {pack.name}")
    console.report(f"send it over: sima-vision push {pack}")
    return 0


def reference_pack() -> Path | None:
    """A published pack, to copy out of. Any of them will do.

    Two things are taken from one: the compile recipe, and the pipeline files
    that the Model SDK's own output does not carry. Both are the same in every
    pack, so the first one on disk is as good as any.
    """
    packs = sorted(models_dir().glob("*.tar.gz"))
    return packs[0] if packs else None


def finish_pack(pack: Path, step) -> None:
    """Add the pipeline files the board reads, when the compile left them out.

    The Model SDK writes the ELF and the manifest. What the board's preprocess
    planner looks for first is `pipeline_sequence.json` and the two plugin
    configs beside it, and a pack without them fails on the board rather than
    here, a minute into a run, with a message about a missing MLA stage.
    """
    reference = reference_pack()
    if reference is None:
        step.note("no published pack here to check this one against")
        return
    added = complete_pack(pack, reference)
    if added:
        step.detail(f"added {', '.join(added)} from {reference.name}")


def write_recipe(out_dir: Path, step) -> Path | None:
    """Copy a published pack's own compile script next to the ONNX.

    Taken from a pack rather than written here, because the settings that
    matter -- bfloat16, MSE calibration, the MLA tessellation layouts -- are
    the ones SiMa actually shipped, and a paraphrase of them would drift.

    A pack is downloaded when there is none to read. It used to be fetched only
    by the caller that was about to compile, on the reasoning that 21 MB is not
    worth spending on a machine that was going to stop at the ONNX anyway. What
    that actually bought was the worst guidance of the three: a machine with no
    Model SDK printed "compile it with the Model SDK" instead of the exact
    command, because the recipe it would have named was the thing it had
    skipped. The pack is public -- a plain GET off the GitHub release, no login
    -- and it is cached, so it is a one-time cost either way.

    Args:
        out_dir: Where the recipe is written, beside the ONNX.
        step: The console step to report under.
    """
    pack = reference_pack()
    if pack is None:
        # Every pack carries the same recipe, so the smallest one will do.
        step.detail("no pack here to take a recipe from, fetching the nano one")
        try:
            ensure_model(default_model_path("detect"), "detect", step)
        except RuntimeError as exc:
            step.note(str(exc))
        pack = reference_pack()
    if pack is None:
        # Not a login problem, whatever the download said: the default packs
        # are on the public release. Something stopped the GET, and the only
        # other way to a recipe is a pack that is already on a machine.
        step.note(
            f"no model pack in {models_dir()} to copy a recipe from, and the recipe\n"
            "comes inside one. The download above is the usual way to get one; if it\n"
            "cannot reach the release from here, bring a pack over instead:\n"
            "  sima-vision push <any pack>       # from a machine that has one"
        )
        return None
    try:
        script = compile_recipe(pack)
    except (RuntimeError, OSError, tarfile.TarError) as exc:
        step.note(f"could not read a recipe from {pack.name}: {exc}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compile_modelsdk.py"
    path.write_text(script, encoding="utf-8")
    step.detail(f"recipe from {pack.name} -> {path}")
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
