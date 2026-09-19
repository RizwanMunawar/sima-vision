"""Turning a trained YOLO26 ``.pt`` into something the DevKit can run.

Two stages, and only the first can happen on an ordinary machine.

**Export.** Ultralytics' own ONNX export ends in the decode: one
``[1, 84, 8400]`` tensor with the boxes already assembled. The board does that
part itself, in ``neatobjectdecode``, and it expects the six raw head tensors
instead -- ``Configured for subtensors: 6`` in a run's log is this. So the
export here stops at the head and emits what the head produces:

===============  ==================  =========================
output           shape at 640x640    from
===============  ==================  =========================
``bbox_0``       ``[1, 4, 80, 80]``  ``Detect.cv2[0]``
``bbox_1``       ``[1, 4, 40, 40]``  ``Detect.cv2[1]``
``bbox_2``       ``[1, 4, 20, 20]``  ``Detect.cv2[2]``
``class_logit_0``  ``[1, 80, 80, 80]``  ``Detect.cv3[0]``
``class_logit_1``  ``[1, 80, 40, 40]``  ``Detect.cv3[1]``
``class_logit_2``  ``[1, 80, 20, 20]``  ``Detect.cv3[2]``
===============  ==================  =========================

A segmentation head adds four more, from the same place: ``mask_coeff_0..2``
off ``Segment.cv4``, 32 coefficients a level, and ``mask_proto`` off
``Segment.proto`` at a quarter of the input side. ``Segment`` subclasses
``Detect``, so a seg model exported as a detector passes every check here and
produces six of the ten, which compiles into a pack that draws boxes and no
masks.

Which *branch* those come off matters as much as which tensors. A YOLO26 head
is end2end and carries two full sets: ``cv2``/``cv3``/``cv4``, the one2many
branch that exists to supervise training and that ``fuse()`` deletes, and
``one2one_cv2``/``one2one_cv3``/``one2one_cv4``, which is what a prediction is
actually made of. They are the same shape, so taking the wrong one is invisible
until the detections are compared against the model they came from.

Those names and that order are not invented here. They are read out of a
working pack's own ``*_mpk.json``, where the final PassThrough carries exactly
``bbox_0..2``, ``class_logit_0..2``, ``mask_coeff_0..2``, ``mask_proto``. Four
box channels rather than 64 is YOLO26 having ``reg_max = 1``: no DFL to unpack.

**The pipeline files.** The Model SDK's compile writes the ELF and the manifest
and stops. What the board reads first is neither: see :mod:`sima_vision.pack`.

**Compile.** ONNX to ``.tar.gz`` is the SiMa Model SDK's job -- quantization to
bfloat16, MLA tessellation, and the ELF. That is the ``afe`` package inside the
Palette container, on x86, and it is not on the DevKit and not on most laptops.
Every published pack ships the exact script that built it, as
``archived_compile_script.*.py``, so :func:`compile_recipe` hands that same
recipe back rather than paraphrasing it.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

#: Output names the board's box decoder expects, in the order a working pack's
#: PassThrough lists them: every box tensor, then every class tensor.
BBOX_OUTPUTS = ("bbox_0", "bbox_1", "bbox_2")
CLASS_OUTPUTS = ("class_logit_0", "class_logit_1", "class_logit_2")
RAW_OUTPUTS = (*BBOX_OUTPUTS, *CLASS_OUTPUTS)

#: What a segmentation head adds, again in the order its pack lists them: the
#: per-level mask coefficients, then the one prototype tensor they weight.
MASK_OUTPUTS = ("mask_coeff_0", "mask_coeff_1", "mask_coeff_2")
PROTO_OUTPUT = "mask_proto"
SEG_OUTPUTS = (*RAW_OUTPUTS, *MASK_OUTPUTS, PROTO_OUTPUT)

#: The prototype masks come out at a quarter of the input side: 160 at 640.
PROTO_STRIDE = 4

#: The input the preprocess contract feeds: one RGB image, letterboxed square.
INPUT_NAME = "images"
DEFAULT_IMGSZ = 640

#: ONNX opset. 17 is what the SDK's importer is happiest with, and it is late
#: enough for everything a YOLO26 graph uses.
DEFAULT_OPSET = 17

#: What the export needs, and what installs each. ``onnx`` is the one that
#: gets missed: torch only reaches for it at the *end* of the trace, so
#: without an up-front check a missing one surfaces a minute in, from inside
#: torch's exporter, reported as though the model were at fault. The Palette
#: container has torch and ultralytics and does not have this.
EXPORT_REQUIREMENTS = {
    "torch": "torch",
    "ultralytics": "ultralytics",
    "onnx": "onnx",
}


def missing_requirements() -> list[str]:
    """Which of the export's imports are not installed here.

    Asked of the import system rather than by importing them: this runs
    before torch is loaded, which is itself several seconds, and the answer
    is wanted before that rather than after.
    """
    import importlib.util

    missing = []
    for module, package in EXPORT_REQUIREMENTS.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # a half-installed package
            found = False
        if not found:
            missing.append(package)
    return missing


#: Name of the compile script inside a published pack.
RECIPE_PREFIX = "archived_compile_script."


class RawHead:
    """Wraps a DetectionModel so its head returns the raw tensors.

    ``Detect.forward`` concatenates each level's box and class branches and
    then decodes them. Both have to go: the concatenation because the board
    wants the branches apart, and the decode because the board does it.
    Replacing the method on the instance is enough -- ``DetectionModel``
    reaches the head through the module list, so the rest of the network is
    untouched and no weights move.
    """

    def __init__(self, net) -> None:
        self.net = net
        self.head = net.model[-1]
        self.branches = head_branches(self.head)
        self.masks = mask_channels(self.head)

    def outputs(self, feats: list) -> list:
        """Every tensor the head produces, in the order the pack expects."""
        head = self.head
        boxes, classes, coefficients = self.branches
        tensors = [boxes[i](feats[i]) for i in range(head.nl)]
        tensors += [classes[i](feats[i]) for i in range(head.nl)]
        if self.masks:
            tensors += [coefficients[i](feats[i]) for i in range(head.nl)]
            tensors.append(proto_tensor(head, feats))
        return tensors

    def __enter__(self):
        self.original = self.head.forward
        self.head.forward = self.outputs
        return self

    def __exit__(self, *exc) -> bool:
        self.head.forward = self.original
        return False


def head_branches(head) -> tuple:
    """The three branches a prediction actually comes out of.

    YOLO26 heads are end2end, and carry two complete sets. ``cv2``/``cv3``/
    ``cv4`` are the one2many branch: supervision during training, and the first
    thing ``fuse()`` deletes for inference. What ``Detect.forward`` runs to
    produce a prediction is ``one2one_cv2``/``one2one_cv3``, and for a
    segmentation head ``one2one_cv4`` as well.

    They have identical shapes, which is what makes this worth a function.
    Exporting the wrong one produces an ONNX that checks out, compiles, loads
    on the board and quietly detects worse than the model it was built from.
    """
    if getattr(head, "one2one_cv2", None) is not None:
        return (
            head.one2one_cv2,
            head.one2one_cv3,
            getattr(head, "one2one_cv4", None),
        )
    return (
        getattr(head, "cv2", None),
        getattr(head, "cv3", None),
        getattr(head, "cv4", None),
    )


def proto_tensor(head, feats: list):
    """The prototype masks, from whichever Proto module the head carries.

    YOLO26's ``Proto26`` refines and sums all three levels, so it takes the
    whole list. The older ``Proto`` takes the finest level on its own. Both are
    called ``proto``, so the module is asked which it is: handing the list to
    the old one indexes a tensor by 1 and reports a size that has nothing to do
    with the mistake.
    """
    if hasattr(head.proto, "feat_refine"):
        return head.proto(feats)
    return head.proto(feats[0])


def mask_channels(head) -> int:
    """How many mask coefficients a head emits, or 0 when it emits none.

    ``Segment`` subclasses ``Detect``, so every check below passes for one and
    the boxes it exports are right. What is not right is stopping there: the
    head also has ``cv4``, a coefficient branch per level, and ``proto``, and a
    pack built without them decodes boxes and no masks. Asked of the head
    rather than of the file name, because a ``-seg`` in the name is not what
    makes it one.
    """
    if head_branches(head)[2] is None or getattr(head, "proto", None) is None:
        return 0
    return int(getattr(head, "nm", 0))


def check_head(net) -> tuple[int, int, int]:
    """Refuse a model whose head cannot produce what the board decodes.

    Returns:
        A ``(levels, classes, masks)`` triple. ``masks`` is 0 for a detection
        head and the coefficient count for a segmentation one.

    Raises:
        RuntimeError: When the head is not a three-level YOLO26 head.
    """
    head = getattr(net, "model", [None])[-1]
    for attribute in ("nl", "nc"):
        if not hasattr(head, attribute):
            raise RuntimeError(
                f"this is not a YOLO detection or segmentation model: its head "
                f"is {type(head).__name__},\n  which has no {attribute}. Pose "
                "and OBB heads have a different box decoder."
            )
    if any(branch is None for branch in head_branches(head)[:2]):
        raise RuntimeError(
            f"this head has no box or class branch to export: {type(head).__name__} "
            "has neither\n  cv2/cv3 nor one2one_cv2/one2one_cv3."
        )
    if head.nl != len(BBOX_OUTPUTS):
        raise RuntimeError(
            f"this head has {head.nl} levels and the board's decoder is built "
            f"for {len(BBOX_OUTPUTS)}."
        )
    reg_max = getattr(head, "reg_max", 1)
    if reg_max != 1:
        raise RuntimeError(
            f"this head has reg_max={reg_max}, so its box branch emits "
            f"{reg_max * 4} channels of DFL bins\n  rather than 4 coordinates. "
            "The board's decoder reads 4. That is a YOLOv8-style head,\n"
            "  not YOLO26."
        )
    return head.nl, head.nc, mask_channels(head)


def expected_shapes(imgsz: int, classes: int,
                    masks: int = 0) -> dict[str, tuple[int, ...]]:
    """What each output should come out as, for checking the export.

    Insertion order is the order the pack's final PassThrough lists them in,
    which is the order the export writes: every box tensor, every class tensor,
    then the mask coefficients and the prototypes they weight.
    """
    sides = [imgsz // stride for stride in (8, 16, 32)]
    shapes = {name: (1, 4, side, side) for name, side in zip(BBOX_OUTPUTS, sides, strict=True)}
    shapes.update(
        {name: (1, classes, side, side) for name, side in zip(CLASS_OUTPUTS, sides, strict=True)}
    )
    if masks:
        shapes.update(
            {name: (1, masks, side, side) for name, side in zip(MASK_OUTPUTS, sides, strict=True)}
        )
        proto = imgsz // PROTO_STRIDE
        shapes[PROTO_OUTPUT] = (1, masks, proto, proto)
    return shapes


def legacy_exporter_kwargs(torch) -> dict:
    """``dynamo=False`` where the installed torch understands it, else nothing.

    torch 2.6 made the dynamo exporter the default, and it renames outputs. The
    board's decoder reads its tensors *by name*, so the TorchScript exporter is
    the one that has to run. ``dynamo=False`` says so.

    Older torch has no such keyword and rejects it outright:
    ``export() got an unexpected keyword argument 'dynamo'``, which is what the
    Palette Model SDK container gives. It does not need telling either, since
    the exporter it has is the one we want. So the argument is passed only where
    it means something, which is asked of the signature rather than guessed from
    a version string.
    """
    import inspect

    try:
        accepted = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C-implemented export
        return {}
    return {"dynamo": False} if "dynamo" in accepted else {}


def export_failure(exc: Exception, head) -> RuntimeError:
    """An export that died inside the model, with the frame it died in.

    Tracing runs the network, so a mistake in the head arrives as whatever
    torch raised at the bottom of it: ``index 1 is out of bounds for dimension
    0 with size 1`` names neither the module nor the line. The last frame does,
    and it is the one worth printing.
    """
    import traceback

    frames = traceback.extract_tb(exc.__traceback__)
    where = ""
    if frames:
        frame = frames[-1]
        where = (
            f"\n  at {Path(frame.filename).name}:{frame.lineno} in {frame.name}"
            f"\n    {(frame.line or '').strip()}"
        )
    return RuntimeError(
        f"the export failed inside the model: {type(exc).__name__}: {exc}"
        f"\n  its head is {type(head).__name__}.{where}"
    )


def export_onnx(weights: Path, out: Path, imgsz: int = DEFAULT_IMGSZ,
                opset: int = DEFAULT_OPSET,
                on_line=None) -> dict[str, tuple[int, ...]]:
    """Write ``weights`` out as a raw-head ONNX at ``out``.

    Args:
        weights: A trained ``.pt``.
        out: Where to write the ONNX.
        imgsz: Square input side. The preprocess contract letterboxes to this.
        opset: ONNX opset version.
        on_line: Called with a short line before each slow part. Importing
            torch, loading the weights and tracing the graph are seconds to a
            minute each, and which of the three is running is the difference
            between a slow export and a stuck one.

    Returns:
        The output name to shape mapping actually produced.

    Raises:
        RuntimeError: When torch or ultralytics is missing, the head is not one
            this can export, or the shapes come out wrong.
    """
    def say(text: str) -> None:
        if on_line is not None:
            on_line(text)

    missing = missing_requirements()
    if missing:
        needs = ", ".join(EXPORT_REQUIREMENTS.values())
        verb = "is" if len(missing) == 1 else "are"
        raise RuntimeError(
            f"exporting a .pt needs {needs}, and {', '.join(missing)} {verb} not "
            f"installed here.\n"
            f"  pip install {' '.join(missing)}\n"
            "  This is a step for your PC or the Palette container, not the DevKit."
        )

    say(f"importing {', '.join(EXPORT_REQUIREMENTS)}")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            # The preflight above catches this in the ordinary case. This is
            # for the one it cannot see: a package that is importable but
            # broken, whose spec is found and whose import still fails.
            f"exporting a .pt needs {', '.join(EXPORT_REQUIREMENTS.values())}, and\n"
            f"importing {exc.name} failed: {exc}\n"
            f"  pip install --force-reinstall {exc.name}\n"
            "  This is a step for your PC or the Palette container, not the DevKit."
        ) from exc

    say(f"loading {weights.name}")
    net = YOLO(str(weights)).model.eval()
    levels, classes, masks = check_head(net)
    head = type(net.model[-1]).__name__
    say(f"head {head}: {levels} levels, {classes} classes, "
        + (f"{masks} mask coefficients" if masks else "no mask branch"))
    wanted = expected_shapes(imgsz, classes, masks)
    say(f"tracing at {imgsz}x{imgsz}, opset {opset}, {len(wanted)} raw outputs")

    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    try:
        with RawHead(net), torch.no_grad():
            torch.onnx.export(
                net,
                dummy,
                str(out),
                input_names=[INPUT_NAME],
                output_names=list(wanted),
                opset_version=opset,
                do_constant_folding=True,
                **legacy_exporter_kwargs(torch),
            )
    except Exception as exc:
        raise export_failure(exc, net.model[-1]) from exc

    say("checking every output against what the board's decoder reads")
    got = onnx_output_shapes(out)
    wrong = {
        name: (wanted[name], got.get(name))
        for name in wanted
        if got.get(name) != wanted[name]
    }
    if wrong:
        raise RuntimeError(
            "the export produced shapes the board's decoder cannot read:\n"
            + "\n".join(f"  {n}: wanted {w}, got {g}" for n, (w, g) in wrong.items())
        )
    return got


def onnx_output_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Every graph output's name and static shape."""
    import onnx

    model = onnx.load(str(path))
    shapes = {}
    for node in model.graph.output:
        dims = tuple(d.dim_value for d in node.type.tensor_type.shape.dim)
        shapes[node.name] = dims
    return shapes


def compile_recipe(pack: Path) -> str:
    """The compile script a published pack was built with.

    Raises:
        RuntimeError: When the pack carries no archived script.
    """
    with tarfile.open(pack) as tar:
        names = [n for n in tar.getnames() if Path(n).name.startswith(RECIPE_PREFIX)]
        if not names:
            raise RuntimeError(
                f"{pack} carries no {RECIPE_PREFIX}*.py, so there is no recipe "
                "to copy.\n  Any of the published packs has one."
            )
        handle = tar.extractfile(names[0])
        return handle.read().decode("utf-8") if handle else ""


#: Where a finished pack lands, relative to the build directory the recipe is
#: given. The archived scripts all end by writing `<name>_mpk.tar.gz`.
PACK_GLOB = "**/*.tar.gz"


#: How long the compile may go without a word before the run says it is still
#: alive. afe's quantization phase is the quiet one: several minutes of nothing
#: is indistinguishable from a hang, and a hang is what people assume.
SILENCE_HEARTBEAT = 30.0

#: Lines of the recipe's own output kept back for the error message. The useful
#: part of an afe failure is the traceback it ends on, not where it began.
TAIL_LINES = 40

#: Where the recipe's full output is written, under the build directory. Kept
#: whatever happens: a compile that worked is worth reading afterwards, and a
#: compile that failed is worth pasting into an issue.
COMPILE_LOG = "compile.log"


def run_recipe(recipe: Path, onnx: Path, build_dir: Path,
               timeout: int = 3600, on_line=None, on_silence=None,
               log_path: Path | None = None,
               python: str | None = None) -> Path:
    """Run a pack's own compile script on an ONNX, and return the pack it built.

    The script is SiMa's, shipped inside the pack for exactly this, and it takes
    ``--model`` and ``--build-dir``. Running it rather than reimplementing it is
    the point: the settings that matter -- bfloat16, MSE calibration, the MLA
    tessellation layouts -- are the ones that produced a pack known to work, and
    a paraphrase of them would drift the first time SiMa changed one.

    Its output is *streamed*, not collected. A compile takes ten to fifteen
    minutes, and the version of this that captured it printed nothing for all of
    them and then threw the whole lot away on success. Every line now goes to
    ``on_line`` as it arrives and to the log either way.

    Args:
        recipe: The ``archived_compile_script.*.py`` written beside the ONNX.
        onnx: The raw-head ONNX to compile.
        build_dir: Where the recipe should put its output.
        timeout: Seconds to allow. Quantization is slow, so this is generous.
        on_line: Called with each line of the recipe's output, without its
            newline, as it is produced.
        on_silence: Called with the seconds elapsed so far, every
            :data:`SILENCE_HEARTBEAT` seconds in which the recipe said nothing.
        log_path: Where to write the full output. Defaults to
            :data:`COMPILE_LOG` inside *build_dir*.
        python: The interpreter to run the recipe with. Defaults to this
            one. It is the SDK that has to be importable to the recipe, not
            to us, so this is what lets a sima-vision installed in one
            virtualenv compile with the SDK in another.

    Returns:
        The pack the recipe produced.

    Raises:
        RuntimeError: When the recipe fails, times out, or finishes without a
            pack. The log's path is named in all three.
    """
    import os
    import queue as queue_lib
    import subprocess
    import sys
    import threading
    from collections import deque

    # Absolute, every one of them. The recipe runs with cwd set to its own
    # directory, so a relative `build/compile_modelsdk.py` resolved against
    # `build/` and the interpreter was handed `build/build/...`, which is not
    # there. The same doubling applied to --model and --build-dir.
    recipe, onnx, build_dir = (p.resolve() for p in (recipe, onnx, build_dir))

    before = {p.resolve() for p in build_dir.glob(PACK_GLOB)}
    build_dir.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path is not None else build_dir / COMPILE_LOG

    process = subprocess.Popen(  # noqa: S603
        # `-u` and PYTHONUNBUFFERED both, because they catch different halves:
        # the flag unbuffers this interpreter's own streams, the variable is
        # what any interpreter afe spawns underneath reads. Without them the
        # child's stdout is a pipe, which Python block-buffers at 8 KB, and a
        # quarter of an hour of progress arrives in one burst at the end --
        # which is the same as not streaming it at all.
        [python or sys.executable, "-u", str(recipe), "--model", str(onnx),
         "--build-dir", str(build_dir)],
        cwd=recipe.parent,
        stdout=subprocess.PIPE,
        # Merged rather than kept apart: afe writes progress to one and warnings
        # to the other, and interleaved in the order they happened is the only
        # way to see which step a warning belongs to.
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # A reader thread and a queue, rather than iterating the pipe here: reading
    # it blocks, and a blocked reader cannot also notice that nothing has been
    # said for two minutes. The thread only ever moves lines; every print still
    # happens on this one, so nothing interleaves mid-line.
    lines: queue_lib.Queue = queue_lib.Queue()

    def pump() -> None:
        try:
            for line in process.stdout:  # type: ignore[union-attr]
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=pump, name="compile-output", daemon=True)
    reader.start()

    tail: deque[str] = deque(maxlen=TAIL_LINES)
    started = time.monotonic()
    deadline = started + timeout
    timed_out = False

    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(
            f"$ {python or sys.executable} {recipe} --model {onnx} "
            f"--build-dir {build_dir}\n"
        )
        while True:
            try:
                line = lines.get(timeout=SILENCE_HEARTBEAT)
            except queue_lib.Empty:
                if time.monotonic() >= deadline:
                    timed_out = True
                    process.kill()
                    break
                if on_silence is not None:
                    on_silence(time.monotonic() - started)
                continue
            if line is None:
                break
            text = line.rstrip("\r\n")
            handle.write(text + "\n")
            tail.append(text)
            if on_line is not None:
                on_line(text)
            if time.monotonic() >= deadline:
                timed_out = True
                process.kill()
                break

    process.wait()
    reader.join(timeout=5)

    if timed_out:
        raise RuntimeError(
            f"the compile recipe ran past its {timeout // 60} minute limit and was "
            f"stopped.\n  Everything it said is in {log}."
        )

    made = sorted(
        (p for p in build_dir.glob(PACK_GLOB) if p.resolve() not in before),
        key=lambda p: p.stat().st_mtime,
    )
    if process.returncode != 0:
        last = [text for text in tail if text.strip()][-6:]
        raise RuntimeError(
            f"the compile recipe exited {process.returncode}.\n  "
            + "\n  ".join(last)
            + f"\n\n  The whole of it is in {log}."
        )
    if not made:
        raise RuntimeError(
            f"the recipe finished but produced no .tar.gz under {build_dir}.\n"
            f"  It exited 0, so it believes it worked. What it actually did is in\n"
            f"  {log}; the pack is what this was for."
        )
    return made[-1]


#: A python to run the compile with, when the one running this cannot. Set it
#: to any interpreter that can import the Model SDK.
MODEL_SDK_PYTHON_ENV = "SIMA_VISION_MODEL_SDK_PYTHON"

#: The Model SDK's top-level module. Importable only inside the container.
SDK_MODULE = "afe"

#: Asked of a candidate interpreter. `find_spec` rather than a plain import:
#: the SDK is heavy, this runs up to four times, and whether it is *there* is
#: the whole question.
SDK_PROBE = (
    "import importlib.util, sys; "
    f"sys.exit(0 if importlib.util.find_spec({SDK_MODULE!r}) else 1)"
)


def sdk_candidates() -> list[str]:
    """Interpreters that might be able to run a compile, best guess first.

    The recipe runs as a subprocess, so the Model SDK never has to be
    importable *here* -- only in whichever python runs it. That makes the
    question "is there a python on this machine that has afe", not "can I
    import afe", and in the container those are routinely different answers:
    `pip install sima-vision` into one virtualenv and `activate-model-compiler`
    switching to another is all it takes. Asking the narrow question is how a
    machine that could finish the job reported that it could not.
    """
    import os
    import shutil
    import sys

    venv = os.environ.get("VIRTUAL_ENV")
    guesses = [
        os.environ.get(MODEL_SDK_PYTHON_ENV),
        sys.executable,
        # What `activate-model-compiler` activates, which is the point of
        # running it: the SDK is in the virtualenv it switches to.
        str(Path(venv) / "bin" / "python") if venv else None,
        shutil.which("python3"),
        shutil.which("python"),
    ]
    seen: set[str] = set()
    candidates: list[str] = []
    for python in guesses:
        if not python:
            continue
        # Resolved for the comparison only. Two names for one interpreter is
        # the normal case here -- `python`, `python3` and the venv's own are
        # usually the same file -- and probing it three times is three imports
        # of a heavy package to learn one thing.
        try:
            key = str(Path(python).resolve())
        except OSError:  # pragma: no cover - an unreadable path is not a python
            key = python
        if key not in seen:
            seen.add(key)
            candidates.append(python)
    return candidates


def has_model_sdk(python: str) -> bool:
    """Whether *python* can import the Model SDK."""
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603
            [python, "-c", SDK_PROBE],
            capture_output=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def model_sdk_python() -> str | None:
    """A python that can compile, or None when this machine has none."""
    for python in sdk_candidates():
        if has_model_sdk(python):
            return python
    return None


def model_sdk_present() -> bool:
    """Whether anything here can run a compile."""
    return model_sdk_python() is not None


def next_steps(onnx_path: Path, recipe_path: Path | None) -> str:
    """What to do with the ONNX, when this machine cannot finish the job.

    The Model SDK quantizes to bfloat16, tessellates for the MLA and emits the
    ELF. It lives in the Palette container on x86 and is not installable here,
    so the honest thing is to hand over the ONNX, the exact recipe, and the two
    commands -- rather than to fail at the last step with a stack trace.
    """
    recipe = (
        f"  3. Compile, with the recipe written beside it:\n"
        f"       python {recipe_path.name} --model {onnx_path.name} --build-dir build\n"
        if recipe_path
        else "  3. Compile it with the Model SDK.\n"
    )
    return (
        "the ONNX is as far as this machine goes. The .tar.gz needs the SiMa "
        "Model SDK,\n"
        "  which quantizes to bfloat16, tessellates for the MLA and emits the "
        "ELF. That is\n"
        "  the `afe` package inside the Palette container, on x86 -- not on the "
        "DevKit, and\n"
        "  not installable beside this one. To see which you are on:\n"
        '       python -c "import afe"\n'
        "\n"
        f"  1. Start Palette, and mount the directory holding {onnx_path.name}.\n"
        "  2. Inside it, install what the recipe imports:\n"
        '       pip install "sima-vision[compile]"\n'
        f"{recipe}"
        "  4. Or simply run this command again in there: with the SDK importable "
        "it does\n     every step and writes the pack itself.\n"
        "  5. Then bring the pack back and run it:\n"
        "       sima-vision push build/best_mpk.tar.gz\n"
        "       sima-vision detect --model best_mpk.tar.gz"
    )
