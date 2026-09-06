"""Finishing a compiled pack so the board's pipeline can read it.

The Model SDK's compile writes three things into the archive: the MLA ELF, the
``*_mpk.json`` manifest describing the compiled graph, and a stats file. What
the board reads before any of that is a *fourth* thing, and a different shape
of thing -- ``pipeline_sequence.json``, which names the two stages a frame goes
through, and the plugin configs those stages point at::

    simaaiprocesspreproc_1   CVU   preproc   0_preproc.json
    simaaiprocessmla_1       MLA   mla       0_process_mla.json

A pack without them gets as far as being opened and then fails::

    preprocess planner: MPK contract is missing an MLA stage for pre route
    selection. Expected a plugin with processor='MLA' or kernel='infer'/'mla'
    in the MPK manifest.

which is the planner having found the manifest and not the pipeline. It happens
at step 6 of a run, on the board, a minute after the compile that caused it
finished on another machine entirely.

Every published pack carries all three files, so they are copied out of one
rather than composed here. ``0_preproc.json`` is the same bytes in every pack.
The other two are the same but for the ELF's name, the MLA output's size, and
one caps entry per model output -- three facts that are read back out of the
new pack's own manifest, so a detection pack gets six and a segmentation pack
gets ten without either number being written down.

Checked the only way it can be: rebuild the segmentation pack's two files from
the detection pack's, using nothing but the segmentation pack's own manifest,
and they come out identical to the ones it ships.
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

#: The manifest the Model SDK writes, matched by suffix: its stem is the model's.
MANIFEST_SUFFIX = "_mpk.json"

#: The pipeline the board reads, and the two plugin configs it points at.
PIPELINE = "pipeline_sequence.json"
PREPROC = "0_preproc.json"
MLA_CONFIG = "0_process_mla.json"
PIPELINE_FILES = (PIPELINE, PREPROC, MLA_CONFIG)

#: The plugin that ends the graph. Its outputs are the model's outputs, which
#: is how many entries the MLA config's caps have to advertise.
FINAL_PLUGIN = "PassThrough"


def read_manifest(tar: tarfile.TarFile) -> dict:
    """The ``*_mpk.json`` inside an open pack.

    Raises:
        RuntimeError: When the archive carries no manifest at all.
    """
    names = [n for n in tar.getnames() if n.endswith(MANIFEST_SUFFIX)]
    if not names:
        raise RuntimeError(
            f"this archive has no *{MANIFEST_SUFFIX}, so it is not a model pack."
        )
    handle = tar.extractfile(names[0])
    return json.loads(handle.read().decode("utf-8")) if handle else {}


def mla_plugin(manifest: dict) -> dict:
    """The one plugin that runs on the accelerator.

    Raises:
        RuntimeError: When there is none, which means the compile put the whole
            graph somewhere else. Nothing downstream can fix that, so it says
            what it found instead.
    """
    plugins = manifest.get("plugins", [])
    for plugin in plugins:
        if plugin.get("processor") == "MLA":
            return plugin
    found = sorted({str(p.get("processor")) for p in plugins}) or ["nothing"]
    raise RuntimeError(
        "this pack has no MLA stage: the compile mapped the graph onto "
        f"{', '.join(found)} instead.\n"
        "  The board runs the model on the MLA and cannot run this pack. It "
        "usually means an\n  operator in the ONNX has no MLA implementation "
        "and the compiler moved the graph off it."
    )


def output_count(manifest: dict) -> int:
    """How many tensors the pack hands back, read off its final plugin."""
    plugins = manifest.get("plugins", [])
    final = next(
        (p for p in plugins if p.get("name") == FINAL_PLUGIN),
        plugins[-1] if plugins else {},
    )
    return len(final.get("output_nodes", []))


def _retune_caps(caps: dict, wanted: int) -> None:
    """Repeat each parenthesised caps entry ``wanted`` times, in place.

    A src pad advertises one bracketed group per tensor it can emit, and the
    published packs differ only in how many: seven groups for six outputs, and
    eleven for ten. The group itself is copied rather than written, because one
    of them is ``(INT8, INT16, INT32)`` and contains the separator.
    """
    for pad in caps.get("src_pads", []):
        for param in pad.get("params", []):
            value = param.get("values", "")
            group = re.match(r"\([^)]*\)", value)
            if group:
                param["values"] = ", ".join([group.group()] * wanted)


def pipeline_json(template: dict, plugin: dict) -> str:
    """``pipeline_sequence.json`` for a pack, from a published one's."""
    elf = plugin.get("resources", {}).get("executable", "")
    for pipeline in template.get("pipelines", []):
        pipeline["name"] = plugin.get("name", pipeline.get("name"))
        for stage in pipeline.get("sequence", []):
            if stage.get("processor") == "MLA":
                stage["executable"] = elf
    return json.dumps(template, indent=2)


def mla_config_json(template: dict, plugin: dict, outputs: int,
                    reference_outputs: int) -> str:
    """``0_process_mla.json`` for a pack, from a published one's.

    Args:
        template: The published pack's own config, parsed.
        plugin: The new pack's MLA plugin, which carries the ELF's name and the
            size of the tensor it writes.
        outputs: How many tensors the new pack emits.
        reference_outputs: How many the template's pack emitted. The difference
            between that and the template's caps count is what gets carried
            over, rather than the caps count being derived from a rule.
    """
    params = template.get("simaai__params", {})
    params["model_path"] = plugin.get("resources", {}).get("executable", "")
    sizes = plugin.get("output_nodes", [])
    if params.get("outputs") and sizes:
        params["outputs"][0]["size"] = sizes[0].get("size")

    caps = template.get("caps", {})
    advertised = 0
    for pad in caps.get("src_pads", []):
        for param in pad.get("params", []):
            advertised = max(advertised, param.get("values", "").count("("))
    if advertised and reference_outputs:
        _retune_caps(caps, outputs + advertised - reference_outputs)
    return json.dumps(template, indent=2)


def missing_files(pack: Path) -> list[str]:
    """Which of the pipeline files a pack does not have."""
    with tarfile.open(pack) as tar:
        have = set(tar.getnames())
    return [name for name in PIPELINE_FILES if name not in have]


def complete_pack(pack: Path, reference: Path) -> list[str]:
    """Add the pipeline files a pack is missing, taken from ``reference``.

    Args:
        pack: The pack the compile produced. Rewritten in place when anything
            is added, and left untouched when nothing is.
        reference: A published pack to copy the files out of.

    Returns:
        The names added, empty when the pack was already complete.

    Raises:
        RuntimeError: When the pack has no MLA stage to describe, or the
            reference carries no pipeline files to copy.
    """
    missing = missing_files(pack)
    if not missing:
        return []

    with tarfile.open(pack) as tar:
        manifest = read_manifest(tar)
    plugin = mla_plugin(manifest)
    outputs = output_count(manifest)

    with tarfile.open(reference) as ref:
        names = set(ref.getnames())
        absent = [name for name in PIPELINE_FILES if name not in names]
        if absent:
            raise RuntimeError(
                f"{reference.name} has no {', '.join(absent)} to copy, so it is "
                "not a pack this can\n  take a pipeline from."
            )
        bodies = {
            name: ref.extractfile(name).read()  # type: ignore[union-attr]
            for name in PIPELINE_FILES
        }
        reference_outputs = output_count(read_manifest(ref))

    made = {PREPROC: bodies[PREPROC]}
    made[PIPELINE] = pipeline_json(
        json.loads(bodies[PIPELINE]), plugin
    ).encode("utf-8")
    made[MLA_CONFIG] = mla_config_json(
        json.loads(bodies[MLA_CONFIG]), plugin, outputs, reference_outputs
    ).encode("utf-8")

    add_to_pack(pack, {name: made[name] for name in missing})
    return missing


def add_to_pack(pack: Path, files: dict[str, bytes]) -> None:
    """Rewrite a ``.tar.gz`` with extra members. gzip cannot be appended to."""
    import io

    temp = pack.parent / (pack.name + ".tmp")
    with tarfile.open(pack) as old, tarfile.open(temp, "w:gz") as new:
        for member in old.getmembers():
            handle = old.extractfile(member) if member.isfile() else None
            new.addfile(member, handle)
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o644
            new.addfile(info, io.BytesIO(body))
    temp.replace(pack)
