"""Finishing a pack the Model SDK left half-described.

The bug this is about had no symptom on the machine that caused it. The
compile ran, printed a report, wrote a `.tar.gz`, and the pack was 35 MB of
perfectly good ELF. It failed on the board, in another building, a minute into
a run, at step 6 of 7:

    preprocess planner: MPK contract is missing an MLA stage for pre route
    selection. Expected a plugin with processor='MLA' or kernel='infer'/'mla'
    in the MPK manifest.

Every published pack carries `pipeline_sequence.json` and the two plugin
configs it points at, and that is where the planner looks. The fixtures here
are cut down from `yolo26n-det-bf16-mla_tess-b1.tar.gz` and
`yolo26n-seg-bf16-mla_tess.tar.gz`, so the numbers that get rewritten are the
numbers those packs actually carry.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from sima_vision import pack as packing

#: The det pack's own figures: MLA_0 writes 1478400 bytes, its ELF is named
#: after the model, and the final PassThrough hands back six tensors.
DET_ELF = "yolo26n_raw_supported_einsum_stage1_mla.elf"
DET_MLA_SIZE = 1478400
DET_OUTPUTS = 6

#: The caps in the det pack's 0_process_mla.json advertise seven groups for
#: those six outputs. The one extra is carried over rather than derived.
DET_CAPS = 7


def manifest(elf: str = DET_ELF, size: int = DET_MLA_SIZE,
             outputs: int = DET_OUTPUTS, processor: str = "MLA") -> dict:
    """A manifest the shape of a real one, down to the plugin names."""
    return {
        "name": "model",
        "plugins": [
            {"name": "cast_0", "processor": "EV74"},
            {
                "name": "MLA_0",
                "processor": processor,
                "output_nodes": [{"name": "MLA_0", "size": size}],
                "resources": {"executable": elf},
            },
            {
                "name": "PassThrough",
                "processor": "EV74",
                "output_nodes": [
                    {"name": f"pass_through_out_{i}"} for i in range(outputs)
                ],
            },
        ],
    }


def pipeline_template(elf: str = DET_ELF) -> dict:
    return {
        "pipelines": [
            {
                "name": "MLA_0",
                "sequence": [
                    {
                        "sequence_id": 1,
                        "name": "simaaiprocesspreproc_1",
                        "processor": "CVU",
                        "kernel": "preproc",
                        "configPath": packing.PREPROC,
                        "executable": None,
                        "input": "decoder",
                    },
                    {
                        "sequence_id": 2,
                        "name": "simaaiprocessmla_1",
                        "processor": "MLA",
                        "kernel": "mla",
                        "configPath": packing.MLA_CONFIG,
                        "executable": elf,
                        "input": "simaaiprocesspreproc_1",
                    },
                ],
                "complete": False,
            }
        ]
    }


def mla_template(elf: str = DET_ELF, size: int = DET_MLA_SIZE,
                 caps: int = DET_CAPS) -> dict:
    groups = ", ".join(["(1 - 4096)"] * caps)
    types = ", ".join(["(INT8, INT16, INT32)"] * caps)
    return {
        "version": 0.1,
        "simaai__params": {
            "outputs": [{"name": "mla_output_tensor", "size": size}],
            "next_cpu": 1,
            "model_path": elf,
        },
        "caps": {
            "sink_pads": [
                {"params": [{"name": "format", "values": "RGB, BGR"}]}
            ],
            "src_pads": [
                {
                    "params": [
                        {"name": "data_type", "values": types},
                        {"name": "width", "values": groups},
                        {"name": "height", "values": groups},
                    ]
                }
            ],
        },
    }


def build(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


def as_json(value) -> bytes:
    return json.dumps(value).encode("utf-8")


@pytest.fixture
def reference(tmp_path) -> Path:
    """A published pack: manifest, ELF, and the three files the board reads."""
    return build(tmp_path / "published.tar.gz", {
        "model_mpk.json": as_json(manifest()),
        DET_ELF: b"elf",
        packing.PIPELINE: as_json(pipeline_template()),
        packing.PREPROC: b'{"graph_name": "preproc", "output_width": 640}',
        packing.MLA_CONFIG: as_json(mla_template()),
    })


def compiled(tmp_path, **kw) -> Path:
    """What the Model SDK writes: the manifest, the ELF, and nothing else."""
    elf = kw.setdefault("elf", "best-raw_stage1_mla.elf")
    return build(tmp_path / "best-raw_mpk.tar.gz", {
        "best-raw_mpk.json": as_json(manifest(**kw)),
        elf: b"elf",
        "best-raw_stage1_mla_stats.yaml": b"stats:\n",
    })


def contents(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as tar:
        return {
            member.name: tar.extractfile(member).read()
            for member in tar.getmembers()
        }


def test_a_pack_from_the_sdk_is_missing_what_the_board_reads(tmp_path):
    assert packing.missing_files(compiled(tmp_path)) == list(packing.PIPELINE_FILES)


def test_the_three_files_are_added_and_nothing_else_is_lost(tmp_path, reference):
    pack = compiled(tmp_path)
    assert packing.complete_pack(pack, reference) == list(packing.PIPELINE_FILES)

    after = contents(pack)
    assert set(packing.PIPELINE_FILES) <= set(after)
    assert after["best-raw_stage1_mla_stats.yaml"] == b"stats:\n"
    assert after["best-raw_stage1_mla.elf"] == b"elf"


def test_the_pipeline_names_this_packs_own_elf(tmp_path, reference):
    """Copying the reference's ELF name over would load the wrong weights."""
    pack = compiled(tmp_path)
    packing.complete_pack(pack, reference)

    sequence = json.loads(contents(pack)[packing.PIPELINE])
    stages = sequence["pipelines"][0]["sequence"]
    mla = next(s for s in stages if s["processor"] == "MLA")
    assert mla["executable"] == "best-raw_stage1_mla.elf"
    assert mla["kernel"] == "mla"


def test_the_mla_config_carries_this_packs_own_output_size(tmp_path, reference):
    pack = compiled(tmp_path, elf="best-raw_stage1_mla.elf", size=3654400)
    packing.complete_pack(pack, reference)

    config = json.loads(contents(pack)[packing.MLA_CONFIG])
    params = config["simaai__params"]
    assert params["outputs"][0]["size"] == 3654400
    assert params["model_path"] == "best-raw_stage1_mla.elf"


def test_the_caps_grow_with_the_number_of_outputs(tmp_path, reference):
    """A segmentation pack emits ten tensors where the reference emits six.

    The published packs advertise one group more than they have outputs --
    seven for six, eleven for ten -- so the difference is taken from the
    reference rather than written down as a rule.
    """
    pack = compiled(tmp_path, elf="best-raw_stage1_mla.elf", outputs=10)
    packing.complete_pack(pack, reference)

    config = json.loads(contents(pack)[packing.MLA_CONFIG])
    for param in config["caps"]["src_pads"][0]["params"]:
        assert param["values"].count("(") == 11, param["name"]
    # The sink pad describes the image coming in, not the tensors going out.
    assert config["caps"]["sink_pads"][0]["params"][0]["values"] == "RGB, BGR"


def test_the_preproc_config_is_copied_untouched(tmp_path, reference):
    """It is the same bytes in every published pack, so it is not rewritten."""
    pack = compiled(tmp_path)
    packing.complete_pack(pack, reference)
    assert contents(pack)[packing.PREPROC] == contents(reference)[packing.PREPROC]


def test_a_pack_that_is_already_complete_is_left_alone(tmp_path, reference):
    before = reference.read_bytes()
    assert packing.complete_pack(reference, reference) == []
    assert reference.read_bytes() == before


def test_a_pack_with_no_mla_stage_says_what_it_got_instead(tmp_path, reference):
    """Nothing downstream can fix this one, so it is named rather than patched.

    A graph that would not map onto the accelerator is compiled for something
    else, and the pack that comes out cannot run on the board at all.
    """
    pack = compiled(tmp_path, processor="EV74")
    with pytest.raises(RuntimeError, match="no MLA stage"):
        packing.complete_pack(pack, reference)


def test_an_archive_with_no_manifest_is_not_a_pack(tmp_path, reference):
    pack = build(tmp_path / "empty.tar.gz", {"readme.txt": b"hello"})
    with pytest.raises(RuntimeError, match="not a model pack"):
        packing.complete_pack(pack, reference)


def test_a_reference_without_the_files_cannot_supply_them(tmp_path):
    """The recipe comes from any pack; these do not. Say which is missing."""
    thin = build(tmp_path / "thin.tar.gz", {"model_mpk.json": as_json(manifest())})
    with pytest.raises(RuntimeError, match=packing.PIPELINE):
        packing.complete_pack(compiled(tmp_path), thin)


def test_only_the_missing_files_are_added(tmp_path, reference):
    """A pack half-finished by a later SDK keeps whatever it already had."""
    pack = build(tmp_path / "half.tar.gz", {
        "half_mpk.json": as_json(manifest(elf="half_stage1_mla.elf")),
        "half_stage1_mla.elf": b"elf",
        packing.PREPROC: b"mine, not the reference's",
    })
    added = packing.complete_pack(pack, reference)

    assert added == [packing.PIPELINE, packing.MLA_CONFIG]
    assert contents(pack)[packing.PREPROC] == b"mine, not the reference's"


# -- a pipeline of the wrong shape --

def _pack_with(tmp_path, name, pipeline, extra=None):
    """A pack carrying a manifest, the pipeline files, and a given pipeline."""
    import io
    import json as _json

    manifest = {
        "plugins": [
            {"name": "MLA_0", "processor": "MLA",
             "resources": {"executable": "model_stage1_mla.elf"},
             "output_nodes": [{"size": 100}]},
            {"name": "PassThrough", "output_nodes": [{}] * 6},
        ]
    }
    bodies = {
        "model_mpk.json": _json.dumps(manifest).encode(),
        packing.PIPELINE: _json.dumps(pipeline).encode(),
        packing.PREPROC: b"{}",
        packing.MLA_CONFIG: _json.dumps({"simaai__params": {"outputs": [{"size": 1}]},
                                      "caps": {}}).encode(),
    }
    bodies.update(extra or {})
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        for member, body in bodies.items():
            info = tarfile.TarInfo(member)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


GOOD_PIPELINE = {"pipelines": [{"name": "MLA_0", "sequence": [
    {"processor": "CVU", "kernel": "preproc"},
    {"processor": "MLA", "kernel": "mla", "executable": "old.elf"},
]}]}

#: What Model SDK 2.1.3 writes for a graph it decides needs no preprocessing.
SDK_PIPELINE = {"use_preproc": False, "pipelines": [{"name": "MLA_0", "sequence": [
    {"processor": "MLA", "kernel": "mla", "executable": "model_stage1_mla.elf"},
    {"processor": "CVU", "kernel": "detessellate"},
]}]}


def test_a_pipeline_without_a_preproc_stage_is_unusable(tmp_path):
    """The board's planner routes through preproc. No preproc, no route.

    Model SDK 2.1.3 writes its own pipeline_sequence.json -- it did not write
    one at all when this module was first needed -- and for a graph it decides
    needs no preprocessing it writes `use_preproc: false` and a sequence of
    `[(MLA, mla), (CVU, detessellate)]`. The file is present and complete and
    the board fails on it exactly as it fails with no file at all.
    """
    sdk = _pack_with(tmp_path, "sdk.tar.gz", SDK_PIPELINE)
    assert packing.unusable_pipeline(sdk) is True


def test_a_published_shaped_pipeline_is_left_alone(tmp_path):
    """Replacing a working pipeline would be the same bug facing the other way."""
    good = _pack_with(tmp_path, "good.tar.gz", GOOD_PIPELINE)
    assert packing.unusable_pipeline(good) is False


def test_an_unreadable_pipeline_is_not_called_unusable(tmp_path):
    """Absent or corrupt is `missing_files`' question, and it has its own answer.

    Saying "unusable" here would replace a file on the strength of not having
    been able to read it, which is a guess dressed as a diagnosis.
    """
    import io

    path = tmp_path / "broken.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        body = b"{ not json"
        info = tarfile.TarInfo(packing.PIPELINE)
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    assert packing.unusable_pipeline(path) is False


def test_the_wrong_shape_is_replaced_from_a_published_pack(tmp_path):
    """And the replacement points at this pack's ELF, not the reference's."""
    import json as _json

    broken = _pack_with(tmp_path, "broken.tar.gz", SDK_PIPELINE)
    reference = _pack_with(tmp_path, "ref.tar.gz", GOOD_PIPELINE)

    assert packing.complete_pack(broken, reference) == [packing.PIPELINE]
    assert packing.unusable_pipeline(broken) is False

    with tarfile.open(broken) as tar:
        body = _json.loads(tar.extractfile(packing.PIPELINE).read().decode())
    stages = body["pipelines"][0]["sequence"]
    assert [s["kernel"] for s in stages] == ["preproc", "mla"]
    mla = next(s for s in stages if s["processor"] == "MLA")
    assert mla["executable"] == "model_stage1_mla.elf", "must name this pack's own ELF"


def test_a_complete_and_correct_pack_is_not_rewritten(tmp_path):
    """Nothing added, nothing touched -- the common case for a published packing."""
    good = _pack_with(tmp_path, "good.tar.gz", GOOD_PIPELINE)
    before = good.read_bytes()
    reference = _pack_with(tmp_path, "ref.tar.gz", GOOD_PIPELINE)
    assert packing.complete_pack(good, reference) == []
    assert good.read_bytes() == before
