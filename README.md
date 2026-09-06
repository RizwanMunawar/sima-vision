<div align="center">

<img src="assets/sima-devkit-docs-logo-home.jpg" alt="sima-vision: live YOLO computer vision on a SiMa Modalix DevKit 3.0">

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix_DevKit_3.0-E63946)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-FF8C00)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-800080)](https://docs.sima.ai)

[![CI](https://github.com/RizwanMunawar/sima-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/RizwanMunawar/sima-projects/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip_install-sima--vision-3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/sima-vision/)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-6C757D)](LICENSE)
[![YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-FFB703&labelColor=333)](https://github.com/ultralytics/ultralytics)

[![Fall detection](https://img.shields.io/badge/Fall-detection-111F68)](https://github.com/ultralytics/ultralytics)
[![Segmentation and blur](https://img.shields.io/badge/Segmentation-blur-FF64DA)](https://github.com/ultralytics/ultralytics)
[![Object detection](https://img.shields.io/badge/Object-detection-042AFF)](https://github.com/ultralytics/ultralytics)

</div>

**Live YOLO26 on the MLA of a SiMa.ai Modalix DevKit 3.0.** Object detection, instance
segmentation and fall detection: three apps, one pipeline, no setup step.

Needs a [Modalix DevKit 3.0](https://devkit.sima.ai/products/development-kit-3-0) and Python 3.10 or later. Everything below runs **on the
board** unless it says otherwise.

## Quickstart

### 1 &middot; Install the SiMa.ai Neat Core

```bash
sima-cli login
sima-cli neat install core@v0.3.0
```

### 2 &middot; Install sima-vision

```bash
pip install sima-vision
```

### 3 &middot; Run it

**Nothing to download first.** Each app fetches its own pretrained YOLO26 pack and a demo
clip into `./assets` on its first run, then runs. Every run after that reuses them.

| App | Fetched for you | Writes |
|:--|:--|:--|
| `detect` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` (20 MB) + a 1080p demo clip (13 MB) | `detections.mp4`, `frames/` |
| `segment` | `yolo26n-seg-bf16-mla_tess.tar.gz` (23 MB) + the same clip | `segmentation.mp4`, `frames/` |
| `fall` | the detection pack again + a shorter clip (1.2 MB) | `falls.mp4`, `frames/`, `alerts/` |

Packs and clips come from one public GitHub release, so no login is involved and
`sima-cli` is not needed to get them. Each download is checked against its published
SHA-256 and discarded if it does not match. Each app fetches only what it needs.

```bash
sima-vision detect
```

<details>
<summary><b>Instance segmentation</b> &nbsp;&middot;&nbsp; per-pixel masks, with an optional blur</summary>

```bash
sima-vision segment
sima-vision segment --blur
sima-vision segment --blur --keep-classes person
```

</details>

<details>
<summary><b>Fall detection</b> &nbsp;&middot;&nbsp; tracks people, with optional email alerts</summary>

```bash
sima-vision fall
sima-vision fall --alert-to ops@example.com
```

Nothing is emailed until you pass `--send`; without it a fall is composed and logged so
you can see what would have gone out.

</details>

> [!TIP]
> **That is the whole setup.** No setup command, no config file, nothing to download by
> hand. A run finds the Neat runtime, puts the board's numpy and OpenCV on the path,
> fetches whatever is missing, and says what it is doing at every stage.
>
> Everything below is optional. Take what you need.

## Moving files between the board and your PC

Output lands beside the run, on the board. Name the board once and neither command needs
`--host`:

```bash
export SIMA_VISION_DEVKIT=sima@<devkit-ip>      # Linux, macOS
$env:SIMA_VISION_DEVKIT="sima@<devkit-ip>"      # Windows PowerShell
```

```bash
sima-vision pull                   # DevKit -> host, whatever the run left
sima-vision pull --into results/   # ...into a directory of your choosing
sima-vision push my-clip.h264      # host -> DevKit
```

## Use a different model

Two published sizes, fetched by name. Nano is the default; small is more accurate:

| Pack | Size | For |
|:--|:--|:--|
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | 20 MB | `detect`, `fall` &mdash; the default |
| `yolo26n-seg-bf16-mla_tess.tar.gz` | 23 MB | `segment` &mdash; the default |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | 35 MB | `detect`, `fall` |
| `yolo26s-seg-bf16-mla_tess.tar.gz` | 39 MB | `segment` |

```bash
sima-vision detect --model yolo26s-det-bf16-mla_tess-b1.tar.gz
```

No path and no URL: a published name is downloaded into `assets/models` and reused.

### Your own model

Train a YOLO26 detector, then convert it **inside the Palette Model SDK container** --
that is where the compiler lives, and it is four commands to get there:

```bash
# On your PC, once: mount the folder holding best.pt into the SDK containers
sima-cli sdk setup --workspace ~/workspace

# Then, any time: open the Model SDK shell
sima-cli sdk model
```

```bash
# Inside that shell, in ~/workspace
pip install sima-vision ultralytics
sima-vision compile best.pt          # -> build/best_mpk.tar.gz
```

That is the whole conversion. It exports the raw-head ONNX the board's box decoder reads
&mdash; six tensors, `bbox_0..2` and `class_logit_0..2`, not ultralytics' assembled
`[1, 84, 8400]` &mdash; then quantizes to bfloat16, tessellates for the MLA and emits the
ELF, using the compile recipe a published pack shipped rather than a paraphrase of it.

> [!NOTE]
> **The compiler is x86-only.** It is the `afe` package, and it exists only in that
> container: not on the DevKit, not on a plain laptop, and not installable next to this.
> `python -c "import afe"` says which side of the line you are on. Run `compile`
> anywhere else and you get the ONNX plus the steps to finish &mdash; never a crash at
> the last one.
>
> If `sima-cli sdk model` is not there, the Model Compiler extension was declined during
> setup. Re-run `sima-cli sdk setup` and accept it; it is a 9 GB download.

Either way, the pack ends up on the board the same way:

```bash
sima-vision push build/best_mpk.tar.gz
sima-vision detect --model best_mpk.tar.gz
```

A URL works too, and is cached under `assets/models`:

```bash
sima-vision detect --model https://example.com/my-model.tar.gz
```

If the head is not YOLO26, say so with `--family`. Get that wrong and the box decoder
reads the output tensor the wrong way, so every detection is noise rather than an error.

## Use your own footage
A path or an `https` URL. Raw H.264 and `.mp4` both work:

```bash
sima-vision push my-clip.mp4
sima-vision detect --source my-clip.mp4
```

The board decodes H.264 in hardware, and a container hits a demuxer bug in Neat 0.3.0, so
an `.mp4` is reframed into a raw stream on the first run and the result cached beside it.
That is a remux, not a re-encode: every coded bit survives, and a 13 MB clip takes about a
second. No `ffmpeg` needed, which matters because the DevKit has none.

H.264 video only, in either case. A fragmented `.mp4`, or one holding something other than
H.264, is refused by name rather than failing halfway through a run.

## Flags worth knowing

`sima-vision <command> --help` lists the rest.

| Flag | What it does |
|:--|:--|
| `--frames 200` | Stop after N frames. The quickest way to try something |
| `--conf 0.5` | Raise the confidence floor. Default `0.30` |
| `--no-video` / `--no-save` | Skip the recording or the stills. Together they are the cheapest possible run, which is how you tell a slow app apart from a stalled graph |
| `--quiet` | Warnings, errors and the closing report only |
| `--profile` | Per-stage timings, when a run is slower than it should be |
| `--model my.tar.gz` | Your own compiled pack instead of the fetched one |
| `--validate` | Resolve and check the settings, then stop. Needs no board |

Settings can also come from a `config.yaml` in the working directory, which is picked up
on its own. Flags win over it, and it wins over the built-in defaults.

## Environment

| Variable | What it does |
|:--|:--|
| `SIMA_VISION_DEVKIT` | The board, as `user@address`, so `push` and `pull` stop asking |
| `SIMA_VISION_ASSETS` | Where clips and models are downloaded. Default `./assets` |
| `SIMA_VISION_PYNEAT` | The `pyneat` virtualenv, when the search does not find it |
| `SIMA_VISION_PYNEAT_INDEX` | A pip index carrying a `pyneat` wheel, if your site publishes one |
| `SIMA_VISION_AUTO_INSTALL` | `0` to look but never install |
| `SIMA_VISION_QUIET` | Non-empty is `--quiet` for every command |
| `SIMA_VISION_COLOR` | `0` or `1` to force colour off or on. `NO_COLOR` also works |
| `FALL_ALERT_SMTP_PASSWORD` | The only place the SMTP password is ever read from |

## Contributing

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
pip install -e ".[dev]"

ruff check sima_vision tests
pytest -q
```

The tests need no board.

## License

The models used here for testing are **Ultralytics YOLO26**, under **AGPL-3.0**. All other
parts of this repository are under **Apache-2.0**. See [LICENSE](LICENSE).

## Credits

- [SiMa.ai](https://github.com/SiMa-ai) for Modalix, the Palette SDK and Neat
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO26 models

<div align="center">

Built by **Muhammad Rizwan Munawar**. If this saved you an afternoon, **star the repo**
and pass it on to someone else bringing up a DevKit.

<a href="https://github.com/RizwanMunawar"><img src="assets/socials/github.svg" width="50" alt="GitHub"></a>
&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/muhammadrizwanmunawar/"><img src="assets/socials/linkedin.svg" width="50" alt="LinkedIn"></a>
&nbsp;&nbsp;
<a href="https://x.com/muhammdrizwanmr"><img src="assets/socials/x.svg" width="50" alt="X"></a>
&nbsp;&nbsp;
<a href="https://www.youtube.com/@muhammadrizwanmunawar"><img src="assets/socials/youtube.svg" width="50" alt="YouTube"></a>
&nbsp;&nbsp;
<a href="https://muhammadrizwanmunawar.medium.com/"><img src="assets/socials/medium.svg" width="50" alt="Medium"></a>

</div>
