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

**Computer vision applications on the SiMa.ai Modalix DevKit 3.0.** Object detection,
instance segmentation, fall detection and more: one pipeline, no setup step.

Needs a [Modalix DevKit 3.0](https://devkit.sima.ai/products/development-kit-3-0) and
Python 3.10 or later.

## 📑 Contents

| | |
|:--|:--|
| [🚀 Quickstart](#-quickstart) | Two commands on the board, nothing to download by hand |
| [🎬 Your own footage](#-your-own-footage) | Point it at an `.mp4` or a raw `.h264` |
| [🧠 Your own model](#-your-own-model) | Convert a trained `.pt` into a DevKit pack |
| [🔁 Moving files](#-moving-files-between-the-board-and-your-pc) | `push` and `pull` between PC and DevKit |
| [🚩 Flags worth knowing](#-flags-worth-knowing) | The seven you will actually reach for |
| [🌱 Environment](#-environment) | Variables that change where things go |
| [🤝 Contributing](#-contributing) | Clone, install, test |

## 🚀 Quickstart

Everything in this section runs **on the DevKit**. No Docker, no WSL, no login.

### 1. Install the SiMa.ai Neat Core

```bash
sima-cli login
sima-cli neat install core@v0.3.0
```

### 2. Install sima-vision and run it

```bash
pip install sima-vision
sima-vision detect
```

That is it. **Two commands, and the second one is the work.**

> [!TIP]
> **Nothing to download by hand.** Each app fetches its own pretrained YOLO26 pack and a
> demo clip into `./assets` on its first run, then runs. Every run after that reuses
> them. Packs and clips come from one public GitHub release, so no login is involved and
> `sima-cli` is not needed to get them.

| App | Fetched for you | Writes |
|:--|:--|:--|
| `detect` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` (21 MB) + a 1080p demo clip (13 MB) | `detections.mp4`, `frames/` |
| `segment` | `yolo26n-seg-bf16-mla_tess.tar.gz` (24 MB) + the same clip | `segmentation.mp4`, `frames/` |
| `fall` | the detection pack again + a shorter clip (1.2 MB) | `falls.mp4`, `frames/`, `alerts/` |

<details>
<summary>🎨 <b>Instance segmentation</b> &nbsp; per-pixel masks, with an optional blur</summary>

```bash
sima-vision segment
sima-vision segment --blur
sima-vision segment --blur --keep-classes person
```

</details>

<details>
<summary>🚨 <b>Fall detection</b> &nbsp; tracks people, with optional email alerts</summary>

```bash
sima-vision fall
sima-vision fall --alert-to ops@example.com
```

Nothing is emailed until you pass `--send`. Without it a fall is composed and logged, so
you can see what would have gone out.

</details>

## 🎬 Your own footage

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

## 🧠 Your own model

Trained a YOLO26 detection or segmentation model and want to run it on the board? It has
to be compiled into a `.tar.gz` pack first.

> [!IMPORTANT]
> **This section is the only place Docker and WSL are needed, and only on your PC.**
> Everything in [Quickstart](#-quickstart) works without either: a pretrained model on
> the DevKit is just the two commands above. Nothing here runs on the board.

The compiler is the `afe` package. It lives inside the Palette Model SDK container, which
is x86 Docker, which on Windows means WSL2. Five steps to get there and convert.

### 1. WSL2

In **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu
wsl -l -v
```

Want `Ubuntu`, `Running`, version `2`.

### 2. Docker Desktop

Install [Docker Desktop](https://docs.docker.com/get-started/get-docker/), then turn on
**Settings → Resources → WSL integration** for Ubuntu. Without that, `docker` works in
PowerShell but is missing inside Ubuntu, which is where it is needed.

### 3. Go into Ubuntu

Everything from here down is typed in that shell, not in PowerShell:

```powershell
wsl -d Ubuntu
```

The prompt changes to something like `you@machine:~$`. `exit` comes back to PowerShell.
Check the whole path now works:

```bash
docker run hello-world
```

Must print `Hello from Docker!`.

### 4. sima-cli, inside Ubuntu

It needs a virtualenv, because Ubuntu refuses a bare `pip install` with
`externally-managed-environment`:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/sima
source ~/sima/bin/activate
pip install sima-cli
sima-cli login
```

> [!NOTE]
> `source ~/sima/bin/activate` is the line to remember. It is Linux, inside WSL, and has
> no PowerShell equivalent. A new Ubuntu shell starts **outside** the venv, so `sima-cli`
> is missing until you run it again. `(sima)` at the front of the prompt is how you know.

Then pull the SDK image. Ten to fifteen minutes, once:

```bash
sima-cli install ghcr:sima-neat/sdk:v2.0.0
```

### 5. Convert

`--workspace` is the folder that appears inside the container, so put your `.pt` there
first:

```bash
sima-cli sdk setup --workspace ~/workspace
sima-cli sdk model
```

That opens the Model SDK shell. Inside it, in `~/workspace`:

```bash
pip install sima-vision ultralytics
sima-vision compile best.pt
```

Out comes `build/best_mpk.tar.gz`. That one command is the whole conversion: it exports
the raw-head ONNX the board's box decoder reads (six tensors, `bbox_0..2` and
`class_logit_0..2`, not ultralytics' assembled `[1, 84, 8400]`), then quantizes to
bfloat16, tessellates for the MLA and emits the ELF, using the compile recipe a published
pack shipped rather than a paraphrase of it.

### 6. Run it on the board

```bash
sima-vision push build/best_mpk.tar.gz
sima-vision detect --model best_mpk.tar.gz
```

A URL works too, and is cached under `assets/models`:

```bash
sima-vision detect --model https://example.com/my-model.tar.gz
```

<details>
<summary>🩹 <b>If something goes wrong</b></summary>

**`sima-cli sdk model` is missing.** The Model Compiler extension was declined during
setup. Re-run `sima-cli sdk setup` and accept it. It is a 9 GB download.

**`compile` produced only an ONNX.** The Model SDK is not importable where you ran it.
`python -c "import afe"` says which side of the line you are on. Run it inside the Model
SDK shell and it does every step; anywhere else it hands you the ONNX and the steps to
finish, rather than crashing at the last one.

**Every detection is noise.** The head is not YOLO26. Say so with `--family`: get it
wrong and the box decoder reads the output tensor the wrong way, which is garbage rather
than an error.

</details>

## 🔁 Moving files between the board and your PC

Output lands beside the run, on the board. Name the board once and neither command needs
`--host`:

```powershell
$env:SIMA_VISION_DEVKIT="sima@<devkit-ip>"
```

```bash
sima-vision pull
sima-vision pull --into results/
sima-vision push my-clip.h264
```

## 🚩 Flags worth knowing

`sima-vision <command> --help` lists the rest.

| Flag | What it does |
|:--|:--|
| `--frames 200` | Stop after N frames. The quickest way to try something |
| `--conf 0.5` | Raise the confidence floor. Default `0.30` |
| `--no-video` / `--no-save` | Skip the recording or the stills |
| `--quiet` | Warnings, errors and the closing report only |
| `--profile` | Per-stage timings, when a run is slower than it should be |
| `--model my.tar.gz` | Your own compiled pack instead of the fetched one |
| `--validate` | Resolve and check the settings, then stop. Needs no board |

Settings can also come from a `config.yaml` in the working directory, which is picked up
on its own. Flags win over it, and it wins over the built-in defaults.

## 🌱 Environment

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

## 🤝 Contributing

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
pip install -e ".[dev]"

ruff check sima_vision tests
pytest -q
```

The tests need no board.

## 📄 License

The models used here for testing are **Ultralytics YOLO26**, under **AGPL-3.0**. All other
parts of this repository are under **Apache-2.0**. See [LICENSE](LICENSE).

## 🙏 Credits

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
