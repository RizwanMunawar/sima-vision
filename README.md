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
instance segmentation, fall detection and more: one pipeline, no setup step. Needs a
[Modalix DevKit 3.0](https://devkit.sima.ai/products/development-kit-3-0) and Python 3.10
or later.

- [Quickstart](#quickstart)
- [Apps arguments](#apps-arguments)
- [Your own model](#your-own-model)
- [Moving files](#moving-files)
- [Environment](#environment)

## Quickstart

![on the DevKit](https://img.shields.io/badge/run_on-DevKit-E63946?style=flat-square)

No Docker, no WSL, no login. Every command here is typed on the board.

```bash
sima-cli login
sima-cli neat install core@v0.3.0    # once per board

pip install sima-vision
sima-vision detect
```

The first run fetches a pretrained YOLO26 pack and a demo clip into `./assets`, then runs.
Every run after that reuses them. Both come from one public GitHub release, so getting
them needs no login and no `sima-cli`.

| App | Fetched for you | Writes |
|:--|:--|:--|
| `detect` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` (21 MB) + a 1080p demo clip (13 MB) | `detections.mp4`, `frames/` |
| `segment` | `yolo26n-seg-bf16-mla_tess.tar.gz` (24 MB) + the same clip | `segmentation.mp4`, `frames/` |
| `fall` | the detection pack again + a shorter clip (1.2 MB) | `falls.mp4`, `frames/`, `alerts/` |

<details>
<summary>🎨 &nbsp;<b>Instance segmentation</b> &nbsp;·&nbsp; per-pixel masks, with an optional blur</summary>

<br>

```bash
sima-vision segment
sima-vision segment --blur
sima-vision segment --blur --keep-classes person
```

</details>

<details>
<summary>🚨 &nbsp;<b>Fall detection</b> &nbsp;·&nbsp; tracks people, with optional email alerts</summary>

<br>

```bash
sima-vision fall
sima-vision fall --alert-to ops@example.com
```

Nothing is emailed until you pass `--send`. Without it a fall is composed and logged, so
you can see what would have gone out.

</details>

Your own video works the same way, as a path or an `https` URL:

```bash
sima-vision push my-clip.mp4             # on your PC
sima-vision detect --source my-clip.mp4  # on the DevKit
```

H.264 only, but the container does not matter. The board decodes H.264 in hardware and
`.mp4` hits a demuxer bug in Neat 0.3.0, so an `.mp4` is reframed into a raw stream on
first use and cached beside it. That is a remux, not a re-encode: every coded bit
survives, a 13 MB clip takes about a second, and no `ffmpeg` is involved, which matters
because the DevKit has none. Anything that is not H.264 is refused by name rather than
failing halfway through a run.

## Apps arguments

Every flag, and which apps take it. All of them are for the three apps, which run on the
DevKit. `sima-vision <app> --help` prints the same list.

| Flag | Apps | What it does |
|:--|:--|:--|
| `--source`, `-s URI` | all | File, `https` URL, RTSP URL, or empty for the sample clip |
| `--source-type` | all | `video`, `rtsp` or `usb`. Default `video` |
| `--fps N` | all | Source frame rate. Default `0`, read from the stream |
| `--width PX` | all | Source width. Default `0`, read from the stream's SPS |
| `--height PX` | all | Source height. Default `0`, read from the stream's SPS |
| `--model`, `-m PATH` | all | Compiled pack, or an `https` URL to one. Empty fetches this app's default |
| `--labels PATH` | all | Newline-separated class names. Defaults to the packaged COCO list |
| `--family NAME` | all | Detection head. Must match the model or every box is noise |
| `--conf T` | all | Minimum confidence. Default `0.30` |
| `--iou T` | all | Non-max suppression IoU. Default `0.60` |
| `--max-det N` | all | Top-K cap per frame. Default `50` |
| `--frames`, `-n N` | all | Stop after N frames. Default `0`, runs until interrupted |
| `--timeout MS` | all | How long to wait for a frame before giving up. Default `20000` |
| `--video-path PATH` | all | Where the annotated recording is written |
| `--no-video` | all | Do not record |
| `--save-dir DIR` | all | Where annotated stills are written |
| `--save-every N` | all | Write every Nth still. Default `10`; `0` disables |
| `--no-save` | all | Do not write stills |
| `--no-hud` | all | Leave the frame-rate badge off the overlay |
| `--insight` | all | Stream to Neat Insight over UDP. Off by default |
| `--insight-host HOST` | all | Insight address as the board sees it. Default `127.0.0.1` |
| `--config`, `-c PATH` | all | Config file. Defaults to `./config.yaml` |
| `--no-config` | all | Ignore any config file and use defaults plus these flags |
| `--validate` | all | Resolve and check the settings, then stop. Needs no board |
| `--quiet`, `-q` | all | Warnings, errors and the closing report only |
| `--profile` | all | Per-stage timings, when a run is slower than it should be |
| `--queue-depth N` | all | Neat's own queue depth. Every slot holds a decoded frame, so raising it makes a starved run worse. Default `1` |
| `--sink-queue-depth N` | all | Finished frames that may wait for the recorder. Host memory only. Default `12` |
| `--sink-queue-mb MB` | all | Memory budget for that backlog, which grows it to fit a known clip |
| `--output-buffers N` | all | Buffers each public output may hold. Default `1` |
| `--decoder-buffers N` | all | Buffers to ask the decoder for. Default `0`, sized from the stream's reference frames |
| `--segment-frames N` | all | Frames per piece when a clip is too long for one decode. Default `150`; `0` runs it whole |
| `--blur` / `--no-blur` | `segment` | Blur the background and keep instances sharp, or draw a plain overlay |
| `--blur-method` | `segment` | `gaussian`, `pixelate` or `none`. Default `gaussian` |
| `--blur-strength PX` | `segment` | Gaussian kernel width at 1080p. Default `41` |
| `--keep-classes CLASS...` | `segment` | Names or ids that stay sharp. Default: every detected class |
| `--anonymise`, `--anonymize` | `segment` | Blur the instances instead of the background |
| `--mask-threshold T` | `segment` | Mask cut-off as a probability. Default `0.5`; lower grows instances |
| `--no-masks` | `segment` | Blur around plain boxes instead. Needs no segment head |
| `--minimal` | `segment` | Pull frames and do nothing else. Tells a slow app from a stalled graph |
| `--classes CLASS...` | `fall` | Classes that can fall. Default `person` |
| `--confirm S` | `fall` | How long a fall signal must hold before it counts. Default `1.5` |
| `--no-fall` | `fall` | Track without judging falls, which is how you tune tracking first |
| `--alert-to EMAIL...` | `fall` | Recipients. Implies `--alerts` |
| `--alert-from EMAIL` | `fall` | From address |
| `--alerts` | `fall` | Enable alerts. Still a dry run until `--send` |
| `--send` | `fall` | Actually connect to the SMTP server |
| `--smtp-host HOST` | `fall` | SMTP server. Default `smtp.gmail.com` |
| `--smtp-port PORT` | `fall` | `587` for STARTTLS, `465` for SSL. Default `587` |
| `--smtp-user USER` | `fall` | SMTP login. The password comes from `$FALL_ALERT_SMTP_PASSWORD` and nowhere else |
| `--site NAME` | `fall` | Camera name, used in the alert subject and body |
| `--test-alert` | `fall` | Send one fake alert and exit. Proves the SMTP settings without a fall, or a board |

A `config.yaml` in the working directory is picked up on its own. Flags beat it, and it
beats the built-in defaults.

## Your own model

![on your PC](https://img.shields.io/badge/run_on-Host_PC-457B9D?style=flat-square)

A trained `.pt` has to be compiled into a `.tar.gz` pack before the board can run it.

> [!IMPORTANT]
> Docker and WSL are needed here and nowhere else, and both live on your PC.
> [Quickstart](#quickstart) needs neither.

<details>
<summary>🧠 &nbsp;<b>Converting a trained <code>.pt</code> into a DevKit pack</b> &nbsp;·&nbsp; six steps, about 30 minutes the first time</summary>

<br>

Needs a [community.sima.ai](https://community.sima.ai) account and ~10 GB of disk. The
compiler is the `afe` package and exists only inside the Palette Model SDK container:
x86 Docker, which on Windows means WSL2.

**1. WSL2** &nbsp; PowerShell, as Administrator

```powershell
wsl --install -d Ubuntu
wsl -l -v                            # want: Ubuntu, Running, 2
```

**2. Docker Desktop** &nbsp; [install it](https://docs.docker.com/get-started/get-docker/),
then **Settings → Resources → WSL integration** → on, for Ubuntu

Without that toggle `docker` works in PowerShell and is missing inside Ubuntu.

**3. Into Ubuntu** &nbsp; everything below is typed there; `exit` comes back

```powershell
wsl -d Ubuntu
```

```bash
docker run hello-world               # must print "Hello from Docker!"
```

**4. sima-cli**

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/sima               # Ubuntu refuses a bare pip install
source ~/sima/bin/activate           # every new shell starts outside it. Look for (sima)
pip install sima-cli
sima-cli login
sima-cli install ghcr:sima-neat/sdk:v2.0.0     # several GB, once per machine
```

**5. Convert** &nbsp; run from the directory holding your `.pt`

```bash
sima-cli sdk setup --workspace .     # . is what gets mounted. 15-20 minutes
activate-model-compiler
```

```bash
pip install sima-vision ultralytics
sima-vision compile best.pt          # -> build/best_mpk.tar.gz, 10-15 minutes
```

One command, four stages: raw-head ONNX, bfloat16 quantization, MLA tessellation, ELF.

**6. Run it** &nbsp; ![on the DevKit](https://img.shields.io/badge/run_on-DevKit-E63946?style=flat-square)

```bash
sima-vision push build/best_mpk.tar.gz
sima-vision detect --model best_mpk.tar.gz
sima-vision detect --model https://example.com/my-model.tar.gz   # a URL works too
```

</details>

## Moving files

![on your PC](https://img.shields.io/badge/run_on-Host_PC-457B9D?style=flat-square)

Output lands beside the run, on the board, and both commands are typed on your PC. Name the board once and neither command needs
`--host`:

```powershell
$env:SIMA_VISION_DEVKIT="sima@<devkit-ip>"
```

```bash
sima-vision pull                     # everything the run left
sima-vision pull --into results/
sima-vision push my-clip.h264
```

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
