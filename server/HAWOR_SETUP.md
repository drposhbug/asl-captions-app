# HaWoR setup (Windows 11, RTX 3050 4 GB, CUDA 12.1 wheels)

Checkout: `C:\Users\thorn\Documents\GitHub\HaWoR` (cloned `--recursive`; submodules eigen + lietorch are present).
Env python: `C:\Users\thorn\miniconda3\envs\hawor\python.exe` (conda env `hawor`, Python 3.10, torch 2.4.1+cu121).

## Finish the install

Everything scripted and re-runnable (downloads resume, finished steps are skipped):

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\thorn\Documents\GitHub\asl hack phs\server\hawor_setup.ps1"
```

It follows the README (torch, requirements, pytorch-lightning, DROID-SLAM, weights) with these Windows substitutions:

| README step | Here |
|---|---|
| torch 1.13+cu117 | torch 2.4.1 / torchvision 0.19.1 +cu121 wheels, downloaded with `curl -C -` into `HaWoR\_downloads\` (2.4 GB; this laptop's link does ~200 KB/s, so ~3 h) |
| `git+pytorch3d` (needs a compiler) | prebuilt `pytorch3d==0.7.8+pt2.4.1cu121` from https://miropsota.github.io/torch_packages_builder |
| `torch-scatter==2.1.2` | prebuilt wheel from data.pyg.org (only DROID-SLAM uses it) |
| `chumpy`, `xtcocotools` | skipped: not imported by `demo.py`'s code path |
| `wget` | `curl.exe` (Hugging Face) and `gdown` (the two Google Drive files) |
| `ffmpeg` on PATH | `conda install ffmpeg` into the env; `run_hawor.py` puts the env's `Library\bin` on the subprocess PATH |

Log: `C:\Users\thorn\Documents\GitHub\HaWoR\_downloads\setup.log`.

## Blocked: DROID-SLAM CUDA extension (`droid_backends`, `lietorch_backends`)

`thirdparty\DROID-SLAM\setup.py` compiles CUDA code. This machine has **no CUDA Toolkit (`nvcc`) and no MSVC (`cl.exe`)**,
so the step fails and `demo.py` stops at the SLAM stage with `ModuleNotFoundError: No module named 'droid'`
(from `lib\pipeline\masked_droid_slam.py:14`). No prebuilt Windows wheel exists. To unblock:

1. Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload.
2. Install CUDA Toolkit 12.1 (must match the torch wheel) from NVIDIA; make sure `nvcc` is on PATH.
3. Re-run `hawor_setup.ps1` (it re-tries the build at the end), or from a "x64 Native Tools" prompt:
   `cd C:\Users\thorn\Documents\GitHub\HaWoR\thirdparty\DROID-SLAM` then `C:\Users\thorn\miniconda3\envs\hawor\python.exe setup.py install`.

Alternative that avoids the build entirely: run HaWoR on Linux/WSL2 (the authors' tested setup) or the
Hugging Face Space (https://huggingface.co/spaces/ThunderVVV/HaWoR).

## MANO (manual, licence-gated)

1. Register at https://mano.is.tue.mpg.de and download `mano_v1_2.zip` (Models & Code).
2. Unzip; copy from `mano_v1_2\models\`:
   - `MANO_RIGHT.pkl` -> `C:\Users\thorn\Documents\GitHub\HaWoR\_DATA\data\mano\MANO_RIGHT.pkl`
   - `MANO_LEFT.pkl`  -> `C:\Users\thorn\Documents\GitHub\HaWoR\_DATA\data_left\mano_left\MANO_LEFT.pkl`

Without them `demo.py` fails inside `lib\models\mano_wrapper.py` / smplx when loading the model
(`FileNotFoundError` / `AssertionError` pointing at `_DATA/data/mano/MANO_RIGHT.pkl`).

## Run

```powershell
$env:HAWOR_DIR = "C:\Users\thorn\Documents\GitHub\HaWoR"
$env:HAWOR_PYTHON = "C:\Users\thorn\miniconda3\envs\hawor\python.exe"
cd "C:\Users\thorn\Documents\GitHub\asl hack phs"
C:\Users\thorn\miniconda3\envs\hawor\python.exe server\run_hawor.py clips\<clip>.mp4
```

`run_hawor.py` saves clean frames to `server\frames\<clip>\`, then runs, with cwd = HaWoR:
`python demo.py --video_path <abs clip>.mp4 --vis_mode world` (these are the only args demo.py takes besides
`--img_focal`, `--input_type`, `--checkpoint`, `--infiller_weight`). HaWoR writes to `<clip dir>\<clip name>\`:
`extracted_images\`, `tracks_0_N\`, `SLAM\`, `world_space_res.pth` (joblib list: trans, rot, hand_pose, betas, valid)
and `vis_0_N\aitviewer\video_0.mp4`. Note `<clip name>` is the file name up to the first dot.

Self-check (opencv + numpy only): `C:\Users\thorn\miniconda3\envs\hawor\python.exe server\run_hawor.py --selftest` -> `selftest ok`.

## Expect

- 4 GB VRAM is tight: HaWoR + Metric3D ViT-L + DROID-SLAM were run by the authors on much bigger GPUs; keep clips short (a few seconds) and expect CUDA OOM on long ones.
- World-view visualisation uses aitviewer headless (moderngl); if it fails to create a GL context, use `--vis_mode cam`.
