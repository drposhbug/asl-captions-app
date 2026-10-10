# HaWoR env + weights on Windows (CUDA 12.1 wheels). Re-runnable: finished steps are skipped, downloads resume.
# Run:  powershell -ExecutionPolicy Bypass -File server\hawor_setup.ps1
$Conda = "C:\Users\thorn\miniconda3\Scripts\conda.exe"
$Hawor = "C:\Users\thorn\Documents\GitHub\HaWoR"
$Py = "C:\Users\thorn\miniconda3\envs\hawor\python.exe"
$Dl = "$Hawor\_downloads"
New-Item -ItemType Directory -Force $Dl, "$Hawor\weights\external", "$Hawor\weights\hawor\checkpoints", "$Hawor\thirdparty\Metric3D\weights" | Out-Null

function Fetch($url, $out) {
    # shortcut: this laptop's link does ~200 KB/s and torch alone is 2.4 GB, so resume (-C -) and retry instead of restarting.
    $need = [int64]((curl.exe -sIL $url | Select-String '^content-length:' | Select-Object -Last 1) -replace '.*:\s*', '')
    if ((Test-Path $out) -and (Get-Item $out).Length -eq $need) { Write-Host "have $out"; return }
    for ($i = 0; $i -lt 50; $i++) {
        curl.exe -L -C - --retry 5 -o $out $url
        if ((Test-Path $out) -and (Get-Item $out).Length -eq $need) { return }
        Start-Sleep 10
    }
    throw "download failed: $url"
}
function Pip { & $Py -m pip install --no-warn-script-location @args; if ($LASTEXITCODE) { throw "pip failed: $args" } }

if (-not (Test-Path $Py)) { & $Conda create -y -n hawor python=3.10 }

# 1. torch first: anything that depends on torch (ultralytics, timm) would otherwise pull the CPU build.
$torch = "torch-2.4.1+cu121-cp310-cp310-win_amd64.whl"; $tv = "torchvision-0.19.1+cu121-cp310-cp310-win_amd64.whl"
Fetch "https://download.pytorch.org/whl/cu121/$($torch -replace '\+','%2B')" "$Dl\$torch"
Fetch "https://download.pytorch.org/whl/cu121/$($tv -replace '\+','%2B')" "$Dl\$tv"
Pip "$Dl\$torch" "$Dl\$tv"

# 2. README requirements.txt minus the ones that need a compiler on Windows (pytorch3d, torch-scatter: prebuilt below;
#    chumpy, xtcocotools: not imported by demo.py).
Pip numpy==1.26.4 opencv-python pyrender scikit-image smplx==0.1.28 yacs mmcv==1.3.9 timm einops pandas hydra-core `
    hydra-submitit-launcher hydra-colorlog pyrootutils rich webdataset ultralytics pulp supervision pycocotools joblib natsort `
    evo pytorch-minimize mmengine==0.10.4 HTML4Vision plyfile aitviewer easydict loguru dill lapx moderngl-window==2.4.6 gdown
Pip pytorch-lightning==2.2.4 --no-deps
Pip lightning-utilities torchmetrics==1.4.0
Pip "https://data.pyg.org/whl/torch-2.4.0%2Bcu121/torch_scatter-2.1.2%2Bpt24cu121-cp310-cp310-win_amd64.whl"
Pip "pytorch3d==0.7.8+pt2.4.1cu121" --extra-index-url https://miropsota.github.io/torch_packages_builder
& $Conda install -y -n hawor -c conda-forge ffmpeg   # demo.py shells out to ffmpeg to split frames

# 3. weights (MANO is manual: see HAWOR_SETUP.md)
Fetch https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt "$Hawor\weights\external\detector.pt"
Fetch https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt "$Hawor\weights\hawor\checkpoints\hawor.ckpt"
Fetch https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt "$Hawor\weights\hawor\checkpoints\infiller.pt"
Fetch https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml "$Hawor\weights\hawor\model_config.yaml"
if (-not (Test-Path "$Hawor\weights\external\droid.pth")) { & $Py -m gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O "$Hawor\weights\external\droid.pth" }
if (-not (Test-Path "$Hawor\thirdparty\Metric3D\weights\metric_depth_vit_large_800k.pth")) { & $Py -m gdown 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr -O "$Hawor\thirdparty\Metric3D\weights\metric_depth_vit_large_800k.pth" }

# 4. DROID-SLAM CUDA extension: needs CUDA Toolkit 12.1 (nvcc) + Visual Studio Build Tools (cl.exe). Not fatal here so the rest completes.
Push-Location "$Hawor\thirdparty\DROID-SLAM"
& $Py setup.py install
if ($LASTEXITCODE) { Write-Warning "DROID-SLAM build failed: install CUDA Toolkit 12.1 and VS 2022 Build Tools (C++ workload), then re-run this script." }
Pop-Location
Write-Host "done"
