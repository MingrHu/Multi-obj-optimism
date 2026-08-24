param(
    [switch]$WithGui,
    [string]$PipIndex = $env:PIP_INDEX
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"
$TorchVersion = "2.10.0"
$TorchCpuIndex = "https://download.pytorch.org/whl/cpu"

if (Get-Command py -ErrorAction SilentlyContinue) {
    $BasePython = "py"
    $BaseArgs = @("-3.12")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $BasePython = "python"
    $BaseArgs = @()
} else {
    throw "未找到 Python。请先安装 Python 3.11 或 3.12（推荐 3.12）。"
}

& $BasePython @BaseArgs -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), '需要 Python 3.11/3.12'; print(sys.version)"
if (-not (Test-Path $VenvDir)) {
    & $BasePython @BaseArgs -m venv $VenvDir
}
$Python = Join-Path $VenvDir "Scripts\python.exe"
$PipArgs = @()
if ($PipIndex) {
    $PipArgs = @("--index-url", $PipIndex)
}

& $Python -m pip install --upgrade pip setuptools wheel @PipArgs
& $Python -m pip install "torch==$TorchVersion" --index-url $TorchCpuIndex
& $Python -m pip install -r (Join-Path $ProjectDir "requirements-dev.txt") @PipArgs
& $Python -m pip install -e $ProjectDir --no-deps
if ($WithGui) {
    & $Python -m pip install -r (Join-Path $ProjectDir "requirements-gui.txt") @PipArgs
}

& $Python -m pip check
& $Python -c "import numpy, pandas, scipy, sklearn, matplotlib, keras, tensorflow, pymoo, gymnasium, stable_baselines3, pyDOE, torch, mobo; print('mobo', mobo.__version__); print('torch', torch.__version__)"

$Pre = Get-Command DEF_PRE_64.exe -ErrorAction SilentlyContinue
$Arm = Get-Command DEF_ARM_CTL.COM -ErrorAction SilentlyContinue
if (-not $Pre -and -not $env:MOBO_DEF_PRE_64) {
    Write-Warning "未找到 DEF_PRE_64.exe；真实求解前请加入 PATH 或设置 MOBO_DEF_PRE_64。"
}
if (-not $Arm -and -not $env:MOBO_DEF_ARM_CTL) {
    Write-Warning "未找到 DEF_ARM_CTL.COM；真实求解前请加入 PATH 或设置 MOBO_DEF_ARM_CTL。"
}

Write-Host "环境安装完成。激活脚本：$VenvDir\Scripts\Activate.ps1"
Write-Host "测试命令：python -m pytest -m 'not slow'"
