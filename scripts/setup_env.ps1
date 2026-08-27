param(
    [switch]$WithGui,
    [switch]$Recreate,
    [string]$PythonPath,
    [string]$PipIndex = $env:PIP_INDEX
)

# 用法
# .\scripts\setup_env.ps1
# .\scripts\setup_env.ps1 -Recreate
# .\scripts\setup_env.ps1 -PythonPath "C:\Python312\python.exe"

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $ProjectDir ".venv"
$TorchVersion = "2.10.0"
$TorchCpuIndex = "https://download.pytorch.org/whl/cpu"

$BasePython = $null
$BaseArgs = @()
$Candidates = @()
if ($PythonPath) {
    $Candidates += ,@($PythonPath)
}
$Candidates += ,@("py", "-3.12")
$Candidates += ,@("py", "-3.11")
$Candidates += ,@("python")
$Candidates += ,@("python3")

foreach ($Candidate in $Candidates) {
    $Command = $Candidate[0]
    $Args = @($Candidate | Select-Object -Skip 1)
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        continue
    }
    & $Command @Args -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $BasePython = $Command
        $BaseArgs = $Args
        break
    }
}
if (-not $BasePython) {
    throw "未找到可用的 Python 3.11 或 3.12，可通过 -PythonPath 指定解释器"
}

& $BasePython @BaseArgs -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), '需要 Python 3.11/3.12'; print(sys.version)"
if ($Recreate -and (Test-Path $VenvDir)) {
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}
if (-not (Test-Path $VenvDir)) {
    & $BasePython @BaseArgs -m venv $VenvDir
}
$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13)" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "现有虚拟环境不可用，请重新运行 .\scripts\setup_env.ps1 -Recreate"
}
$PipArgs = @()
if ($PipIndex) {
    $PipArgs = @("--index-url", $PipIndex)
}

& $Python -m pip install --upgrade pip setuptools wheel @PipArgs
& $Python -m pip install "torch==$TorchVersion" --index-url $TorchCpuIndex
& $Python -m pip install -r (Join-Path $ProjectDir "requirements\dev.txt") @PipArgs
& $Python -m pip install -e $ProjectDir --no-deps
if ($WithGui) {
    & $Python -m pip install -r (Join-Path $ProjectDir "requirements\gui.txt") @PipArgs
}

& $Python -m pip check
& $Python -c "from importlib.metadata import version; import flask, requests, numpy, pandas, scipy, sklearn, matplotlib, keras, tensorflow, pymoo, gymnasium, stable_baselines3, pyDOE, torch, mobo; print('mobo', mobo.__version__); print('flask', version('flask')); print('torch', torch.__version__)"

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
Write-Host "API启动：mobo-api"
Write-Host "完整演示：python -m mobo.api.demo"

