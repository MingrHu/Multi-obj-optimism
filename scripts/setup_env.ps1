param(
    [switch]$WithGui,
    [switch]$Recreate,
    [string]$PythonPath,
    [string]$PipIndex = $env:PIP_INDEX,
    [switch]$NoAutoPython,
    [switch]$SkipVCRuntime,
    [string]$DeformDir
)

# Windows PowerShell 5.1 and PowerShell 7; dot-source to test helpers without installing.
$ErrorActionPreference = 'Stop'

function Get-PythonInfo {
    param([string]$FilePath, [string[]]$PrefixArguments = @())
    # A failed py launcher writes stderr in PowerShell 5.1. It must not abort discovery.
    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false
    $probe = "import json,struct,sys; print(json.dumps(dict(executable=sys.executable,base=getattr(sys,'_base_executable',sys.executable),major=sys.version_info.major,minor=sys.version_info.minor,bits=struct.calcsize('P')*8,version=sys.version)))"
    try {
        $global:LASTEXITCODE = -1
        $output = & $FilePath @PrefixArguments -c $probe 2>$null
        if ($global:LASTEXITCODE -ne 0) { return $null }
        $info = ($output -join "`n") | ConvertFrom-Json -ErrorAction Stop
        if ($info.major -eq 3 -and $info.minor -in @(11, 12) -and $info.bits -eq 64) {
            return $info
        }
    } catch { Write-Verbose "Python probe failed: $FilePath ($_)" }
    return $null
}

function Get-CompatiblePython {
    param([string]$ProjectDir, [string]$PythonPath)
    if ($PythonPath) {
        $info = Get-PythonInfo -FilePath $PythonPath
        if (-not $info) { throw '-PythonPath must point to a working 64-bit Python 3.11/3.12.' }
        return $info
    }
    $candidates = @()
    $candidates += ,@((Join-Path $ProjectDir '.venv\Scripts\python.exe'))
    $candidates += ,@((Join-Path $ProjectDir '.tools\python312\python.exe'))
    $candidates += ,@('python')
    $candidates += ,@('python3')
    # Also find registered runtimes installed without PATH or a launcher.
    foreach ($registryRoot in @('HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore')) {
        foreach ($version in @('3.12', '3.11')) {
            $key = Get-Item -LiteralPath (Join-Path $registryRoot "$version\InstallPath") -ErrorAction SilentlyContinue
            if (-not $key) { continue }
            $registered = $key.GetValue('ExecutablePath')
            if (-not $registered -and $key.GetValue('')) { $registered = Join-Path $key.GetValue('') 'python.exe' }
            if ($registered) { $candidates += ,@($registered) }
        }
    }
    $candidates += ,@('py', '-3.12')
    $candidates += ,@('py', '-3.11')
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate[0] -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        if ($command.Source -like '*\Microsoft\WindowsApps\python*.exe') { continue }
        $prefix = @($candidate | Select-Object -Skip 1)
        $info = Get-PythonInfo -FilePath $candidate[0] -PrefixArguments $prefix
        if ($info) { return $info }
    }
    return $null
}

function Invoke-CheckedCommand {
    param([string]$FilePath, [string[]]$ArgumentList, [string]$Step)
    Write-Host "==> $Step"
    $null = Get-Command $FilePath -ErrorAction Stop
    # Inspect exit codes explicitly on BOTH PowerShell versions; stderr alone is not failure.
    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false
    $global:LASTEXITCODE = -1
    & $FilePath @ArgumentList
    $code = $global:LASTEXITCODE
    if ($code -ne 0) { throw "$Step failed (exit code $code). Installation stopped; see output above." }
}

function Get-VerifiedInstaller {
    param([string]$Url, [string]$Destination, [string]$Publisher)
    $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination)
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if (-not (Test-Path -LiteralPath $Destination)) {
        Write-Host "==> Download $Url"
        $partial = $Destination + '.partial'
        try {
            Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing -TimeoutSec 300
            Move-Item -LiteralPath $partial -Destination $Destination -Force
        } finally {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        }
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $Destination
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch $Publisher) {
        Remove-Item -LiteralPath $Destination -Force
        throw "Installer signature verification failed: $Destination. Retry the download."
    }
    return $Destination
}

function Install-ProjectPython {
    param([string]$ProjectDir)
    # Last Python 3.12 full-maintenance Windows installer, signed by the PSF.
    $installer = Get-VerifiedInstaller -Url 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' `
        -Destination (Join-Path $ProjectDir '.tools\downloads\python-3.12.10-amd64.exe') -Publisher 'Python Software Foundation'
    $target = Join-Path $ProjectDir '.tools\python312'
    Write-Host '==> Install project-local Python 3.12 (no system PATH change)'
    $arguments = @('/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0',
        'Include_test=0', 'Include_doc=0', 'Shortcuts=0', 'AssociateFiles=0', "TargetDir=`"$target`"")
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) { throw "Python installer failed: $($process.ExitCode)" }
    $info = Get-PythonInfo -FilePath (Join-Path $target 'python.exe')
    if (-not $info) { throw 'Python installation did not produce a working interpreter.' }
    return $info
}

function Install-VCRuntime {
    param([string]$ProjectDir)
    $systemDir = Join-Path $env:SystemRoot 'System32'
    $missing = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $systemDir $_))
    }
    if (-not $missing) { return }
    $installer = Get-VerifiedInstaller -Url 'https://aka.ms/vc14/vc_redist.x64.exe' `
        -Destination (Join-Path $ProjectDir '.tools\downloads\vc_redist.x64.exe') -Publisher 'Microsoft Corporation'
    Write-Host '==> Install Microsoft Visual C++ x64 runtime (Windows may request administrator approval)'
    $process = Start-Process -FilePath $installer -ArgumentList @('/install', '/quiet', '/norestart') `
        -Verb RunAs -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 1638, 3010)) { throw "Visual C++ runtime installation failed: $($process.ExitCode)" }
    if ($process.ExitCode -eq 3010) { Write-Warning 'Visual C++ runtime requests a reboot. Restart Windows if DLL imports fail.' }
}

function Backup-ProjectVenv {
    param([string]$ProjectDir)
    $root = [IO.Path]::GetFullPath($ProjectDir).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $venv = [IO.Path]::GetFullPath((Join-Path $root '.venv'))
    if ((Split-Path -Parent $venv) -ne $root) { throw 'Refusing to move an environment outside the project.' }
    if ((Get-Item -LiteralPath $venv -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Refusing to move a linked .venv. Remove the link manually before recreating.'
    }
    $backup = Join-Path $root ('.venv.backup-' + [Guid]::NewGuid().ToString('N'))
    if ((Split-Path -Parent $backup) -ne $root) { throw 'Invalid environment backup path.' }
    Move-Item -LiteralPath $venv -Destination $backup
    Write-Host "==> Previous environment preserved: $backup"
}

function Set-DeformEnvironment {
    param([string]$DeformDir)
    $pairs = @()
    if ($DeformDir) {
        $specified = @((Join-Path $DeformDir 'DEF_PRE_64.exe'), (Join-Path $DeformDir 'DEF_ARM_CTL.COM'))
        if (-not ((Test-Path -LiteralPath $specified[0] -PathType Leaf) -and (Test-Path -LiteralPath $specified[1] -PathType Leaf))) {
            throw '-DeformDir must contain DEF_PRE_64.exe and DEF_ARM_CTL.COM.'
        }
        $pairs += ,$specified
    }
    if ($env:MOBO_DEF_PRE_64 -and $env:MOBO_DEF_ARM_CTL) { $pairs += ,@($env:MOBO_DEF_PRE_64, $env:MOBO_DEF_ARM_CTL) }
    $pre = Get-Command DEF_PRE_64.exe -ErrorAction SilentlyContinue
    $arm = Get-Command DEF_ARM_CTL.COM -ErrorAction SilentlyContinue
    if ($pre -and $arm) { $pairs += ,@($pre.Source, $arm.Source) }
    foreach ($programRoot in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $programRoot) { continue }
        $deformRoot = Join-Path $programRoot 'SFTC\DEFORM'
        if (Test-Path -LiteralPath $deformRoot) {
            foreach ($version in (Get-ChildItem -LiteralPath $deformRoot -Directory | Sort-Object Name -Descending)) {
                $pairs += ,@((Join-Path $version.FullName '3D\DEF_PRE_64.exe'), (Join-Path $version.FullName '3D\DEF_ARM_CTL.COM'))
            }
        }
    }
    foreach ($pair in $pairs) {
        if ((Test-Path -LiteralPath $pair[0] -PathType Leaf) -and (Test-Path -LiteralPath $pair[1] -PathType Leaf)) {
            [Environment]::SetEnvironmentVariable('MOBO_DEF_PRE_64', $pair[0], 'User')
            [Environment]::SetEnvironmentVariable('MOBO_DEF_ARM_CTL', $pair[1], 'User')
            $env:MOBO_DEF_PRE_64 = $pair[0]
            $env:MOBO_DEF_ARM_CTL = $pair[1]
            Write-Host "==> DEFORM configured: $($pair[0])"
            return
        }
    }
    if ($DeformDir) { throw '-DeformDir must contain DEF_PRE_64.exe and DEF_ARM_CTL.COM.' }
    Write-Warning 'DEFORM not found. Install and license DEFORM separately; Python/API/UI can still be used.'
}

function Invoke-MoboSetup {
    param([switch]$WithGui, [switch]$Recreate, [string]$PythonPath, [string]$PipIndex = $env:PIP_INDEX,
          [switch]$NoAutoPython, [switch]$SkipVCRuntime, [string]$DeformDir)
    if (-not [Environment]::Is64BitProcess) { throw 'Run this script in 64-bit Windows PowerShell.' }
    $project = Split-Path -Parent $PSScriptRoot
    $venv = Join-Path $project '.venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    $info = Get-CompatiblePython -ProjectDir $project -PythonPath $PythonPath
    if (-not $info) {
        if ($NoAutoPython) { throw 'No compatible Python found. Supply -PythonPath or allow automatic installation.' }
        $info = Install-ProjectPython -ProjectDir $project
    }
    Write-Host "==> Python $($info.version) [$($info.executable)]"
    if (-not $SkipVCRuntime) { Install-VCRuntime -ProjectDir $project }
    $venvInfo = Get-PythonInfo -FilePath $python
    if ((Test-Path -LiteralPath $venv) -and ($Recreate -or -not $venvInfo)) {
        # Select the BASE interpreter BEFORE moving an active .venv, including when called from (.venv).
        $baseInfo = Get-PythonInfo -FilePath $info.base
        if (-not $baseInfo) { throw 'Cannot locate a working base Python. Supply -PythonPath before recreating.' }
        $info = $baseInfo
        Backup-ProjectVenv -ProjectDir $project
    }
    if (-not (Test-Path -LiteralPath $venv)) {
        Invoke-CheckedCommand -FilePath $info.executable -ArgumentList @('-m', 'venv', $venv) -Step 'Create .venv'
    }
    if (-not (Get-PythonInfo -FilePath $python)) { throw 'Project .venv did not pass Python 3.11/3.12 x64 validation.' }
    $pipArgs = @()
    if ($PipIndex) { $pipArgs = @('--index-url', $PipIndex) }
    Invoke-CheckedCommand -FilePath $python -ArgumentList (@('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel') + $pipArgs) -Step 'Update pip tooling'
    Invoke-CheckedCommand -FilePath $python -ArgumentList @('-m', 'pip', 'install', 'torch==2.10.0', '--no-deps', '--index-url', 'https://download.pytorch.org/whl/cpu') -Step 'Install CPU PyTorch'
    $requirements = @('-m', 'pip', 'install', '-r', (Join-Path $project 'requirements\dev.txt'))
    if ($WithGui) { $requirements += @('-r', (Join-Path $project 'requirements\gui.txt')) }
    Invoke-CheckedCommand -FilePath $python -ArgumentList ($requirements + $pipArgs) -Step 'Install locked runtime, development and optional GUI dependencies'
    Invoke-CheckedCommand -FilePath $python -ArgumentList @('-m', 'pip', 'install', '-e', $project, '--no-deps') -Step 'Install mobo'
    Invoke-CheckedCommand -FilePath $python -ArgumentList @('-m', 'pip', 'check') -Step 'Check dependency consistency'
    $check = 'from importlib.metadata import version; import flask,requests,numpy,pandas,scipy,sklearn,matplotlib,keras,tensorflow,pymoo,gymnasium,stable_baselines3,pyDOE,moocore,torch,mobo; from mobo.api.readiness import load_runtime_dependencies; print(load_runtime_dependencies()); assert torch.version.cuda is None; print(''mobo'',version(''mobo''),''torch'',torch.__version__)'
    Invoke-CheckedCommand -FilePath $python -ArgumentList @('-c', $check) -Step 'Validate all algorithm and API imports'
    if ($WithGui) {
        Invoke-CheckedCommand -FilePath $python -ArgumentList @('-c', 'import PySide6; from PySide6.QtCharts import QChart; print(''PySide6'',PySide6.__version__,QChart.__name__)') -Step 'Validate GUI imports'
    }
    Invoke-CheckedCommand -FilePath $python -ArgumentList @((Join-Path $project 'tools\check_docs.py')) -Step 'Check repository documentation'
    Set-DeformEnvironment -DeformDir $DeformDir
    Write-Host 'SUCCESS: all requested dependencies installed and verified.'
    Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
    Write-Host "API: .\.venv\Scripts\mobo-api.exe"
    if ($WithGui) { Write-Host 'UI: .\.venv\Scripts\python.exe .\work_platform\main.py' }
}

if ($MyInvocation.InvocationName -ne '.') {
    try { Invoke-MoboSetup @PSBoundParameters }
    catch { Write-Host "INSTALLATION FAILED: $_" -ForegroundColor Red; exit 1 }
}
