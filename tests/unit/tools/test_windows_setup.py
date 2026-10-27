"""Execute Windows installer regressions in PowerShell 5.1 and 7 without downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts' / 'setup_env.ps1'
SHELLS = [entry for entry in ('powershell.exe', 'pwsh.exe') if shutil.which(entry)]
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows installer')


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.fixture(params=SHELLS or ['powershell.exe'])
def run_ps(request, tmp_path):
    def run(body, expected=0):
        test_script = tmp_path / 'installer-test.ps1'
        test_script.write_text(
            "$ErrorActionPreference='Stop'\n. " + quote(SCRIPT) + '\n' + body,
            encoding='utf-8-sig',
        )
        result = subprocess.run(
            [request.param, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(test_script)],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout
    return run


def test_existing_python_is_detected(run_ps):
    output = run_ps(f"Get-PythonInfo -FilePath {quote(sys.executable)} | ConvertTo-Json -Compress")
    info = json.loads(output)
    assert info['bits'] == 64 and info['minor'] in (11, 12)


def test_probe_stderr_and_nonzero_exit_do_not_abort_discovery(run_ps, tmp_path):
    bad = tmp_path / 'missing-runtime.py'
    bad.write_text("import sys; sys.stderr.write('No suitable Python runtime found\\n'); sys.exit(103)")
    output = run_ps(f"""
$bad = Get-PythonInfo -FilePath {quote(sys.executable)} -PrefixArguments @({quote(bad)})
if ($bad) {{ throw 'Bad runtime accepted' }}
$good = Get-PythonInfo -FilePath {quote(sys.executable)}
if (-not $good) {{ throw 'Fallback probe did not run' }}
Write-Output 'FALLBACK_OK'
""")
    assert 'FALLBACK_OK' in output


def test_launcher_falls_back_from_312_to_311(run_ps, tmp_path):
    output = run_ps(f"""
function Get-Command {{ param($Name); if ($Name -eq 'py') {{ return [pscustomobject]@{{Source='py.exe'}} }} }}
function Get-PythonInfo {{
    param($FilePath, $PrefixArguments)
    if ($PrefixArguments -contains '-3.11') {{ return [pscustomobject]@{{minor=11;bits=64}} }}
    return $null
}}
$found = Get-CompatiblePython -ProjectDir {quote(tmp_path)}
if ($found.minor -ne 11) {{ throw 'Missing launcher fallback' }}
Write-Output 'LAUNCHER_OK'
""")
    assert 'LAUNCHER_OK' in output


def test_native_failure_throws_and_prevents_following_steps(run_ps):
    output = run_ps(f"""
try {{
    Invoke-CheckedCommand -FilePath {quote(sys.executable)} -ArgumentList @('-c','import sys; sys.exit(17)') -Step 'pip test'
    throw 'Command failure ignored'
}} catch {{
    if ($_ -notmatch 'exit code 17') {{ throw }}
    Write-Output 'STOPPED_ON_FAILURE'
}}
""")
    assert 'STOPPED_ON_FAILURE' in output


def test_native_stderr_with_zero_exit_is_success(run_ps, tmp_path):
    good = tmp_path / 'pip-warning.py'
    good.write_text("import sys; sys.stderr.write('a non-fatal warning\\n')")
    output = run_ps(f"""
Invoke-CheckedCommand -FilePath {quote(sys.executable)} -ArgumentList @({quote(good)}) -Step 'pip warning'
Write-Output 'SUCCESS_AFTER_WARNING'
""")
    assert 'SUCCESS_AFTER_WARNING' in output


def test_recreate_preserves_old_environment(run_ps, tmp_path):
    project = tmp_path / 'project with spaces'
    (project / '.venv').mkdir(parents=True)
    (project / '.venv' / 'keep.txt').write_text('original environment')
    run_ps(f'Backup-ProjectVenv -ProjectDir {quote(project)}')
    assert not (project / '.venv').exists()
    assert next(project.glob('.venv.backup-*/keep.txt')).read_text() == 'original environment'


def test_unsigned_installer_is_rejected(run_ps, tmp_path):
    installer = tmp_path / 'unsigned.exe'
    installer.write_bytes(b'not an installer')
    output = run_ps(f"""
function Get-AuthenticodeSignature {{ return [pscustomobject]@{{Status='NotSigned';SignerCertificate=$null}} }}
try {{
    Get-VerifiedInstaller -Url 'https://www.python.org/test' -Destination {quote(installer)} -Publisher 'Python Software Foundation'
    throw 'Signature bypassed'
}} catch {{
    if ($_ -notmatch 'signature verification failed') {{ throw }}
    Write-Output 'UNSIGNED_REJECTED'
}}
""")
    assert 'UNSIGNED_REJECTED' in output
    assert not installer.exists()


def mock_environment():
    return f"""
$script:steps = [System.Collections.Generic.List[object]]::new()
function Get-CompatiblePython {{ return [pscustomobject]@{{version='3.12';executable={quote(sys.executable)};base={quote(sys.executable)}}} }}
function Get-PythonInfo {{ return [pscustomobject]@{{version='3.12';executable={quote(sys.executable)};base={quote(sys.executable)}}} }}
function Install-VCRuntime {{ $script:steps.Add('VC_RUNTIME') }}
function Set-DeformEnvironment {{ $script:steps.Add('DEFORM') }}
function Invoke-CheckedCommand {{
    param($FilePath,$ArgumentList,$Step)
    $script:steps.Add([pscustomobject]@{{step=$Step;args=$ArgumentList}})
}}
"""


def test_complete_install_includes_gui_cpu_torch_and_import_validation(run_ps):
    output = run_ps(mock_environment() + """
Invoke-MoboSetup -WithGui
Write-Output ('STEPS_JSON=' + (ConvertTo-Json -InputObject $script:steps -Depth 5 -Compress))
""")
    steps = json.loads(output.split('STEPS_JSON=')[1])
    assert steps[0] == 'VC_RUNTIME' and steps[-1] == 'DEFORM'
    commands = [row for row in steps if isinstance(row, dict)]
    assert any('https://download.pytorch.org/whl/cpu' in row['args'] and '--no-deps' in row['args'] for row in commands)
    dependencies = next(row for row in commands if row['step'].startswith('Install locked'))
    assert any(arg.endswith('dev.txt') for arg in dependencies['args'])
    assert any(arg.endswith('gui.txt') for arg in dependencies['args'])
    assert any(row['args'] == ['-m', 'pip', 'check'] for row in commands)
    assert any('load_runtime_dependencies' in ' '.join(row['args']) for row in commands)
    assert any('QtCharts' in ' '.join(row['args']) for row in commands)
    assert 'SUCCESS: all requested' in output


def test_failed_pipeline_never_prints_success(run_ps):
    output = run_ps(mock_environment() + """
function Invoke-CheckedCommand { throw 'SIMULATED_PIP_FAILURE' }
try { Invoke-MoboSetup -WithGui; throw 'Failure ignored' }
catch {
    if ($_ -notmatch 'SIMULATED_PIP_FAILURE') { throw }
    Write-Output 'INSTALL_FAILED_AS_EXPECTED'
}
""")
    assert 'INSTALL_FAILED_AS_EXPECTED' in output
    assert 'SUCCESS: all requested' not in output
    assert 'DEFORM' not in output


def test_cmd_entry_preserves_failure_exit_code(run_ps, tmp_path):
    project = tmp_path / 'entry with spaces'
    (project / 'scripts').mkdir(parents=True)
    entry = project / 'install_windows.cmd'
    shutil.copy2(ROOT / 'install_windows.cmd', entry)
    (project / 'scripts' / 'setup_env.ps1').write_text('exit 23')
    output = run_ps(f"$env:MOBO_SETUP_NO_PAUSE='1'\n& {quote(entry)}\nexit $global:LASTEXITCODE", expected=23)
    assert 'Installation failed' in output


@pytest.mark.parametrize('minor,bits', [(10, 64), (13, 64), (12, 32)])
def test_incompatible_python_is_rejected(run_ps, tmp_path, minor, bits):
    fake = tmp_path / 'unsupported.py'
    fake.write_text('import json; print(json.dumps(' + repr({'major': 3, 'minor': minor, 'bits': bits}) + '))')
    output = run_ps(f"""
$found = Get-PythonInfo -FilePath {quote(sys.executable)} -PrefixArguments @({quote(fake)})
if ($found) {{ throw 'Incompatible Python accepted' }}
Write-Output 'REJECTED'
""")
    assert 'REJECTED' in output


def test_missing_python_bootstraps_before_installing_dependencies(run_ps):
    output = run_ps(mock_environment() + f"""
function Get-CompatiblePython {{ return $null }}
function Install-ProjectPython {{
    $script:steps.Add('BOOTSTRAP_PYTHON')
    return [pscustomobject]@{{version='3.12';executable={quote(sys.executable)};base={quote(sys.executable)}}}
}}
Invoke-MoboSetup -WithGui
Write-Output ('STEPS_JSON=' + (ConvertTo-Json -InputObject $script:steps -Depth 5 -Compress))
""")
    steps = json.loads(output.split('STEPS_JSON=')[1])
    assert steps[:2] == ['BOOTSTRAP_PYTHON', 'VC_RUNTIME']
    assert 'SUCCESS: all requested' in output


def test_no_auto_python_does_not_download(run_ps):
    output = run_ps(mock_environment() + """
function Get-CompatiblePython { return $null }
function Install-ProjectPython { throw 'UNEXPECTED_DOWNLOAD' }
try { Invoke-MoboSetup -NoAutoPython; throw 'Missing Python ignored' }
catch {
    if ($_ -notmatch 'No compatible Python found') { throw }
    Write-Output 'NO_DOWNLOAD'
}
""")
    assert 'NO_DOWNLOAD' in output


def test_python_installer_failure_is_not_accepted(run_ps, tmp_path):
    output = run_ps(f"""
function Get-VerifiedInstaller {{ return 'signed-installer.exe' }}
function Start-Process {{ return [pscustomobject]@{{ExitCode=1603}} }}
try {{ Install-ProjectPython -ProjectDir {quote(tmp_path)}; throw 'Installer failure ignored' }}
catch {{
    if ($_ -notmatch 'Python installer failed: 1603') {{ throw }}
    Write-Output 'INSTALLER_FAILED'
}}
""")
    assert 'INSTALLER_FAILED' in output


def test_registered_python_is_found_without_path_or_launcher(run_ps, tmp_path):
    output = run_ps(f"""
function Get-Command {{
    param($Name)
    if ($Name -eq 'registered-python.exe') {{ return [pscustomobject]@{{Source=$Name}} }}
}}
function Get-Item {{
    param($LiteralPath)
    if ($LiteralPath -like '*PythonCore*3.12*') {{
        $key = [pscustomobject]@{{}}
        $key | Add-Member -MemberType ScriptMethod -Name GetValue -Value {{
            param($name)
            if ($name -eq 'ExecutablePath') {{ return 'registered-python.exe' }}
            return $null
        }}
        return $key
    }}
}}
function Get-PythonInfo {{
    param($FilePath)
    if ($FilePath -eq 'registered-python.exe') {{ return [pscustomobject]@{{minor=12;bits=64}} }}
}}
$info = Get-CompatiblePython -ProjectDir {quote(tmp_path)}
if ($info.minor -ne 12) {{ throw 'Registered runtime not discovered' }}
Write-Output 'REGISTRY_OK'
""")
    assert 'REGISTRY_OK' in output
