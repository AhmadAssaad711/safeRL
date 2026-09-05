param(
    [string]$OutputDir = 'C:\agv_ppo_pilot_50k',
    [string]$Python = 'C:\agv312\Scripts\python.exe',
    [switch]$Resume,
    [switch]$Foreground
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Get-PathIdentity {
    param([string]$Path)
    $Normalized = [System.IO.Path]::GetFullPath($Path).ToLowerInvariant()
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Normalized)
        $Digest = $Hasher.ComputeHash($Bytes)
        return ([System.BitConverter]::ToString($Digest) -replace '-', '').Substring(0, 12).ToLowerInvariant()
    } finally {
        $Hasher.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not $Resume -and (Test-Path -LiteralPath $OutputDir)) {
    $Existing = @(Get-ChildItem -LiteralPath $OutputDir -Force -ErrorAction Stop)
    if ($Existing.Count -gt 0) {
        throw "Fresh PPO pilot output is not empty: $OutputDir"
    }
}

$PilotArgs = @(
    '-m', 'scripts.training.run_nominal_ppo_parameter_pilot',
    '--stage', 'screen',
    '--configs',
        'Q0_current_aligned',
        'Q1_stable',
        'Q2_exploratory',
        'Q3_conservative_update',
    '--seeds', '307',
    '--timesteps', '50000',
    '--checkpoint-interval', '10000',
    '--eval-seeds',
        '900000', '900001', '900002', '900003', '900004',
        '900005', '900006', '900007', '900008', '900009',
    '--eval-timesteps', '800',
    '--strict-checkpoint-retention', '2',
    '--device', 'cpu',
    '--output-dir', $OutputDir
)
if ($Resume) {
    $PilotArgs += '--resume'
}

if ($Foreground) {
    Push-Location $ProjectRoot
    try {
        & $Python @PilotArgs
        $ExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    exit $ExitCode
}

$LogDir = Join-Path $ProjectRoot 'artifacts\pilot_run_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$OutputIdentity = Get-PathIdentity -Path $OutputDir
$Stdout = Join-Path $LogDir "ppo50k_${OutputIdentity}_$Stamp.stdout.log"
$Stderr = Join-Path $LogDir "ppo50k_${OutputIdentity}_$Stamp.stderr.log"
$LatestManifest = Join-Path $LogDir "ppo50k_${OutputIdentity}.latest.json"

$ProcessArgs = @($PilotArgs)
$OutputArgumentIndex = [Array]::IndexOf($ProcessArgs, '--output-dir') + 1
$ProcessArgs[$OutputArgumentIndex] = '"' + $ProcessArgs[$OutputArgumentIndex] + '"'
$Process = Start-Process -FilePath $Python `
    -ArgumentList $ProcessArgs `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

$LaunchRecord = [ordered]@{
    output_directory = [System.IO.Path]::GetFullPath($OutputDir)
    invocation_root_pid = $Process.Id
    stdout_log = $Stdout
    stderr_log = $Stderr
    resume = [bool]$Resume
    launched_at = (Get-Date).ToString('o')
}
$LaunchRecord | ConvertTo-Json | Set-Content -LiteralPath $LatestManifest -Encoding UTF8

[pscustomobject]@{
    InvocationRootPid = $Process.Id
    OutputDirectory = $OutputDir
    StdoutLog = $Stdout
    StderrLog = $Stderr
    LatestLaunchManifest = $LatestManifest
    Resume = [bool]$Resume
    Note = 'The launcher starts the prepared pilot; this setup script does not run it automatically.'
}
