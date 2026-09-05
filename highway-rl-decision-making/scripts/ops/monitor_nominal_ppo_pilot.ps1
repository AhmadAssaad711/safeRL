param(
    [string]$OutputDir = 'C:\agv_ppo_pilot_50k',
    [switch]$Watch,
    [int]$IntervalSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 5) {
    throw 'IntervalSeconds must be at least 5.'
}

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

function Get-PpoPilotStatus {
    param([string]$Path)

    $ScriptNeedle = 'scripts.training.run_nominal_ppo_parameter_pilot'
    $Matching = @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object {
                $_.CommandLine -like "*$ScriptNeedle*" -and
                $_.CommandLine -like "*$Path*"
            }
    )
    $PidSet = @{}
    foreach ($Process in $Matching) {
        $PidSet[[int]$Process.ProcessId] = $true
    }
    $InvocationRoots = @(
        $Matching | Where-Object { -not $PidSet.ContainsKey([int]$_.ParentProcessId) }
    )

    $Progress = @()
    foreach ($Config in @(
        'Q0_current_aligned',
        'Q1_stable',
        'Q2_exploratory',
        'Q3_conservative_update'
    )) {
        $RunDir = Join-Path $Path "seed_307\$Config"
        $Episodes = Join-Path $RunDir 'training_episodes.csv'
        $Diagnostics = Join-Path $RunDir 'checkpoint_diagnostics.csv'
        $CheckpointPointer = Join-Path $RunDir 'latest_checkpoint.json'
        $ProgressStep = 0
        $ProgressSource = 'not_started'
        if (Test-Path -LiteralPath $CheckpointPointer) {
            $Checkpoint = Get-Content -LiteralPath $CheckpointPointer -Raw | ConvertFrom-Json
            $ProgressStep = [int]$Checkpoint.timestep
            $ProgressSource = 'strict_checkpoint'
        } elseif (Test-Path -LiteralPath $Episodes) {
            $Rows = @()
            $Rows = @(Import-Csv -LiteralPath $Episodes)
            if ($Rows.Count -gt 0) {
                $ProgressStep = [int]$Rows[-1].global_timestep
                $ProgressSource = 'last_completed_episode'
            }
        }
        $DiagnosticRows = @()
        if (Test-Path -LiteralPath $Diagnostics) {
            $DiagnosticRows = @(Import-Csv -LiteralPath $Diagnostics)
        }
        $Progress += [pscustomobject]@{
            Config = $Config
            ProgressStep = $ProgressStep
            ProgressSource = $ProgressSource
            Evaluations = $DiagnosticRows.Count
            LastEvaluation = if ($DiagnosticRows.Count -gt 0) {
                [int]$DiagnosticRows[-1].model_timestep
            } else { 0 }
            FinalModel = Test-Path -LiteralPath (Join-Path $RunDir 'model_final.zip')
        }
    }

    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $LogDir = Join-Path $ProjectRoot 'artifacts\pilot_run_logs'
    $OutputIdentity = Get-PathIdentity -Path $Path
    $LatestManifest = Join-Path $LogDir "ppo50k_${OutputIdentity}.latest.json"
    $Launch = if (Test-Path -LiteralPath $LatestManifest) {
        Get-Content -LiteralPath $LatestManifest -Raw | ConvertFrom-Json
    } else { $null }
    $LatestStdout = if ($Launch -and (Test-Path -LiteralPath $Launch.stdout_log)) {
        Get-Item -LiteralPath $Launch.stdout_log
    } else { $null }
    $LatestStderr = if ($Launch -and (Test-Path -LiteralPath $Launch.stderr_log)) {
        Get-Item -LiteralPath $Launch.stderr_log
    } else { $null }

    $RankingReady = Test-Path -LiteralPath (Join-Path $Path 'ranking_final_three.csv')
    $LaunchRecorded = $null -ne $Launch
    $LaunchAgeSeconds = if ($LaunchRecorded) {
        ((Get-Date) - [datetime]$Launch.launched_at).TotalSeconds
    } else { 0.0 }
    $RunFailed = $LaunchRecorded -and $LaunchAgeSeconds -ge 15.0 -and
        $InvocationRoots.Count -eq 0 -and -not $RankingReady

    [pscustomobject]@{
        Timestamp = Get-Date
        InvocationCount = $InvocationRoots.Count
        InvocationRootPids = (($InvocationRoots | ForEach-Object { $_.ProcessId }) -join ',')
        AllMatchingPythonPids = (($Matching | ForEach-Object { $_.ProcessId }) -join ',')
        LaunchRecorded = $LaunchRecorded
        LaunchAgeSeconds = [math]::Round($LaunchAgeSeconds, 1)
        RankingReady = $RankingReady
        RunFailed = $RunFailed
        StdoutLog = if ($LatestStdout) { $LatestStdout.FullName } else { '' }
        StderrLog = if ($LatestStderr) { $LatestStderr.FullName } else { '' }
        StderrBytes = if ($LatestStderr) { $LatestStderr.Length } else { 0 }
        Progress = $Progress
    }
}

do {
    $Status = Get-PpoPilotStatus -Path $OutputDir
    $Status | Select-Object Timestamp, InvocationCount, InvocationRootPids,
        AllMatchingPythonPids, LaunchRecorded, RankingReady, RunFailed,
        LaunchAgeSeconds, StderrBytes, StdoutLog, StderrLog |
        Format-List
    $Status.Progress | Format-Table -AutoSize
    if ($Watch -and -not $Status.RankingReady -and -not $Status.RunFailed) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Watch -and -not $Status.RankingReady -and -not $Status.RunFailed)

if ($Status.RunFailed) {
    Write-Error "PPO pilot process exited without producing ranking_final_three.csv. Inspect the reported logs."
}
