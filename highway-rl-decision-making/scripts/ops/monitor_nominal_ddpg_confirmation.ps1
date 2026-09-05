param(
    [string]$OutputDir = 'C:\agv_pilot_confirm_final',
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

function Get-ConfirmationStatus {
    param([string]$Path)

    $ScriptNeedle = 'scripts.training.run_nominal_ddpg_parameter_pilot'
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
    foreach ($Seed in @(307, 1307, 2307)) {
        foreach ($Config in @('P0_current', 'P2_more_exploration')) {
            $RunDir = Join-Path $Path "seed_$Seed\$Config"
            $Episodes = Join-Path $RunDir 'training_episodes.csv'
            $Diagnostics = Join-Path $RunDir 'checkpoint_diagnostics.csv'
            $Calibration = Join-Path $RunDir 'critic_calibration_samples.csv'
            $CheckpointPointer = Join-Path $RunDir 'latest_checkpoint.json'
            $ProgressStep = 0
            $ProgressSource = 'not_started'
            if (Test-Path -LiteralPath $CheckpointPointer) {
                $Checkpoint = Get-Content -LiteralPath $CheckpointPointer -Raw | ConvertFrom-Json
                $ProgressStep = [int]$Checkpoint.timestep
                $ProgressSource = 'strict_checkpoint'
            } elseif (Test-Path -LiteralPath $Episodes) {
                $LastEpisode = Get-Content -LiteralPath $Episodes -Tail 1
                if ($LastEpisode) {
                    $ProgressStep = [int](($LastEpisode -split ',')[2])
                    $ProgressSource = 'last_completed_episode'
                }
            }
            $Evaluations = 0
            $LastEvaluation = 0
            if (Test-Path -LiteralPath $Diagnostics) {
                $DiagnosticRows = @(Import-Csv -LiteralPath $Diagnostics)
                $Evaluations = $DiagnosticRows.Count
                if ($Evaluations -gt 0) {
                    $LastEvaluation = [int]$DiagnosticRows[-1].model_timestep
                }
            }
            $Progress += [pscustomobject]@{
                Seed = $Seed
                Config = $Config
                ProgressStep = $ProgressStep
                ProgressSource = $ProgressSource
                Evaluations = $Evaluations
                LastEvaluation = $LastEvaluation
                CalibrationBytes = if (Test-Path -LiteralPath $Calibration) {
                    (Get-Item -LiteralPath $Calibration).Length
                } else { 0 }
                FinalModel = Test-Path -LiteralPath (Join-Path $RunDir 'model_final.zip')
            }
        }
    }

    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $LogDir = Join-Path $ProjectRoot 'artifacts\pilot_run_logs'
    $OutputIdentity = Get-PathIdentity -Path $Path
    $LatestManifest = Join-Path $LogDir "confirm_${OutputIdentity}.latest.json"
    $Launch = if (Test-Path -LiteralPath $LatestManifest) {
        Get-Content -LiteralPath $LatestManifest -Raw | ConvertFrom-Json
    } else { $null }
    $LatestStdout = if ($Launch -and (Test-Path -LiteralPath $Launch.stdout_log)) {
        Get-Item -LiteralPath $Launch.stdout_log
    } else { $null }
    $LatestStderr = if ($Launch -and (Test-Path -LiteralPath $Launch.stderr_log)) {
        Get-Item -LiteralPath $Launch.stderr_log
    } else { $null }

    [pscustomobject]@{
        Timestamp = Get-Date
        InvocationCount = $InvocationRoots.Count
        InvocationRootPids = ($InvocationRoots.ProcessId -join ',')
        AllMatchingPythonPids = ($Matching.ProcessId -join ',')
        RankingReady = Test-Path -LiteralPath (Join-Path $Path 'ranking_final_three.csv')
        PairedSummaryReady = Test-Path -LiteralPath (Join-Path $Path 'paired_difference_summary.csv')
        StdoutLog = if ($LatestStdout) { $LatestStdout.FullName } else { '' }
        StderrLog = if ($LatestStderr) { $LatestStderr.FullName } else { '' }
        StderrBytes = if ($LatestStderr) { $LatestStderr.Length } else { 0 }
        Progress = $Progress
    }
}

do {
    $Status = Get-ConfirmationStatus -Path $OutputDir
    $StatusFields = @(
        'Timestamp',
        'InvocationCount',
        'InvocationRootPids',
        'AllMatchingPythonPids',
        'RankingReady',
        'PairedSummaryReady',
        'StderrBytes',
        'StdoutLog',
        'StderrLog'
    )
    $Status | Select-Object -Property $StatusFields |
        Format-List
    $Status.Progress | Format-Table -AutoSize
    if ($Watch -and -not $Status.RankingReady) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Watch -and -not $Status.RankingReady)
