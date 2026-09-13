<#
.SYNOPSIS
  Runs the four lateral-target trainings sequentially, in the foreground.

.DESCRIPTION
  One run at a time, each a full `scripts.training.run_ppo_cbf_progression`
  invocation on the canonical laneless contract:

    1_n_ctr    ppo_nominal            gap target, centre fallback
    2_n_field  ppo_nominal            occupancy-field target
    3_h_noy    ppo_nominal +          no lateral tracking cost (wy = 0); the
               ppo_hocbf_reward_raw   nominal is the fixed psi-scale
                                      calibration policy the runner requires
    4_h_field  ppo_hocbf_reward_raw   field target, run 3's psi scale reused

  Everything else is the canonical contract: mtm traffic, 40 vehicles,
  100/20/20, 32D observation, Q1_stable, 20 envs x 1000 steps, seed 307,
  `--potential-field-weight 0`, then a 200+200 post-train evaluation from seed
  1100000.  The protocol itself lives in the Python runner; this launcher only
  sequences, supervises, and resumes it.

  Hardening, against the faults that killed the earlier launches:
    - every argument is quoted, so the spaces in the project path survive
      Start-Process;
    - the intermittent native worker access violation is detected and the run
      relaunched from its newest checkpoint (25k granularity);
    - a run that stops writing for -StallMinutes is treated as a hung pool;
    - a run counts as complete only when every variant it trains appears in its
      post_train_200ep_kpis.csv, so run 3 cannot hand over after its nominal
      evaluation and leave two trainings running at once;
    - `--force-retrain` is passed only when the run directory holds no
      checkpoint, so a relaunch resumes instead of discarding progress.

  Results are generated output and never belong in the repository, so -OutBase
  defaults to a directory under %LOCALAPPDATA%\Temp.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\ops\run_lateral_target_ladder.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\ops\run_lateral_target_ladder.ps1 -Only 3_h_noy,4_h_field
#>
param(
  [int]$Timesteps         = 250000,
  [int]$MaxAttemptsPerRun = 100,
  [int]$StallMinutes      = 25,
  [string]$Only           = "",
  [string]$Device         = "cuda",
  [string]$Python         = "C:\Program Files\Python39\python.exe",
  [string]$OutBase        = ""
)

$ErrorActionPreference = "Continue"

# scripts/ops -> scripts -> safeRL_workspace
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($OutBase -eq "") {
  $OutBase = Join-Path $env:LOCALAPPDATA ("Temp\saferl_runs\lateral_ladder_{0}k" -f [int]($Timesteps / 1000))
}
$ladder = Join-Path $OutBase "LADDER.log"
$lock   = Join-Path $OutBase "LADDER.lock"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python executable not found: $Python"
}
New-Item -ItemType Directory -Force -Path $OutBase | Out-Null

function Log([string]$m) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Write-Host $line
  Add-Content -Path $ladder -Value $line -Encoding utf8
}

# One ladder per output base. A lock left behind by a crashed or blue-screened
# run is reclaimed; a lock held by a live powershell is not.
if (Test-Path $lock) {
  $old = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
  $alive = $null
  if ($old) { $alive = Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue }
  if ($alive -and $alive.ProcessName -match 'powershell') {
    Log "ABORT: another ladder is alive at PID $old"
    exit 1
  }
  Log "reclaiming stale lock from PID $old"
}
Set-Content -Path $lock -Value $PID -Encoding utf8

$base = @(
  "-u", "-m", "scripts.training.run_ppo_cbf_progression",
  "--project-root", $project,
  "--traffic-model", "mtm", "--device", $Device, "--timesteps", "$Timesteps",
  "--ppo-config", "Q1_stable", "--n-envs", "20", "--n-steps", "1000",
  "--batch-size", "100", "--n-epochs", "10", "--checkpoint-freq", "25000",
  "--seeds", "307",
  "--remove-vehicle-dimensions", "--expose-target-y",
  "--task-distance-m", "1000", "--task-max-policy-steps", "3000",
  "--post-train-eval-episodes", "200", "--post-train-eval-workers", "20",
  "--post-train-eval-seed-start", "1100000", "--post-train-evaluate-reused",
  "--skip-evaluation", "--skip-counterfactual",
  "--potential-field-weight", "0"
)

# Need = the variants whose KPI rows must exist before the run counts as done.
$runs = @(
  @{ Label = "1_n_ctr";   Need = @("ppo_nominal")
     Extra = @("--variants","ppo_nominal",
               "--lateral-target","gap","--lateral-target-fallback","center") },
  @{ Label = "2_n_field"; Need = @("ppo_nominal")
     Extra = @("--variants","ppo_nominal","--lateral-target","field") },
  @{ Label = "3_h_noy";   Need = @("ppo_nominal","ppo_hocbf_reward_raw")
     Extra = @("--variants","ppo_nominal","ppo_hocbf_reward_raw",
               "--lateral-y-weight","0","--hocbf-calibration-steps","200") },
  @{ Label = "4_h_field"; Need = @("ppo_hocbf_reward_raw")
     Extra = @("--variants","ppo_hocbf_reward_raw","--lateral-target","field") }
)

if ($Only -ne "") {
  $keep = $Only.Split(",") | ForEach-Object { $_.Trim() }
  $runs = @($runs | Where-Object { $keep -contains $_.Label })
}

# Start-Process does not quote array elements, so do it here.
function Quote([string[]]$a) {
  ($a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
}

function Run-Complete([string]$dir, [string[]]$need) {
  $kpi = Join-Path $dir "post_train_200ep_kpis.csv"
  if (-not (Test-Path $kpi)) { return $false }
  $txt = Get-Content $kpi -Raw -ErrorAction SilentlyContinue
  foreach ($v in $need) { if ($txt -notmatch [regex]::Escape($v)) { return $false } }
  return $true
}

function Has-Checkpoint([string]$dir) {
  $c = Get-ChildItem -Path $dir -Recurse -Filter "*.zip" -ErrorAction SilentlyContinue |
       Select-Object -First 1
  return [bool]$c
}

function Newest-Write([string]$dir) {
  $f = Get-ChildItem -Path $dir -Recurse -File -ErrorAction SilentlyContinue |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($f) { return $f.LastWriteTime }
  return (Get-Date)
}

function Kill-Tree([int]$parentId) {
  $kids = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
          Where-Object { $_.ParentProcessId -eq $parentId }
  foreach ($k in $kids) { Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue }
  Stop-Process -Id $parentId -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 5
}

Log "=== ladder starting: $($runs.Count) run(s), $Timesteps steps each, PID $PID, out $OutBase ==="

$psiScale = $null

foreach ($run in $runs) {
  $label  = $run.Label
  $outDir = Join-Path $OutBase $label
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  if (Run-Complete $outDir $run.Need) { Log "$label already complete, skipping"; continue }

  $extra = @() + $run.Extra

  # Runs 3 and 4 must share one psi normalization. Run 3 calibrates it from
  # its own fixed nominal policy; run 4 reuses that number, never its own.
  if ($label -eq "4_h_field") {
    if ($psiScale -eq $null) {
      $calib = Join-Path (Join-Path $OutBase "3_h_noy") "hocbf_scale_calibration.json"
      if (Test-Path $calib) {
        $psiScale = (Get-Content $calib -Raw | ConvertFrom-Json).selected_psi_scale
      }
    }
    if ($psiScale -eq $null) {
      Log "$label SKIPPED: no psi scale from 3_h_noy (hocbf_scale_calibration.json missing)"
      continue
    }
    $extra += @("--hocbf-psi-scale", ("{0:R}" -f [double]$psiScale))
    Log "$label using shared psi scale $psiScale from 3_h_noy"
  }

  Log "--- $label starting ---"
  $done = $false

  for ($attempt = 1; $attempt -le $MaxAttemptsPerRun; $attempt++) {
    $argv = $base + $extra + @("--output-dir", $outDir, "--tensorboard-run-label", $label)
    # Train clean only when there is nothing to resume from; a retry after a
    # crash must keep the checkpoints it already paid for.
    if (-not (Has-Checkpoint $outDir)) { $argv += "--force-retrain" }

    $so = Join-Path $outDir "attempt$attempt.out"
    $se = Join-Path $outDir "attempt$attempt.err"
    Log "$label attempt $attempt starting"

    $started = Get-Date
    $proc = Start-Process -FilePath $Python -ArgumentList (Quote $argv) `
              -WorkingDirectory $project -RedirectStandardOutput $so `
              -RedirectStandardError $se -PassThru -WindowStyle Hidden

    while ($true) {
      Start-Sleep -Seconds 30
      if (Run-Complete $outDir $run.Need) { $done = $true; break }

      $txt = ""
      if (Test-Path $so) { $txt += (Get-Content $so -Raw -ErrorAction SilentlyContinue) }
      if (Test-Path $se) { $txt += (Get-Content $se -Raw -ErrorAction SilentlyContinue) }

      if ($txt -match "access violation" -or $txt -match "error return without exception") {
        Log "$label attempt $attempt FAULT: native worker fault, killing pool"
        Kill-Tree $proc.Id
        break
      }
      # A config error dies on arrival. A crash after real progress is
      # transient and must still be retried, so only the fast exit aborts.
      $alive = (New-TimeSpan -Start $started -End (Get-Date)).TotalSeconds
      if ($proc.HasExited -and $proc.ExitCode -ne 0 -and $alive -lt 120) {
        Log "$label attempt $attempt CONFIG ERROR, exit $($proc.ExitCode) after $([int]$alive)s. Not retrying."
        Log ("    " + (($txt -split "`n" | Where-Object { $_.Trim() } | Select-Object -Last 3) -join " | "))
        $attempt = $MaxAttemptsPerRun
        break
      }
      $idle = (New-TimeSpan -Start (Newest-Write $outDir) -End (Get-Date)).TotalMinutes
      if ($idle -gt $StallMinutes) {
        Log "$label attempt $attempt STALLED: no writes for $([int]$idle) min, killing pool"
        Kill-Tree $proc.Id
        break
      }
      if ($proc.HasExited) {
        Log "$label attempt $attempt exited with $($proc.ExitCode)"
        break
      }
    }

    if ($done) { break }
    if (Run-Complete $outDir $run.Need) { $done = $true; break }
  }

  if ($done) {
    Log "--- $label COMPLETE ---"
    if ($label -eq "3_h_noy") {
      $calib = Join-Path $outDir "hocbf_scale_calibration.json"
      if (Test-Path $calib) {
        $psiScale = (Get-Content $calib -Raw | ConvertFrom-Json).selected_psi_scale
        Log "3_h_noy calibrated psi scale = $psiScale"
      } else {
        Log "WARNING: 3_h_noy left no hocbf_scale_calibration.json; run 4 cannot start"
      }
    }
  } else {
    Log "--- $label GAVE UP after $MaxAttemptsPerRun attempts, continuing ---"
  }
}

Log "=== ladder finished ==="
Remove-Item $lock -Force -ErrorAction SilentlyContinue
