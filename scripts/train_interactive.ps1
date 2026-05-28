<#
.SYNOPSIS  Phase-2 interactive trainer for CDPR_SIMULATOR.

.DESCRIPTION
    Asks for a CSV path (or URL), then a model selection (1-7 per the
    directive), then runs the appropriate train_from_csv / compare_models
    commands. Validates numeric inputs and rejects invalid menu choices
    without aborting --- the user gets a re-prompt loop until they enter
    something sensible. Opens the result folder when done.

    The CSV argument can be:
      * a local file path,
      * an http:// or https:// URL,
      * empty (Enter accepts the default).

    Optional ``--RobotConfig`` forwards a JSON robot description to both
    train_from_csv and compare_models so replay / PPO / SAC work on any
    CSV --- not just ones produced by run_simulation.py.

.PARAMETER Csv
    Optional positional CSV path or URL; overrides the interactive prompt.

.PARAMETER RobotConfig
    Optional JSON robot description (forwarded with --robot-config).

.PARAMETER ColumnMap
    Optional canonical-name=source-column overrides (forwarded with
    --column-map). Example: "px=Position X,py=Position Y,pz=Position Z".

.EXAMPLE
    .\scripts\train_interactive.ps1
    .\scripts\train_interactive.ps1 -Csv "out\foo\timeseries.csv"
    .\scripts\train_interactive.ps1 -Csv "https://example.com/data.csv"
#>
[CmdletBinding()]
param(
    [string]$Csv = "",
    [string]$RobotConfig = "",
    [string]$ColumnMap = "",
    [string]$DefaultCsv = ""
)

$ErrorActionPreference = "Stop"

# -- helpers ---------------------------------------------------------------

function Read-Int {
    param(
        [string]$Prompt,
        [int]$Default,
        [int]$Min = 1,
        [int]$Max = [int]::MaxValue
    )
    while ($true) {
        $raw = Read-Host ("{0} [{1}]" -f $Prompt, $Default)
        if (-not $raw) { return $Default }
        $parsed = 0
        if ([int]::TryParse($raw, [ref]$parsed)) {
            if ($parsed -ge $Min -and $parsed -le $Max) { return $parsed }
        }
        Write-Host ("    ! must be an integer in [{0}, {1}]. try again." -f $Min, $Max) -ForegroundColor Yellow
    }
}

function Read-MenuChoice {
    param([int]$NumOptions)
    while ($true) {
        $raw = Read-Host ("Enter choice 1-{0}" -f $NumOptions)
        $parsed = 0
        if ([int]::TryParse($raw, [ref]$parsed) -and $parsed -ge 1 -and $parsed -le $NumOptions) {
            return $parsed
        }
        Write-Host "    ! invalid choice. try again." -ForegroundColor Yellow
    }
}

function Test-CsvAvailable {
    param([string]$Path)
    if ($Path -match '^https?://') { return $true }                  # let the python loader fetch
    return (Test-Path $Path)
}

# -- 1) CSV path -----------------------------------------------------------
if (-not $Csv) {
    if (-not $DefaultCsv) {
        $DefaultCsv = "out\dissertation_8cable-circle-20260528-181124\timeseries.csv"
    }
    $Csv = Read-Host ("Enter CSV path or http(s) URL (Enter for default: {0})" -f $DefaultCsv)
    if (-not $Csv) { $Csv = $DefaultCsv }
}
if (-not (Test-CsvAvailable $Csv)) {
    Write-Host ("ERROR: CSV not found at {0}" -f $Csv) -ForegroundColor Red
    exit 1
}
if ($Csv -match '^https?://') {
    Write-Host ("Using CSV: {0}  (will be fetched)" -f $Csv) -ForegroundColor Cyan
} else {
    Write-Host ("Using CSV: {0}" -f (Resolve-Path $Csv)) -ForegroundColor Cyan
}

# -- 2) Model menu --------------------------------------------------------
Write-Host ""
Write-Host "Select model(s):"
Write-Host "  [1] PPO"
Write-Host "  [2] SAC"
Write-Host "  [3] PINN"
Write-Host "  [4] PPO + SAC"
Write-Host "  [5] PPO + PINN"
Write-Host "  [6] SAC + PINN"
Write-Host "  [7] PPO + SAC + PINN"
$choice = Read-MenuChoice -NumOptions 7
$models = switch ($choice) {
    1 { @("ppo") }
    2 { @("sac") }
    3 { @("pinn") }
    4 { @("ppo","sac") }
    5 { @("ppo","pinn") }
    6 { @("sac","pinn") }
    7 { @("ppo","sac","pinn") }
}

# -- 3) Hyper-parameters --------------------------------------------------
$epochs   = Read-Int -Prompt "Epochs (PINN/MLP)"      -Default 80    -Min 1     -Max 100000
$rlSteps  = Read-Int -Prompt "RL training steps"      -Default 5000  -Min 0     -Max 10000000
$evalEps  = Read-Int -Prompt "RL eval episodes"       -Default 3     -Min 1     -Max 1000

# -- 4) Run ---------------------------------------------------------------
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out   = Join-Path "out" ("compare-{0}" -f $stamp)
Write-Host ""
Write-Host ("[run] models      = {0}" -f ($models -join ", ")) -ForegroundColor Yellow
Write-Host ("[run] out         = {0}" -f $out)                 -ForegroundColor Yellow
Write-Host ("[run] epochs      = {0}" -f $epochs)              -ForegroundColor Yellow
Write-Host ("[run] rl_steps    = {0}" -f $rlSteps)             -ForegroundColor Yellow
Write-Host ("[run] eval_eps    = {0}" -f $evalEps)             -ForegroundColor Yellow
if ($RobotConfig) { Write-Host ("[run] robot_cfg   = {0}" -f $RobotConfig) -ForegroundColor Yellow }
if ($ColumnMap)   { Write-Host ("[run] column_map  = {0}" -f $ColumnMap)   -ForegroundColor Yellow }
Write-Host ""

# Build the argument array dynamically so we can include optional flags.
$singleArgs = @(
    "scripts\train_from_csv.py",
    "--input", $Csv,
    "--model", $models[0],
    "--epochs", "$epochs",
    "--rl-steps", "$rlSteps",
    "--eval-episodes", "$evalEps"
)
$compareArgs = @(
    "scripts\compare_models.py",
    "--input", $Csv,
    "--out", $out,
    "--models"
) + $models + @(
    "--epochs", "$epochs",
    "--rl-steps", "$rlSteps",
    "--eval-episodes", "$evalEps"
)
if ($RobotConfig) {
    $singleArgs  += @("--robot-config", $RobotConfig)
    $compareArgs += @("--robot-config", $RobotConfig)
}
if ($ColumnMap) {
    $singleArgs  += @("--column-map", $ColumnMap)
    $compareArgs += @("--column-map", $ColumnMap)
}

if ($models.Count -eq 1) {
    & python @singleArgs
} else {
    & python @compareArgs
    Write-Host ""
    Write-Host "=== ranking ===" -ForegroundColor Cyan
    $rankingPath = Join-Path $out "ranking.json"
    if (Test-Path $rankingPath) {
        Get-Content $rankingPath | ConvertFrom-Json | Format-Table
    } else {
        Write-Host "(ranking.json was not produced --- compare_models likely crashed)" -ForegroundColor Red
    }
    if (Test-Path $out) {
        Start-Process $out
    }
}
