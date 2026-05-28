<#
.SYNOPSIS
    Drive the live Render FastAPI from PowerShell --- run a simulation,
    request a plot, save the PNG to disk, open it in the default viewer.

.DESCRIPTION
    The deployed backend at https://cdpr-api.onrender.com exposes:
      GET  /health        -- liveness check
      GET  /robots        -- catalogue
      POST /simulate      -- forward integration (returns JSON timeseries)
      POST /plot          -- render a plot (returns {"png_base64": "..."})
      POST /workspace     -- WCW / WFW grid scan

    The free tier sleeps after 15 min idle. The first request can take
    ~50 s while the container wakes; later requests respond in under a
    second. This script reuses the cached connection so the cold start
    only happens once per PowerShell session.

.PARAMETER Action
    One of: health, robots, simulate, plot, workspace.

.PARAMETER Robot
    Reference robot name. Defaults to ipanema_class.

.PARAMETER Kind
    Trajectory kind: hold, line, circle, lissajous. Default circle.

.PARAMETER Duration
    Simulation length in seconds. Default 1.5.

.PARAMETER Dt
    Integration step in seconds. Default 2e-3.

.PARAMETER PlotKind
    For Action=plot: which plot to render (position, cable_tensions,
    cable_lengths, tracking_error, condition_number,
    trajectory_xy / xz / yz).

.PARAMETER OutDir
    Where to drop response JSON / PNG. Defaults to ./out/render-<stamp>/.

.PARAMETER OpenPng
    For Action=plot: open the saved PNG in the default viewer.

.PARAMETER BaseUrl
    Override the API root. Defaults to the live Render URL.

.EXAMPLE
    # Liveness check (also a good way to warm the worker before a big call).
    .\scripts\call_render.ps1 -Action health

.EXAMPLE
    # Run a circle and save the JSON response.
    .\scripts\call_render.ps1 -Action simulate -Robot ipanema_class -Kind circle

.EXAMPLE
    # Render a tension plot and pop it open.
    .\scripts\call_render.ps1 -Action plot -PlotKind cable_tensions -OpenPng

#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("health", "robots", "simulate", "plot", "workspace")]
    [string]$Action,

    [ValidateSet("point_mass_3d", "planar_translational", "ipanema_class", "cogiro_class")]
    [string]$Robot = "ipanema_class",

    [ValidateSet("hold", "line", "circle", "lissajous")]
    [string]$Kind = "circle",

    [double]$Duration = 1.5,
    [double]$Dt = 2e-3,
    [double]$PayloadMass = 0.0,

    [ValidateSet("min_norm", "centered", "preferred")]
    [string]$Objective = "centered",

    [ValidateSet(
        "position", "velocity", "angular_velocity",
        "cable_lengths", "cable_tensions",
        "tracking_error", "condition_number",
        "trajectory_xy", "trajectory_xz", "trajectory_yz"
    )]
    [string]$PlotKind = "position",

    [string]$OutDir = "",
    [switch]$OpenPng,
    [string]$BaseUrl = "https://cdpr-api.onrender.com"
)

$ErrorActionPreference = "Stop"

# Resolve output directory.
if (-not $OutDir) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutDir = Join-Path -Path (Get-Location) -ChildPath "out\render-$stamp"
}
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

Write-Host ("[call_render] base   = {0}" -f $BaseUrl)
Write-Host ("[call_render] action = {0}" -f $Action)
Write-Host ("[call_render] outdir = {0}" -f $OutDir)

function Write-Json {
    param([string]$Path, $Object)
    $Object | ConvertTo-Json -Depth 12 | Out-File -FilePath $Path -Encoding utf8
    Write-Host ("[call_render] wrote  -> {0}" -f $Path)
}

# Build the per-action request body.
$body = $null
switch ($Action) {
    "health"  { $url = "$BaseUrl/health";  $method = "GET" }
    "robots"  { $url = "$BaseUrl/robots";  $method = "GET" }
    "simulate" {
        $url = "$BaseUrl/simulate"; $method = "POST"
        $params = switch ($Kind) {
            "line" {
                @{ start = @(0.0, 0.0, 0.5); end = @(0.3, 0.0, 0.5) }
            }
            "circle" {
                @{
                    center = @(0.0, 0.0, 0.5)
                    radius = 0.2
                    axis = @(0.0, 0.0, 1.0)
                    angle_span = [math]::PI * 2.0
                }
            }
            "lissajous" {
                @{
                    center = @(0.0, 0.0, 0.5)
                    amplitudes = @(0.2, 0.2, 0.0)
                    frequencies = @(1.0, 2.0, 0.0)
                    phases = @(0.0, [math]::PI / 2.0, 0.0)
                }
            }
            default { @{} }
        }
        $body = @{
            robot = $Robot
            payload_mass = $PayloadMass
            duration = $Duration
            dt = $Dt
            tension_objective = $Objective
            gravity = @(0.0, 0.0, -9.81)
            trajectory = @{ kind = $Kind; duration = $Duration; params = $params }
        }
    }
    "plot" {
        $url = "$BaseUrl/plot"; $method = "POST"
        $body = @{
            kind = $PlotKind
            simulate = @{
                robot = $Robot
                payload_mass = $PayloadMass
                duration = $Duration
                dt = $Dt
                tension_objective = $Objective
                gravity = @(0.0, 0.0, -9.81)
                trajectory = @{
                    kind = $Kind
                    duration = $Duration
                    params = @{
                        center = @(0.0, 0.0, 0.5)
                        radius = 0.2
                        axis = @(0.0, 0.0, 1.0)
                        angle_span = [math]::PI * 2.0
                    }
                }
            }
        }
    }
    "workspace" {
        $url = "$BaseUrl/workspace"; $method = "POST"
        $body = @{
            robot = $Robot
            xlim = @(-0.6, 0.6); ylim = @(-0.4, 0.4); zlim = @(-0.4, 0.4)
            resolution = 8
            kind = "wcw"
        }
    }
}

# Invoke. Render's free tier cold-start can exceed PowerShell's default
# 100-s WebRequest timeout the first time you hit /simulate; bump it.
$invokeArgs = @{ Uri = $url; Method = $method; TimeoutSec = 180 }
if ($body) {
    $invokeArgs.Body = ($body | ConvertTo-Json -Depth 12)
    $invokeArgs.ContentType = "application/json"
}

Write-Host ("[call_render] {0} {1} (timeout 180s --- cold start can take ~50 s)" -f $method, $url)
$t0 = Get-Date
try {
    $response = Invoke-RestMethod @invokeArgs
} catch {
    Write-Host ("[call_render] HTTP error: {0}" -f $_.Exception.Message) -ForegroundColor Red
    if ($_.Exception.Response) {
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $errBody = $reader.ReadToEnd()
        Write-Host ("[call_render] response body:`n{0}" -f $errBody) -ForegroundColor Red
    }
    exit 1
}
$elapsed = (Get-Date) - $t0
Write-Host ("[call_render] {0:N2} s" -f $elapsed.TotalSeconds)

# Persist the response.
$jsonPath = Join-Path $OutDir "$Action.json"
Write-Json -Path $jsonPath -Object $response

# Special handling for /plot: decode the base64 PNG to disk.
if ($Action -eq "plot" -and $response.png_base64) {
    $pngPath = Join-Path $OutDir "$PlotKind.png"
    $bytes = [Convert]::FromBase64String($response.png_base64)
    [IO.File]::WriteAllBytes($pngPath, $bytes)
    Write-Host ("[call_render] PNG    -> {0}" -f $pngPath)
    if ($OpenPng) {
        Start-Process $pngPath
    }
}

Write-Host "[call_render] done."
