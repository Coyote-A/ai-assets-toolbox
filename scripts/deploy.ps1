# Build and push Docker images for RunPod serverless workers (Windows PowerShell)
# Usage: .\scripts\deploy.ps1 -DockerHubUsername <username> [-Worker upscale|caption|all] [-Tag <tag>]
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - Logged in to Docker Hub: docker login
#   - Run from the repository root

[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0, HelpMessage = "Your Docker Hub username")]
    [string]$DockerHubUsername,

    [Parameter(Mandatory = $false, Position = 1, HelpMessage = "Worker to deploy: upscale, caption, or all (default: all)")]
    [ValidateSet("upscale", "caption", "all")]
    [string]$Worker = "all",

    [Parameter(Mandatory = $false, Position = 2, HelpMessage = "Image tag (default: latest)")]
    [string]$Tag = "latest",

    [Parameter(Mandatory = $false, HelpMessage = "CivitAI API token for baking LoRAs into the upscale image at build time")]
    [string]$CivitaiApiToken = $env:CIVITAI_API_TOKEN
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

# ---------------------------------------------------------------------------
# Helper: build and push a single worker
# ---------------------------------------------------------------------------
function Build-And-Push {
    param([string]$WorkerName)

    $ImageName  = "ai-assets-${WorkerName}"
    $FullImage  = "${DockerHubUsername}/${ImageName}:${Tag}"
    $Context    = Join-Path $RepoRoot "workers\${WorkerName}"
    $Dockerfile = Join-Path $Context "Dockerfile"

    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  AI Assets Toolbox — ${WorkerName} Worker Deployment" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Image : $FullImage"
    Write-Host "  Source: $Context"
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------
    Write-Host "[1/3] Building Docker image..." -ForegroundColor Yellow
    if ($WorkerName -eq "upscale" -and $CivitaiApiToken) {
        Write-Host "  Passing CIVITAI_API_TOKEN build arg to bake LoRAs into image..." -ForegroundColor DarkCyan
        docker build `
            --platform linux/amd64 `
            --build-arg "CIVITAI_API_TOKEN=${CivitaiApiToken}" `
            -t $FullImage `
            -f $Dockerfile `
            $Context
    } else {
        docker build `
            --platform linux/amd64 `
            -t $FullImage `
            -f $Dockerfile `
            $Context
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    # -----------------------------------------------------------------------
    # Tag
    # -----------------------------------------------------------------------
    Write-Host ""
    Write-Host "[2/3] Tagging image as $FullImage..." -ForegroundColor Yellow
    docker tag $FullImage $FullImage

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker tag failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    # -----------------------------------------------------------------------
    # Push
    # -----------------------------------------------------------------------
    Write-Host ""
    Write-Host "[3/3] Pushing image to Docker Hub..." -ForegroundColor Yellow
    docker push $FullImage

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker push failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    # -----------------------------------------------------------------------
    # Next steps
    # -----------------------------------------------------------------------
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  ${WorkerName} worker build and push complete!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps — create a RunPod Serverless Endpoint:"
    Write-Host ""
    Write-Host "  1. Go to https://www.runpod.io/console/serverless"
    Write-Host "  2. Click 'New Endpoint'"
    Write-Host "  3. Set Docker image to: $FullImage"
    if ($WorkerName -eq "upscale") {
        Write-Host "  4. Attach your Network Volume (mount path: /runpod-volume/)"
        Write-Host "  5. Select GPU: A100 80GB SXM (recommended)"
        Write-Host "  6. Save and copy the Endpoint ID into your frontend\.env"
        Write-Host ""
        Write-Host "  RUNPOD_UPSCALE_ENDPOINT_ID=<your_endpoint_id>"
    } else {
        Write-Host "  4. Select GPU: A100 80GB SXM (recommended)"
        Write-Host "  5. Save and copy the Endpoint ID into your frontend\.env"
        Write-Host ""
        Write-Host "  RUNPOD_CAPTION_ENDPOINT_ID=<your_endpoint_id>"
    }
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Worker) {
    "upscale" { Build-And-Push "upscale" }
    "caption" { Build-And-Push "caption" }
    "all"     { Build-And-Push "upscale"; Build-And-Push "caption" }
}
