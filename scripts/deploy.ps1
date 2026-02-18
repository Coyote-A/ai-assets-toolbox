# Build and push Docker image for RunPod serverless (Windows PowerShell)
# Usage: .\scripts\deploy.ps1 -DockerHubUsername <username> [-Tag <tag>]
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - Logged in to Docker Hub: docker login
#   - Run from the repository root

[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0, HelpMessage = "Your Docker Hub username")]
    [string]$DockerHubUsername,

    [Parameter(Mandatory = $false, Position = 1, HelpMessage = "Image tag (default: latest)")]
    [string]$Tag = "latest"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ImageName  = "ai-assets-toolbox-backend"
$FullImage  = "${DockerHubUsername}/${ImageName}:${Tag}"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$Dockerfile = Join-Path $RepoRoot "backend\Dockerfile"
$Context    = Join-Path $RepoRoot "backend"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AI Assets Toolbox — Backend Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Image : $FullImage"
Write-Host "  Source: $Context"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
Write-Host "[1/3] Building Docker image..." -ForegroundColor Yellow
docker build `
    --platform linux/amd64 `
    -t $FullImage `
    -f $Dockerfile `
    $Context

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/3] Tagging image as $FullImage..." -ForegroundColor Yellow
docker tag $FullImage $FullImage

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker tag failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/3] Pushing image to Docker Hub..." -ForegroundColor Yellow
docker push $FullImage

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker push failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Build and push complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps — create a RunPod Serverless Endpoint:"
Write-Host ""
Write-Host "  1. Go to https://www.runpod.io/console/serverless"
Write-Host "  2. Click 'New Endpoint'"
Write-Host "  3. Set Docker image to: $FullImage"
Write-Host "  4. Attach your Network Volume (mount path: /runpod-volume/)"
Write-Host "  5. Select GPU: A100 80GB SXM (recommended)"
Write-Host "  6. Set environment variables if needed (see .env.example)"
Write-Host "  7. Save and copy the Endpoint ID into your frontend\.env"
Write-Host ""
Write-Host "  RUNPOD_ENDPOINT_ID=<your_endpoint_id>"
Write-Host ""
