#!/usr/bin/env pwsh
# AI Assets Toolbox - Deploy to Modal
# Usage: .\deploy.ps1
#
# Runs full setup (idempotent) then deploys to Modal production.

. "$PSScriptRoot\scripts\common.ps1"

Write-Host "AI Assets Toolbox - Deploy" -ForegroundColor Magenta
Write-Host ""

Ensure-Setup

Write-Step "Deploying to Modal (production)"
Write-Info "Running: modal deploy src/app.py"
Write-Host ""

try {
    modal deploy src/app.py
    if ($LASTEXITCODE -ne 0) {
        throw "modal deploy exited with code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Ok "Deployment successful!"
    Write-Info "Your Gradio UI URL is shown above (look for the 'https://' link in the output)."
} catch {
    Write-Err "Deployment failed: $_"
    exit 1
}

Write-Host ""
Write-Host "🎉 Done! Your AI Assets Toolbox is live on Modal." -ForegroundColor Green
Write-Host ""
