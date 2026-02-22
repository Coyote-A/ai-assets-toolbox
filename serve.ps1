#!/usr/bin/env pwsh
# AI Assets Toolbox - Serve on Modal (dev mode)
# Usage: .\serve.ps1
#
# Runs full setup (idempotent) then starts a Modal dev server with hot-reload.
# Press Ctrl+C to stop.

. "$PSScriptRoot\scripts\common.ps1"

Write-Host "AI Assets Toolbox - Dev Server" -ForegroundColor Magenta
Write-Host ""

Ensure-Setup

Write-Step "Starting Modal dev server (hot-reload)"
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

modal serve src/app.py
