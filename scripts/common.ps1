# Shared setup functions for deploy.ps1 and serve.ps1
# Dot-source this: . "$PSScriptRoot\scripts\common.ps1"

$ErrorActionPreference = "Stop"

function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Skip { param($msg) Write-Host "  → $msg (already done)" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  ℹ $msg" -ForegroundColor White }

function Get-PythonCommand {
    foreach ($cmd in @("python", "python3")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 11)) {
                    return $cmd
                }
            }
        } catch { }
    }
    return $null
}

function Ensure-Setup {
    # -------------------------------------------------------------------------
    # STEP 1: Check Python 3.11+
    # -------------------------------------------------------------------------
    Write-Step "Checking Python 3.11+"

    $pythonCmd = Get-PythonCommand

    if ($pythonCmd) {
        $ver = & $pythonCmd --version 2>&1
        Write-Skip "Python already installed: $ver"
    } else {
        Write-Info "Python 3.11+ not found. Attempting to install via winget..."
        try {
            winget install --id Python.Python.3.11 --source winget --accept-package-agreements --accept-source-agreements
            Write-Ok "Python 3.11 installed via winget."
            Write-Info "Please restart this terminal and re-run the script to continue."
            Write-Info "Or add Python to PATH manually if winget did not do it automatically."
            exit 0
        } catch {
            Write-Err "winget install failed: $_"
            Write-Host ""
            Write-Host "  Please install Python 3.11+ manually from: https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            exit 1
        }
    }

    # Re-resolve after potential install
    $pythonCmd = Get-PythonCommand
    if (-not $pythonCmd) {
        Write-Err "Python 3.11+ still not found after install attempt. Please restart your terminal."
        exit 1
    }

    # -------------------------------------------------------------------------
    # STEP 2: Check pip
    # -------------------------------------------------------------------------
    Write-Step "Checking pip"

    try {
        $pipVer = & $pythonCmd -m pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Skip "pip is available: $pipVer"
        } else {
            throw "pip not found"
        }
    } catch {
        Write-Info "pip not found. Installing pip via ensurepip..."
        try {
            & $pythonCmd -m ensurepip --upgrade
            Write-Ok "pip installed."
        } catch {
            Write-Err "Failed to install pip: $_"
            exit 1
        }
    }

    # -------------------------------------------------------------------------
    # STEP 3: Install / upgrade modal
    # -------------------------------------------------------------------------
    Write-Step "Checking Modal CLI"

    $modalInstalled = $false
    try {
        $modalVer = modal --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $modalInstalled = $true
        }
    } catch { }

    if ($modalInstalled) {
        Write-Info "Modal is installed ($modalVer). Upgrading to latest..."
        try {
            & $pythonCmd -m pip install --upgrade modal --quiet
            $modalVer = modal --version 2>&1
            Write-Ok "Modal upgraded: $modalVer"
        } catch {
            Write-Err "Failed to upgrade modal: $_"
            exit 1
        }
    } else {
        Write-Info "Installing modal..."
        try {
            & $pythonCmd -m pip install modal --quiet
            $modalVer = modal --version 2>&1
            Write-Ok "Modal installed: $modalVer"
        } catch {
            Write-Err "Failed to install modal: $_"
            exit 1
        }
    }

    # -------------------------------------------------------------------------
    # STEP 4: Modal authentication
    # -------------------------------------------------------------------------
    Write-Step "Checking Modal authentication"

    $authenticated = $false

    # Check ~/.modal.toml first (fastest check)
    $modalToml = Join-Path $HOME ".modal.toml"
    if (Test-Path $modalToml) {
        $tomlContent = Get-Content $modalToml -Raw
        if ($tomlContent -match "\[") {
            # Has at least one profile section
            $authenticated = $true
        }
    }

    # Double-check with modal profile current
    if (-not $authenticated) {
        try {
            $profileResult = modal profile current 2>&1
            if ($LASTEXITCODE -eq 0) {
                $authenticated = $true
            }
        } catch { }
    }

    if ($authenticated) {
        Write-Skip "Modal is already authenticated"
    } else {
        Write-Info "Modal is not authenticated. Opening browser for login..."
        Write-Info "A browser window will open. Please log in or sign up at modal.com."
        Write-Host ""
        try {
            modal token new
            Write-Ok "Modal authentication complete."
        } catch {
            Write-Err "Modal authentication failed: $_"
            exit 1
        }
    }

    # -------------------------------------------------------------------------
    # STEP 5: Modal secret 'ai-toolbox-secrets'
    # -------------------------------------------------------------------------
    Write-Step "Checking Modal secret 'ai-toolbox-secrets'"

    $secretName = "ai-toolbox-secrets"
    $secretExists = $false

    try {
        $secrets = modal secret list 2>&1
        if ($secrets -match $secretName) {
            $secretExists = $true
        }
    } catch {
        Write-Err "Failed to list Modal secrets: $_"
        exit 1
    }

    if ($secretExists) {
        Write-Skip "Secret '$secretName' already configured"
    } else {
        Write-Info "Secret '$secretName' not found. You need a CivitAI API token."
        Write-Info "Get your token at: https://civitai.com/user/account (API Keys section)"
        Write-Host ""
        $civitaiToken = Read-Host "  Enter your CivitAI API token"
        if ([string]::IsNullOrWhiteSpace($civitaiToken)) {
            Write-Err "No token provided. Cannot create secret."
            exit 1
        }
        try {
            modal secret create $secretName "CIVITAI_API_TOKEN=$civitaiToken"
            Write-Ok "Secret '$secretName' created."
        } catch {
            Write-Err "Failed to create secret: $_"
            exit 1
        }
    }

    # -------------------------------------------------------------------------
    # STEP 6: Modal volume 'ai-toolbox-loras'
    # -------------------------------------------------------------------------
    Write-Step "Checking Modal volume 'ai-toolbox-loras'"

    $volumeName = "ai-toolbox-loras"
    $volumeExists = $false

    try {
        $volumes = modal volume list 2>&1
        if ($volumes -match $volumeName) {
            $volumeExists = $true
        }
    } catch {
        Write-Err "Failed to list Modal volumes: $_"
        exit 1
    }

    if ($volumeExists) {
        Write-Skip "Volume '$volumeName' already exists"
    } else {
        Write-Info "Creating Modal volume '$volumeName'..."
        try {
            modal volume create $volumeName
            Write-Ok "Volume '$volumeName' created."
        } catch {
            Write-Err "Failed to create volume: $_"
            exit 1
        }
    }
}
