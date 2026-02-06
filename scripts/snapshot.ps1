<#
.SYNOPSIS
    Creates a project snapshot before major changes.
.DESCRIPTION
    1. Physically copies all source code to snapshots/<timestamp>/
    2. Records dependency versions (pip freeze, npm list)
    3. Generates a detailed state record markdown
    4. Creates a git commit + tag for easy rollback
.EXAMPLE
    powershell scripts/snapshot.ps1 -Desc "before realtime ASR rewrite"
    powershell scripts/snapshot.ps1 -Desc "before switching audio library"
.NOTES
    Rollback methods:
      - Git:  git reset --soft snapshot/<timestamp>
      - File: manually copy from snapshots/<timestamp>/backup/
#>
param(
    [Parameter(Mandatory=$true)]
    [Alias("Desc")]
    [string]$Description
)

# Use Continue for git commands (git writes warnings to stderr which PowerShell treats as errors)
$ErrorActionPreference = "Continue"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dateStr = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$safeName = ($Description -replace '[^\w\-]', '_')
if ($safeName.Length -gt 50) { $safeName = $safeName.Substring(0, 50) }
$snapshotBase = "snapshots"
$snapshotDir = Join-Path $snapshotBase "${timestamp}_before_${safeName}"
$backupDir = Join-Path $snapshotDir "backup"
$tagName = "snapshot/${timestamp}"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  SNAPSHOT: before $Description" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# --- Create snapshot directory ---
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# =============================================================
# STEP 1: Physical file backup (the insurance policy)
# =============================================================
Write-Host "[1/4] Backing up source files..." -ForegroundColor Yellow

$backupItems = @(
    @{ Src = "frontend/src";          Dest = "backup/frontend_src" },
    @{ Src = "frontend/package.json"; Dest = "backup/frontend_package.json" },
    @{ Src = "frontend/package-lock.json"; Dest = "backup/frontend_package-lock.json" },
    @{ Src = "frontend/vite.config.ts"; Dest = "backup/frontend_vite.config.ts" },
    @{ Src = "frontend/tsconfig.json"; Dest = "backup/frontend_tsconfig.json" },
    @{ Src = "frontend/tailwind.config.js"; Dest = "backup/frontend_tailwind.config.js" },
    @{ Src = "frontend/postcss.config.js"; Dest = "backup/frontend_postcss.config.js" },
    @{ Src = "frontend/index.html";   Dest = "backup/frontend_index.html" },
    @{ Src = "backend/src";           Dest = "backup/backend_src" },
    @{ Src = "backend/pyproject.toml"; Dest = "backup/backend_pyproject.toml" },
    @{ Src = "backend/.env";          Dest = "backup/backend_.env" },
    @{ Src = "backend/.env.example";  Dest = "backup/backend_.env.example" },
    @{ Src = "backend/environment.yml"; Dest = "backup/backend_environment.yml" },
    @{ Src = "backend/environment-cuda.yml"; Dest = "backup/backend_environment-cuda.yml" },
    @{ Src = "CLAUDE.md";             Dest = "backup/CLAUDE.md" },
    @{ Src = "README.md";             Dest = "backup/README.md" },
    @{ Src = ".gitignore";            Dest = "backup/.gitignore" }
)

$copiedCount = 0
$totalSize = 0
foreach ($item in $backupItems) {
    $srcPath = $item.Src
    $destPath = Join-Path $snapshotDir $item.Dest
    if (Test-Path $srcPath) {
        if ((Get-Item $srcPath).PSIsContainer) {
            # Directory: recursive copy
            Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
        } else {
            # File: ensure parent directory exists
            $parentDir = Split-Path $destPath -Parent
            if (-not (Test-Path $parentDir)) {
                New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            }
            Copy-Item -Path $srcPath -Destination $destPath -Force
        }
        $copiedCount++
    }
}

# Calculate backup size
$backupSize = (Get-ChildItem -Recurse $backupDir -File | Measure-Object -Property Length -Sum).Sum
$backupSizeMB = [Math]::Round($backupSize / 1MB, 2)
Write-Host "  Backed up $copiedCount items ($backupSizeMB MB) -> $backupDir" -ForegroundColor Green

# =============================================================
# STEP 2: Record exact dependency versions
# =============================================================
Write-Host "[2/4] Recording dependency versions..." -ForegroundColor Yellow

# Python: pip freeze
$pipFreeze = "(pip not available)"
try {
    $pipFreeze = pip freeze 2>$null
    if ($pipFreeze) {
        $pipFreeze = $pipFreeze -join "`n"
        $pipFreeze | Out-File -FilePath (Join-Path $snapshotDir "pip_freeze.txt") -Encoding UTF8
        $pipCount = ($pipFreeze -split "`n").Count
        Write-Host "  pip freeze: $pipCount packages -> pip_freeze.txt" -ForegroundColor Green
    }
} catch {
    Write-Host "  pip freeze: skipped (not available)" -ForegroundColor DarkYellow
}

# Python: conda list (if available)
try {
    $condaList = conda list 2>$null
    if ($condaList) {
        ($condaList -join "`n") | Out-File -FilePath (Join-Path $snapshotDir "conda_list.txt") -Encoding UTF8
        Write-Host "  conda list: saved -> conda_list.txt" -ForegroundColor Green
    }
} catch {
    # conda not available, skip silently
}

# Node: npm list
$npmList = "(npm not available)"
if (Test-Path "frontend/package.json") {
    try {
        $npmList = npm list --prefix frontend 2>$null
        if ($npmList) {
            $npmList = $npmList -join "`n"
            $npmList | Out-File -FilePath (Join-Path $snapshotDir "npm_list.txt") -Encoding UTF8
            Write-Host "  npm list: saved -> npm_list.txt" -ForegroundColor Green
        }
    } catch {
        Write-Host "  npm list: skipped (not available)" -ForegroundColor DarkYellow
    }
}

# =============================================================
# STEP 3: Generate state record markdown
# =============================================================
Write-Host "[3/4] Generating state record..." -ForegroundColor Yellow

$gitBranch = git branch --show-current 2>$null
if (-not $gitBranch) { $gitBranch = "(detached HEAD)" }

$gitLastCommit = git log -1 --oneline 2>$null
if (-not $gitLastCommit) { $gitLastCommit = "(no commits)" }

$gitStatus = (git status --short 2>$null) -join "`n"
$gitDiffStat = (git diff --stat 2>$null) -join "`n"

# File trees
$frontendTree = "(not found)"
if (Test-Path "frontend/src") {
    $frontendTree = (Get-ChildItem -Recurse "frontend/src" -File | Sort-Object FullName |
        ForEach-Object { $_.FullName.Replace((Resolve-Path "frontend/src").Path + "\", "") }) -join "`n"
}

$backendTree = "(not found)"
if (Test-Path "backend/src") {
    $backendTree = (Get-ChildItem -Recurse "backend/src" -File | Sort-Object FullName |
        ForEach-Object { $_.FullName.Replace((Resolve-Path "backend/src").Path + "\", "") }) -join "`n"
}

# Key config content
$packageJsonContent = "(not found)"
if (Test-Path "frontend/package.json") {
    $packageJsonContent = Get-Content "frontend/package.json" -Raw
}

$pyprojectContent = "(not found)"
if (Test-Path "backend/pyproject.toml") {
    $pyprojectContent = Get-Content "backend/pyproject.toml" -Raw
}

$envContent = "(not found)"
if (Test-Path "backend/.env") {
    $envContent = Get-Content "backend/.env" -Raw
} elseif (Test-Path "backend/.env.example") {
    $envContent = "(.env not found, showing .env.example)`n" + (Get-Content "backend/.env.example" -Raw)
}

$viteConfig = "(not found)"
if (Test-Path "frontend/vite.config.ts") {
    $viteConfig = Get-Content "frontend/vite.config.ts" -Raw
}

$md = @"
# Snapshot: Before $Description

| Field | Value |
|-------|-------|
| Created | $dateStr |
| Branch | $gitBranch |
| Last Commit | $gitLastCommit |
| Tag | ``$tagName`` |
| Backup Size | $backupSizeMB MB |
| Backup Path | ``$backupDir/`` |

## How to Rollback

### Method 1: Git (recommended)
``````bash
# View changes since this snapshot
git diff ${tagName}..HEAD

# Soft rollback (keep changes as unstaged)
git reset --soft $tagName

# Hard rollback (DISCARD all changes after snapshot)
git reset --hard $tagName

# Restore a single file from snapshot
git checkout $tagName -- path/to/file
``````

### Method 2: Physical file restore (if git is broken)
``````powershell
# Restore entire frontend source
Copy-Item -Path "$snapshotDir\backup\frontend_src\*" -Destination "frontend\src\" -Recurse -Force

# Restore entire backend source
Copy-Item -Path "$snapshotDir\backup\backend_src\*" -Destination "backend\src\" -Recurse -Force

# Restore a specific config file
Copy-Item -Path "$snapshotDir\backup\frontend_package.json" -Destination "frontend\package.json" -Force
Copy-Item -Path "$snapshotDir\backup\backend_pyproject.toml" -Destination "backend\pyproject.toml" -Force

# Reinstall dependencies after restore
cd frontend && npm install
cd backend && pip install -e ".[stream,enhance]"
``````

## Git Status

``````
$gitStatus
``````

## Git Diff Stats

``````
$gitDiffStat
``````

## Frontend Source Tree (frontend/src/)

``````
$frontendTree
``````

## Backend Source Tree (backend/src/)

``````
$backendTree
``````

## package.json (frontend)

``````json
$packageJsonContent
``````

## pyproject.toml (backend)

``````toml
$pyprojectContent
``````

## vite.config.ts

``````ts
$viteConfig
``````

## .env Config (backend)

``````env
$envContent
``````

## Dependency Files in This Snapshot

| File | Description |
|------|-------------|
| ``pip_freeze.txt`` | Python packages (exact versions) |
| ``conda_list.txt`` | Conda environment (if available) |
| ``npm_list.txt`` | Node.js packages (dependency tree) |
| ``backup/`` | Physical copy of all source code + configs |

---
*Auto-generated by scripts/snapshot.ps1 at $dateStr*
"@

$md | Out-File -FilePath (Join-Path $snapshotDir "STATE.md") -Encoding UTF8
Write-Host "  State record: $snapshotDir/STATE.md" -ForegroundColor Green

# =============================================================
# STEP 4: Git commit + tag
# =============================================================
Write-Host "[4/4] Creating git commit + tag..." -ForegroundColor Yellow

git add -A 2>$null

try {
    git commit -m "SNAPSHOT: before $Description" 2>&1 | Out-Null
    Write-Host "  Committed: SNAPSHOT: before $Description" -ForegroundColor Green
} catch {
    git commit -m "SNAPSHOT: before $Description" --allow-empty 2>&1 | Out-Null
    Write-Host "  Committed (empty): SNAPSHOT: before $Description" -ForegroundColor Green
}

# Create tag (silently handle if exists)
git tag $tagName 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Tagged: $tagName" -ForegroundColor Green
} else {
    Write-Host "  Tag already exists, skipped" -ForegroundColor DarkYellow
}

# Session marker for hook
$markerPath = Join-Path $env:TEMP "meeting_ai_snapshot_done"
$dateStr | Out-File -FilePath $markerPath -Encoding UTF8 -Force

# =============================================================
# Summary
# =============================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  SNAPSHOT COMPLETE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Description : before $Description" -ForegroundColor White
Write-Host "  Tag         : $tagName" -ForegroundColor White
Write-Host "  Backup      : $backupDir/ ($backupSizeMB MB)" -ForegroundColor White
Write-Host "  State record: $snapshotDir/STATE.md" -ForegroundColor White
Write-Host "  Dependencies: pip_freeze.txt, npm_list.txt" -ForegroundColor White
Write-Host ""
Write-Host "  Git rollback : git reset --soft $tagName" -ForegroundColor Yellow
Write-Host "  File rollback: see STATE.md for copy commands" -ForegroundColor Yellow
Write-Host ""
