<#
.SYNOPSIS
    Hook helper: checks if a snapshot reminder is needed.
    Called automatically by Claude Code before Write operations.
    Outputs a reminder if no snapshot was taken recently.
#>

$marker = Join-Path $env:TEMP "meeting_ai_snapshot_done"

try {
    if (-not (Test-Path $marker)) {
        Write-Host "[SNAPSHOT REMINDER] This session has no snapshot yet. If you are about to make MAJOR changes (restructuring, rewriting, architecture changes), first run:"
        Write-Host "  powershell scripts/snapshot.ps1 -Desc 'description of upcoming changes'"
    } else {
        $age = (Get-Date) - (Get-Item $marker).LastWriteTime
        if ($age.TotalHours -gt 4) {
            Write-Host "[SNAPSHOT REMINDER] Last snapshot was $([Math]::Round($age.TotalHours, 1)) hours ago. Consider a new snapshot if making major changes."
            Write-Host "  powershell scripts/snapshot.ps1 -Desc 'description'"
        }
    }
} catch {
    # Silently ignore errors - never block the operation
}

exit 0
