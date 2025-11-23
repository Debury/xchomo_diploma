# Climate ETL Pipeline - Start Services
# Starts both Dagster and FastAPI in separate terminals

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Climate ETL Pipeline - Starting Services" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Set environment
$env:DAGSTER_HOME = Join-Path $PSScriptRoot ".dagster_home"
Write-Host "Setting DAGSTER_HOME: $env:DAGSTER_HOME" -ForegroundColor Yellow

# Create directories
$directories = @(
    ".dagster_home",
    ".dagster_home\storage",
    ".dagster_home\compute_logs", 
    ".dagster_home\history",
    "logs",
    "data\raw",
    "data\processed",
    "qdrant_data"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "[1/2] Starting Dagster dev server on http://localhost:3000" -ForegroundColor Cyan
Write-Host ""

# Start Dagster in new terminal
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$env:DAGSTER_HOME='$env:DAGSTER_HOME';
Write-Host 'Starting Dagster Dev Server...' -ForegroundColor Green;
Write-Host 'UI: http://localhost:3000' -ForegroundColor Yellow;
Write-Host '';
python -m dagster dev -w dagster_project/workspace.yaml -h 0.0.0.0 -p 3000
"@

Write-Host "Waiting for Dagster to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "[2/2] Starting FastAPI web server on http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

# Start FastAPI in new terminal
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host 'Starting FastAPI Web Server...' -ForegroundColor Green;
Write-Host 'Docs: http://localhost:8000/docs' -ForegroundColor Yellow;
Write-Host 'Health: http://localhost:8000/health' -ForegroundColor Yellow;
Write-Host '';
python -m uvicorn web_api.main:app --reload --host 0.0.0.0 --port 8000
"@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Services Started!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dagster UI:  " -NoNewline; Write-Host "http://localhost:3000" -ForegroundColor Cyan
Write-Host "API Docs:    " -NoNewline; Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "API Health:  " -NoNewline; Write-Host "http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Wait 10 seconds for services to fully start, then run:" -ForegroundColor Yellow
Write-Host "  python scripts\test_source_flow.py" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to exit (services will keep running in separate windows)" -ForegroundColor Gray
Write-Host ""

# Keep this window open
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "Exiting..." -ForegroundColor Yellow
}
