$ErrorActionPreference = "Stop"

# Configuration for fast E2E
$env:FL_MIN_CLIENTS = "2"
$env:FL_MIN_FIT_CLIENTS = "2"
$env:FL_NUM_ROUNDS = "1"
$env:LOCAL_EPOCHS = "1"
$env:FL_MOCK_DATASET = "true"  # Tell client to use mock tiny dataset
$env:FL_SERVER_ADDRESS = "127.0.0.1:8081"
$env:API_PORT = "8001"
$env:DB_URL = "sqlite:///./sql_app_e2e.db"
$env:FL_USE_TLS = "false"
$env:PYTHONPATH = ".;./server;./shared"

$PYTHON_EXEC = "python"

Write-Host "Cleaning up old E2E database..."
if (Test-Path "./server/sql_app_e2e.db") {
    Remove-Item "./server/sql_app_e2e.db" -Force
}

Write-Host "Migrating and seeding E2E database..."
Set-Location -Path "server"
$env:DB_URL = "sqlite:///../sql_app_e2e.db"
& $PYTHON_EXEC -m alembic upgrade head
& $PYTHON_EXEC scripts/seed.py
Set-Location -Path ".."
$env:DB_URL = "sqlite:///./sql_app_e2e.db"

Write-Host "Starting FastAPI Backend..."
$backend = Start-Process -PassThru -NoNewWindow -FilePath $PYTHON_EXEC -ArgumentList "-m uvicorn server.app.main:app --port 8001" -RedirectStandardOutput "e2e_backend.log" -RedirectStandardError "e2e_backend_error.log"

Write-Host "Waiting 5 seconds for backend to initialize..."
Start-Sleep -Seconds 5

Write-Host "Starting Flower server..."
$fl_server = Start-Process -PassThru -NoNewWindow -FilePath $PYTHON_EXEC -ArgumentList "fl/server.py" -RedirectStandardOutput "e2e_server.log" -RedirectStandardError "e2e_server_error.log"

Write-Host "Waiting 3 seconds for FL server..."
Start-Sleep -Seconds 3

Write-Host "Starting 2 clients..."
$clients = @()
for ($i = 1; $i -le 2; $i++) {
    Write-Host "Starting Client $i..."
    $env:CLIENT_PARTITION_ID = $i
    $process = Start-Process -PassThru -NoNewWindow -FilePath $PYTHON_EXEC -ArgumentList "fl/client/client.py" -RedirectStandardOutput "e2e_client_$i.log" -RedirectStandardError "e2e_client_${i}_error.log"
    $clients += $process
}

Write-Host "Waiting for FL server and clients to complete the round..."
$fl_server | Wait-Process
$clients | Wait-Process

Write-Host "FL Round Complete. Running Pytest verification..."
$pytest_proc = Start-Process -PassThru -NoNewWindow -FilePath $PYTHON_EXEC -ArgumentList "-m pytest tests/test_e2e.py -v" -RedirectStandardOutput "e2e_pytest.log" -RedirectStandardError "e2e_pytest_error.log"

$pytest_proc | Wait-Process

if ($pytest_proc.ExitCode -eq 0) {
    Write-Host "E2E Test PASSED" -ForegroundColor Green
} else {
    Write-Host "E2E Test FAILED" -ForegroundColor Red
}

Write-Host "Cleaning up processes..."
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue

exit $pytest_proc.ExitCode
