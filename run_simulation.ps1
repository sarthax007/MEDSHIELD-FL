$ErrorActionPreference = "Stop"

# Configuration
$env:FL_MIN_CLIENTS = "3"
$env:FL_NUM_ROUNDS = "3"
$env:FL_SERVER_ADDRESS = "127.0.0.1:8080"
$PYTHON_EXEC = "python" # Adjust if you use a specific virtual environment python executable

Write-Host "Starting Flower server..."
Start-Process -NoNewWindow -FilePath $PYTHON_EXEC -ArgumentList "fl/server.py" -RedirectStandardOutput "server.log" -RedirectStandardError "server_error.log"

Write-Host "Waiting 5 seconds for the server to initialize..."
Start-Sleep -Seconds 5

Write-Host "Starting 3 clients..."
$clients = @()
for ($i = 1; $i -le 3; $i++) {
    Write-Host "Starting Client $i..."
    $env:CLIENT_PARTITION_ID = $i
    
    # Start client process (Notice the separate error log file here!)
    $process = Start-Process -PassThru -NoNewWindow -FilePath $PYTHON_EXEC -ArgumentList "fl/client/client.py" -RedirectStandardOutput "client_$i.log" -RedirectStandardError "client_${i}_error.log"
    $clients += $process
}

Write-Host "All clients started. Waiting for them to complete..."
$clients | Wait-Process

Write-Host "Simulation complete. Check server.log and client_*.log for metrics."