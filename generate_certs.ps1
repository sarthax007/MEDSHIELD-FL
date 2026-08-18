$ErrorActionPreference = "Stop"

$CertDir = ".certificates"
if (-Not (Test-Path -Path $CertDir)) {
    New-Item -ItemType Directory -Path $CertDir | Out-Null
}

Write-Host "Generating Root CA..."
# Generate CA key and certificate
openssl req -x509 -sha256 -nodes -days 365 -newkey rsa:2048 -subj "/O=MedShield/CN=MedShield Root CA" -keyout "$CertDir\ca.key" -out "$CertDir\ca.crt"

Write-Host "Generating Server Key and CSR..."
# Generate server key and CSR
openssl req -new -nodes -newkey rsa:2048 -subj "/O=MedShield/CN=127.0.0.1" -keyout "$CertDir\server.key" -out "$CertDir\server.csr"

Write-Host "Signing Server Certificate with Root CA..."
# Create an extfile for Subject Alternative Name
$ExtFile = "$CertDir\extfile.cnf"
"subjectAltName=DNS:localhost,IP:127.0.0.1" | Out-File -FilePath $ExtFile -Encoding ASCII

# Sign the server certificate
openssl x509 -req -sha256 -days 365 -in "$CertDir\server.csr" -CA "$CertDir\ca.crt" -CAkey "$CertDir\ca.key" -CAcreateserial -extfile $ExtFile -out "$CertDir\server.pem"

Write-Host "Certificates generated successfully in $CertDir directory:"
Write-Host "- CA Certificate: $CertDir\ca.crt"
Write-Host "- Server Certificate: $CertDir\server.pem"
Write-Host "- Server Key: $CertDir\server.key"
