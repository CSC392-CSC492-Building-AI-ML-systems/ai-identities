# Run this PowerShell script as Administrator the first time.

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$root = Split-Path -Parent $scriptDir
$proxy = Join-Path $root "dev-proxy"
$certs = Join-Path $proxy "certs"
$caddyfile = Join-Path $proxy "Caddyfile"

# 1) Ensure mkcert + caddy
function Ensure-Cmd($name, $wingetId) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    Write-Host "Installing $name..."
    winget install --id $wingetId -e --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine")
  }
}

Ensure-Cmd "mkcert.exe" "FiloSottile.mkcert"
Ensure-Cmd "caddy.exe"   "caddyserver.caddy"

# 2) mkcert local CA + host certs
New-Item -ItemType Directory -Force -Path $certs | Out-Null
mkcert -install
if (-not (Test-Path (Join-Path $certs "app.crt"))) {
  mkcert -key-file (Join-Path $certs "app.key")  -cert-file (Join-Path $certs "app.crt")  app.llm.test
}
if (-not (Test-Path (Join-Path $certs "wiki.crt"))) {
  mkcert -key-file (Join-Path $certs "wiki.key") -cert-file (Join-Path $certs "wiki.crt") wiki.llm.test
}

# 3) hosts entries
$hostsPath = "$env:SystemRoot\System32\drivers\etc\hosts"
$hostsTxt = Get-Content $hostsPath -Raw
if ($hostsTxt -notmatch "app\.llm\.test") {
  Add-Content $hostsPath "`n127.0.0.1 app.llm.test"
}
if ($hostsTxt -notmatch "wiki\.llm\.test") {
  Add-Content $hostsPath "`n127.0.0.1 wiki.llm.test"
}
Write-Host "Hosts updated."

# 4) Run Caddy
Write-Host "Starting Caddy with $caddyfile ..."
caddy run --config $caddyfile --watch
