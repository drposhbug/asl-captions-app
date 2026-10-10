# Public https URL for the hub (needed for phone install / service worker). Run the hub first, then:
#   powershell -ExecutionPolicy Bypass -File server\tunnel.ps1
# Ctrl+C stops the tunnel. The URL changes on every run (free trycloudflare quick tunnel).
$exe = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
foreach ($p in "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe", "$env:ProgramFiles\cloudflared\cloudflared.exe", "$PSScriptRoot\cloudflared.exe") {
    if (-not $exe -and (Test-Path $p)) { $exe = $p }
}
if (-not $exe) { Write-Error "cloudflared not found: winget install --id Cloudflare.cloudflared -e"; exit 1 }

$log = Join-Path $env:TEMP "asl-tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue
# cloudflared prints the URL on stderr, so log it to a file and watch the file
$proc = Start-Process $exe -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://localhost:8766" `
    -RedirectStandardError $log -NoNewWindow -PassThru
try {
    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url -and -not $proc.HasExited; $i++) {
        Start-Sleep 1
        if (Test-Path $log) {
            $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 20; Write-Error "no tunnel URL after 60 s"; exit 1 }
    Write-Host ""
    Write-Host "  PUBLIC HUB URL:  $url" -ForegroundColor Green
    Write-Host "  phone: open it in Chrome -> menu -> Add to Home screen -> open app -> ARM ALERTS"
    Write-Host ""
    $proc.WaitForExit()
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
