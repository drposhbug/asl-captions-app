# Sign-to-speech on the laptop: chest camera (or a recorded clip) -> hand landmarks -> sign classifier -> sentence -> ElevenLabs voice.
# Usage:  powershell -ExecutionPolicy Bypass -File server\start_signing.ps1 1            # camera index 1 (the Ace Pro 2 in webcam mode)
#         powershell -ExecutionPolicy Bypass -File server\start_signing.ps1 clip.mp4 none   # replay a recorded portrait clip, no rotation
param([string]$Camera = "1", [string]$Rotate = "cw")
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$hubUp = (netstat -an | Select-String ':8766 ' | Select-String LISTENING) -ne $null
if (-not $hubUp) {
  Start-Process powershell -WorkingDirectory $here -ArgumentList '-NoExit','-Command',"`$host.UI.RawUI.WindowTitle = 'ASL hub - keep open'; `$env:ELEVENLABS_API_KEY = [Environment]::GetEnvironmentVariable('ELEVENLABS_API_KEY','User'); `$env:ELEVENLABS_VOICE_ID = 'mBfxa8ysADIVOqWwpBvN'; `$env:STT_BACKEND = 'local'; `$env:STT_LOCAL_MODEL = 'small.en'; `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; `$env:TF_CPP_MIN_LOG_LEVEL = '2'; `$env:TF_ENABLE_ONEDNN_OPTS = '0'; & 'C:\Users\thorn\AppData\Local\Programs\Python\Python313\python.exe' -u hub.py --sounds --yamnet --stereo"
  Write-Host "hub starting (Tyler's voice, offline captions)"
} else { Write-Host "hub already running" }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'hands_server' -and $_.Name -eq 'python.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "stopped old hand server $($_.ProcessId)" }
Start-Process powershell -WorkingDirectory $here -ArgumentList '-NoExit','-Command',"`$host.UI.RawUI.WindowTitle = 'ASL hands - keep open'; while ($true) { & 'C:\Users\thorn\miniconda3\envs\wilor\python.exe' hands_server.py --save-frames --camera $Camera --rotate $Rotate; Write-Host 'hand server exited, restarting in 3 s'; Start-Sleep 3 }"
Write-Host "hand server starting on $Camera (rotate $Rotate). Needs server\signs\model.pkl to recognise signs: build it with the wilor python (C:\Users\thorn\miniconda3\envs\wilor\python.exe) via 'python server\import_sheet.py' then 'python server\signs.py train', or train from the dashboard."
Write-Host "Watch: http://127.0.0.1:8766/ (dashboard)  http://127.0.0.1:8766/?speaker (big spoken-sentence screen)"
