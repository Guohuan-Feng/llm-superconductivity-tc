param([string]$TaskName)
$ErrorActionPreference='Stop'
$p1Root=(Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if($p1Root -match '(?i)onedrive'){throw 'OneDrive path forbidden'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=$p1Root
$env:OMP_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:TEMP="$p1Root\work\p1-cache\tmp"
$env:TMP=$env:TEMP
$env:TORCH_HOME="$p1Root\work\p1-cache\torch"
$env:MPLCONFIGDIR="$p1Root\work\p1-cache\matplotlib"
$env:XDG_CACHE_HOME="$p1Root\work\p1-cache\xdg"
$p1Deadline=[DateTimeOffset]::Parse((Get-Content -LiteralPath "$p1Root\P1\config.json" -Raw | ConvertFrom-Json).deadline_utc)
$p1Remaining=[int][Math]::Max(1,($p1Deadline-[DateTimeOffset]::UtcNow).TotalMilliseconds)
$p1Process=Start-Process -FilePath "$p1Root\work\p1-env\Scripts\python.exe" -ArgumentList @('-B',"$p1Root\run_p1.py",'run','--task-name',$TaskName) -WorkingDirectory "$p1Root\work\p1-cache" -WindowStyle Hidden -PassThru
if(-not $p1Process.WaitForExit($p1Remaining)){
    Add-Content -LiteralPath "$p1Root\P1\run.log" -Value 'One-workday deadline reached; terminating only this P1 process tree. No Stage5 dispatch.'
    & taskkill.exe /PID $p1Process.Id /T /F | Out-Null
    exit 124
}
exit $p1Process.ExitCode
