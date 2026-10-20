param([string]$TaskName,[int]$RestartIndex=0)
$ErrorActionPreference='Stop'
$g5Root=(Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if($g5Root -match '(?i)onedrive'){throw 'OneDrive path forbidden'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
Set-Location -LiteralPath $g5Root
& 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -B "$g5Root\g5_run.py" run --task-name $TaskName --restart-index $RestartIndex --g4-completed-handoff
exit $LASTEXITCODE
