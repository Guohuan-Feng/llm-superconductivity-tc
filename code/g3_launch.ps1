param([string]$TaskName,[int]$RestartIndex=0)
$ErrorActionPreference='Stop'
$g3Root=(Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if($g3Root -match '(?i)onedrive'){throw 'OneDrive path forbidden'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
Set-Location -LiteralPath $g3Root
& 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -B "$g3Root\g3_run.py" run --task-name $TaskName --restart-index $RestartIndex
exit $LASTEXITCODE
