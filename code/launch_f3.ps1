param([string]$TaskName,[int]$RestartIndex=0)
$ErrorActionPreference='Stop'
$stageRoot=(Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if($stageRoot -match '(?i)onedrive'){throw 'OneDrive path forbidden'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
Set-Location -LiteralPath $stageRoot
& 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -B "$stageRoot\run_f3.py" run --task-name $TaskName --restart-index $RestartIndex
exit $LASTEXITCODE
