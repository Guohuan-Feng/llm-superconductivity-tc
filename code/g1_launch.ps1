$ErrorActionPreference = 'Stop'
$g1Root = 'C:\Users\28908\Codex\fung-research\v2-diagnosis'
Set-Location -LiteralPath $g1Root
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$g1Python = Join-Path $g1Root 'work\p1-env\Scripts\python.exe'
$g1Runner = Join-Path $g1Root 'g1_run.py'
& $g1Python -X utf8 -B $g1Runner --run *>> (Join-Path $g1Root 'work\g1\scheduled_process.log')
exit $LASTEXITCODE
