$ErrorActionPreference='Stop'
$g7Root=(Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if($g7Root -match '(?i)onedrive'){throw 'OneDrive path forbidden'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
Set-Location -LiteralPath $g7Root
& 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -B "$g7Root\g7_eval.py" run
exit $LASTEXITCODE
