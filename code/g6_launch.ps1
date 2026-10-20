$ErrorActionPreference='Stop'
$root='C:\Users\28908\Codex\fung-research\v2-diagnosis'
Set-Location -LiteralPath $root
$env:PYTHONDONTWRITEBYTECODE='1'
$py='C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -X utf8 -B (Join-Path $root 'g6_local.py') run *> (Join-Path $root 'work\G6_runtime\process.log')
exit $LASTEXITCODE
