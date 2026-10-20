$ErrorActionPreference = 'Stop'
$stageRoot = 'C:\Users\28908\Codex\fung-research\v2-diagnosis'
Set-Location -LiteralPath $stageRoot
& 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -B "$stageRoot\g_supplement_run.py" *> "$stageRoot\work\G_supplement_runtime\launcher.log"
exit $LASTEXITCODE
