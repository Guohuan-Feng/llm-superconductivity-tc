param([ValidateSet('E0','E1')][string]$Experiment, [string]$TaskName)
$ErrorActionPreference = 'Stop'
$taskRoot = (Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if ($taskRoot -match '(?i)onedrive') { throw 'OneDrive path forbidden' }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'C:\Users\28908\Documents\Codex\2026-10-05\tc-performance-push\env\Lib\site-packages'
$env:OMP_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$taskProcess = Start-Process -FilePath 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -ArgumentList @('-B', "$taskRoot\run_stage1.py", $Experiment, '--task-name', $TaskName) -WorkingDirectory $taskRoot -WindowStyle Hidden -Wait -PassThru
exit $taskProcess.ExitCode
