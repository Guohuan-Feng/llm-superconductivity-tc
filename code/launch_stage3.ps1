param([string]$TaskName)
$ErrorActionPreference='Stop'
$stageRoot=(Get-Item -LiteralPath 'C:\Users\28908\Codex\fung-research\v2-diagnosis').FullName
if($stageRoot -match '(?i)onedrive'){throw 'OneDrive path forbidden'}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='C:\Users\28908\Documents\Codex\2026-10-05\tc-performance-push\env\Lib\site-packages'
$env:OMP_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$stageProcess=Start-Process -FilePath 'C:\Users\28908\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -ArgumentList @('-B',"$stageRoot\run_stage3.py",'run','--task-name',$TaskName) -WorkingDirectory $stageRoot -WindowStyle Hidden -Wait -PassThru
exit $stageProcess.ExitCode
