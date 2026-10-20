"""Record and unregister exactly the three finite G6/G7 tasks from this run."""
from common import ROOT
from pathlib import Path
import argparse,json,subprocess

SPECS={
 'inference':('Codex.V2Diagnosis.G6.20261019',ROOT/'G6_local_cutoff/config.json'),
 'statistics':('Codex.V2Diagnosis.G6.Stats.20261019',ROOT/'G6_local_cutoff/config.json'),
 'G7_local':('Codex.V2Diagnosis.G7.Local.20261019',ROOT/'work/G7_eval/local_run.json')}
PS=Path(r'C:/Users/28908/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe')
def shell(command):
    r=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf8',errors='replace',check=True)
    return r.stdout.strip()
def main(kind):
    task,path=SPECS[kind]
    raw=shell("$t=Get-ScheduledTask -TaskName '"+task+"';$i=Get-ScheduledTaskInfo -TaskName '"+task+"';[pscustomobject]@{task_name=$t.TaskName;state=[string]$t.State;LastTaskResult=$i.LastTaskResult;last_run_utc=$i.LastRunTime.ToUniversalTime().ToString('o');execution_time_limit=$t.Settings.ExecutionTimeLimit;restart_count=$t.Settings.RestartCount;trigger_count=@($t.Triggers|Where-Object{$null -ne $_}).Count}|ConvertTo-Json -Compress")
    meta=json.loads(raw);assert meta['state']=='Ready' and meta['LastTaskResult']==0,meta
    assert meta['execution_time_limit']=='PT0S' and meta['restart_count']==0 and meta['trigger_count']==0,meta
    obj=json.loads(path.read_text(encoding='utf8'));obj.setdefault('finite_task_completion_audits',{})[kind]=meta
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    shell("Unregister-ScheduledTask -TaskName '"+task+"' -Confirm:$false")
    remain=shell("@(Get-ScheduledTask -TaskName '"+task+"' -ErrorAction SilentlyContinue).Count")
    assert remain=='0',remain
    meta['unregistered_after_recording']=True
    obj=json.loads(path.read_text(encoding='utf8'));obj['finite_task_completion_audits'][kind]=meta
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=list(SPECS));a=p.parse_args();main(a.kind)
