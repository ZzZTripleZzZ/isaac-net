# Task grid: E x R x {off, L0 reference, L2-legacy graph, L2-legacy triton}; one process per configuration.
# Needs a context with GPU access (on the lab box: a SYSTEM scheduled task, scripts\windows\systask.ps1).
param([string]$Out = 'benchmarks\isaac\results\grid_new.jsonl', [int]$Steps = 150, [int]$Warmup = 15, [int]$MaxTime = 45,
      [string]$Py = 'C:\isaac5g\env_isaaclab\Scripts\python.exe')
. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$runs = @(@('off','graph'), @('L0','reference'), @('L2-legacy','graph'), @('L2-legacy','triton'))
foreach ($R in @(16, 32)) {
  foreach ($E in @(64, 256, 1024)) {
    foreach ($rb in $runs) {
      "== E=$E R=$R $($rb[0]) $($rb[1]) $(Get-Date)"
      & $Py benchmarks\isaac\bench.py --num_envs $E --num_robots $R --level $rb[0] --backend $rb[1] --steps $Steps --warmup $Warmup --max_time $MaxTime --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:'
    }
  }
}
# eager reference engines at two points: the legacy slot model and the NR engine
foreach ($lvl in @('L2-legacy', 'L2')) {
  foreach ($E in @(64, 1024)) {
    "== E=$E R=16 $lvl reference $(Get-Date)"
    & $Py benchmarks\isaac\bench.py --num_envs $E --num_robots 16 --level $lvl --backend reference --steps $Steps --warmup 3 --max_time $MaxTime --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:'
  }
}
"DONE $(Get-Date)"
