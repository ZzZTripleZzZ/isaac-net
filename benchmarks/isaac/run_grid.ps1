# Task-4 grid: E x R x {off, L0 (ref engine), L2 graph, L2 triton}; one process per configuration.
# Must run in a context with GPU access (on the lab box: a SYSTEM scheduled task, see install/README).
param([string]$Out = 'results\grid.jsonl', [int]$Steps = 150, [int]$Warmup = 15, [int]$MaxTime = 45)
. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location C:\isaac5g\demo
New-Item -ItemType Directory -Force results | Out-Null
$py = 'C:\isaac5g\env_isaaclab\Scripts\python.exe'
$runs = @(@('off','graph'), @('L0','ref'), @('L2','graph'), @('L2','triton'))
foreach ($R in @(16, 32)) {
  foreach ($E in @(64, 256, 1024)) {
    foreach ($rb in $runs) {
      "== E=$E R=$R $($rb[0]) $($rb[1]) $(Get-Date)"
      & $py benchmarks\bench.py --num_envs $E --num_robots $R --rung $rb[0] --backend $rb[1] --steps $Steps --warmup $Warmup --max_time $MaxTime --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:'
    }
  }
}
# reference engine L2 (eager, launch-bound) at two points
foreach ($E in @(64, 1024)) {
  "== E=$E R=16 L2 ref $(Get-Date)"
  & $py benchmarks\bench.py --num_envs $E --num_robots 16 --rung L2 --backend ref --steps $Steps --warmup 3 --max_time $MaxTime --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:'
}
"GRID_DONE $(Get-Date)"
