# Scale sweep (headline): largest E x R at interactive speed, network off vs L2-legacy (triton, graph).
param([string]$Out = 'benchmarks\isaac\results\scale_new.jsonl', [int]$Steps = 100, [int]$Warmup = 10, [int]$MaxTime = 45,
      [string]$Py = 'C:\isaac5g\env_isaaclab\Scripts\python.exe')
. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$cfgs = @(@(2048,32), @(4096,32), @(8192,32), @(1024,64), @(2048,64), @(4096,64), @(1024,128), @(2048,128), @(4096,128), @(8192,128))
foreach ($c in $cfgs) {
  foreach ($rb in @(@('off','graph'), @('L2-legacy','triton'), @('L2-legacy','graph'))) {
    "== E=$($c[0]) R=$($c[1]) $($rb[0]) $($rb[1]) $(Get-Date)"
    & $Py benchmarks\isaac\bench.py --num_envs $c[0] --num_robots $c[1] --level $rb[0] --backend $rb[1] --steps $Steps --warmup $Warmup --max_time $MaxTime --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:|Out of memory|CUDA'
  }
}
"DONE $(Get-Date)"
