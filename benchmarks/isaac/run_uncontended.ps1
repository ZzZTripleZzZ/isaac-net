# Uncontended Isaac Lab scale runs: network off / L2-legacy triton / L2 (NR) triton at 2048, 4096, 8192 x 128.
# One process per configuration, 3 timed windows each (median reported). Waits before every run until the GPU is idle
# (utilization below 5% in three samples), and records what it saw. Run as SYSTEM: scripts\windows\systask.ps1.
param([string]$Out = 'benchmarks\isaac\results\scale_uncontended.jsonl', [int]$Steps = 50, [int]$Warmup = 10,
      [int]$MaxTime = 60, [int]$Repeats = 3, [string]$Py = 'C:\isaac5g\env_isaaclab\Scripts\python.exe',
      [string]$Sizes = '2048x128,4096x128,8192x128')
. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$smi = 'C:\Windows\System32\nvidia-smi.exe'
function Wait-Idle {
  for ($i = 0; $i -lt 360; $i++) {
    $u = @(1..3 | ForEach-Object { Start-Sleep -Milliseconds 500; [int](& $smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) })
    $mx = ($u | Measure-Object -Maximum).Maximum
    if ($mx -lt 5) { return "idle util max $mx% after $($i*2) s" }
    Start-Sleep 2
  }
  return "NOT IDLE (util $mx%)"
}
foreach ($sz in $Sizes.Split(',')) {
  $E, $R = $sz.Split('x')
  foreach ($rb in @(@('off','graph'), @('L2-legacy','triton'), @('L2','triton'))) {
    $w = Wait-Idle
    "== E=$E R=$R $($rb[0]) $($rb[1]) $(Get-Date) $w"
    & $smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    & $smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    & $Py benchmarks\isaac\bench.py --num_envs $E --num_robots $R --level $rb[0] --backend $rb[1] --steps $Steps --warmup $Warmup --max_time $MaxTime --repeats $Repeats --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:|Out of memory|CUDA'
  }
}
"DONE $(Get-Date)"
