# Isaac Lab scale runs with process repeats: network off and the NR engine (L2, triton) in a chosen configuration,
# $Repeats separate processes each, interleaved (off, L2, off, L2, ...) so slow drift of the host hits both arms.
# Each process: $Warmup warm-up steps, then 3 timed windows of $Steps steps (median window reported by bench.py).
# Before every process: wait until the GPU is idle (utilization below 5% in three samples) and no other python / kit
# process runs, and log the Windows CPU load and the busiest processes (the Isaac step is host-bound).
# Run as SYSTEM: scripts\windows\systask.ps1. The 5G-LENA EESM tables (bler_source="lena") come from $LenaTables.
param([string]$Out = 'benchmarks\isaac\results\scale_repeats.jsonl', [int]$Steps = 50, [int]$Warmup = 10,
      [int]$MaxTime = 60, [int]$Repeats = 3, [string]$Py = 'C:\isaac5g\env_isaaclab\Scripts\python.exe',
      [string]$Sizes = '2048x128,4096x128,8192x128', [string]$NrCfg = 'v2l', [string]$LenaTables = '')
. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
if ($LenaTables) { $env:ISAAC_NET_LENA_TABLES = $LenaTables }
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$smi = 'C:\Windows\System32\nvidia-smi.exe'
function Wait-Idle {
  for ($i = 0; $i -lt 360; $i++) {
    $u = @(1..3 | ForEach-Object { Start-Sleep -Milliseconds 500; [int](& $smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) })
    $mx = ($u | Measure-Object -Maximum).Maximum
    $py = @(Get-Process python*, kit* -ErrorAction SilentlyContinue)
    if ($mx -lt 5 -and $py.Count -eq 0) { return "idle util max $mx%, no python/kit process, after $($i*2) s" }
    Start-Sleep 2
  }
  return "NOT IDLE (util $mx%, $($py.Count) python/kit processes)"
}
function Cpu-State {
  # Windows CPU load and the five processes that used the most CPU over 5 s (in cores); WSL jobs show as vmmem*
  $l = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
  $p1 = @{}; Get-Process | ForEach-Object { if ($_.CPU) { $p1[$_.Id] = $_.CPU } }
  Start-Sleep 5
  $top = Get-Process | Where-Object { $_.CPU -and $p1.ContainsKey($_.Id) } |
    Select-Object ProcessName, @{n = 'c'; e = { ($_.CPU - $p1[$_.Id]) / 5 } } | Sort-Object c -Descending |
    Select-Object -First 5 | ForEach-Object { "$($_.ProcessName)=$([math]::Round($_.c, 2))" }
  return "cpu_load_pct $l; top cores over 5 s: $($top -join ' ')"
}
foreach ($sz in $Sizes.Split(',')) {
  $E, $R = $sz.Split('x')
  for ($k = 1; $k -le $Repeats; $k++) {
    foreach ($rb in @(@('off', 'graph'), @('L2', 'triton'))) {
      $w = Wait-Idle
      "== E=$E R=$R $($rb[0]) $($rb[1]) repeat $k $(Get-Date -Format s) $w"
      "CPU " + (Cpu-State)
      & $smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
      & $smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
      & $Py benchmarks\isaac\bench.py --num_envs $E --num_robots $R --level $rb[0] --backend $rb[1] --nr_cfg $NrCfg --steps $Steps --warmup $Warmup --max_time $MaxTime --repeats 3 --out $Out 2>&1 | Select-String 'RESULT|Error|Traceback|error:|Out of memory|CUDA'
      "CPU after " + (Cpu-State)
    }
  }
}
"DONE $(Get-Date -Format s)"
