# PPO training with the network in the loop (rsl_rl through the Isaac Lab 3.0 wrapper).
param([int]$NumEnvs = 1024, [int]$NumRobots = 16, [string]$Level = 'L2-legacy', [string]$Backend = 'triton',
      [int]$Iters = 30, [string]$Py = 'C:\isaac5g\env_isaaclab\Scripts\python.exe')
. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
"== train E=$NumEnvs R=$NumRobots $Level $Backend $(Get-Date)"
& $Py benchmarks\isaac\train_ppo.py --num_envs $NumEnvs --num_robots $NumRobots --level $Level --backend $Backend --iters $Iters 2>&1 | Select-String 'Learning iteration|Steps per second|Mean reward|Episode/|Collection time|Learning time|TRAIN_DONE|Error|Traceback|error:|line \d+'
"DONE $(Get-Date)"
