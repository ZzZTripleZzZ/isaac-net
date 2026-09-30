. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location C:\isaac5g\IsaacLab
"== whoami $(whoami) $(Get-Date)"
"== cartpole physx $(Get-Date)"
& C:\isaac5g\env_isaaclab\Scripts\isaaclab.exe train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 4096 --max_iterations 30 physics=isaacsim_physx 2>&1
"== exit $LASTEXITCODE $(Get-Date)"
"DONE $(Get-Date)"
