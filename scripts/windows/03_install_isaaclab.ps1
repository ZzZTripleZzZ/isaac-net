. C:\isaac5g\env.ps1
$ErrorActionPreference = 'Continue'
Set-Location C:\isaac5g\IsaacLab
"== isaaclab -i $(Get-Date)"
cmd /c "isaaclab.bat -i 2>&1"
"== exit $LASTEXITCODE $(Get-Date)"
& C:\isaac5g\env_isaaclab\Scripts\python.exe -m pip list 2>&1 | Select-String 'isaaclab|rsl|torch |warp|newton|skrl|rl-games|gymnasium'
"DONE $(Get-Date)"
