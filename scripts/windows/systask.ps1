param([string]$Script, [string]$Log, [string]$Name = 'Isaac5gJob')
# Run $Script as SYSTEM (the only context on this box where CUDA works without a console logon).
# The task is one-shot; call with -Name and remove it with systask_rm.ps1 once DONE appears.
$cmd = "-NoProfile -ExecutionPolicy Bypass -Command `"& { & '$Script' *> '$Log' }`""
$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $cmd -WorkingDirectory 'C:\isaac5g'
$pr = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$st = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6) -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName $Name -Action $a -Principal $pr -Settings $st -Force | Out-Null
Start-ScheduledTask -TaskName $Name
"started task $Name : $Script -> $Log"
