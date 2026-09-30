param([string]$Script, [string]$Log)
# Win32_Process.Create starts the process outside the ssh session's job object, so it survives disconnect.
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"& { & '$Script' *> '$Log' }`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = 'C:\isaac5g' }
"launched pid=$($r.ProcessId) rc=$($r.ReturnValue) $Script -> $Log"
