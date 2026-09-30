param([string]$Log, [string]$Name = '', [int]$Max = 540, [string]$Pattern = '^DONE')
# Lab-side bounded wait: poll the log for $Pattern (max $Max s), then print the tail; remove the task if finished.
for ($i = 0; $i -lt $Max; $i += 10) {
  if ((Test-Path $Log) -and (Select-String -Path $Log -Pattern $Pattern -Quiet)) { break }
  Start-Sleep 10
}
Get-Content $Log -Tail 25 -ErrorAction SilentlyContinue
if ($Name -and (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue)) {
  $s = (Get-ScheduledTask -TaskName $Name).State
  "TASK $Name state=$s"
  if ($s -ne 'Running') { Unregister-ScheduledTask -TaskName $Name -Confirm:$false; "task removed" }
}
