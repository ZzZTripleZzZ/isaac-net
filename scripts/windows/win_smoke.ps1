# Windows -> WSL2 smoke test of one netslot-bridge worker (the path Isaac Sim on Windows uses).
# Spawns the worker inside WSL via wsl.exe, connects over localhost TCP, runs 20 steps.
$port = 46123
$bin = "/home/zzhang66/experiments/bridge_parallel/bin/netslot-bridge"
$init = "20:20:-1,30:10:-1,10:40:-1,50:5:-1"
$wargs = "-d Ubuntu-20.04 -- $bin --nUe=4 --io=tcp:$port --bindAll=1 --init=$init --outDir=/tmp --macTraces=0 --pktLog=0 --flowmon=0 --ueUeFilter=1"
$p = Start-Process -FilePath wsl.exe -ArgumentList $wargs -WindowStyle Hidden -PassThru
$c = $null
for ($i = 0; $i -lt 300; $i++) {
  try { $c = New-Object System.Net.Sockets.TcpClient("127.0.0.1", $port); break } catch { Start-Sleep -Milliseconds 100 }
}
if ($c -eq $null) { "CONNECT FAILED"; exit 1 }
$c.NoDelay = $true
$s = $c.GetStream()
$r = New-Object System.IO.StreamReader($s)
$w = New-Object System.IO.StreamWriter($s)
$w.NewLine = "`n"
$w.AutoFlush = $true
"hello: " + $r.ReadLine()
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$delivered = 0
for ($t = 0; $t -lt 20; $t++) {
  $w.WriteLine("S $t 0 4 0 $t 4000 1 $t 4000 2 $t 4000 3 $t 4000")
  $line = $r.ReadLine()
  $delivered += [int]($line.Split(" ")[3])
}
$ms = $sw.ElapsedMilliseconds
$w.WriteLine("Q")
$c.Close()
"steps 20, frames sent 80, delivered $delivered, wall $ms ms ($([math]::Round($ms/20,1)) ms/step)"
