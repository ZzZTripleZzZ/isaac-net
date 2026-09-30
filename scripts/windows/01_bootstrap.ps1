$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path C:\isaac5g\tools, C:\isaac5g\logs | Out-Null
# Long paths (required by Isaac Lab Windows docs)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force | Out-Null
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem').LongPathsEnabled
# uv, standalone zip into C:\isaac5g\tools\uv
curl.exe -L -s -o C:\isaac5g\tools\uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
Expand-Archive -Force C:\isaac5g\tools\uv.zip C:\isaac5g\tools\uv
& C:\isaac5g\tools\uv\uv.exe --version
# MinGit portable
$rel = Invoke-RestMethod -UseBasicParsing https://api.github.com/repos/git-for-windows/git/releases/latest
$a = $rel.assets | Where-Object { $_.name -match '^MinGit-.*-64-bit\.zip$' -and $_.name -notmatch 'busybox' } | Select-Object -First 1
$a.name
curl.exe -L -s -o C:\isaac5g\tools\mingit.zip $a.browser_download_url
Expand-Archive -Force C:\isaac5g\tools\mingit.zip C:\isaac5g\tools\git
& C:\isaac5g\tools\git\cmd\git.exe --version
