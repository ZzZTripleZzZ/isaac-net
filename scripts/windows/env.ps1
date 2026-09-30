# Common environment for all Isaac processes (dot-source this). Everything lives in C:\isaac5g.
$env:PATH = "C:\isaac5g\tools\uv;C:\isaac5g\tools\git\cmd;C:\isaac5g\env_isaaclab\Scripts;" + $env:PATH
$env:UV_CACHE_DIR = "C:\isaac5g\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "C:\isaac5g\python"
$env:UV_LINK_MODE = "copy"
$env:PIP_CACHE_DIR = "C:\isaac5g\cache\pip"
$env:VIRTUAL_ENV = "C:\isaac5g\env_isaaclab"
$env:OMNI_KIT_ACCEPT_EULA = "YES"   # accepted on the user's instruction to install Isaac
$env:PYTHONUTF8 = "1"
# Keep every per-user cache (Kit/Omniverse, pip, temp) inside C:\isaac5g, whichever account runs this.
New-Item -ItemType Directory -Force -Path C:\isaac5g\home\AppData\Local, C:\isaac5g\home\AppData\Roaming, C:\isaac5g\home\tmp | Out-Null
$env:USERPROFILE = "C:\isaac5g\home"
$env:HOME = "C:\isaac5g\home"
$env:LOCALAPPDATA = "C:\isaac5g\home\AppData\Local"
$env:APPDATA = "C:\isaac5g\home\AppData\Roaming"
$env:TEMP = "C:\isaac5g\home\tmp"
$env:TMP = "C:\isaac5g\home\tmp"
