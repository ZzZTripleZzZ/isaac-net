# Optional: community Triton build for Windows (needed only for the fast engine's `triton` backend).
. C:\isaac5g\env.ps1
uv pip install "triton-windows" 2>&1
& C:\isaac5g\env_isaaclab\Scripts\python.exe -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)" 2>&1
"DONE $(Get-Date)"
