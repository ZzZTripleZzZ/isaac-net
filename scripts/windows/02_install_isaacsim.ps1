# Isaac Sim 6.1 (pip) + PyTorch 2.12 cu130 into C:\isaac5g\env_isaaclab, then clone Isaac Lab release/3.0.0.
# Commands follow the Isaac Lab 3.0 install page, "Python environment with Isaac Sim", Windows, uv.
# Run detached with launch.ps1 (about 8 min on the lab box). No GPU needed.
. C:\isaac5g\env.ps1
Set-Location C:\isaac5g
uv venv --python 3.12 --seed C:\isaac5g\env_isaaclab 2>&1
uv pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow 2>&1
"== isaacsim exit $LASTEXITCODE"
uv pip install -U torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130 2>&1
"== torch exit $LASTEXITCODE"
# MinGit has no git-lfs; the repo's LFS files are docs media / test images, so skip the smudge filter.
git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false clone https://github.com/isaac-sim/IsaacLab.git --branch release/3.0.0 C:\isaac5g\IsaacLab 2>&1
git -C C:\isaac5g\IsaacLab log -1 --format="%H %cd" 2>&1
"DONE $(Get-Date)"
