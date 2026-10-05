# Isaac Lab on Linux (HPC recipe)

This page is the Linux counterpart of [isaac-lab.md](isaac-lab.md), which covers the Windows-native install. It was worked out on 2026-09-29 on the NCSU Hazel cluster (RHEL 9.8, glibc 2.34, Slurm, Apptainer, no internet on compute nodes), which is the hard case for Isaac: an older glibc than NVIDIA's wheels require, no root, no Docker, and offline GPU nodes. On an Ubuntu 22.04/24.04 workstation the same steps work without the container (see [Ubuntu without a container](#ubuntu-without-a-container)).

**Result.** The core package and its GPU tests work natively. Isaac Lab 3.0 works in its *kit-less* mode (Newton or OV PhysX physics, no Isaac Sim), inside an Apptainer container that supplies a newer glibc. There, the official cartpole smoke run trains, our fleet environment runs on both physics backends, and all 9 Isaac-marked tests pass on both. The Isaac Sim (Kit) path does not install natively on glibc 2.34. The official NGC container, which needs no API key, runs Kit and NVIDIA's own cartpole smoke run, but our fleet env fails on it, because the only 3.0 image is a release candidate that predates `release/3.0.0` (see [Isaac Sim with Kit](#isaac-sim-with-kit-physx-through-isaac-sim)).

| Route | Status on Hazel | Why |
|:---|:---|:---|
| Core package (`pip install -e .[dev]`), torch cu126 + Triton | works, native | plain manylinux wheels |
| Isaac Lab 3.0 kit-less, native venv | fails at install | `omniverseclient` 2.74.0 (a core Isaac Lab dependency) ships only `manylinux_2_35` wheels, and its `libomniverse_connection.so` really imports `GLIBC_2.35` symbols; `usd-exchange` and `ovstage` are also 2.35-only |
| Isaac Lab 3.0 kit-less in Apptainer (Debian bookworm, glibc 2.36) | **works** | the container supplies glibc; the GPU driver comes from the host through `--nv` |
| Isaac Sim 6.1 pip (`--extra isaacsim`), native | fails at install | all 25 `isaacsim` 6.1.0.0 wheels are `manylinux_2_35` only |
| Isaac Sim through the NGC `isaac-lab:3.0.0-rc1` container | Kit and its cartpole smoke work, our fleet env fails | the only published 3.0 image is rc1 (Isaac Lab 17.0.2); on it our env module loads `pxr` before `SimulationApp` starts (see below) |

## Versions

| Component | Core env (`envs/core`) | Isaac Lab env (`envs/lab_ctr`) |
|:---|:---|:---|
| Host | RHEL 9.8, glibc 2.34, NVIDIA driver 595.58.03 (CUDA 13.2) | same host, container userland `python:3.12-bookworm` (glibc 2.36) under Apptainer 1.4.2 |
| GPU | L40 46 GB (gpu14), one GPU per job | same |
| Python | 3.11.16 (uv-managed) | 3.12.14 (uv-managed) |
| PyTorch | 2.14.0+cu126 | 2.12.0+cu130 (pinned by Isaac Lab) |
| Triton | 3.8.0 (bundled with the torch wheel) | 3.7.0 |
| Isaac Lab | not installed | `release/3.0.0` at `60e28c1` (package `isaaclab` 25.0.0, `isaaclab_tasks` 20.0.0) |
| Physics | | newton 1.6.0 (MuJoCo-Warp 3.12.0), warp-lang 1.17.0, ovphysx 0.6.3, `isaaclab_physx` 7.3.0 |
| RL | | rsl-rl-lib 5.5.1 |
| Tools | uv 0.12.21 standalone binary | same |

The cu130 torch build that Isaac Lab pins needs driver 580.65.06 or newer on Linux; Hazel's 595.58 is fine. Older clusters with a 55x/57x driver would need the cu126/cu128 build instead, which Isaac Lab 3.0 does not pin.

## Cluster rules this recipe follows

Everything lives under one directory `B` on the shared file system (on Hazel `/share/hpcproject/<user>/isaacnet`), because `/home` is small. Installs run on the internet-connected transfer partition (`sbatch -p xfer`), and every test runs in a GPU batch job. Nothing heavy runs on the login node, and no job ssh-es to a compute node. The templates set `PYTHONNOUSERSITE=1`, and keep the uv, pip, Triton, Warp and XDG caches under `B`.

The templates are in `scripts/hazel/` in the repository. They hard-code the Hazel paths in a `B=` line and in the `#SBATCH -o` line, and the account, QOS and GPU type in the `#SBATCH` header; change those for another cluster.

| Step | Template | Partition | What it does | Time |
|:---|:---|:---|:---|---:|
| 0 | (by hand) | login | copy the code: `git archive main \| ssh hazel "tar x -C $B/repo"` (or `git clone https://github.com/ZzZTripleZzZ/isaac-net` on the login node, since the repository is public) | |
| 1 | `install_core.sbatch` | xfer | fetches the uv binary, creates a Python 3.11 venv, installs torch cu126 and `isaac-net[dev]` | 1.7 min |
| 2 | `core_gpu.sbatch` | gpu | `pytest -m gpu`, `pytest -m "not gpu"`, `benchmarks/bench.py` fast and reference backends, `bench_nr.py` | 10 min |
| 3 | `install_isaaclab_kitless.sbatch` | xfer | clones Isaac Lab `release/3.0.0`, pulls `python:3.12-bookworm` as a SIF, runs `uv sync --extra rsl-rl --extra ovphysx` inside it, adds `isaac-net` (editable, `--no-deps`) and pytest | 6 min |
| 4 | `prefetch_assets.sbatch` | xfer | downloads the cloud USD assets the runs need (ground plane, cartpole) into a local mirror | 1 min |
| 5 | `isaac_gpu.sbatch` | gpu | official cartpole smoke on Newton and OV PhysX, the fleet env on every physics backend, `pytest -m isaac` on Newton and on OV PhysX, the GPU suite in the Isaac env | 15 min |
| 6 | `isaac_scale.sbatch` (optional) | gpu | fleet-env throughput grid on both kit-less backends, and `bench_nr.py` without the GPL presets | 25 min |
| 7 | `pull_isaaclab_container.sbatch` | xfer | pulls the official `nvcr.io/nvidia/isaac-lab:3.0.0-rc1` image (Isaac Sim included) as a 12 GB SIF | 50 min |
| 8 | `isaacsim_ctr_gpu.sbatch` | gpu | Kit cartpole smoke, then the fleet env check, the bench smoke and `pytest -m isaac` on Kit PhysX (`LAB=release` binds the `release/3.0.0` checkout over the image's sources) | 8 min |

Step 4 copies the mirrored tree to `$B/assets` by hand once (`cp -r $B/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets $B/assets/`), and the GPU templates set `ISAACSIM_ASSET_ROOT=$B/assets/Assets/Isaac/6.1`.

## The kit-less recipe

Isaac Lab 3.0's recommended install is `uv run` / `uv sync` from a source checkout, which creates the environment from the checkout's `uv.lock` without Isaac Sim. The legacy `./isaaclab.sh --install` path from the kit-less page also works in principle, but on a machine without `cmake` in `PATH` it runs `sudo apt-get update`, which fails without a terminal (and should not be attempted on a shared cluster). `uv sync` does not do that.

The commands inside the container are:

```bash
cd $B/IsaacLab            # git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git (GIT_LFS_SKIP_SMUDGE=1)
UV_PROJECT_ENVIRONMENT=$B/envs/lab_ctr uv sync --extra rsl-rl --extra ovphysx
uv pip install --python $B/envs/lab_ctr/bin/python --no-deps -e $B/repo      # isaac-net, keeps Isaac Lab's torch
uv pip install --python $B/envs/lab_ctr/bin/python pytest
```

The container is started with `apptainer exec --nv --bind /gpfs_common,/gpfs_common/share:/share $B/sif/py312-bookworm.sif ...`. `--nv` binds the host driver. Both binds are needed on Hazel because `/share` is a symlink into `/gpfs_common`, and without them the container cannot see the venv or the checkout. The venv's interpreter is uv's standalone CPython under `$B`, so the image only has to supply glibc, git (for the two git-sourced Isaac Lab dependencies) and a shell; `python:3.12-bookworm` (350 MB as a SIF) does. The env is only valid inside the container.

A GPU job then runs, from the `isaac-net` checkout:

```bash
X="apptainer exec --nv --bind /gpfs_common,/gpfs_common/share:/share --pwd $B/repo \
   --env TMPDIR=$B/tmp,ISAACSIM_ASSET_ROOT=$B/assets/Assets/Isaac/6.1,PATH=$B/envs/lab_ctr/bin:/usr/local/bin:/usr/bin:/bin \
   $B/sif/py312-bookworm.sif"
$X isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=newton_mjwarp
$X env ISAAC_NET_PHYSICS=newton python -m pytest -m isaac tests/test_isaac_env.py
$X env ISAAC_NET_PHYSICS=newton python benchmarks/isaac/bench.py --num_envs 256 --num_robots 16 --level L2-legacy --backend triton
```

The templates also set `OMNI_KIT_ACCEPT_EULA=YES` for these processes only. This records the project's acceptance of the NVIDIA Omniverse EULA, given for the Windows install and confirmed for these templates. Whether kit-less runs need the variable was not tested.

### Choosing the physics backend of the fleet env

The fleet env (`isaac_net/examples/isaac_fleet_env.py`) used to hard-code `PhysxCfg`, which is PhysX *through Isaac Sim*, so kit-less Isaac Lab refused it with "Isaac Sim is not installed or not found on PYTHONPATH. ... PhysX backend and Kit visualizer currently requires Isaac Sim." `make_cfg(..., physics=...)`, or the environment variable `ISAAC_NET_PHYSICS` when it is not passed, now selects one of:

| `physics` | Isaac Lab config | Needs Isaac Sim |
|:---|:---|:---|
| `isaacsim_physx` (default, unchanged) | `isaaclab_physx.physics.PhysxCfg()` | yes |
| `ovphysx` | `isaaclab_ov.physics.OvPhysxCfg()` | no |
| `newton` | `NewtonCfg(solver_cfg=MJWarpSolverCfg())` | no |

The sphere assets keep `PhysxRigidBodyCfg(disable_gravity=True, ...)`, which Isaac Lab's own tasks also use under Newton and OV PhysX. For the two kit-less backends `make_cfg` also sets world gravity to zero, because Newton ignores the per-body `disable_gravity` flag (a PhysX schema attribute): without the fix the spheres fell from 0.5 m to the ground (height 0.30–0.32 m after 120 steps) and slid against friction, while with it they stay at 0.5 m. OV PhysX honors the flag either way. With a static ground and only gravity-free spheres, zero world gravity is the same scene. The 9 Isaac tests pass with and without the fix, because they check the network, not the trajectories. Because the test scripts and `benchmarks/isaac/bench.py` build the env through `make_cfg`, the environment variable reaches them (and the subprocesses of `tests/test_isaac_env.py`) without new command-line flags. `bench.py` now also records `physics` and the min and max sphere height in its `RESULT` line.

### Offline GPU nodes and cloud assets

Isaac Lab loads task assets (the default ground plane, the cartpole USD) from NVIDIA's S3 bucket. With no route to the internet, the first run on a GPU node hung in environment creation for over 9 minutes after "Successfully created S3 provider", in `omni.client.stat` (Isaac Lab checks every locally cached copy against the server before using it), and was cancelled. Isaac Lab's cache under `$TMPDIR` alone therefore does not make GPU jobs work offline. What works is to download the assets on the xfer node with `isaaclab.utils.assets.retrieve_file_path(url)` (`prefetch_assets.sbatch`), which mirrors the S3 layout under `$TMPDIR/https/<host>/...`, copy that tree to `$B/assets`, and point `ISAACSIM_ASSET_ROOT` at `$B/assets/Assets/Isaac/6.1`. Asset paths are then local files, relative references inside the USD files resolve, and no network call is made. A task that needs other assets needs them added to the prefetch list.

## Results (Hazel L40, dedicated)

All numbers below are from one L40 allocated to the job by Slurm, with no other process on it (the fleet runs recorded 0% utilization and 3 MiB in use before starting). Unlike the Windows numbers in [performance.md](performance.md) and [isaac-lab.md](isaac-lab.md), they are not affected by contention. They are single runs.

### Core package

| Suite | Env | Result | Time |
|:---|:---|:---|---:|
| `pytest -m gpu` | core (torch 2.14 cu126, Triton 3.8) | 26 passed | 95 s |
| `pytest -m "not gpu"` | core | 231 passed, 11 skipped | 315 s |
| `pytest -m gpu` | Isaac Lab env (torch 2.12 cu130, Triton 3.7) | 26 passed | 94 s |

The 11 CPU skips are the 9 Isaac tests (no Isaac Lab in the core env), a Sionna comparison (Sionna not installed) and a 5G-LENA table test (tables not generated, by design). `benchmarks/bench_nr.py` stops at the `lena_like` presets for the same reason; `isaac_scale.sbatch` runs it with those two presets removed.

`benchmarks/bench.py`, ms per control step (submit + step), R = 16 unless stated. The level names are the prototype's (`L2` here is the slot-level engine that `make_engine` calls `L2-legacy`).

| E × R | L0 `graph` | L1 `graph` | L1 `triton` | L2 `graph` | L2 `triton` | L2 `reference` | L2 `compile` |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 16 × 16 | | | | | | 118.6 | 0.66 |
| 256 × 16 | 0.41 | 2.12 | 0.43 | 11.0 | 0.53 | 122.0 | 0.74 |
| 1024 × 16 | 0.55 | 5.84 | 0.57 | 15.1 | 0.71 | | |
| 4096 × 16 | 1.64 | 17.3 | 1.77 | 28.3 | 2.21 | | |
| 1024 × 64 | 1.64 | 17.3 | 1.69 | 28.3 | 2.36 | | |

The `triton` backend costs 0.4–2.4 ms per 100 ms control step up to 65,536 robots, 5–21× less than `graph`, which replays thousands of small kernels per step. The eager reference of the slot-level engine takes 119–122 ms per step at 16 × 16 and 256 × 16. Under the contention of the Windows measurements the same step took about 1,400 ms, so the "up to about 20×" pessimism stated in performance.md is borne out. `compile` reaches 0.4–0.7 ms but pays up to 37 s of compilation per configuration. `bench_nr.py` (reference NR engine, R = 16): legacy NetSlot 90 ms, NR compat 250 ms and NR default (13 subbands, EESM) 318 ms per step at E = 64–256, flat in E because the reference engine is launch-bound.

### Isaac Lab smoke runs (kit-less, in the container)

| Run | Result | Wall time of the job step |
|:---|:---|---:|
| `isaaclab train ... Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=newton_mjwarp` | trains, 886 steps/s, 47.5 s training | 163 s (first run, includes Warp kernel compilation) |
| same, `physics=ovphysx` | trains, about 540 steps/s, 7.2 s training | 74 s |
| same, `physics=newton_mjwarp`, 4,096 envs × 30 iterations | trains, about 197,000 steps/s, 15.8 s training | 62 s |
| fleet env, `ISAAC_NET_PHYSICS=isaacsim_physx` | fails as expected: "Isaac Sim is not installed" | 13 s |
| fleet env, `newton`, 16 × 4, L2-legacy `triton` | runs, finite observations | 37 s |
| fleet env, 64 × 16, sphere height after 120 steps | Newton 0.30–0.32 m before the gravity fix, 0.5 m after; OV PhysX 0.5 m both ways | 17–33 s |
| fleet env, `ovphysx`, 16 × 4, L2-legacy `triton` | runs, finite observations | 18 s |
| `pytest -m isaac tests/test_isaac_env.py`, `ISAAC_NET_PHYSICS=newton` | **9 passed** | 179 s |
| same, `ISAAC_NET_PHYSICS=ovphysx` | **9 passed** | 107 s |

The 9 tests include the three bitwise checks: the fleet env's network, fed by live physics poses, equals a reference-engine replay of the recorded poses, sends, tags and RNG stream through partial resets made by DirectRLEnv, at L2-legacy, L1 and L0DR. The network side is therefore validated on Linux with both kit-less physics backends. The physics side is a different simulator from the Windows validation (PhysX through Isaac Sim), so robot trajectories are not comparable across the two.

### Fleet env throughput (kit-less)

`benchmarks/bench.py` in the container, random actions, 100 timed control steps after 20 warm-up steps, network L2-legacy on `triton`. "Off" is the same env without a network. Steps/s are control steps (0.1 s of simulated time each) per second of wall time.

| Physics | E × R | Robots | Startup (s) | Off (steps/s) | Network on (steps/s) | Network-only (ms/step) | Robot-steps/s, network on |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Newton | 256 × 16 | 4,096 | 6–7 | 243 | 157 | 2.0 | 642,122 |
| Newton | 1024 × 16 | 16,384 | 7 | 221 | 156 | 2.0 | 2,553,802 |
| Newton | 1024 × 64 | 65,536 | 11–12 | 16.7 | 16.1 | 2.8 | 1,054,808 |
| Newton | 4096 × 32 | 131,072 | 13–14 | 29.4 | 25.2 | 5.6 | 3,308,432 |
| OV PhysX | 256 × 16 | 4,096 | 4–7 | 27.3 | 24.2 | 2.6 | 99,012 |
| OV PhysX | 1024 × 16 | 16,384 | 14 | 19.8 | 18.8 | 2.6 | 307,792 |
| OV PhysX | 1024 × 64 | 65,536 | 126–135 | 10.1 | 10.5 | 2.9 | 685,479 |
| OV PhysX | 4096 × 32 | 131,072 | 365 | 6.2 | 6.3 | 5.7 | 828,455 |

On a dedicated GPU the `triton` network costs 2.0–5.7 ms per control step up to 131,072 robots. Where physics is slow, which is OV PhysX at every size and Newton from 65k robots up, that is 3–14% of the step. Newton steps 4k–16k robots in about 4 ms, so there the network is about 30% of the step and lowers throughput by 29–36%. Newton starts in seconds at every size, while OV PhysX startup grows with the robot count (365 s at 131k robots), much like the PhysX-through-Kit startup on Windows. Newton with R = 64 is slower than with R = 32 at twice the robots, probably because robot-robot contact pairs grow with R per env. The Newton rows were taken after the zero-gravity fix; a first grid with gravity on (spheres resting on the ground) was 3–40% slower and is superseded. These throughputs are not comparable with the Windows table in isaac-lab.md (different physics, GPU and contention).

## Isaac Sim with Kit (PhysX through Isaac Sim)

Isaac Sim 6.1's pip wheels need glibc 2.35, so on RHEL 9 the only Kit route is a container. NVIDIA publishes `nvcr.io/nvidia/isaac-lab` with Isaac Sim included; tags on 2026-09-29 were `2.0.0`–`2.3.2`, `3.0.0-beta1`, `3.0.0-beta2`, `3.0.0-beta2-post1`, `3.0.0-rc1` and `3.0.0-rc1-kitless` (no final `3.0.0`). The registry issues an anonymous pull token, so **no NGC API key is needed**. The Isaac Sim image `nvcr.io/nvidia/isaac-sim` (tags up to `6.1.0`) also lists anonymously; it was not pulled.

`pull_isaaclab_container.sbatch` pulls `3.0.0-rc1` into a 12.1 GB SIF in about 50 minutes on the xfer partition. Its first attempt failed because `mksquashfs` was killed at the partition's default memory; 22 GB (the xfer QOS allows 24 GB per user) was enough. `isaacsim_ctr_gpu.sbatch` runs it with `apptainer exec --nv --writable-tmpfs`, `HOME` and Kit's `cache`, `data` and `logs` directories bound to `$B/kit/`, the local asset root, and `OMNI_KIT_ACCEPT_EULA=YES`. The image contains Isaac Sim `6.1.0-rc.26`, Isaac Lab `3.0.0` rc1 (package `isaaclab` 17.0.2), torch 2.11.0+cu128, Triton 3.6.0 and pytest, and its Python is `/isaac-sim/python.sh`. An NVIDIA Vulkan ICD file (`/etc/vulkan/icd.d/nvidia_icd.json`) is present inside the container. The findings were:

- **Kit starts headless and the official smoke run trains**: `isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=isaacsim_physx` reached about 600 steps/s, with 14.8 s of training and 116 s for the whole job step. There was no EGL, Vulkan, driver or NGC-authentication problem. Kit only warned that it could not open an X display.
- **Our fleet env does not start on this image.** `SimulationApp` fails with `AttributeError: module 'omni.usd' has no attribute 'get_context'`, after the warning "Please check to make sure no extra omniverse or pxr modules are imported before the call to SimulationApp(...)". With rc1's Isaac Lab, importing `isaac_net.examples.isaac_fleet_env` loads `pxr` (an import probe showed that `isaaclab.sim`, `isaaclab_physx` and our mixins alone do not), and our scripts import the env module before they launch the app, because they need its config to choose the launcher. With `release/3.0.0` the same import does not load `pxr`, which is why the Windows install and the kit-less runs are unaffected. Kit then exits with status 0, so only the missing `CHECK`/`RESULT` line shows the failure, and `pytest -m isaac` reports all 9 cases as failed.
- **Binding the `release/3.0.0` checkout over the image's `/workspace/isaaclab` does not help.** The newer Isaac Lab needs a newer Warp than the image ships, and `import isaaclab` fails in `warp.struct` with `TypeError: issubclass() arg 1 must be a class`.

So the Kit route on this cluster needs an image with Isaac Lab `release/3.0.0` on Isaac Sim 6.1.0: either a final `isaac-lab:3.0.0` tag when NVIDIA publishes one, or an image built with Isaac Lab's own `docker/container.py` on a machine with Docker and converted with `apptainer build x.sif docker-archive://x.tar`. The alternative is to make the fleet env module importable without `pxr` on older Isaac Lab versions. Neither was tried. Until then, use the kit-less route on RHEL-type clusters.

## Ubuntu without a container

On Ubuntu 22.04 or 24.04 (glibc 2.35/2.39), skip Apptainer and run step 3's commands directly: install uv, clone Isaac Lab `release/3.0.0`, `uv sync --extra rsl-rl --extra ovphysx` (add `--extra isaacsim` for Kit), then `uv pip install --no-deps -e <isaac-net>`. This was not run here; the only difference from the tested path is the missing container.

## Failure modes seen

1. `./isaaclab.sh --install` runs `sudo apt-get update` when `cmake` is not on `PATH`, which fails in a batch job ("sudo: a terminal is required") and is inappropriate on a shared machine. Use `uv sync`.
2. Native `uv sync` on glibc 2.34: "Distribution `omniverseclient==2.74.0` can't be installed because it doesn't have a source distribution or wheel for the current platform ... You're on Linux (`manylinux_2_34_x86_64`)". Forcing the wheel would not help: its `libomniverse_connection.so` imports `GLIBC_2.35` symbols (checked with `objdump -T`). `ovphysx` and `usd-exchange` only import up to `GLIBC_2.34`, but `ovstage` bundles the same library.
3. Apptainer on Hazel: `/share` is a symlink, so the container needs `--bind /gpfs_common,/gpfs_common/share:/share`, otherwise "cd: ... No such file or directory".
4. Offline GPU nodes: without a local asset root, environment creation hangs in `omni.client.stat` (see above).
5. `apptainer pull` of the Isaac Lab image: `mksquashfs` was killed (out of memory) at the xfer partition's default job memory. The xfer QOS allows at most 4 CPUs and 24 GB per user, and `--mem=22G` works.
6. Kit in the `3.0.0-rc1` image: `omni.usd` has no attribute `get_context` when `pxr` was imported before `SimulationApp`. Our fleet env triggers this on rc1's Isaac Lab, and Kit still exits with status 0 (see above).
7. A30 jobs stayed pending with `QOSGrpGRES` for over an hour (the gpu QOS caps the whole group at 4 A30s), so all GPU results here are on the L40.
8. `benchmarks/isaac/bench.py` printed a `SyntaxWarning: invalid escape sequence '\i'` from a Windows path in its docstring, now fixed.
