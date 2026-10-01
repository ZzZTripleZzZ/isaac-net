"""Radio map from the Isaac Lab stage: the scene's shelves and walls become the network's channel.

    from isaac_net.isaac import IsaacNetCfg
    from isaac_net.isaac.scene_map import SceneRadioMapCfg

    isaac = IsaacNetCfg(pose_asset="robots", gnb_pos=((10.0, 20.0, 6.0),),
                        scene_map=SceneRadioMapCfg(root="/World/envs/env_0", exclude=("*/robot_*",)))
    self.net_setup("L2", R, nr, "reference", isaac=isaac)          # in _setup_scene, after the scene is spawned

With IsaacNetCfg.scene_map set, NetEnvMixin.net_setup calls resolve_scene_map before it builds the network:

    1. export   the subtree `root` of the current stage (tools/scene/usd_export: meshes, world transforms, ITU
                materials) into the map frame: the frame of prim `frame` (default env_0, i.e. env-local
                coordinates) plus IsaacNetCfg.pose_offset_m, the frame of the poses the network sees
    2. key      sha256 of the exported geometry and materials plus every bake parameter
    3. load     <cache_dir>/<key>.pt if it exists (a stage that did not change is not baked again)
    4. bake     otherwise run Sionna RT: in-process when `sionna.rt` imports, else in a separate interpreter
                (`python`, or $ISAAC_NET_SIONNA_PYTHON) with `python -m isaac_net.tools.scene.bake`
    5. apply    NRConfig(channel="radio_map", radio_map_path=<file>, n_cells=C, cell_layout="custom",
                cell_positions_m=<gNB xy>) and IsaacNetCfg(radio="engine"): the engine's radio samples the map

The transmitters are IsaacNetCfg.gnb_positions(nr) (x, y and mast height) unless SceneRadioMapCfg.tx is set. The map
is static: robots are not part of it (exclude them), and their bodies are handled at run time by the blockage
add-on (NRConfig.blockage). Every env shares env_0's map, so the envs must be clones of one layout.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SceneRadioMapCfg:
    """How to turn the stage into a radio map (module docstring). None fields come from the NRConfig / IsaacNetCfg."""
    root: str = "/World/envs/env_0"            # subtree to export
    frame: Optional[str] = "/World/envs/env_0"  # its frame is the map frame (env-local); None = stage world
    offset_m: Optional[Sequence[float]] = None  # None: IsaacNetCfg.pose_offset_m
    include: Sequence[str] = ()                 # fnmatch patterns on prim paths (usd_export)
    exclude: Sequence[str] = ("*/robot*",)      # the robots are not part of the static map
    material_map: dict = field(default_factory=dict)   # {pattern: ITU material}
    default_material: str = "concrete"
    thickness: dict = field(default_factory=dict)      # {material: per-face slab thickness m}
    crop: Optional[Sequence[float]] = None      # (x0, y0, x1, y1) map frame: drop geometry outside
    z_max: Optional[float] = None
    tx: Optional[Sequence[Sequence[float]]] = None     # None: IsaacNetCfg.gnb_positions(nr)
    fc_ghz: Optional[float] = None             # None: NRConfig.carrier_ghz
    ue_height_m: Optional[float] = None        # None: NRConfig.ue_height_m
    cell_m: float = 1.0
    bounds: Optional[Sequence[float]] = None   # area of the map; None: footprint of the exported geometry
    samples: int = 1_000_000
    depth: int = 4
    diffraction: bool = False
    los_map: bool = True
    seed: int = 42
    variant: Optional[str] = None              # Mitsuba variant ("cuda", "llvm", ...); None = Sionna's default
    python: Optional[str] = None               # interpreter with sionna-rt when this one has none
    cache_dir: Optional[str] = None            # None: $ISAAC_NET_RADIO_MAPS or ~/.cache/isaac_net/radio_maps
    rebake: bool = False                       # ignore a cached map
    timeout_s: float = 3600.0


def cache_dir(cfg: Optional[SceneRadioMapCfg] = None) -> str:
    d = (cfg.cache_dir if cfg is not None and cfg.cache_dir else None) or os.environ.get("ISAAC_NET_RADIO_MAPS") \
        or os.path.join(os.path.expanduser("~"), ".cache", "isaac_net", "radio_maps")
    os.makedirs(d, exist_ok=True)
    return d


def current_stage():
    """The USD stage of the running Isaac Sim app."""
    try:
        from isaaclab.sim.utils import get_current_stage
        return get_current_stage()
    except ImportError:
        pass
    try:
        from isaaclab.sim.utils.stage import get_current_stage
        return get_current_stage()
    except ImportError:
        pass
    import omni.usd
    return omni.usd.get_context().get_stage()


def _sionna_available() -> bool:
    try:
        import sionna.rt  # noqa: F401
        return True
    except Exception:
        return False


def bake_stage_map(stage, scfg: SceneRadioMapCfg, tx: Sequence[Sequence[float]], fc_ghz: float, ue_h: float,
                   offset_m: Sequence[float] = (0.0, 0.0, 0.0), verbose: bool = True) -> tuple:
    """Export `stage` (object or path), then load the cached map or bake it. Returns (map path, info dict)."""
    import time

    from ..tools.scene.bake import bake_key, bake_scene, write_map
    from ..tools.scene.usd_export import export_usd

    root = cache_dir(scfg)
    t0 = time.perf_counter()
    scene_dir = tempfile.mkdtemp(prefix="scene_", dir=root)
    res = export_usd(stage, scene_dir, root=scfg.root, frame=scfg.frame, offset_m=offset_m,
                     material_map=scfg.material_map, default_material=scfg.default_material,
                     thickness=scfg.thickness, include=scfg.include, exclude=scfg.exclude, crop=scfg.crop,
                     z_max=scfg.z_max)
    t_export = time.perf_counter() - t0
    bounds = tuple(scfg.bounds) if scfg.bounds is not None else \
        (res.bounds_min[0], res.bounds_min[1], res.bounds_max[0], res.bounds_max[1])
    key = bake_key(res.scene_hash, tx, fc_ghz, bounds, scfg.cell_m, ue_h, scfg.samples, scfg.depth, True,
                   scfg.diffraction, scfg.los_map, 3, scfg.seed)
    path = os.path.join(root, f"{key[:24]}.pt")
    final_scene = os.path.join(root, f"scene_{res.scene_hash[:16]}")
    if os.path.exists(final_scene):                  # same geometry: refresh the files (the writer may have changed)
        import shutil
        shutil.rmtree(final_scene, ignore_errors=True)
    os.replace(scene_dir, final_scene)
    xml = os.path.join(final_scene, "scene.xml")
    info = dict(path=path, key=key, scene_hash=res.scene_hash, export_s=t_export, summary=res.summary(),
                coverage=res.coverage(), cached=False, scene_xml=xml)
    if os.path.exists(path) and not scfg.rebake:
        info["cached"] = True
        if verbose:
            print(f"[scene_map] {res.summary()}\n[scene_map] cached map {path}", flush=True)
        return path, info
    t1 = time.perf_counter()
    if _sionna_available():
        import json
        m = bake_scene(xml, tx, fc_ghz, bounds, scfg.cell_m, ue_h, scfg.samples, scfg.depth,
                       diffraction=scfg.diffraction, los_map=scfg.los_map, seed=scfg.seed, variant=scfg.variant)
        m.update(scene_hash=res.scene_hash, bake_key=key, source="sionna-rt RadioMapSolver, stage " + scfg.root,
                 materials=json.dumps(dict(materials=res.materials, sources=res.sources, coverage=res.coverage())))
        write_map(path, m)
    else:
        py = scfg.python or os.environ.get("ISAAC_NET_SIONNA_PYTHON")
        if not py:
            raise RuntimeError("sionna-rt is not importable here: set SceneRadioMapCfg.python (or "
                               "$ISAAC_NET_SIONNA_PYTHON) to an interpreter with sionna-rt and torch, or bake "
                               f"offline with python -m isaac_net.tools.scene.bake --scene-xml {xml}")
        cmd = [py, "-m", "isaac_net.tools.scene.bake", "--scene-xml", xml, "--out", path, "--fc", str(fc_ghz),
               "--cell", str(scfg.cell_m), "--ue-height", str(ue_h), "--samples", str(int(scfg.samples)),
               "--depth", str(scfg.depth), "--seed", str(scfg.seed), "--bounds", *map(str, bounds)]
        for p in tx:
            cmd += ["--tx", *map(str, p)]
        if scfg.los_map:
            cmd.append("--los-map")
        if scfg.diffraction:
            cmd.append("--diffraction")
        if scfg.variant:
            cmd += ["--variant", scfg.variant]
        env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=scfg.timeout_s)
        if r.returncode != 0 or not os.path.exists(path):
            raise RuntimeError(f"bake failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        info["bake_log"] = r.stdout.strip().splitlines()[-1:] if r.stdout else []
    info["bake_s"] = time.perf_counter() - t1
    if verbose:
        print(f"[scene_map] {res.summary()}\n[scene_map] baked {path} in {info['bake_s']:.1f} s", flush=True)
    return path, info


def apply_scene_radio_map(nr, isaac, path: str):
    """(NRConfig, IsaacNetCfg) that use the map at `path`: channel="radio_map" with one cell per map gNB at the map's
    gNB positions, and the engine's radio (radio="engine")."""
    from ..core.channels.radio_map import RadioMap

    m = RadioMap.load(path)
    gxy = m.meta.get("gnb_xy")
    if gxy is None:
        raise ValueError(f"{path} has no gnb_xy metadata")
    gxy = [tuple(float(v) for v in row) for row in gxy.reshape(-1, 2)]
    nr2 = nr.with_(channel="radio_map", radio_map_path=path, n_cells=len(gxy), cell_layout="custom",
                   cell_positions_m=tuple(gxy))
    isaac2 = isaac.with_(radio="engine", scene_map=None)
    return nr2, isaac2


def resolve_scene_map(nr, isaac, stage=None, verbose: bool = True):
    """The hook behind IsaacNetCfg.scene_map: bake or load the map of the current stage and return the (NRConfig,
    IsaacNetCfg) that use it. nr None means NRConfig()."""
    from ..core.config import NRConfig

    scfg = isaac.scene_map
    if not isinstance(scfg, SceneRadioMapCfg):
        raise TypeError("IsaacNetCfg.scene_map must be a SceneRadioMapCfg")
    nr = nr if nr is not None else NRConfig()
    tx = [tuple(float(v) for v in p) for p in (scfg.tx if scfg.tx is not None else isaac.gnb_positions(nr))]
    fc = float(scfg.fc_ghz if scfg.fc_ghz is not None else nr.carrier_ghz)
    ue_h = float(scfg.ue_height_m if scfg.ue_height_m is not None else nr.ue_height_m)
    offset = tuple(scfg.offset_m) if scfg.offset_m is not None else tuple(isaac.pose_offset_m)
    stage = stage if stage is not None else current_stage()
    path, info = bake_stage_map(stage, scfg, tx, fc, ue_h, offset, verbose=verbose)
    nr2, isaac2 = apply_scene_radio_map(nr, isaac, path)
    resolve_scene_map.last_info = info
    return nr2, isaac2


resolve_scene_map.last_info = None
