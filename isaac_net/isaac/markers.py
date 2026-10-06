"""Viewport overlays of the network in Isaac Lab: links coloured by SINR, gNBs with coverage, AoI bars, access state.

    from isaac_net.isaac import NetMarkersCfg
    self.net_setup("L2-legacy", R, nr, "graph", isaac=isc, markers=NetMarkersCfg(update_every=2))   # Direct mixin
    NetManagerCfg(..., markers=NetMarkersCfg())                                                    # manager-based

NetMarkers(env, net, cfg) draws, when something displays the stage (a Kit viewport with `--viz kit`, a Newton
visualizer, or offscreen video capture):
  * a line from every robot to its serving gNB, coloured by the SINR of the step in 5 bins (red < 0 dB ... green
    >= 20 dB); links whose line of sight is blocked are dimmer and dashed
  * a mast per gNB with a translucent coverage disc (radius where the nominal SNR falls to coverage_snr_db)
  * a vertical bar above every robot whose height and colour encode its age of information
  * a sphere above robots that are idle, in random access, DRX-dormant or in radio link failure (level L2 with
    NRConfig(rach=True) / drx / rlf)

Shapes are isaaclab.markers.VisualizationMarkers prototypes (one PointInstancer, 21 prototypes); links can instead go
through the Kit debug-draw interface (NetMarkersCfg.line_backend = "debug_draw"). The geometry is computed in torch on
the network's device by isaac/marker_geometry.py every `update_every` env steps and moved to the host in one copy.
Headless (no GUI, no visualizer, no offscreen rendering), NetMarkers is inert: update() returns at once, nothing is
computed and nothing from Isaac Lab is imported.
"""
from __future__ import annotations

from typing import Optional

import torch

from .marker_geometry import (NetMarkersCfg, OverlayFrame, overlay_frame, prototypes, radio_coverage,  # noqa: F401
                              radio_gnb)


def display_active(env) -> bool:
    """True if anything can show markers: a GUI, an active visualizer, offscreen rendering (video). False for a
    plain headless run, and for an env without a simulation context (CPU tests)."""
    sim = getattr(env, "sim", None)
    if sim is None:
        return False
    try:
        if bool(getattr(sim, "has_gui", False)) or bool(getattr(sim, "has_offscreen_render", False)):
            return True
        f = getattr(sim, "has_active_visualizers", None)
        return bool(f()) if callable(f) else False
    except Exception:
        return False


class NetMarkers:
    """Draw the network state of a NetModule `net` living in Isaac Lab env `env` (see the module docstring)."""

    def __init__(self, env, net, cfg: Optional[NetMarkersCfg] = None, force: bool = False):
        self.env, self.net, self.cfg = env, net, cfg or NetMarkersCfg()
        self.active = bool(self.cfg.enable) and (force or display_active(env))
        self._tick = 0
        self._vis = None
        self._draw = None
        self.updates = 0
        self.last_frame: Optional[OverlayFrame] = None

    # ------------------------------------------------------------------ geometry (pure torch)
    def frame(self, poses: torch.Tensor, out: Optional[dict]) -> OverlayFrame:
        """The overlay of the current state, on the network's device."""
        net = self.net
        origins = getattr(getattr(self.env, "scene", None), "env_origins", None)
        if origins is None:
            origins = torch.zeros(net.E, 3, device=net.dev)
        cov = radio_coverage(net, self.cfg) if (self.cfg.gnbs and self.cfg.coverage) else None
        return overlay_frame(self.cfg, poses, out, radio_gnb(net), origins, net.isaac.pose_offset_m, cov)

    # ------------------------------------------------------------------ drawing
    def update(self, poses: Optional[torch.Tensor], out: Optional[dict], force: bool = False) -> bool:
        """Redraw if due (every cfg.update_every calls). poses [E,R,3] radio frame; out: NetModule.step dict.
        Returns True if something was drawn. Inert when headless."""
        if not self.active or poses is None:
            return False
        self._tick += 1
        if not force and (self._tick - 1) % self.cfg.update_every != 0:
            return False
        with torch.no_grad():
            f = self.frame(poses, out)
            if self.cfg.host_transfer:
                f = f.to_host()
        self.last_frame = f
        self._draw_markers(f)
        if self.cfg.line_backend == "debug_draw":
            self._draw_lines(f)
        self.updates += 1
        return True

    def _markers(self):
        if self._vis is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            protos = {}
            for name, shape, rgb, opacity in prototypes(self.cfg):
                mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(x) for x in rgb), opacity=float(opacity),
                                                  roughness=1.0)
                if shape == "sphere":
                    protos[name] = sim_utils.SphereCfg(radius=1.0, visual_material=mat)
                else:
                    protos[name] = sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="Z", visual_material=mat)
            self._vis = VisualizationMarkers(VisualizationMarkersCfg(prim_path=self.cfg.prim_path, markers=protos))
        return self._vis

    def _draw_markers(self, f: OverlayFrame):
        vis = self._markers()
        if f.num_markers == 0:
            vis.set_visibility(False)
            return
        if not vis.is_visible():
            vis.set_visibility(True)
        vis.visualize(translations=f.translations, orientations=f.orientations, scales=f.scales,
                      marker_indices=f.indices.to(torch.int32))

    def _debug_draw(self):
        if self._draw is None:
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                try:
                    import omni.kit.app
                    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                        "isaacsim.util.debug_draw", True)
                    from isaacsim.util.debug_draw import _debug_draw
                except Exception:
                    from omni.isaac.debug_draw import _debug_draw            # Isaac Sim 4.x name
            self._draw = _debug_draw.acquire_debug_draw_interface()
        return self._draw

    def _draw_lines(self, f: OverlayFrame):
        dd = self._debug_draw()
        dd.clear_lines()
        if f.num_lines == 0:
            return
        a, b, c = (x.cpu() for x in (f.line_a, f.line_b, f.line_rgba))
        w = self.cfg.line_width_px
        dd.draw_lines([tuple(p) for p in a.tolist()], [tuple(p) for p in b.tolist()],
                      [tuple(p) for p in c.tolist()], [w] * len(a))

    def set_visibility(self, visible: bool):
        if self._vis is not None:
            self._vis.set_visibility(visible)
        if not visible and self._draw is not None:
            self._draw.clear_lines()


def make_markers(env, net, cfg) -> Optional[NetMarkers]:
    """NetMarkers for net_setup(..., markers=cfg): None for cfg None / False or a network that is off; True means
    NetMarkersCfg()."""
    if cfg is None or cfg is False or net is None:
        return None
    if cfg is True:
        cfg = NetMarkersCfg()
    return NetMarkers(env, net, cfg)
