"""USD scene -> Sionna RT scene -> radio map (docs/scene-radio-map.md).

    materials       ITU-R P.2040 materials, assignment rules (user mapping, semantic label, USD material, prim name),
                    slab-transmission and free-space formulas for checks
    mitsuba_writer  Mitsuba XML + binary PLY per material (numpy only)
    usd_export      export_usd(stage, out_dir, ...): meshes with world transforms in the map frame (needs pxr)
    synthetic       small USD scenes with known geometry (needs pxr)
    bake            Sionna RT RadioMapSolver -> the radio_map file; CLI `python -m isaac_net.tools.scene.bake`
    sector          the TR 38.901 sector antenna pattern applied to a baked (isotropic) map (bake --gnb-antenna)

Nothing here is imported by the engine; pxr and sionna-rt are imported only when a function needs them.
"""
