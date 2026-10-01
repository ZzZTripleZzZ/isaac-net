"""Write triangle meshes grouped by ITU material as a Sionna RT scene: Mitsuba XML plus binary PLY meshes.

Layout of an exported scene directory:

    scene.xml            <scene version="2.1.0">, one itu-radio-material bsdf per material (id = material name,
                         thickness = the per-face slab thickness), one ply shape per material
    meshes/<mat>.ply     binary little-endian PLY, float32 x y z, faces as uchar count + int32 indices
    manifest.json        what the exporter did (written by usd_export)

Coordinates are metres, Z up, in the map frame (see usd_export). Only numpy is needed.
"""
from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from .materials import DEFAULT_THICKNESS, check_material


def write_ply(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    v = np.ascontiguousarray(vertices, dtype="<f4").reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    assert f.size == 0 or (f.min() >= 0 and f.max() < len(v)), "face index out of range"
    rec = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    rec["n"] = 3
    rec["i"] = f
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(v.tobytes())
        fh.write(rec.tobytes())


def read_ply(path: str) -> tuple:
    """(vertices [N,3] float32, faces [M,3] int32) of a PLY written by write_ply."""
    with open(path, "rb") as fh:
        data = fh.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode("ascii").split("\n")
    nv = int(next(h for h in head if h.startswith("element vertex")).split()[-1])
    nf = int(next(h for h in head if h.startswith("element face")).split()[-1])
    v = np.frombuffer(data, dtype="<f4", count=nv * 3, offset=end).reshape(nv, 3)
    rec = np.frombuffer(data, dtype=[("n", "u1"), ("i", "<i4", (3,))], count=nf, offset=end + nv * 12)
    return v.copy(), rec["i"].astype(np.int32)


def write_scene(out_dir: str, groups: Mapping[str, tuple], thickness: Mapping[str, float] | None = None,
                comment: str = "") -> str:
    """groups {material: (vertices [N,3], faces [M,3])} -> out_dir/scene.xml (returned) and out_dir/meshes/*.ply.
    thickness {material: m} overrides DEFAULT_THICKNESS."""
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    th = dict(DEFAULT_THICKNESS)
    th.update(thickness or {})
    mats, shapes = [], []
    for mat in sorted(groups):
        v, f = groups[mat]
        if len(f) == 0:
            continue
        check_material(mat)
        write_ply(os.path.join(out_dir, "meshes", f"{mat}.ply"), v, f)
        mats.append(f'  <bsdf type="itu-radio-material" id="{mat}">\n'
                    f'    <string name="type" value="{mat}"/>\n'
                    f'    <float name="thickness" value="{float(th[mat]):.6g}"/>\n  </bsdf>')
        shapes.append(f'  <shape type="ply" id="mesh-{mat}_objects">\n'
                      f'    <string name="filename" value="meshes/{mat}.ply"/>\n'
                      f'    <boolean name="face_normals" value="true"/>\n'
                      f'    <ref id="{mat}" name="bsdf"/>\n  </shape>')
    if not shapes:
        raise ValueError("no triangles to write")
    head = f"<!-- {comment} -->\n" if comment else ""
    xml = head + '<scene version="2.1.0">\n' + "\n".join(mats) + "\n" + "\n".join(shapes) + "\n</scene>\n"
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w") as fh:
        fh.write(xml)
    return path
