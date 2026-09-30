"""ITU-R P.2040 radio materials for exported scenes, and the rules that assign them to USD prims.

Assignment order (the first rule that gives a material wins):

    1. user     the user mapping {pattern: material}: fnmatch patterns, case-insensitive, tested against the prim
                path, then the prim's semantic labels, then the name of its bound USD material
    2. semantic semantic labels of the prim or of an ancestor (UsdSemantics LabelsAPI, or the older Isaac
                `semantic:*:params:semanticData` attributes), through the keyword table below
    3. material the bound USD material's name, through the keyword table
    4. name     the prim path's names, through the keyword table
    5. default  DEFAULT_MATERIAL (concrete)

Keywords are matched against name tokens: a name is split at non-alphanumerics and camel-case humps ("SM_RackShelf_01"
-> sm, rack, shelf, 01), and a token matches a keyword when it starts with it. Words that name a material (steel,
wood, glass, ...) win over words that name an object (rack, pallet, window, ...), so "WoodenRack" is wood. Within each
tier the table order decides. There is no ITU-R P.2040 entry for plastic or cardboard: both map to wood, the closest
low-permittivity material in the table (plastic bins, crates, bottles and cones; cardboard boxes).

Thickness is the thickness of the slab Sionna RT puts behind every face a ray crosses. A closed solid (a box wall) is
crossed twice, so the defaults are half of a typical physical thickness for the materials that come as solids
(concrete 0.1 m for a 0.2 m wall, as in Sionna's own scenes) and the sheet thickness for the others.
"""
from __future__ import annotations

import fnmatch
import math
import re
from typing import Iterable, Mapping, Optional

DEFAULT_MATERIAL = "concrete"

# ITU-R P.2040-3 Table 3 (as tabulated in Sionna RT 2.2 radio_materials/itu.py), 1-100 GHz unless noted:
# eps_r = a f^b, sigma = c f^d (f in GHz, sigma in S/m).
ITU_PARAMS = {
    "concrete": (5.24, 0.0, 0.0462, 0.7822),
    "brick": (3.91, 0.0, 0.0238, 0.16),               # 1-40 GHz
    "plasterboard": (2.73, 0.0, 0.0085, 0.9395),
    "wood": (1.99, 0.0, 0.0047, 1.0718),
    "glass": (6.31, 0.0, 0.0036, 1.3394),
    "ceiling_board": (1.48, 0.0, 0.0011, 1.0750),
    "chipboard": (2.58, 0.0, 0.0217, 0.7800),
    "plywood": (2.71, 0.0, 0.33, 0.0),                # 1-40 GHz
    "marble": (7.074, 0.0, 0.0055, 0.9262),           # 1-60 GHz
    "floorboard": (3.66, 0.0, 0.0044, 1.3515),        # 50-100 GHz
    "metal": (1.0, 0.0, 1e7, 0.0),
    "very_dry_ground": (3.0, 0.0, 0.00015, 2.52),     # 1-10 GHz
    "medium_dry_ground": (15.0, -0.1, 0.035, 1.63),   # 1-10 GHz
    "wet_ground": (30.0, -0.4, 0.15, 1.30),           # 1-10 GHz
}
MATERIALS = tuple(ITU_PARAMS)

# per-face slab thickness (m), see the module docstring
DEFAULT_THICKNESS = {"concrete": 0.1, "brick": 0.1, "plasterboard": 0.0125, "wood": 0.02, "glass": 0.006,
                     "ceiling_board": 0.015, "chipboard": 0.015, "plywood": 0.012, "marble": 0.02,
                     "floorboard": 0.02, "metal": 0.002, "very_dry_ground": 0.5, "medium_dry_ground": 0.5,
                     "wet_ground": 0.5}

# tier 1: words that name a material
MATERIAL_WORDS = (
    ("glass", ("glass", "glazing")),
    ("metal", ("metal", "steel", "iron", "alumin", "chrome", "zinc", "galvan")),
    ("plasterboard", ("plaster", "drywall", "gypsum", "sheetrock")),
    ("wood", ("wood", "timber", "plywood", "osb", "oak", "pine", "cardboard", "carton", "paper", "plastic",
              "polymer", "rubber")),
    ("concrete", ("concrete", "cement", "asphalt", "stone", "brick", "masonry")),
)
# tier 2: words that name an object (warehouse vocabulary)
OBJECT_WORDS = (
    ("glass", ("window", "windshield")),
    ("metal", ("rack", "shelf", "shelv", "bracket", "forklift", "beam", "girder", "truss", "pipe", "duct",
               "conveyor", "cage", "locker", "cabinet", "container", "ladder", "railing", "guardrail", "bollard",
               "rollup", "shutter", "vent", "lamp", "light", "wire", "fence", "barrel", "barel", "drum", "fuse",
               "extinguisher", "cart", "trolley", "vehicle", "truck")),
    ("wood", ("pallet", "palette", "crate", "plank", "box", "boxes", "package", "parcel", "bin", "bucket",
              "bottle", "cone", "note")),
    ("plasterboard", ("partition", "office")),
    ("concrete", ("wall", "floor", "ground", "ceiling", "roof", "pillar", "column", "slab", "foundation",
                  "stair", "ramp", "dock", "curb")),
)

_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokens(name: str) -> list:
    """Lower-case tokens of a name: split at non-alphanumerics and camel-case humps."""
    out = []
    for part in _SPLIT.split(name):
        out += [t.lower() for t in _CAMEL.findall(part)]
    return out


def keyword_material(names: Iterable[str]) -> Optional[str]:
    """The material the keyword tables give for a set of names (None: no keyword matched)."""
    toks = [t for n in names for t in tokens(n)]
    if not toks:
        return None
    for table in (MATERIAL_WORDS, OBJECT_WORDS):
        for mat, words in table:
            if any(t.startswith(w) for t in toks for w in words):
                return mat
    return None


def check_material(name: str) -> str:
    if name not in ITU_PARAMS:
        raise ValueError(f"unknown ITU radio material {name!r}; one of {MATERIALS}")
    return name


class MaterialRules:
    """Assigns an ITU material to a prim from its path, semantic labels and bound material name (module docstring).

    mapping: {fnmatch pattern: material}, tried in order; default: the fallback material; keywords: False turns
    the keyword tables off (then only the mapping and the default apply)."""

    def __init__(self, mapping: Optional[Mapping[str, str]] = None, default: str = DEFAULT_MATERIAL,
                 keywords: bool = True):
        self.mapping = [(p.lower(), check_material(m)) for p, m in (mapping or {}).items()]
        self.default = check_material(default)
        self.keywords = keywords

    def assign(self, path: str, labels: Iterable[str] = (), material_name: Optional[str] = None) -> tuple:
        """(material, source) with source one of user / semantic / material / name / default."""
        labels = [str(x) for x in labels]
        cands = [path.lower()] + [x.lower() for x in labels] + ([material_name.lower()] if material_name else [])
        for pat, mat in self.mapping:
            if any(fnmatch.fnmatchcase(c, pat) for c in cands):
                return mat, "user"
        if self.keywords:
            if labels:
                m = keyword_material(labels)
                if m:
                    return m, "semantic"
            if material_name:
                m = keyword_material([material_name])
                if m:
                    return m, "material"
            m = keyword_material([s for s in path.split("/") if s])
            if m:
                return m, "name"
        return self.default, "default"


# ------------------------------------------------------------------------------------------------ slab model
def complex_permittivity(material: str, fc_ghz: float) -> complex:
    """eta = eps_r - j sigma / (eps_0 omega) of an ITU material at fc_ghz (ITU-R P.2040-3 eqs. 9a-9b)."""
    a, b, c, d = ITU_PARAMS[check_material(material)]
    eps_r = a * fc_ghz ** b
    sigma = c * fc_ghz ** d
    return complex(eps_r, -17.98 * sigma / fc_ghz)


def slab_transmission_db(material: str, thickness_m: float, fc_ghz: float) -> float:
    """Power loss (dB, positive) of a plane wave through one slab at normal incidence, ITU-R P.2040-3 eqs. 37-38:
    T = (1 - R'^2) e^{-jq} / (1 - R'^2 e^{-j2q}), R' the air-slab Fresnel coefficient, q = 2 pi d sqrt(eta) / lambda."""
    eta = complex_permittivity(material, fc_ghz)
    n = eta ** 0.5
    r = (1 - n) / (1 + n)
    lam = 299_792_458.0 / (fc_ghz * 1e9)
    q = 2 * math.pi * thickness_m * n / lam
    import cmath
    t = (1 - r * r) * cmath.exp(-1j * q) / (1 - r * r * cmath.exp(-2j * q))
    return -20.0 * math.log10(max(abs(t), 1e-30))


def free_space_gain_db(d_m, fc_ghz: float):
    """Free-space path gain (dB, negative) between isotropic antennas: -20 log10(4 pi d / lambda)."""
    import numpy as np
    lam = 299_792_458.0 / (fc_ghz * 1e9)
    return -20.0 * np.log10(4 * np.pi * np.maximum(np.asarray(d_m, dtype=float), 1e-9) / lam)
