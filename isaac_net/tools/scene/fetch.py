"""Download a USD asset and every layer it composes (sublayers, references, payloads) from an HTTP(S) location.

Plain `usd-core` cannot open the http URLs Isaac Lab uses for its assets (ISAAC_NUCLEUS_DIR is an S3 bucket; inside
Isaac Sim the Omniverse resolver handles them). The files are mirrored under out_dir/<host>/<url path>, so relative
asset paths such as ../../Props/x.usd resolve locally exactly as on the server. Textures and MDL files are not
needed for the geometry and are skipped.

    python -m isaac_net.tools.scene.fetch \\
        https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Warehouse/warehouse.usd \\
        assets/
"""
from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request

USD_EXT = (".usd", ".usda", ".usdc", ".usdz")


def _layer_deps(path: str) -> list:
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(path)
    if layer is None:
        return []
    if hasattr(layer, "GetCompositionAssetDependencies"):
        return [str(d) for d in layer.GetCompositionAssetDependencies()]
    from pxr import UsdUtils
    sub, refs, pay = UsdUtils.ExtractExternalReferences(path)
    return [str(d) for d in (*sub, *refs, *pay)]


def local_path(url: str, out_dir: str) -> str:
    u = urllib.parse.urlparse(url)
    return os.path.join(out_dir, u.netloc, *[p for p in urllib.parse.unquote(u.path).split("/") if p])


def fetch_usd(url: str, out_dir: str, timeout: float = 60.0, verbose: bool = True) -> str:
    """Download `url` and the layers it composes (recursively) below out_dir; returns the local root layer path."""
    todo, seen = [url], set()
    n = 0
    while todo:
        u = todo.pop()
        if u in seen:
            continue
        seen.add(u)
        dst = local_path(u, out_dir)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with urllib.request.urlopen(u, timeout=timeout) as r, open(dst + ".part", "wb") as fh:
                fh.write(r.read())
            os.replace(dst + ".part", dst)
            n += 1
        for d in _layer_deps(dst):
            if not d.lower().split("?")[0].endswith(USD_EXT):
                continue
            if "://" in d and not d.startswith(("http://", "https://")):
                continue
            todo.append(d if "://" in d else urllib.parse.urljoin(u, d))
    if verbose:
        print(f"{len(seen)} layers ({n} downloaded) under {out_dir}", file=sys.stderr)
    return local_path(url, out_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    print(fetch_usd(argv[0], argv[1]))


if __name__ == "__main__":
    main()
