# Cutting a release

This page is the checklist for a release of `isaac-net`. **Publishing to PyPI is deferred until the project is public.** Until then a release is a git tag plus the wheel and sdist attached to it (or handed to collaborators directly), and nobody runs `twine upload`.

## Versioning

The version has one source of truth: `__version__` in `isaac_net/__init__.py`. `pyproject.toml` reads it (`dynamic = ["version"]`, `[tool.setuptools.dynamic]`), so never write a version anywhere else. The project follows semantic versioning. While the major version is 0, a minor release (0.x.0) may change the API and a patch release (0.x.y) only fixes bugs. Every API change is listed under **Changed** in [CHANGELOG.md](CHANGELOG.md).

## Checklist

1. **Branch.** Start from an up-to-date `main` on a branch `release/X.Y.Z`. Everything meant for the release is merged.
2. **Version and changelog.** Set `__version__ = "X.Y.Z"` in `isaac_net/__init__.py`. In `CHANGELOG.md`, replace `unreleased` with the date (`YYYY-MM-DD`), check that every merged branch since the last release has its entry, and start a new empty `## [Unreleased]` section above it.
3. **Docs.** Update `docs/STATUS.md` (date, `main` hash, merged work, open items) and the README roadmap. Numbers in the README come from the docs pages, and a table measured on a busy GPU says so.
4. **Lint and docs build** (fine on any machine):

   ```bash
   uvx ruff check .
   uvx --with mkdocs-material --with "mkdocstrings[python]" --with mkdocs-jupyter mkdocs build --strict
   ```

5. **Tests on a GPU machine.** The CPU suite and the GPU equivalence tests must pass from a clean checkout:

   ```bash
   pip install -e ".[dev]"
   python -m pytest -m "not gpu"
   python -m pytest -m gpu
   ```

   The `isaac` tests (Isaac Lab installs, [Windows](docs/isaac-lab.md) or [Linux](docs/isaac-lab-linux.md)), the `mjx` tests and `tests/bridges/` (ns-3 or OAI) are run when their area changed.
6. **Build.**

   ```bash
   rm -rf dist build *.egg-info
   python -m build                                    # dist/isaac_net-X.Y.Z-py3-none-any.whl and .tar.gz
   python -m zipfile -l dist/isaac_net-X.Y.Z-py3-none-any.whl
   ```

   Check the wheel contents: the package, `core/data/` (Sionna tables, their Apache-2.0 license, the synthetic radio map), the ns-3 bridge sources and the OAI compose file. There must be no `*lena_eesm_tables*`, no `.so`, `.pt` or `.ckpt` file, no datasets and no results. The wheel is below 1 MB.
7. **Install from the wheel.** In a fresh venv on Linux, with the CPU build of torch, install the wheel and run the suite from outside the source tree, so the tests import the installed package:

   ```bash
   python -m venv /tmp/rel && . /tmp/rel/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install "dist/isaac_net-X.Y.Z-py3-none-any.whl[dev]"
   cp -r tests /tmp/rel-tests && cd /tmp/rel-tests      # plus a pytest.ini with pythonpath = tests and the markers
   python -m pytest -m "not gpu"
   isaac-net-bench --help && isaac-net-bake --help && isaac-net-measure --help
   ```

   `tests/test_package.py::test_prototype_shims_alias_the_package_modules` needs the source tree (`prototype/` is not packaged) and is expected to fail there.
8. **Tag.** Merge the release branch, then tag the merge commit and push the tag (only when the maintainer says so):

   ```bash
   git tag -a vX.Y.Z -m "isaac-net X.Y.Z"
   git push origin vX.Y.Z
   ```

9. **Artifacts.** Attach the wheel and sdist to a GitHub release for the tag, with the changelog section as its notes.
10. **After the release.** Bump `__version__` to the next development version (`X.Y.(Z+1).dev0`) on `main`.

## When the project is public

Publishing to PyPI starts only after the repository is public and the paper describing the engine is out. At that point: register the `isaac-net` name, publish to TestPyPI first and install from it in a fresh venv, then publish with a trusted-publisher GitHub Actions workflow triggered by the tag, and update the project URLs in `pyproject.toml` and the Citation section of the README.
