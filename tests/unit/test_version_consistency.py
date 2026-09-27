# Copyright (c) 2026 ysnrfd
"""Version single-sourcing enforcement (QUALITY_AUDIT Q-018).

The release version is declared in five places that must agree:

* ``pyproject.toml`` -> ``project.version`` (packaging source of truth, a
  plain literal so ``tomllib`` can read it without building the project),
* ``jevx/__init__.py`` -> ``__version__``,
* ``jevx/modeling_jevx.py`` -> ``__version__``,
* ``JevXConfig().version`` -> checkpoint-format version metadata,
* ``importlib.metadata.version("jevx")`` -> the installed distribution.

Any drift fails loudly here so a version bump has to be applied everywhere
at once. This test is the enforcement deliverable; the three in-package
declarations are owned elsewhere and are asserted, never edited, from here.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import jevx
import jevx.modeling_jevx
from jevx import JevXConfig

# The release version under test. When cutting a release, update this together
# with every source of truth listed in the module docstring.
EXPECTED_VERSION = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    """Read ``project.version`` from pyproject.toml with stdlib tomllib."""
    with PYPROJECT_TOML.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


def test_pyproject_exists_and_declares_plain_literal_version():
    assert PYPROJECT_TOML.is_file(), f"missing packaging metadata: {PYPROJECT_TOML}"
    text = PYPROJECT_TOML.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, (
        "pyproject.toml must declare project.version as a plain string literal "
        '(version = "0.1.0"), not a dynamic/non-quoted value'
    )


def test_pyproject_version_matches_expected():
    assert _pyproject_version() == EXPECTED_VERSION


def test_package_version_triple_matches_expected():
    """jevx.__version__, modeling_jevx.__version__ and JevXConfig().version."""
    init_version = jevx.__version__
    modeling_version = jevx.modeling_jevx.__version__
    config_version = JevXConfig().version

    assert init_version == EXPECTED_VERSION, (
        f"jevx.__version__ drifted: {init_version!r} != {EXPECTED_VERSION!r}"
    )
    assert modeling_version == EXPECTED_VERSION, (
        f"jevx.modeling_jevx.__version__ drifted: {modeling_version!r} != {EXPECTED_VERSION!r}"
    )
    assert config_version == EXPECTED_VERSION, (
        f"JevXConfig().version drifted: {config_version!r} != {EXPECTED_VERSION!r}"
    )
    # Chained equality so the failure message shows the actual three-way drift.
    assert init_version == modeling_version == config_version


def test_all_sources_agree():
    """The four in-repo declarations plus pyproject.toml are single-sourced."""
    pyproject_version = _pyproject_version()
    assert (
        pyproject_version
        == jevx.__version__
        == jevx.modeling_jevx.__version__
        == JevXConfig().version
        == EXPECTED_VERSION
    ), (
        "version drift: pyproject.toml={!r} jevx.__version__={!r} "
        "modeling_jevx.__version__={!r} JevXConfig().version={!r}".format(
            pyproject_version,
            jevx.__version__,
            jevx.modeling_jevx.__version__,
            JevXConfig().version,
        )
    )


def test_installed_distribution_version_matches():
    """Installed metadata (editable install) must agree with the package."""
    try:
        dist_version = metadata.version("jevx")
    except metadata.PackageNotFoundError:
        pytest.skip(
            "jevx not pip-installed (editable install not applied): "
            "run `pip install -e .` to enable this check"
        )
    assert dist_version == jevx.__version__, (
        f"importlib.metadata.version('jevx')={dist_version!r} != "
        f"jevx.__version__={jevx.__version__!r}"
    )
    assert dist_version == EXPECTED_VERSION
