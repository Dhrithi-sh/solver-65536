"""Sanity checks for the build configuration (pyproject.toml / setup.py)."""

import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_build_system_uses_setuptools_and_pybind11():
    bs = _pyproject()["build-system"]
    assert bs["build-backend"] == "setuptools.build_meta"
    requires = " ".join(bs["requires"]).lower()
    assert "setuptools" in requires and "pybind11" in requires


def test_pybind11_is_a_build_dependency_only():
    project = _pyproject()["project"]
    runtime = project["dependencies"] + [d for group in project.get("optional-dependencies", {}).values() for d in group]
    assert not any(d.lower().startswith("pybind11") for d in runtime)


def test_cpp_sources_are_not_packaged_as_python_packages():
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert "engine._core" in find["exclude"]


def test_setup_py_declares_the_optional_extension():
    with open(os.path.join(ROOT, "setup.py"), encoding="utf-8") as f:
        text = f.read()
    for needle in ("engine._cengine", "engine/_core/binding.cpp", "engine/_core/core.cpp", "optional=True"):
        assert needle in text, needle
    for rel in ("engine/_core/binding.cpp", "engine/_core/core.cpp", "engine/_core/core.hpp", "engine/_core/pyglue.hpp"):
        assert os.path.exists(os.path.join(ROOT, rel)), rel
