"""Build the C++ search extension (agents.hybrid._hsearch).

From the repo root, on Windows, macOS or Linux:

    pip install pybind11 setuptools
    python agents/hybrid/cpp/setup.py build_ext

The compiled module lands next to this package (agents/hybrid/_hsearch*.so or
.pyd). If it is missing, HybridAgent falls back to the slow pure-Python
reference, so nothing breaks, it is just slow.

Requirements: a C++17 compiler (g++ / clang++, or "Build Tools for Visual
Studio" on Windows; on macOS run `xcode-select --install`).
"""

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Build in this folder (it has no pyproject.toml to confuse setuptools) and put
# the finished module in <repo>/agents/hybrid/.
os.chdir(HERE)
if "build_ext" in sys.argv and not any(a.startswith("--build-lib") or a == "--inplace" for a in sys.argv):
    sys.argv += ["--build-lib", str(ROOT)]

if sys.platform == "win32":
    compile_args = ["/O2"]
    link_args = []
else:
    # -ffp-contract=off: no fused multiply-add, so C++ and the Python reference
    # round identically (tests compare them to ~1e-9).
    compile_args = ["-O3", "-ffp-contract=off"]
    link_args = ["-pthread"]

ext = Pybind11Extension(
    "agents.hybrid._hsearch",
    ["hsearch.cpp"],
    cxx_std=17,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(name="hybrid-search", version="0.1", ext_modules=[ext], cmdclass={"build_ext": build_ext}, packages=[])
