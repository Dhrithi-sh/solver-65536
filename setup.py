"""Optional C++ extension for the 2048 engine: ``engine._cengine``.

All project metadata lives in pyproject.toml; this file only declares the
compiled extension. Build it in place (so ``python -m pytest`` from the repo
root sees it) with:

    pip install "setuptools>=64" wheel "pybind11>=2.11"
    python setup.py build_ext --inplace

The extension is OPTIONAL by design:
  * optional=True  -> if the C++ compiler is missing or the compile fails,
    setuptools prints a warning and carries on without it;
  * no pybind11 installed -> no extension is declared at all.
In both cases the pure-Python engine is used. To make a build failure loud
(e.g. in CI), check ``python -c "from engine import game; print(game.backend())"``
afterwards: it must print ``cpp``.
"""

import sys

from setuptools import setup

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("WARNING: pybind11 is not installed; skipping the C++ extension "
          "(the pure-Python engine will be used).", file=sys.stderr)
else:
    ext_modules = [
        Pybind11Extension(
            "engine._cengine",
            sources=["engine/_core/binding.cpp", "engine/_core/core.cpp"],
            include_dirs=["engine/_core"],
            depends=["engine/_core/core.hpp", "engine/_core/pyglue.hpp"],
            cxx_std=17,
            # core.cpp throws std::invalid_argument; MSVC needs /EHsc for that.
            extra_compile_args=["/EHsc"] if sys.platform == "win32" else [],
            optional=True,
        )
    ]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
