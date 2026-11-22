"""numba import with a pure-Python fallback.

If numba is installed, ``njit`` compiles functions to machine code. If it is
not (e.g. it failed to install on someone's machine), ``njit`` is a no-op and
the same functions run as ordinary Python: much slower, but identical results.
"""

from __future__ import annotations

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op stand-in for numba.njit; supports @njit and @njit(...)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


__all__ = ["njit", "HAVE_NUMBA"]
