"""Compatibility shim: provide importlib.resources.files() for Python 3.7.

Usage: import this module BEFORE importing anything from hcu_train_simulator.
"""
import os as _os
import importlib as _il
import importlib.resources as _ir

try:
    _ir.files  # noqa: F401
    _HAS_FILES = True
except AttributeError:
    _HAS_FILES = False

if not _HAS_FILES:
    class _CompatFiles(object):
        """Filesystem-backed stand-in for importlib.resources.files()."""

        def __init__(self, package, parts=()):
            self._package = package
            self._parts = tuple(parts)

        def joinpath(self, *more):
            return _CompatFiles(self._package, self._parts + tuple(more))

        @property
        def name(self):
            return _os.path.join(*self._parts) if self._parts else ""

        def _abs_path(self):
            pkg = _il.import_module(self._package)
            base = _os.path.dirname(getattr(pkg, "__file__", "") or "")
            return _os.path.join(base, *self._parts) if self._parts else base

        def read_text(self, encoding="utf-8"):
            with open(self._abs_path(), "r", encoding=encoding) as fh:
                return fh.read()

        def read_bytes(self):
            with open(self._abs_path(), "rb") as fh:
                return fh.read()

        def iterdir(self):
            for n in _os.listdir(self._abs_path()):
                yield type(self)(self._package, self._parts + (n,))

        def __truediv__(self, other):
            return self.joinpath(other)

        def __str__(self):
            return self._abs_path()

        def __fspath__(self):
            return self._abs_path()

    def _files(package):
        return _CompatFiles(package)

    _ir.files = _files

    if not hasattr(_ir, "abc"):
        class _Abc(object):
            class Traversable(object):
                pass
        _ir.abc = _Abc
    elif not hasattr(_ir.abc, "Traversable"):
        class _Traversable(object):
            pass
        _ir.abc.Traversable = _Traversable
