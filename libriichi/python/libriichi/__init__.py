"""
libriichi — Rust-backed mahjong engine (pyo3 extension).

pyo3 编译的子模块需要手动注册到 sys.modules，
否则 `from libriichi.arena import OneVsThree` 等形式的 import 会失败。
"""
import sys
from .libriichi import *  # noqa: F401, F403
from . import libriichi as _inner

_pkg = __name__
for _submod_name in ("arena", "consts", "dataset", "mjai", "stat", "state"):
    _key = f"{_pkg}.{_submod_name}"
    if _key not in sys.modules:
        sys.modules[_key] = getattr(_inner, _submod_name)

__doc__     = _inner.__doc__
__version__ = _inner.__version__
__profile__ = _inner.__profile__
if hasattr(_inner, "__all__"):
    __all__ = _inner.__all__
