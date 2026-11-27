"""Tests for what the `pushhdc` package gives at its top level."""

import importlib
import pkgutil
from types import ModuleType

import pushhdc

# Tests


def test_no_top_level_name_hides_a_module() -> None:
    """each module of the package stays a module at the top level, so
    `import pushhdc.levels as levels` gives the module and its functions.

    fails if the package gives a function or class with the name of one of
    its modules, as `levels` once did.
    """
    for module in pkgutil.iter_modules(pushhdc.__path__):
        importlib.import_module(f"pushhdc.{module.name}")
        assert isinstance(getattr(pushhdc, module.name), ModuleType)
