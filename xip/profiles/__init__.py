"""Profile discovery for XiP.

To add a new instrument profile:
  1. Create a new .py file in this ``profiles/`` directory.
  2. Subclass :class:`~xip.profiles.base.XiPProfile` and implement all
     abstract methods.
  3. Define a **module-level** ``PROFILE`` variable that holds an instance
     of your subclass::

       PROFILE = MyInstrumentProfile()

  4. Restart the GUI — your profile will appear in the drop-down automatically.

No other files need to be edited.
"""

import importlib
import pkgutil
from pathlib import Path

from xip.profiles.base import XiPProfile  # noqa: F401 — re-exported for convenience


def load_profiles() -> list[XiPProfile]:
    """Return all profiles discovered in this package.

    Profiles are sorted so that *eROSITA* comes first and *Default* last;
    any additional user profiles appear in alphabetical order in between.
    """
    profiles: list[XiPProfile] = []
    pkg_dir = Path(__file__).parent
    for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name == "base":
            continue
        try:
            mod = importlib.import_module(f".{module_name}", package=__package__)
            profile = getattr(mod, "PROFILE", None)
            if isinstance(profile, XiPProfile):
                profiles.append(profile)
        except Exception:
            pass

    # Stable ordering: eROSITA first, Default last, others alphabetically.
    def _sort_key(p: XiPProfile) -> tuple:
        if p.name == "eROSITA":
            return (0, "")
        if p.name == "Default":
            return (2, "")
        return (1, p.name.lower())

    profiles.sort(key=_sort_key)
    return profiles
