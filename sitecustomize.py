"""Small Tkinter safety hook for FaceTrack desktop shutdown/rebuild flows.

Tkinter widgets can be destroyed when switching back to the access-code page.
A later shutdown callback may still hold a reference to one of those widgets.
Ignore only the TclError produced when that already-destroyed widget is touched.
"""

from __future__ import annotations

import tkinter as tk


_original_configure = tk.Misc.configure


def _safe_configure(self, cnf=None, **kw):
    try:
        return _original_configure(self, cnf, **kw)
    except tk.TclError as error:
        message = str(error)
        if "invalid command name" in message:
            return None
        raise


tk.Misc.configure = _safe_configure
# Tkinter's config() is an alias in normal builds, but assign it explicitly so
# both forms use the same teardown-safe behavior.
tk.Misc.config = _safe_configure
