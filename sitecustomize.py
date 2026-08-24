"""Small Tkinter safety and camera-switching hooks for FaceTrack desktop."""

from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk


_original_configure = tk.Misc.configure
_original_combobox_bind = ttk.Combobox.bind


def _safe_configure(self, cnf=None, **kw):
    try:
        return _original_configure(self, cnf, **kw)
    except tk.TclError as error:
        if "invalid command name" in str(error):
            return None
        raise


tk.Misc.configure = _safe_configure
tk.Misc.config = _safe_configure


def _bind(self, sequence=None, func=None, add=None):
    if (
        sequence == "<<ComboboxSelected>>"
        and callable(func)
        and getattr(func, "__name__", "") == "_on_camera_selected"
    ):
        callback = func

        def wrapped(event):
            result = callback(event)
            root = event.widget.winfo_toplevel()
            if getattr(root, "running", False):
                root.after(0, root.start)
            return result

        func = wrapped
    return _original_combobox_bind(self, sequence, func, add)


# Allows the camera dropdown to switch cameras while the scanner is running.
ttk.Combobox.bind = _bind
