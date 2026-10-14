"""Plotting helpers (live-updating figures in notebooks without ipympl)."""

from __future__ import annotations

import io


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


class LiveFigure:
    """Display a matplotlib figure once and update it in place (PNG), so long
    loops can show their progress like MATLAB figures that refresh."""

    def __init__(self):
        self._handle = None

    def update(self, fig):
        import matplotlib.pyplot as plt
        if not in_notebook():
            plt.close(fig)
            return
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        img = Image(data=buf.getvalue(), format="png")
        if self._handle is None:
            self._handle = display(img, display_id=True)
        else:
            self._handle.update(img)
