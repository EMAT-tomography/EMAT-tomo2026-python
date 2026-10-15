"""Interactive viewers based on stackview (replacing MATLAB ``sliceViewer`` and
``orthosliceViewer``).

The data is optionally downscaled for display (like the ``imresize(..., 0.25)``
calls of ``tomography.m``) and shown with a common intensity range for all
slices, as in MATLAB.
"""

from __future__ import annotations

import numpy as np

from .series import TiltSeries

__all__ = ["slice_viewer", "orthoslice_viewer", "side_by_side_viewer", "curtain_viewer",
           "downscale"]


def _arr(x):
    return np.asarray(x.data if isinstance(x, TiltSeries) else x)


def downscale(stack, scale, volume=False):
    """Resize for display: the last two axes of a stack (``volume=False``,
    like MATLAB ``imresize``) or all three axes (``volume=True``, like
    ``imresize3``), with linear interpolation and anti-aliasing."""
    from skimage.transform import resize
    a = np.asarray(stack, dtype=np.float32)
    if scale is None or scale == 1:
        return a
    if volume:
        shape = tuple(max(1, int(np.ceil(s * scale))) for s in a.shape)
    else:
        shape = a.shape[:-2] + tuple(max(1, int(np.ceil(s * scale))) for s in a.shape[-2:])
    return resize(a, shape, order=1, anti_aliasing=scale < 1, preserve_range=True).astype(np.float32)


def _display_params(a, display_size, zoom_factor, display_min, display_max, percentiles):
    if zoom_factor is None:
        zoom_factor = 1.0
        if display_size:
            zoom_factor = max(1.0, display_size / max(a.shape[-2:]))
    finite = a[np.isfinite(a)]
    if display_min is None:
        display_min = float(np.percentile(finite, percentiles[0])) if percentiles else float(finite.min())
    if display_max is None:
        display_max = float(np.percentile(finite, percentiles[1])) if percentiles else float(finite.max())
    if display_max <= display_min:
        display_max = display_min + 1
    return zoom_factor, display_min, display_max


def slice_viewer(stack, scale=None, display_size=512, zoom_factor=None, colormap=None,
                 display_min=None, display_max=None, percentiles=None, axis=0):
    """Browse a stack slice by slice (``sliceViewer``).

    Parameters
    ----------
    stack : ndarray or TiltSeries
        ``(n, rows, cols)``; slices are taken along ``axis``.
    scale : float, optional
        Downscale the images before display (e.g. 0.25 or 0.5 as in
        ``tomography.m``). By default large images are reduced to
        ``display_size``.
    display_size : int
        Target size (px) of the displayed image; small images are zoomed.
    percentiles : (low, high), optional
        Use percentiles of the data for the display range instead of min/max.
    """
    import stackview
    a = _arr(stack).astype(np.float32, copy=False)
    if axis != 0:
        a = np.moveaxis(a, axis, 0)
    if a.ndim == 2:
        a = a[None]
    if scale is None and display_size and max(a.shape[-2:]) > display_size:
        scale = display_size / max(a.shape[-2:])
    a = downscale(a, scale)
    zf, vmin, vmax = _display_params(a, display_size, zoom_factor, display_min, display_max, percentiles)
    return stackview.slice(a, continuous_update=True, zoom_factor=zf, colormap=colormap,
                           display_min=vmin, display_max=vmax, slider_text="Slice")


def orthoslice_viewer(volume, scale=None, display_size=100, zoom_factor=None, colormap=None,
                      display_min=None, display_max=None, percentiles=None):
    """Three orthogonal slices through a volume (``orthosliceViewer``).
    ``scale`` downsizes all three axes (``imresize3``)."""
    import stackview
    a = _arr(volume).astype(np.float32, copy=False)
    if scale is None and display_size and max(a.shape) > display_size:
        scale = display_size / max(a.shape)
    a = downscale(a, scale, volume=True)
    zf, vmin, vmax = _display_params(a, display_size, zoom_factor, display_min, display_max, percentiles)
    return stackview.orthogonal(a, continuous_update=True, zoom_factor=zf, colormap=colormap,
                                display_min=vmin, display_max=vmax)


def side_by_side_viewer(stack1, stack2, scale=None, display_size=384, zoom_factor=None,
                        colormap=None, display_min=None, display_max=None):
    """Two stacks of the same shape next to each other, with a common slider."""
    import stackview
    a = _arr(stack1).astype(np.float32, copy=False)
    b = _arr(stack2).astype(np.float32, copy=False)
    if scale is None and display_size and max(a.shape[-2:]) > display_size:
        scale = display_size / max(a.shape[-2:])
    a, b = downscale(a, scale), downscale(b, scale)
    zf, vmin, vmax = _display_params(a, display_size, zoom_factor, display_min, display_max, None)
    if display_min is None:
        vmin = min(vmin, float(np.nanmin(b)))
    if display_max is None:
        vmax = max(vmax, float(np.nanmax(b)))
    cmap = colormap or "Greys_r"
    return stackview.side_by_side(a, b, continuous_update=True, zoom_factor=zf,
                                  colormap=[cmap, cmap], display_min=vmin, display_max=vmax)


def curtain_viewer(stack1, stack2, scale=None, display_size=512, zoom_factor=None):
    """Compare two stacks with a draggable curtain (stackview.curtain)."""
    import stackview
    a = _arr(stack1).astype(np.float32, copy=False)
    b = _arr(stack2).astype(np.float32, copy=False)
    if scale is None and display_size and max(a.shape[-2:]) > display_size:
        scale = display_size / max(a.shape[-2:])
    a, b = downscale(a, scale), downscale(b, scale)
    zf, vmin, vmax = _display_params(a, display_size, zoom_factor, None, None, None)
    _, vmin2, vmax2 = _display_params(b, display_size, zoom_factor, None, None, None)
    return stackview.curtain(a, b, continuous_update=True, zoom_factor=zf, display_min=vmin,
                             display_max=vmax, curtain_display_min=vmin2, curtain_display_max=vmax2)
