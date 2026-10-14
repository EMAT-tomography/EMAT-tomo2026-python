"""Pre-processing of tilt series.

Ports of ``rg_normalize_intensities``, ``rg_gaussfilter_slice``,
``rg_find_edges_stack``, the non-interactive parts of ``crop_stack``
(tomohawk), ``wh_background_subtract`` / ``wh_background_mask`` /
``wh_background_fit`` (wh-tomo-tools, W. Heyvaert) and ``remove_bkg_median``
(tomohawk).

All stacks are ``(n_angles, n_rows, n_cols)``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ._matlab import (graythresh, imgaussfilt, imgradient_sobel, iround,
                      rescale)
from .series import TiltSeries

__all__ = [
    "rescale", "normalize_intensities", "gaussfilter_slice", "find_edges_stack",
    "imresize_stack", "rot90_stack", "auto_crop_rect", "apply_roi", "crop_stack",
    "background_subtract", "background_mask", "background_fit", "remove_bkg_median",
    "graythresh",
]


def _data(obj):
    return obj.data if isinstance(obj, TiltSeries) else np.asarray(obj)


def _like(obj, data):
    """Return ``data`` wrapped like ``obj`` (TiltSeries in -> TiltSeries out)."""
    return obj.with_data(data) if isinstance(obj, TiltSeries) else data


def normalize_intensities(stack):
    """Normalize each projection to mean 1 and standard deviation 1, then
    rescale the whole stack to [0, 1] (``rg_normalize_intensities``).

    Note: assumes the total intensity of each projection should be constant,
    which is not always correct."""
    s = np.array(_data(stack), dtype=np.float32, copy=True)
    for i in range(s.shape[0]):
        im = s[i].astype(np.float64)
        s[i] = (im - im.mean()) / im.std(ddof=1) + 1.0
    return _like(stack, rescale(s))


def gaussfilter_slice(stack, sigma):
    """Gaussian filter of each projection (MATLAB ``imgaussfilt`` defaults);
    ``sigma == 0`` returns the input unchanged (``rg_gaussfilter_slice``)."""
    if sigma == 0:
        return stack
    s = _data(stack).astype(np.float32, copy=False)
    return _like(stack, imgaussfilt(s, sigma, axes=(1, 2)))


def find_edges_stack(stack, sigma=4.0, method="sobel"):
    """Gradient magnitude of the Gaussian-filtered projections
    (``rg_find_edges_stack``)."""
    s = _data(stack).astype(np.float32, copy=False)
    out = np.empty_like(s)
    for i in range(s.shape[0]):
        im = imgaussfilt(s[i], sigma) if sigma > 0 else s[i]
        if method == "sobel":
            out[i] = imgradient_sobel(im)
        elif method == "prewitt":
            gx = ndi.prewitt(im, axis=-1, mode="nearest")
            gy = ndi.prewitt(im, axis=-2, mode="nearest")
            out[i] = np.hypot(gx, gy)
        elif method == "central":
            gy, gx = np.gradient(im)
            out[i] = np.hypot(gx, gy)
        else:
            raise ValueError(f"Unknown gradient method '{method}'")
    return _like(stack, out)


def imresize_stack(stack, scale, order=3):
    """Resize each projection like MATLAB ``imresize(stack, scale)``: output size
    ``ceil(scale * size)``, bicubic interpolation, anti-aliasing when shrinking."""
    from skimage.transform import resize
    s = _data(stack)
    n, ny, nx = s.shape
    out_shape = (n, int(np.ceil(ny * scale)), int(np.ceil(nx * scale)))
    out = resize(s.astype(np.float32, copy=False), out_shape, order=order,
                 anti_aliasing=scale < 1, preserve_range=True, mode="edge").astype(np.float32)
    if isinstance(stack, TiltSeries):
        res = stack.with_data(out)
        res.pixel_size = stack.pixel_size / scale
        return res
    return out


def rot90_stack(stack, k=1):
    """Rotate every projection by k x 90 degrees counter-clockwise (MATLAB ``rot90``)."""
    s = _data(stack)
    return _like(stack, np.ascontiguousarray(np.rot90(s, k, axes=(1, 2))))


# --------------------------------------------------------------------------- cropping
def auto_crop_rect(stack):
    """Initial ROI proposed by ``crop_stack`` (``get_rect_position``): bounding box
    of the largest Otsu object in the average projection, enlarged by 40 % of its
    width on each side and 15 % of its height top and bottom.

    Returns ``(x0, y0, width, height)`` as integers in 0-based pixel indices,
    clipped to the image."""
    from skimage.measure import label, regionprops
    s = _data(stack)
    ny, nx = s.shape[-2:]
    im = rescale(s.mean(axis=0) if s.ndim == 3 else s)
    bw = im > graythresh(im)
    props = regionprops(label(bw, connectivity=2))
    if not props:
        rect = [nx / 5, ny / 5, 3 * nx / 5, 3 * ny / 5]
        x0, y0, w, h = (iround(v) for v in rect)
        return _clip_rect((x0 - 1, y0 - 1, w, h), ny, nx)
    p = max(props, key=lambda r: r.area)
    r0, c0, r1, c1 = p.bbox
    # MATLAB BoundingBox: [ul_x, ul_y, w, h] with 1-based pixel-edge corner
    rect = [c0 + 0.5, r0 + 0.5, c1 - c0, r1 - r0]
    h_add = rect[3] * 0.15
    w_add = rect[2] * 0.4
    rect = [rect[0] - w_add, rect[1] - h_add, rect[2] + 2 * w_add, rect[3] + 2 * h_add]
    if rect[0] < 0 or rect[1] < 0:
        rect = [nx / 5, ny / 5, 3 * nx / 5, 3 * ny / 5]
    # apply_roi: MATLAB 1-based start round(rect(1)) -> 0-based start round(rect(1)) - 1
    x0, y0, w, h = iround(rect[0]) - 1, iround(rect[1]) - 1, iround(rect[2]), iround(rect[3])
    return _clip_rect((x0, y0, w, h), ny, nx)


def _clip_rect(rect, ny, nx):
    x0, y0, w, h = (int(v) for v in rect)
    x0 = min(max(x0, 0), nx - 1)
    y0 = min(max(y0, 0), ny - 1)
    w = max(1, min(w, nx - x0))
    h = max(1, min(h, ny - y0))
    return (x0, y0, w, h)


def apply_roi(stack, rect):
    """Crop all projections to ``rect = (x0, y0, width, height)`` (0-based)."""
    s = _data(stack)
    ny, nx = s.shape[-2:]
    x0, y0, w, h = _clip_rect(rect, ny, nx)
    return _like(stack, np.ascontiguousarray(s[..., y0:y0 + h, x0:x0 + w]))


def crop_stack(stack, rect=None):
    """Crop a stack with a rectangular ROI (non-interactive ``crop_stack``).

    ``rect = (x0, y0, width, height)`` in 0-based pixels; if omitted the
    automatic ROI of :func:`auto_crop_rect` is used. For the interactive version
    see :func:`rgtomo.crop_stack_gui`."""
    if rect is None:
        rect = auto_crop_rect(stack)
    return apply_roi(stack, rect)


# --------------------------------------------------------------------------- background
def background_mask(im, threshold=0.5, smooth_sigma=10.0, smooth_threshold=0.1):
    """Background mask (``wh_background_mask``): pixels far from anything brighter
    than ``threshold * Otsu``."""
    t = threshold * graythresh(im)
    return imgaussfilt((im > t).astype(np.float64), smooth_sigma) < smooth_threshold


def background_fit(im, mask):
    """Least-squares plane ``a*x + b*y + c`` fitted on the masked pixels
    (``wh_background_fit``)."""
    ny, nx = im.shape
    x, y = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1))
    if mask.sum() < 3:
        return np.zeros_like(im, dtype=np.float64)
    A = np.column_stack([x[mask], y[mask], np.ones(mask.sum())])
    abc, *_ = np.linalg.lstsq(A, im[mask].astype(np.float64), rcond=None)
    return abc[0] * x + abc[1] * y + abc[2]


def background_subtract(stack, threshold=0.5, smooth_sigma=10.0, smooth_threshold=0.1):
    """Fit and subtract a planar background for every projection, then set the
    background region to zero (``wh_background_subtract``).

    ``threshold`` is the factor of the Otsu threshold separating object and
    background: higher removes more."""
    from tqdm.auto import tqdm
    s = np.array(_data(stack), dtype=np.float32, copy=True)
    for i in tqdm(range(s.shape[0]), desc="Removing background", leave=False):
        mask = background_mask(s[i], threshold, smooth_sigma, smooth_threshold)
        bgfit = background_fit(s[i], mask)
        s[i] = (~mask) * (s[i] - bgfit)
    return _like(stack, s)


def remove_bkg_median(stack):
    """Subtract the median intensity of each projection and clip negative values
    (``remove_bkg_median``)."""
    s = _data(stack).astype(np.float32, copy=False)
    out = s - np.median(s, axis=(-2, -1), keepdims=True)
    out[out < 0] = 0
    return _like(stack, out)
