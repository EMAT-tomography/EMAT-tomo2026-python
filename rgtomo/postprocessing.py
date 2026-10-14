"""Post-processing of reconstructions and intensity corrections.

- :func:`gaussian_filter_3d`, :func:`median_filter_3d`: ``imgaussfilt3`` /
  ``medfilt3`` with MATLAB defaults.
- :func:`align_z_pca`: port of ``align_z_pca`` (major axis along the tilt axis).
- :func:`rotational_slices`: port of ``rg_rotational_slices``.
- :func:`cupping_correction`, :func:`intensity_range`: the intensity
  linearisation of ``cupping_correction_gui`` (tomohawk), without the GUI.

Volumes are ``(n_slices, n, n)`` as returned by :func:`rgtomo.reconstruct`,
i.e. the first axis is the tilt axis.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ._matlab import graythresh, imgaussfilt, prctile, rescale
from .series import TiltSeries

__all__ = ["gaussian_filter_3d", "median_filter_3d", "align_z_pca", "rotational_slices",
           "cupping_correction", "intensity_range"]


def gaussian_filter_3d(vol, sigma):
    """3D Gaussian filter (``imgaussfilt3``: size 2*ceil(2*sigma)+1, replicate)."""
    return imgaussfilt(np.asarray(vol, dtype=np.float32), sigma)


def median_filter_3d(vol, size=(3, 3, 3)):
    """3D median filter (``medfilt3``, symmetric padding)."""
    return ndi.median_filter(np.asarray(vol, dtype=np.float32), size=size, mode="reflect")


# --------------------------------------------------------------------------- PCA alignment
def _pca_axes(points):
    """Principal axes (rows, decreasing variance) with MATLAB ``pca`` sign
    convention (largest absolute component of each axis positive)."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    for i in range(vt.shape[0]):
        if vt[i, np.argmax(np.abs(vt[i]))] < 0:
            vt[i] = -vt[i]
    return vt


def align_z_pca(rec, T=(1, 99), plot=False, center=True):
    """Align a particle with a clear orientation so that its major axis
    (first principal component of its surface) lies along the first axis of
    the volume (the tilt axis), port of ``align_z_pca``.

    Parameters
    ----------
    rec : ndarray (n_slices, n, n)
    T : (low, high)
        Percentiles for outlier removal of the surface points (e.g. to ignore
        other objects attached to the rod).
    plot : bool
        Scatter plot of the surface points and principal axes.
    center : bool
        Also move the intensity-weighted centre of mass of the largest object
        to the centre of the volume (the intent of the MATLAB code).
    """
    from skimage.measure import label, regionprops
    from skimage.transform import rescale as skrescale
    rec = np.asarray(rec, dtype=np.float32)
    ctr = (np.asarray(rec.shape) - 1) / 2

    # centre of mass of the largest object (strongly filtered volume)
    im = imgaussfilt(rescale(rec), 10)
    bw = im > graythresh(im)
    com = ctr.copy()
    if center and bw.any():
        props = regionprops(label(bw, connectivity=3), intensity_image=im)
        biggest = max(props, key=lambda p: p.area)
        com = np.asarray(biggest.centroid_weighted, dtype=float)

    # surface points of the binarized, downsampled volume
    small = skrescale(rescale(rec), 0.25, order=1, anti_aliasing=True, preserve_range=True)
    bw_s = small > graythresh(small)
    edges = bw_s & ~ndi.binary_erosion(bw_s)
    pts = np.argwhere(edges).astype(float)
    if pts.shape[0] < 3:
        raise ValueError("Not enough surface points for the PCA")
    lo = np.percentile(pts, T[0], axis=0)
    hi = np.percentile(pts, T[1], axis=0)
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    pcs = _pca_axes(pts)

    if plot:
        import matplotlib.pyplot as plt
        c = pts - pts.mean(axis=0)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(c[:, 2], c[:, 1], c[:, 0], s=1, alpha=0.3)
        L = np.abs(c).max()
        for v, col in zip(pcs, "rgb"):
            ax.quiver(0, 0, 0, *(v[::-1] * L), color=col)
        ax.set(xlabel="x", ylabel="z (beam)", zlabel="tilt axis")
        plt.show()

    # rows of R: new axis 0 <- PC1, axis 1 <- PC3, axis 2 <- PC2 (as in MATLAB)
    R = np.stack([pcs[0], pcs[2], pcs[1]])
    if np.linalg.det(R) < 0:
        R[1] = -R[1]
    # output o: (o - ctr) = R (i - com)  ->  i = R^T (o - ctr) + com
    offset = com - R.T @ ctr
    return ndi.affine_transform(rec, R.T, offset=offset, order=3, mode="constant",
                                cval=0.0).astype(np.float32)


# --------------------------------------------------------------------------- rotational slices
def rotational_slices(rec, spacing=1.0):
    """Central slices through the volume while rotating it about the first
    axis (the tilt/major axis), port of ``rg_rotational_slices``.

    Returns ``(n_slices, n_slices_vol, n)``: one image per rotation angle
    (0 to 360 deg, step ``spacing``), with the rotation axis vertical. Only the
    needed planes are interpolated (linear), equivalent to rotating the
    volume with ``scipy.ndimage.rotate(rec, angle, axes=(1, 2))`` and taking
    the central slice."""
    from tqdm.auto import tqdm
    rec = np.asarray(rec, dtype=np.float32)
    n0, n1, n2 = rec.shape
    c1 = int(np.floor(n1 / 2)) - 1                 # MATLAB floor(nx/2), 1-based
    ctr = np.array([(n1 - 1) / 2, (n2 - 1) / 2])
    thetas = np.arange(0, 360, spacing)
    out = np.zeros((thetas.size, n0, n2), dtype=np.float32)
    k = np.arange(n2)
    a0 = np.arange(n0)
    for i, th in enumerate(tqdm(thetas, desc="Rotational slices", leave=False)):
        c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        o1, o2 = c1 - ctr[0], k - ctr[1]
        p1 = c * o1 + s * o2 + ctr[0]
        p2 = -s * o1 + c * o2 + ctr[1]
        coords = np.stack([np.broadcast_to(a0[:, None], (n0, n2)),
                           np.broadcast_to(p1[None, :], (n0, n2)),
                           np.broadcast_to(p2[None, :], (n0, n2))])
        out[i] = ndi.map_coordinates(rec, coords, order=1, mode="constant", cval=0.0)
    return out


# --------------------------------------------------------------------------- cupping
def intensity_range(data, mode="full"):
    """Intensity range used by the cupping correction: ``'full'`` (min/max) or
    ``'robust'`` (1st percentile of the background, 99th of the Otsu object)."""
    I = np.asarray(data.data if isinstance(data, TiltSeries) else data)
    if mode == "full":
        return float(I.min()), float(I.max())
    if mode == "robust":
        mask = I > graythresh(I)
        return float(prctile(I[~mask], 1)), float(prctile(I[mask], 99))
    raise ValueError("mode must be 'full' or 'robust'")


def cupping_correction(series, strength, i_min=None, i_max=None):
    """Linearise HAADF-STEM intensities to correct cupping artefacts
    (``cupping_correction_gui``, Beer-Lambert type correction, see
    10.1016/j.ultramic.2012.03.005).

    ``strength`` is defined through ``I0 = (1 + 10**(2 - strength)) * I_max``
    (strength 0 gives I0/I_max ~ 100); the corrected intensities
    ``-log(1 - I/I0)`` are mapped back to [I_min, I_max]."""
    data = series.data if isinstance(series, TiltSeries) else series
    I = np.asarray(data, dtype=np.float64)
    if i_min is None or i_max is None:
        fmin, fmax = intensity_range(I, "full")
        i_min = fmin if i_min is None else i_min
        i_max = fmax if i_max is None else i_max
    I = np.clip(I, i_min, i_max)
    I0 = (1 + 10 ** (2 - strength)) * i_max
    with np.errstate(divide="ignore", invalid="ignore"):
        I = -np.log(1 - I / I0)
    out = rescale(I, i_min, i_max).astype(np.float32)
    if isinstance(series, TiltSeries):
        res = series.with_data(out)
        res.meta["cupping_correction"] = {"strength": strength, "i_min": i_min, "i_max": i_max}
        return res
    return out
