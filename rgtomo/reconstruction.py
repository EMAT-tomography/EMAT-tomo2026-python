"""Tomographic reconstruction with the ASTRA toolbox.

Ports of ``reconstruct``, ``fp``, ``bp`` and ``rec_em_cpu`` (tomohawk).

Geometry: parallel beam, single tilt axis along the image rows. Each detector
row (image row) is an independent 2D problem, reconstructed slice by slice
with ASTRA's 2D algorithms exactly like the MATLAB version (``EM_CUDA``,
``SIRT_CUDA``, ``FBP_CUDA``; CPU fall-backs without a GPU).

Volume layout: ``(n_rows, n_cols, n_cols)`` = (tilt axis, beam direction at
0 degrees, x), i.e. the reconstructed slices are stacked along the first axis.
When written with :func:`rgtomo.write_rec` this gives the same file layout as
the MATLAB toolbox.

Note: ASTRA's 2D ``parallel`` and 3D ``parallel3d`` geometries have opposite
y-axis conventions; the 3D projectors used here for speed are wrapped so that
they are identical to the 2D ones.
"""

from __future__ import annotations

import numpy as np

from .csaxs import circular_mask
from .series import TiltSeries

__all__ = ["reconstruct", "fp", "bp", "has_cuda"]

_HAS_CUDA = None


def has_cuda() -> bool:
    """True if ASTRA can use a CUDA GPU."""
    global _HAS_CUDA
    if _HAS_CUDA is None:
        import astra
        try:
            _HAS_CUDA = bool(astra.use_cuda())
        except Exception:
            _HAS_CUDA = False
    return _HAS_CUDA


def _progress(n, desc, enabled):
    if not enabled:
        return range(n)
    from tqdm.auto import tqdm
    return tqdm(range(n), desc=desc, leave=False)


def reconstruct(tilt_series, algorithm="em", n_iterations=None, apodization=True,
                pad_size=5, print_time=True, use_cuda=None):
    """Reconstruct a volume from a tilt series (port of ``reconstruct.m``).

    Parameters
    ----------
    tilt_series : TiltSeries
        Projections ``(n_angles, n_rows, n_cols)`` and angles (degrees). A
        single sinogram ``(n_angles, n_cols)`` is also accepted.
    algorithm : {'em', 'sirt', 'fbp'}
        Expectation maximization, SIRT or filtered back-projection.
    n_iterations : int, optional
        Default 15 for EM and 150 for SIRT; ignored for FBP.
    apodization : bool
        Pad the sinogram edges (``pad_size`` replicated columns on each side)
        and multiply the result by a circular mask to avoid bright rings.
    pad_size : int
        Padding used with ``apodization``.
    print_time : bool
        Show a progress bar.

    Returns
    -------
    ndarray, shape (n_rows, n_cols, n_cols) (or (n_cols, n_cols) for a single sinogram)
    """
    import astra

    if isinstance(tilt_series, TiltSeries):
        data, angles = tilt_series.data, tilt_series.angles
    else:
        raise TypeError("Insufficient data. The input must provide a series of images "
                        "and the corresponding tilt angles (TiltSeries)")
    if data is None or angles is None:
        raise ValueError("The tilt series needs both data and angles")
    data = np.asarray(data, dtype=np.float32)
    single = data.ndim == 2
    if single:
        data = data[:, None, :]
    n_angles, n_rows, n_columns = data.shape
    if n_angles != len(angles):
        raise ValueError("Number of specified angles is different from number of projections in stack.")

    algorithm = algorithm.lower()
    if use_cuda is None:
        use_cuda = has_cuda()

    mask = None
    if apodization:
        mask = circular_mask(n_columns, n_columns / 50)
        data = np.pad(data, [(0, 0), (0, 0), (pad_size, pad_size)], mode="edge")
        n_columns = n_columns + 2 * pad_size

    if algorithm == "em":
        if n_iterations is None:
            n_iterations = 15
        if not use_cuda:
            rec = _rec_em_cpu(data, angles, n_iterations, print_time)
            return _finish(rec, apodization, pad_size, mask, single)
        alg_name, initial_value = "EM_CUDA", 1.0
    elif algorithm == "sirt":
        if n_iterations is None:
            n_iterations = 150
        alg_name, initial_value = ("SIRT_CUDA" if use_cuda else "SIRT"), 0.0
    elif algorithm == "fbp":
        if n_iterations not in (None, 1):
            import warnings
            warnings.warn("More then 1 iteration was requested, whereas FBP algorithm is not "
                          "iterative. Resetting the number of iterations to 1.")
        n_iterations = 1
        alg_name, initial_value = ("FBP_CUDA" if use_cuda else "FBP"), 0.0
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}' (use 'em', 'sirt' or 'fbp')")

    ang = np.deg2rad(np.asarray(angles, dtype=float))
    proj_geom = astra.create_proj_geom("parallel", 1.0, n_columns, ang)
    vol_geom = astra.create_vol_geom(n_columns, n_columns)
    sino = np.ascontiguousarray(np.transpose(data, (1, 0, 2)))   # (n_rows, n_angles, n_det)
    rec = np.zeros((n_rows, n_columns, n_columns), dtype=np.float32)

    sino_id = astra.data2d.create("-sino", proj_geom, sino[0])
    rec_id = astra.data2d.create("-vol", vol_geom, initial_value)
    cfg = astra.astra_dict(alg_name)
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    proj_id = None
    if not use_cuda:
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)
        cfg["ProjectorId"] = proj_id
    alg_id = astra.algorithm.create(cfg)
    try:
        if print_time:
            print(f"Performing reconstruction using {algorithm.upper()} algorithm...")
        for sl in _progress(n_rows, f"{algorithm.upper()} reconstruction", print_time and n_rows > 1):
            astra.data2d.store(sino_id, sino[sl])
            astra.data2d.store(rec_id, initial_value)
            astra.algorithm.run(alg_id, n_iterations)
            rec[sl] = astra.data2d.get(rec_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([sino_id, rec_id])
        if proj_id is not None:
            astra.projector.delete(proj_id)
    return _finish(rec, apodization, pad_size, mask, single)


def _finish(rec, apodization, pad_size, mask, single):
    if apodization:
        rec = rec[:, pad_size:-pad_size, pad_size:-pad_size] * mask
    return rec[0] if single else rec


# --------------------------------------------------------------------------- projectors
def fp(volume, angles):
    """Forward projection (port of ``fp.m``).

    ``volume`` is ``(n_slices, n, n)`` (the output of :func:`reconstruct`) or a
    single ``(n, n)`` slice. Returns projections in tilt-series layout
    ``(n_angles, n_slices, n)`` (or ``(n_angles, n)`` for a slice), identical
    to ASTRA's 2D slice-by-slice projector."""
    import astra
    vol = np.asarray(volume, dtype=np.float32)
    single = vol.ndim == 2
    if single:
        vol = vol[None]
    n_slices, n_r, n_c = vol.shape
    ang = np.deg2rad(np.asarray(angles, dtype=float))
    if has_cuda():
        # 3D projector on the y-flipped volume == 2D 'parallel' projector per slice
        proj_geom = astra.create_proj_geom("parallel3d", 1.0, 1.0, n_slices, max(n_r, n_c), ang)
        vol_geom = astra.create_vol_geom(n_r, n_c, n_slices)
        sid, sino = astra.create_sino3d_gpu(np.ascontiguousarray(vol[:, ::-1, :]), proj_geom, vol_geom)
        astra.data3d.delete(sid)
        out = np.transpose(sino, (1, 0, 2))             # (n_angles, n_slices, n_det)
    else:
        proj_geom = astra.create_proj_geom("parallel", 1.0, n_r, ang)
        vol_geom = astra.create_vol_geom(n_r, n_c)
        pid = astra.create_projector("linear", proj_geom, vol_geom)
        out = np.empty((len(ang), n_slices, n_r), dtype=np.float32)
        try:
            for s in range(n_slices):
                sid, sino = astra.create_sino(vol[s], pid)
                astra.data2d.delete(sid)
                out[:, s, :] = sino
        finally:
            astra.projector.delete(pid)
    out = np.ascontiguousarray(out, dtype=np.float32)
    return out[:, 0, :] if single else out


def bp(projections, angles):
    """Back-projection (port of ``bp.m``), adjoint layout of :func:`fp`:
    ``(n_angles, n_slices, n)`` -> ``(n_slices, n, n)``."""
    import astra
    p = np.asarray(projections, dtype=np.float32)
    single = p.ndim == 2
    if single:
        p = p[:, None, :]
    n_angles, n_slices, n_det = p.shape
    ang = np.deg2rad(np.asarray(angles, dtype=float))
    if has_cuda():
        proj_geom = astra.create_proj_geom("parallel3d", 1.0, 1.0, n_slices, n_det, ang)
        vol_geom = astra.create_vol_geom(n_det, n_det, n_slices)
        bid, vol = astra.create_backprojection3d_gpu(
            np.ascontiguousarray(np.transpose(p, (1, 0, 2))), proj_geom, vol_geom)
        astra.data3d.delete(bid)
        vol = np.ascontiguousarray(vol[:, ::-1, :])
    else:
        proj_geom = astra.create_proj_geom("parallel", 1.0, n_det, ang)
        vol_geom = astra.create_vol_geom(n_det, n_det)
        pid = astra.create_projector("linear", proj_geom, vol_geom)
        vol = np.empty((n_slices, n_det, n_det), dtype=np.float32)
        try:
            for s in range(n_slices):
                bid, v = astra.create_backprojection(np.ascontiguousarray(p[:, s, :]), pid)
                astra.data2d.delete(bid)
                vol[s] = v
        finally:
            astra.projector.delete(pid)
    return vol[0] if single else vol


def _rec_em_cpu(data, angles, n_iterations, print_time):
    """EM reconstruction without CUDA (port of ``rec_em_cpu.m``)."""
    data = np.asarray(data, dtype=np.float32)
    n_angles, n_rows, n_cols = data.shape
    rec = np.ones((n_rows, n_cols, n_cols), dtype=np.float32)
    R = bp(np.ones_like(data), angles)
    R[R < 1e-6] = np.inf
    R = 1.0 / R
    for _ in _progress(n_iterations, "EM (CPU)", print_time):
        with np.errstate(divide="ignore", invalid="ignore"):
            Wp = data / fp(rec, angles)
        Wp[~np.isfinite(Wp) | (Wp < 1e-6)] = 0
        rec = rec * R * bp(Wp, angles)
    return rec
