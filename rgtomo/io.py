"""Reading and writing tilt series and reconstructions.

Ports of ``rg_uniloader``, ``wh_read_fasttomo``, ``read_emi``/``read_ser``/
``emi2stack`` (tomohawk), ``read_emd``/``emd2stack``, ``ilm_tomo_dir``,
``write_rec``/``write_tiff`` (tomohawk) and ``rg_get_series_names``.

Orientation note: the MATLAB TIA reader returns every image transposed with
respect to the way TIA/rosettasciio display it, and the Velox reader returns
single images transposed but image series not. These quirks are reproduced so
that the "rotate by 90 degrees for the Titans, 0 for the Osiris" convention of
``tomography.m`` still holds.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from ._matlab import im2double, rescale
from .series import TiltSeries

__all__ = [
    "uniloader", "read_tif_series", "tomo_dir", "load_angles", "read_fasttomo",
    "read_emi", "emi2stack", "read_emd", "emd2stack", "read_mat_series",
    "write_rec", "read_rec", "write_tiff", "write_angles_csv", "get_series_names",
]


# =========================================================================== loader
def uniloader(fname, fangles=None, ftype="tif") -> TiltSeries:
    """Load a tilt series (``rg_uniloader``).

    Parameters
    ----------
    fname : str
        Path to the tilt-series file or folder.
    fangles : str or None
        Optional angle file. If omitted the naming rules below are assumed.
    ftype : {'fast', 'manual', 'tif', 'mat'}
        - ``'fast'``: one file ``TomoX.emi`` with a ``TomoX.log`` (or an angle
          file given in ``fangles``), or a multi-frame Velox ``.emd``.
        - ``'manual'``: one folder with one ``.emi``/``.ser`` or ``.emd`` per angle.
        - ``'tif'``: one multi-page ``.tif`` (+ optional angle file), or a folder
          of ``.tif`` images whose names encode the angles.
        - ``'mat'``: MATLAB file containing a struct with ``data`` and ``angles``
          (e.g. the FastAutomatedTomography datasets).
    """
    fname = str(fname)
    ftype = str(ftype)
    series = TiltSeries()
    if ftype == "fast":
        if fangles is None and fname.lower().endswith(".emi") and os.path.isfile(fname[:-4] + ".log"):
            series.data, series.angles, series.pixel_size = read_fasttomo(fname, fname[:-4] + ".log")
        elif fangles is None and fname.lower().endswith(".emd"):
            data, series.pixel_size, series.angles = read_emd(fname)
            series.data = data
        elif fangles is not None:
            series.data, series.angles, series.pixel_size = read_fasttomo(fname, fangles)
        else:
            raise FileNotFoundError("No correct file found for the angles")
    elif ftype == "manual":
        has_emi = bool(glob.glob(os.path.join(fname, "*.emi")))
        has_emd = bool(glob.glob(os.path.join(fname, "*.emd")))
        if has_emi and has_emd:
            raise ValueError("The folder contains both .emi and .emd files. "
                             "Make sure only one type is present.")
        if has_emi:
            series = emi2stack(fname)
        elif has_emd:
            series = emd2stack(fname)
        else:
            raise FileNotFoundError(f"No .emi or .emd files found in {fname}")
    elif ftype == "tif":
        series = read_tif_series(fname, fangles)
    elif ftype == "mat":
        series = read_mat_series(fname)
        if fangles is not None:
            series.angles = load_angles(fangles)
    else:
        raise ValueError(f"Unknown ftype '{ftype}' (use 'fast', 'manual', 'tif' or 'mat')")
    series.data = np.ascontiguousarray(series.data, dtype=np.float32)
    print(f"Loaded {series.data.shape[0]} projections with {series.pixel_size:.3f} nm pixel size")
    return series


def load_angles(fangles) -> np.ndarray:
    """Read a text/CSV file of angles (any of comma, semicolon, tab or space
    separated), like MATLAB ``load``."""
    with open(fangles, "r") as f:
        txt = f.read()
    vals = [float(v) for v in re.split(r"[,\s;]+", txt.strip()) if v]
    return np.asarray(vals, dtype=float)


# =========================================================================== TIF
def read_tif_series(fname, fangles=None) -> TiltSeries:
    """Multi-page TIF (``'tif'`` mode of ``rg_uniloader``) or folder of TIFs."""
    import tifffile
    series = TiltSeries(pixel_size=1.0)
    if ".tif" in fname.lower() and os.path.isfile(fname):
        data = tifffile.imread(fname)
        if data.ndim == 2:
            data = data[None]
        series.data = rescale(data.astype(np.float32))
        series.angles = load_angles(fangles) if fangles is not None else None
    elif os.path.isdir(fname):
        files, angles = tomo_dir(fname, "*.tif*")
        series.data = rescale(np.stack([tifffile.imread(f).astype(np.float32) for f in files]))
        series.angles = angles
    else:
        raise FileNotFoundError(fname)
    return series


def _str2double(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def tomo_dir(path_dir, pattern="*.tif*"):
    """List the files of a folder and extract the tilt angle from each file name
    (``ilm_tomo_dir``). Returns (sorted file list, sorted angles)."""
    files = sorted(glob.glob(os.path.join(path_dir, pattern)))
    names = [os.path.basename(f) for f in files]
    fs = pattern.replace("*", "")
    angles = np.zeros(len(names))
    for ik, name in enumerate(names):
        s = name.replace(fs, "").strip().lower()
        if s and s[0] in "mp" and not np.isnan(_str2double(s[1:])):
            s = "-" + s[1:] if s[0] == "m" else s[1:]
        else:
            s = "".join(ch for ch in s if not ch.isalpha())
            if "-" in s:
                s = s[s.rfind("-"):]
            elif "+" in s:
                s = s[s.rfind("+") + 1:]
            elif "_" in s:
                s_t = s[:s.rfind("_")]
                s = s.replace("_", "") if np.isnan(_str2double(s_t)) else s_t
        if "_" in s:
            s = s.split("_")[0]
        angles[ik] = _str2double(s)
    order = np.argsort(angles, kind="stable")
    return [files[i] for i in order], angles[order]


# =========================================================================== TIA (.emi/.ser)
def _find_key(d, *names):
    """Find a value in a metadata dict, ignoring case, spaces, underscores and unit suffixes."""
    if not isinstance(d, dict):
        return None
    norm = {re.sub(r"[\s_]", "", k).lower(): v for k, v in d.items()}
    for name in names:
        n = re.sub(r"[\s_]", "", name).lower()
        for k, v in norm.items():
            if k == n or k.startswith(n):
                return v
    return None


def _scan_rotation_k(scan_rotation):
    """k for np.rot90 reproducing ``rotate_data`` of read_emi.m."""
    if scan_rotation is None:
        return None
    r = float(scan_rotation)
    if 45 < r <= 135:
        return 2
    if -45 < r <= 45:
        return 3
    if -135 < r <= -45:
        return 0
    return 1


def read_emi(filename, normalize=True) -> dict:
    """Read a TIA ``.emi`` file and its ``.ser`` data (``read_emi``).

    Returns a dict with ``data`` (list, one stack per detector, each
    ``(n_frames, n_rows, n_cols)`` in MATLAB orientation), ``pixel_size`` (nm),
    ``stage_a`` (deg), ``time`` (s since midnight) and ``scan_rotation``."""
    from rsciio.tia import file_reader
    sigs = file_reader(filename)
    out = {"data": [], "pixel_size": None, "stage_a": None, "time": None, "scan_rotation": None}
    for sig in sigs:
        om = sig.get("original_metadata", {}) or {}
        obj = om.get("ObjectInfo", {}) or {}
        exp = obj.get("ExperimentalDescription", {}) or {}
        if out["stage_a"] is None:
            v = _find_key(exp, "Stage A")
            out["stage_a"] = None if v is None else float(v)
        if out["scan_rotation"] is None:
            v = _find_key(exp, "Stem rotation")
            out["scan_rotation"] = None if v is None else float(v)
        if out["time"] is None and "AcquireDate" in obj:
            try:
                hh, mm, ss = str(obj["AcquireDate"]).split()[3].split(":")
                out["time"] = int(hh) * 3600 + int(mm) * 60 + float(ss)
            except Exception:
                out["time"] = None
        data = np.asarray(sig["data"])
        if data.ndim == 2:
            data = data[None]
        # MATLAB read_ser returns each image transposed (fread [sizeX sizeY])
        data = np.swapaxes(data, -1, -2)
        k = _scan_rotation_k(out["scan_rotation"])
        if k:
            data = np.rot90(data, k, axes=(1, 2))
        data = im2double(data) if normalize else data
        out["data"].append(np.ascontiguousarray(data))
        if out["pixel_size"] is None:
            for ax in sig.get("axes", []):
                if ax.get("name") == "x":
                    scale = float(ax.get("scale", 1.0))
                    units = str(ax.get("units", "nm"))
                    out["pixel_size"] = scale * (1e9 if units in ("m", "meters") else 1.0)
    if out["pixel_size"] is None:
        out["pixel_size"] = 1.0
    return out


def read_fasttomo(fname_preview, fname_angles):
    """Read a fast-tomography acquisition (``wh_read_fasttomo``): the ``.emi``
    time series and a log/CSV with columns ``Time`` and ``Angle``. Each frame
    gets the angle of the last log entry before its (uniformly spaced)
    acquisition time.

    Returns (data, angles, pixel_size)."""
    import pandas as pd
    emi = read_emi(fname_preview)
    proj = emi["data"][0]
    table = pd.read_csv(fname_angles, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    t = table["Time"].to_numpy(dtype=float)
    a = table["Angle"].to_numpy(dtype=float)
    times = np.linspace(t[0], t[-1], proj.shape[0])
    idx = np.clip(np.searchsorted(t, times, side="right") - 1, 0, len(t) - 1)
    return proj.astype(np.float32), a[idx], emi["pixel_size"]


def emi2stack(dirname, normalize=True, detector=0) -> TiltSeries:
    """Read a folder with one ``.emi``/``.ser`` pair per tilt angle (``emi2stack``).
    Images are sorted by stage tilt. Time series are averaged over time; for
    multi-detector files ``detector`` selects which one is returned."""
    from tqdm.auto import tqdm
    files = sorted(glob.glob(os.path.join(dirname, "*.emi")), key=lambda s: s.lower())
    if not files:
        raise FileNotFoundError("No EMI files were found in the specified folder")
    first = read_emi(files[0], normalize)
    series = TiltSeries(pixel_size=first["pixel_size"])
    imgs, angles, times = [], [], []
    for f in tqdm(files, desc="Reading EMI files"):
        emi = read_emi(f, normalize)
        d = emi["data"][min(detector, len(emi["data"]) - 1)]
        imgs.append(d.mean(axis=0) if d.shape[0] > 1 else d[0])
        angles.append(emi["stage_a"] if emi["stage_a"] is not None else np.nan)
        times.append(emi["time"] if emi["time"] is not None else np.nan)
    order = np.argsort(angles, kind="stable")
    series.data = np.stack(imgs)[order].astype(np.float32)
    series.angles = np.asarray(angles)[order]
    t = np.asarray(times)[order]
    series.times = t - np.nanmin(t) if np.any(np.isfinite(t)) else t
    return series


# =========================================================================== Velox (.emd)
def read_emd(fpath, verbose=True):
    """Read a Velox ``.emd`` image or image series (``read_emd``).

    Returns (image, pixel_size_nm, alpha_tilt_deg). For series the data is a
    rescaled ``(n_frames, n_rows, n_cols)`` stack with one tilt per frame."""
    import h5py
    with h5py.File(fpath, "r") as f:
        grp = f["Data/Image"]
        name = list(grp.keys())[0]
        meta = np.asarray(grp[name]["Metadata"])          # (L, n_frames)
        dset = grp[name]["Data"]                           # (H, W, n_frames)
        nframes = meta.shape[1]

        def decode(col):
            raw = bytes(np.asarray(col, dtype=np.uint8)).rstrip(b"\x00")
            return json.loads(raw.decode("utf-8", errors="ignore"))

        if nframes > 1:
            if verbose:
                print(f"Reading a series with {nframes} frames ...")
            alpha = np.zeros(nframes)
            image = np.empty((nframes,) + dset.shape[:2], dtype=np.float32)
            for i in range(nframes):
                mj = decode(meta[:, i])
                alpha[i] = np.rad2deg(float(mj["Stage"]["AlphaTilt"]))
                image[i] = dset[:, :, i]
            image = rescale(image)
        else:
            if verbose:
                print("Reading a single image ...")
            mj = decode(meta[:, 0])
            alpha = float(np.rad2deg(float(mj["Stage"]["AlphaTilt"])))
            # MATLAB's squeeze(h5read(...)) returns single images transposed
            image = np.asarray(dset[:, :, 0]).T
        pixel_size = float(mj["BinaryResult"]["PixelSize"]["height"]) * 1e9
    return image, pixel_size, alpha


def emd2stack(dirname) -> TiltSeries:
    """Read a folder with one Velox ``.emd`` per tilt angle (``emd2stack``)."""
    from tqdm.auto import tqdm
    files = sorted(glob.glob(os.path.join(dirname, "*.emd")), key=lambda s: s.lower())
    if not files:
        raise FileNotFoundError("No EMD files were found in the specified folder")
    imgs, angles = [], []
    pixel_size = 1.0
    for f in tqdm(files, desc="Reading EMD files"):
        img, pixel_size, alpha = read_emd(f, verbose=False)
        imgs.append(np.asarray(img, dtype=np.float32))
        angles.append(alpha)
    order = np.argsort(angles, kind="stable")
    return TiltSeries(data=np.stack(imgs)[order], angles=np.asarray(angles)[order],
                      pixel_size=pixel_size)


# =========================================================================== MATLAB .mat
def read_mat_series(fname, var=None) -> TiltSeries:
    """Read a tilt series stored in a MATLAB ``.mat`` file as a struct with
    fields ``data`` (rows x cols x angles) and ``angles`` (and optionally
    ``pixelsize``/``pixel_size``)."""
    try:
        import scipy.io as sio
        m = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
        keys = [k for k in m if not k.startswith("__")]
        s = m[var or keys[0]]
        data = np.asarray(s.data)
        data = np.moveaxis(data, -1, 0)
        angles = np.asarray(s.angles, dtype=float).ravel()
        px = float(getattr(s, "pixel_size", getattr(s, "pixelsize", 1.0)))
    except NotImplementedError:
        import h5py
        with h5py.File(fname, "r") as f:
            key = var or [k for k in f.keys() if not k.startswith("#")][0]
            s = f[key]
            data = np.swapaxes(np.asarray(s["data"]), -1, -2)    # HDF5 stores (P, N, M)
            angles = np.asarray(s["angles"], dtype=float).ravel()
            px = 1.0
            for k in ("pixel_size", "pixelsize"):
                if k in s:
                    px = float(np.asarray(s[k]).ravel()[0])
    return TiltSeries(data=im2double(data).astype(np.float32), angles=angles, pixel_size=px)


# =========================================================================== writers
FEI_LABEL = "Fei Company (C) Copyright 2003"


def write_rec(array, filename, voxel_size=1.0, normalize=True):
    """Save a 3D array as a REC (MRC) file (``write_rec``).

    The array is written as ``(nz, ny, nx)`` sections, which is the same file
    layout as the MATLAB toolbox for reconstructions from :func:`reconstruct`
    (sections along the tilt axis). With ``normalize`` (default) the data is
    stretched to [0, 65535] and stored as uint16. An empty 128 kB extended
    header and the FEI label are written for Amira/Avizo compatibility, as in
    the MATLAB version, but the header is a standard MRC2014 header readable by
    ``mrcfile``, IMOD, FIJI, etc."""
    import mrcfile
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    if normalize:
        a = a.astype(np.float32)
        a = a - np.nanmin(a)
        mx = np.nanmax(a)
        a = a / mx if mx > 0 else a
        a = np.round(65535 * a).astype(np.uint16)
    elif a.dtype == np.float64:
        a = a.astype(np.float32)
    elif a.dtype not in (np.uint16, np.uint8, np.int16, np.float32, np.int8):
        raise TypeError("Unsupported data type for writing in REC file.")
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(a))
        mrc.set_extended_header(np.zeros(4 * 32 * 1024, dtype=np.uint8))
        mrc.voxel_size = float(voxel_size)
        mrc.header.label[0] = FEI_LABEL
        mrc.header.nlabl = 1
    return filename


def read_rec(filename) -> np.ndarray:
    """Read a REC/MRC file (also the ones written by the MATLAB toolbox, whose
    header lacks the MRC2014 map id; the related warnings are silenced)."""
    import warnings
    import mrcfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with mrcfile.open(filename, permissive=True) as mrc:
            return np.array(mrc.data)


def write_tiff(array, filename, pixel_size=1.0, normalize=True, compress=True):
    """Save a 2D/3D array as a (multi-page) TIFF (``write_tiff``).

    With ``normalize`` (default) the data is stretched to uint16. Otherwise only
    float32 and uint8 are supported (Amira/Avizo). PackBits compression is used
    when ``compress`` is True. The ImageJ description carries the pixel size in
    nm so FIJI opens the stack calibrated."""
    import tifffile
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    if normalize:
        a = np.round(65535 * rescale(a.astype(np.float32))).astype(np.uint16)
    elif a.dtype == np.float64:
        a = a.astype(np.float32)
    elif a.dtype not in (np.float32, np.uint8):
        raise TypeError("Only single or uint8 data types are supported by Amira/Avizo.")
    nz = a.shape[0]
    desc = (f"ImageJ=1.54f\nimages={nz}\nslices={nz}\nunit=nm\n"
            f"spacing={pixel_size}\nloop=false\n")
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    tifffile.imwrite(filename, a, photometric="minisblack",
                     compression="packbits" if compress else None,
                     resolution=(1.0 / pixel_size, 1.0 / pixel_size),
                     description=desc, metadata=None)
    return filename


def write_angles_csv(angles, filename):
    """Write the angles as one comma-separated row (like MATLAB ``csvwrite``)."""
    vals = ",".join(np.format_float_positional(float(a), trim="-") for a in np.ravel(angles))
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w") as f:
        f.write(vals + "\n")
    return filename


def get_series_names(fname, save_folder, from_root=None):
    """Root name for exported files (``rg_get_series_names``).

    - With ``from_root``: ``<save_folder>/<from_root>_``.
    - Otherwise the input is assumed to follow ``...\\Sample\\TomoX\\...``: the
      files go to ``<save_folder>/Sample/Sample_PX_`` (the ``Sample`` subfolder
      is created).

    The algorithm and number of iterations are appended by the caller, e.g.
    ``rec_name + f"{alg}{it}"``."""
    fname = str(fname)
    save_folder = str(save_folder)
    if from_root:
        return os.path.join(save_folder, f"{from_root}_")
    sep = "\\" if "\\" in fname else "/"
    k = [i for i, ch in enumerate(fname) if ch == sep]
    if len(k) < 3:
        raise ValueError("Cannot derive a sample name from the path; please provide from_root")
    rec_folder = fname[k[-3]:k[-2] + 1]            # e.g. '\Sample\'
    sample = rec_folder[1:-1]
    os.makedirs(os.path.join(save_folder, sample), exist_ok=True)
    p = [m.start() for m in re.finditer("Tomo", fname)] or [m.start() for m in re.finditer("tomo", fname)]
    p_num = fname[p[-1] + 4] if p and p[-1] + 4 < len(fname) else ""
    return os.path.join(save_folder, sample, f"{sample}_P{p_num}_")
