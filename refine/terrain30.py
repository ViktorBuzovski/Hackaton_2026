"""Terrain statistics on a 30 m grid, identical for Mars and Earth.

AI disclosure: this file was written with Claude Code (Anthropic). NASA Space Apps 2026.

Same definitions as the Ground Truth app (analog-finder/pipeline/terrainlib.py), applied at 30 m
instead of 1.85 km:

  slope   slope in degrees from central differences (baseline = 2 pixels)
  rough   RMS height difference between each pixel and its 5 x 5 neighbourhood mean (m)
  relief  95th minus 5th percentile of elevation inside the box (m)

Slope is also reported after block-averaging to 90 m and 300 m pixels, because stereo elevation
models carry a few metres of vertical noise that inflates slopes measured over one 30 m pixel.
"""
import numpy as np

PIXEL_M = 30.0


def _box5(z):
    """5 x 5 box mean that ignores NaN (edge pixels use the neighbours they have)."""
    ok = np.isfinite(z)
    zz = np.where(ok, z, 0.0)

    def boxsum(a):
        p = np.pad(a.astype(np.float64), 2)
        c = np.pad(np.cumsum(np.cumsum(p, 0), 1), ((1, 0), (1, 0)))
        return c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5]

    n = boxsum(ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, boxsum(zz) / n, np.nan)


def _block_mean(z, k):
    """Average k x k blocks; a block is valid when at least half its pixels are."""
    h, w = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    b = z[:h, :w].reshape(h // k, k, w // k, k)
    ok = np.isfinite(b)
    n = ok.sum((1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, b, 0.0).sum((1, 3)) / n
    return np.where(n >= k * k / 2, m, np.nan)


def slope_deg(z, pixel_m):
    gy, gx = np.gradient(z, pixel_m, pixel_m)   # NaN neighbours give NaN slopes, which are dropped
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def stats(z, pixel_m=PIXEL_M):
    """z: 2D float array of elevations in metres on a square grid, NaN = no data.
    Returns a dict of terrain statistics, or None if under 60 % of the box has data."""
    ok = np.isfinite(z)
    cover = float(ok.mean())
    if cover < 0.6:
        return None
    out = dict(coverage=round(cover, 3))
    for k in (1, 3, 10):
        zk = z if k == 1 else _block_mean(z, k)
        s = slope_deg(zk, pixel_m * k)
        s = s[np.isfinite(s)]
        b = int(pixel_m * k)
        out[f"slope{b}_mean"] = float(s.mean())
        out[f"slope{b}_median"] = float(np.median(s))
        out[f"slope{b}_p95"] = float(np.percentile(s, 95))
    resid = z - _box5(z)
    out["rough"] = float(np.sqrt(np.nanmean(resid ** 2)))
    p5, p95 = np.nanpercentile(z, [5, 95])
    out["relief"] = float(p95 - p5)
    out["elev_mean"] = float(np.nanmean(z))
    return out
