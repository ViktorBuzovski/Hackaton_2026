"""Mars reference terrain at 30 m from USGS CTX stereo elevation models (NASA MRO).

AI disclosure: this file was written with Claude Code (Anthropic). NASA Space Apps 2026.

Finds every CTX elevation model in the USGS Astrogeology archive that overlaps a box around a
point on Mars, resamples each one to a 30 m grid centred on that point, and computes the terrain
statistics in terrain30.py for each model separately. Models are not mosaicked, because the
archive warns that neighbouring models may not line up; the spread between models is reported
instead, as a measure of how far the numbers can be trusted.

Data: https://stac.astrogeology.usgs.gov/docs/data/mars/ctxdtms (public domain, no login).
The models are read over HTTP, a window at a time; nothing large is downloaded.

Usage: python mars_reference.py <name> <lat> <lon_east> [box_km]
  e.g. python mars_reference.py mawrth-vallis 24.01 -18.97 20
Writes out/<name>_reference.json and out/<name>_<model>.npy (30 m elevation grids).
"""
import sys, os, json, urllib.request
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import terrain30

API = "https://stac.astrogeology.usgs.gov/api/collections/mro_ctx_controlled_usgs_dtms/items"
MARS_R = 3396190.0
KM_PER_DEG = np.pi * MARS_R / 180 / 1000


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "stac-client"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def main(name, lat, lon, box_km=20.0):
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "out")
    os.makedirs(out_dir, exist_ok=True)
    half_lat = box_km / 2 / KM_PER_DEG
    half_lon = half_lat / np.cos(np.radians(lat))
    bbox = f"{lon - half_lon},{lat - half_lat},{lon + half_lon},{lat + half_lat}"
    items = get_json(f"{API}?bbox={bbox}&limit=200")["features"]
    print(f"{len(items)} CTX models overlap the {box_km:g} km box at {lat}, {lon}")

    # 30 m grid on a local equirectangular projection, so pixels are square on the ground here
    n = int(round(box_km * 1000 / terrain30.PIXEL_M))
    dst_crs = f"+proj=eqc +lat_ts={lat} +lat_0=0 +lon_0={lon} +R={MARS_R} +units=m +no_defs"
    y0 = np.radians(lat) * MARS_R
    dst_tf = from_origin(-n * terrain30.PIXEL_M / 2, y0 + n * terrain30.PIXEL_M / 2, terrain30.PIXEL_M, terrain30.PIXEL_M)

    models = []
    for it in items:
        z = np.full((n, n), np.nan, np.float32)
        with rasterio.open(it["assets"]["dtm"]["href"]) as ds:
            reproject(rasterio.band(ds, 1), z, dst_transform=dst_tf, dst_crs=dst_crs, dst_nodata=np.nan,
                      resampling=Resampling.average)
        s = terrain30.stats(z)
        if s is None:
            print(f"  skip {it['id'][:40]}  (covers {np.isfinite(z).mean():.0%} of the box)")
            continue
        np.save(os.path.join(out_dir, f"{name}_{it['id']}.npy"), z)
        models.append(dict(id=it["id"], **s))
        print(f"  {it['id'][:40]}  cover {s['coverage']:.0%}  slope30 {s['slope30_mean']:.2f}  slope90 {s['slope90_mean']:.2f}"
              f"  slope300 {s['slope300_mean']:.2f}  rough {s['rough']:.2f} m  relief {s['relief']:.0f} m")
    if not models:
        sys.exit("no model covers enough of the box")
    keys = [k for k in models[0] if k not in ("id", "coverage")]
    arr = {k: np.array([m[k] for m in models]) for k in keys}
    summary = {k: dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max())) for k, v in arr.items()}
    res = dict(name=name, body="mars", lat=lat, lon=lon, box_km=box_km, pixel_m=terrain30.PIXEL_M,
               source="USGS Astrogeology ARD, MRO CTX controlled DTMs (20 m), NASA/JPL/MSSS/USGS, CC0",
               n_models=len(models), summary=summary, models=models,
               generated_with="Claude Code (Anthropic), AI-assisted")
    path = os.path.join(out_dir, f"{name}_reference.json")
    json.dump(res, open(path, "w"), indent=1)
    print("\nmedian across models (min to max):")
    for k, v in summary.items():
        print(f"  {k:16s} {v['median']:8.2f}  ({v['min']:.2f} to {v['max']:.2f})")
    print("wrote", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    main(a[0], float(a[1]), float(a[2]), float(a[3]) if len(a) > 3 else 20.0)
