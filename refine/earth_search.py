"""Search an area on Earth at 30 m for the spot whose terrain best matches a Mars reference.

AI disclosure: this file was written with Claude Code (Anthropic). NASA Space Apps 2026.

Fetches AWS Terrain Tiles (zoom 12, about 30 m; SRTM between 60 N and 56 S) for a square area,
resamples them to a 30 m grid on a local projection, slides a box of the same size as the Mars
reference across it and computes the same statistics (terrain30.py) in every position.

PROVISIONAL scoring rule (a team decision, easy to change in score()):
  per metric: 1 - |log2(earth / mars)|, clipped to 0..1 (equal = 1, a factor of two apart = 0)
  score = 100 x mean over METRICS

Data: https://registry.opendata.aws/terrain-tiles/ (open, no login).

Usage: python earth_search.py <reference.json> <name> <lat> <lon> [area_km] [step_km]
  e.g. python earth_search.py out/mawrth-vallis_reference.json sistan 30.125 60.625 60 2
Writes out/<name>_search.json and caches tiles in out/tiles/.
"""
import sys, os, json, math, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import terrain30

ZOOM = 12
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
METRICS = ("slope90_mean", "slope300_mean", "rough", "relief")
WEB_R = 6378137.0
HERE = os.path.dirname(os.path.abspath(__file__))


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def fetch_tile(z, x, y):
    path = os.path.join(HERE, "out", "tiles", f"{z}_{x}_{y}.png")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(6):                 # transient DNS / network errors are common under parallel requests
            try:
                data = urllib.request.urlopen(TILE_URL.format(z=z, x=x, y=y), timeout=60).read()
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(1 + attempt)
        with open(path, "wb") as f:
            f.write(data)
    with MemoryFile(open(path, "rb").read()) as mf, mf.open() as ds:
        r, g, b = (ds.read(i).astype(np.float32) for i in (1, 2, 3))
    return r * 256 + g + b / 256 - 32768     # Terrarium encoding, metres


def earth_grid(lat, lon, area_km):
    """30 m elevation grid, area_km on a side, centred on lat/lon."""
    half = area_km / 2 / 111.2 * 1.1
    hl = half / math.cos(math.radians(lat))
    x0, y0 = (int(v) for v in tile_xy(lat + half, lon - hl, ZOOM))
    x1, y1 = (int(v) for v in tile_xy(lat - half, lon + hl, ZOOM))
    jobs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(8) as ex:
        tiles = list(ex.map(lambda xy: fetch_tile(ZOOM, *xy), jobs))
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.float32)
    for (x, y), t in zip(jobs, tiles):
        mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = t
    world = 2 * math.pi * WEB_R
    px = world / (2 ** ZOOM * 256)
    src_tf = from_origin(-world / 2 + x0 * 256 * px, world / 2 - y0 * 256 * px, px, px)
    n = int(round(area_km * 1000 / terrain30.PIXEL_M))
    dst_crs = f"+proj=eqc +lat_ts={lat} +lat_0=0 +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    yc = math.radians(lat) * WEB_R
    dst_tf = from_origin(-n * 15.0, yc + n * 15.0, terrain30.PIXEL_M, terrain30.PIXEL_M)
    z = np.full((n, n), np.nan, np.float32)
    reproject(mosaic, z, src_transform=src_tf, src_crs="EPSG:3857", dst_transform=dst_tf, dst_crs=dst_crs,
              dst_nodata=np.nan, resampling=Resampling.bilinear)
    return z, len(jobs)


def score(s, ref):
    sims = {}
    for k in METRICS:
        e, m = max(s[k], 1e-6), max(ref[k]["median"], 1e-6)
        sims[k] = float(np.clip(1 - abs(math.log2(e / m)), 0, 1))
    return 100 * sum(sims.values()) / len(sims), sims


def main(ref_path, name, lat, lon, area_km=60.0, step_km=2.0):
    ref = json.load(open(ref_path))
    box_km, R = ref["box_km"], ref["summary"]
    z, nt = earth_grid(lat, lon, area_km)
    print(f"{nt} tiles, grid {z.shape[0]} x {z.shape[1]} px at 30 m, elevation {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m")
    np.save(os.path.join(HERE, "out", f"{name}_earth30.npy"), z)
    w = int(round(box_km * 1000 / terrain30.PIXEL_M)); st = int(round(step_km * 1000 / terrain30.PIXEL_M))
    n = z.shape[0]
    wins = []
    for r in range(0, n - w + 1, st):
        for c in range(0, n - w + 1, st):
            s = terrain30.stats(z[r:r + w, c:c + w])
            if s is None:
                continue
            sc, sims = score(s, R)
            dy = (n / 2 - (r + w / 2)) * 30.0; dx = ((c + w / 2) - n / 2) * 30.0      # metres north / east of centre
            wlat = lat + dy / 111200.0; wlon = lon + dx / (111200.0 * math.cos(math.radians(lat)))
            wins.append(dict(lat=round(wlat, 4), lon=round(wlon, 4), score=round(sc, 1), sims=sims,
                             **{k: round(s[k], 3) for k in s}))
    wins.sort(key=lambda d: -d["score"])
    sc_all = np.array([d["score"] for d in wins])
    print(f"{len(wins)} boxes of {box_km:g} km scored; best {sc_all.max():.1f}, median {np.median(sc_all):.1f}, worst {sc_all.min():.1f}")
    print(f"Mars reference: " + ", ".join(f"{k} {R[k]['median']:.2f}" for k in METRICS))
    for d in wins[:5]:
        print(f"  {d['lat']:.4f}, {d['lon']:.4f}  score {d['score']:.1f}  " + ", ".join(f"{k} {d[k]:.2f}" for k in METRICS))
    out = dict(name=name, body="earth", centre=dict(lat=lat, lon=lon), area_km=area_km, step_km=step_km, box_km=box_km,
               reference=ref["name"], metrics=METRICS, rule="1 - |log2(earth/mars)| per metric, mean x 100 (provisional)",
               source="AWS Terrain Tiles (Mapzen Terrarium, zoom 12; SRTM)", n_boxes=len(wins),
               score_median=float(np.median(sc_all)), best=wins[:20], generated_with="Claude Code (Anthropic), AI-assisted")
    path = os.path.join(HERE, "out", f"{name}_search.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 4:
        sys.exit(__doc__)
    main(a[0], a[1], float(a[2]), float(a[3]), *(float(v) for v in a[4:6]))
