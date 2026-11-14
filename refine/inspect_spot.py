"""Check a 30 m match against imagery and land cover, and draw it beside the Mars reference.

AI disclosure: this file was written with Claude Code (Anthropic). NASA Space Apps 2026.
The figures it writes are plots of real data; each carries a visible note that AI-assisted code drew it.

For the best box of one searched cell (out/<slug>_search.json) this:
  * reads ESA WorldCover 2021 (10 m) for the box and reports the share of each land-cover class
  * fetches a Sentinel-2 cloudless image of the box (EOX, 2016 layer, CC BY 4.0)
  * reads the CTX orthoimage of the Mars reference box (USGS, CC0)
  * draws both boxes at the same ground scale: shaded relief from the 30 m grids on top, imagery below

Usage: python inspect_spot.py <reference.json> <slug> [more slugs...]
Writes out/fig_<slug>.png and out/<slug>_inspect.json. Needs the .npy grids from the earlier steps.
"""
import sys, os, io, json, math, urllib.request, warnings
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import terrain30

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MARS_R, EARTH_R = 3396190.0, 6378137.0
CTX = "https://astrogeo-ard.s3-us-west-2.amazonaws.com/mars/mro/ctx/controlled/usgs/{id}/{id}_orthoimage.tif"
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
S2 = ("https://tiles.maps.eox.at/wms?service=WMS&version=1.1.1&request=GetMap&layers=s2cloudless&styles=&srs=EPSG:4326"
      "&bbox={w},{s},{e},{n}&width={px}&height={px}&format=image/jpeg")
CLASSES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse vegetation",
           70: "Snow and ice", 80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
IMG_PX = 1000


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            if attempt == 4:
                raise


def hillshade(z, pixel_m, exaggeration=3.0, az=315.0, alt=45.0):
    gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)) * exaggeration, pixel_m, pixel_m)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(gy, -gx)
    a, h = math.radians(360 - az + 90), math.radians(alt)
    hs = np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect)
    hs = np.clip(hs, 0, 1)
    hs[~np.isfinite(z)] = np.nan
    return hs


def box_bounds(lat, lon, box_km, km_per_deg):
    hl = box_km / 2 / km_per_deg
    ho = hl / math.cos(math.radians(lat))
    return lon - ho, lat - hl, lon + ho, lat + hl


def local_grid(lat, lon, box_km, px, radius, datum):
    crs = f"+proj=eqc +lat_ts={lat} +lat_0=0 +lon_0={lon} {datum} +units=m +no_defs"
    size = box_km * 1000 / px
    tf = from_origin(-box_km * 500, math.radians(lat) * radius + box_km * 500, size, size)
    return crs, tf


def land_cover(lat, lon, box_km):
    w, s, e, n = box_bounds(lat, lon, box_km, 111.2)
    tiles = {(math.floor(a / 3) * 3, math.floor(o / 3) * 3) for a in (s, n) for o in (w, e)}
    crs, tf = local_grid(lat, lon, box_km, 2000, EARTH_R, "+datum=WGS84")      # 10 m pixels
    lc = np.zeros((2000, 2000), np.uint8)
    for la, lo in tiles:
        t = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
        with rasterio.open(WC.format(t=t)) as ds:
            reproject(rasterio.band(ds, 1), lc, dst_transform=tf, dst_crs=crs, resampling=Resampling.nearest,
                      init_dest_nodata=False, src_nodata=0, dst_nodata=0)
    vals, counts = np.unique(lc[lc > 0], return_counts=True)
    tot = counts.sum()
    return {CLASSES.get(int(v), str(v)): round(100 * c / tot, 1) for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])}


def earth_box(search, slug):
    """Cut the best box out of the saved 30 m search grid."""
    z = np.load(os.path.join(OUT, f"{slug}_earth30.npy")); n = z.shape[0]
    b, c = search["best"][0], search["centre"]
    w = int(round(search["box_km"] * 1000 / 30))
    r0 = int(round(n / 2 - (b["lat"] - c["lat"]) * 111200.0 / 30 - w / 2))
    c0 = int(round(n / 2 + (b["lon"] - c["lon"]) * 111200.0 * math.cos(math.radians(c["lat"])) / 30 - w / 2))
    return z[r0:r0 + w, c0:c0 + w], b


def main(ref_path, slug):
    ref = json.load(open(ref_path)); box = ref["box_km"]
    search = json.load(open(os.path.join(OUT, f"{slug}_search.json")))
    ze, best = earth_box(search, slug)
    elat, elon = best["lat"], best["lon"]
    # --- Earth: land cover and Sentinel-2
    lc = land_cover(elat, elon, box)
    w, s, e, n = box_bounds(elat, elon, box, 111.2)
    s2 = plt.imread(io.BytesIO(fetch(S2.format(w=w, s=s, e=e, n=n, px=IMG_PX))), format="jpeg")
    # --- Mars: the model that covers the box best, and its orthoimage
    m = max(ref["models"], key=lambda x: x["coverage"])
    zm = np.load(os.path.join(OUT, f"{ref['name']}_{m['id']}.npy"))
    crs, tf = local_grid(ref["lat"], ref["lon"], box, IMG_PX, MARS_R, f"+R={MARS_R}")
    ortho = np.full((IMG_PX, IMG_PX), np.nan, np.float32)
    with rasterio.open(CTX.format(id=m["id"])) as ds:
        reproject(rasterio.band(ds, 1), ortho, dst_transform=tf, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    lo, hi = np.nanpercentile(ortho, [1, 99])
    # --- figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 11.6), dpi=130)
    ext = [0, box, 0, box]
    ax[0, 0].imshow(hillshade(zm, 30), cmap="gray", vmin=0.2, vmax=1, extent=ext)
    ax[0, 1].imshow(hillshade(ze, 30), cmap="gray", vmin=0.2, vmax=1, extent=ext)
    ax[1, 0].imshow(ortho, cmap="gray", vmin=lo, vmax=hi, extent=ext)
    ax[1, 1].imshow(s2, extent=ext)
    R = ref["summary"]
    st = lambda d: f"slope over 90 m {d['slope90_mean']:.1f}°, roughness {d['rough']:.1f} m, relief {d['relief']:.0f} m"
    ax[0, 0].set_title(f"Mars: {ref['name'].replace('-', ' ').title()}  ({ref['lat']:.2f}°N, {abs(ref['lon']):.2f}°W)\n"
                       + st({k: v['median'] for k, v in R.items()}), fontsize=10)
    ax[0, 1].set_title(f"Earth: {search['name'][3:].replace('-', ' ').title()}  ({elat:.3f}, {elon:.3f})  score {best['score']:.0f}\n" + st(best), fontsize=10)
    ax[1, 0].set_title("CTX orthoimage (NASA MRO, USGS)", fontsize=10)
    ax[1, 1].set_title("Sentinel-2 cloudless 2016 (EOX, Copernicus)", fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
        a.plot([1, 6], [1, 1], color="white", lw=4, solid_capstyle="butt"); a.plot([1, 6], [1, 1], color="black", lw=1.5, solid_capstyle="butt")
        a.text(3.5, 1.5, "5 km", color="white", ha="center", va="bottom", fontsize=9,
               bbox=dict(facecolor="black", alpha=0.5, pad=1.5, edgecolor="none"))
    top = ", ".join(f"{k} {v:.0f}%" for k, v in list(lc.items())[:3])
    fig.suptitle(f"Same scale, same 30 m pixels, same lighting ({box:g} km boxes, relief exaggerated 3x)", fontsize=12, y=0.985)
    fig.text(0.5, 0.008, f"Earth land cover in the box (ESA WorldCover 2021): {top}.\n"
             "Earth elevation: AWS Terrain Tiles (SRTM). Mars elevation: USGS CTX stereo models.\n"
             "Figure drawn by AI-assisted code (Claude Code). The data are real observations.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.065, wspace=0.04, hspace=0.09)
    path = os.path.join(OUT, f"fig_{slug}.png")
    fig.savefig(path); plt.close(fig)
    info = dict(slug=slug, lat=elat, lon=elon, score=best["score"], elev_mean_m=round(float(np.nanmean(ze))),
                land_cover_percent=lc, figure=os.path.basename(path), mars_model=m["id"],
                generated_with="Claude Code (Anthropic), AI-assisted")
    json.dump(info, open(os.path.join(OUT, f"{slug}_inspect.json"), "w"), indent=1)
    print(f"{slug}: {elat:.4f}, {elon:.4f}  mean elevation {info['elev_mean_m']} m  land cover {lc}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for sl in sys.argv[2:]:
        main(sys.argv[1], sl)
