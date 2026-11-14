"""Run earth_search.py on every cell of a shortlist, in parallel, and print a summary table.

AI disclosure: this file was written with Claude Code (Anthropic). NASA Space Apps 2026.

Usage: python run_shortlist.py <reference.json> <shortlist.json> [area_km] [step_km]
Writes out/<slug>_search.json per cell and out/<target>_summary.json.
"""
import sys, os, json, io, contextlib, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
import earth_search

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    ref, c, area, step = args
    with contextlib.redirect_stdout(io.StringIO()):
        earth_search.main(ref, c["slug"], c["lat"], c["lon"], area, step)
    d = json.load(open(os.path.join(HERE, "out", f"{c['slug']}_search.json")))
    b = d["best"][0]
    half = (area - d["box_km"]) / 2 / 111.2          # how far a box centre can sit from the area centre, degrees of latitude
    edge = abs(b["lat"] - c["lat"]) > half * 0.97 or abs(b["lon"] - c["lon"]) * __import__("math").cos(__import__("math").radians(c["lat"])) > half * 0.97
    return dict(**c, best_lat=b["lat"], best_lon=b["lon"], best_score=b["score"], area_median=round(d["score_median"], 1),
                on_edge=bool(edge), **{k: b[k] for k in d["metrics"]})


if __name__ == "__main__":
    ref, sl = sys.argv[1], json.load(open(sys.argv[2]))
    area = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    with ProcessPoolExecutor(5) as ex:
        rows = list(ex.map(run, [(ref, c, area, step) for c in sl["cells"]]))
    R = json.load(open(ref))["summary"]
    rows.sort(key=lambda r: -r["best_score"])
    print(f"Mars reference {sl['target']}: slope90 {R['slope90_mean']['median']:.2f}, slope300 {R['slope300_mean']['median']:.2f}, rough {R['rough']['median']:.2f}, relief {R['relief']['median']:.0f}")
    for r in rows:
        print(f"{r['best_score']:5.1f} (area median {r['area_median']:4.1f}) app {r['app_score']:.1f}  {r['place'][:38]:38s} {r['best_lat']:.4f},{r['best_lon']:.4f}"
              f"  s90 {r['slope90_mean']:.2f} s300 {r['slope300_mean']:.2f} rough {r['rough']:.2f} relief {r['relief']:.0f}{'  EDGE' if r['on_edge'] else ''}")
    json.dump(dict(target=sl["target"], area_km=area, step_km=step, rows=rows, generated_with="Claude Code (Anthropic), AI-assisted"),
              open(os.path.join(HERE, "out", f"{sl['target']}_summary.json"), "w"), indent=1)
