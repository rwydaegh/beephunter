"""Publish an ANONYMISED overview bundle to a public slug folder.

- Measurements are relabelled #1 / #2 / #3 by chronological order (no dates/hours).
- Measurement #2 (the off-GPS fluke) is NOT published.
- The bundle defaults to the latest good walk, #3, and lets you switch to #1.
- The home marker is genericised to "HOME" (house number dropped). The walk's
  GPS track and home coordinate remain visible (the map is inherently spatial).

Run:  py -3.14 desktop/publish_overview.py [slug]      (slug defaults to "report")
Writes:  <repo>/<slug>/index.html  (= #3)  and  <repo>/<slug>/m1.html  (= #1)
"""
import contextlib, glob, io, json, os, sys
from overview_map import analyze, TEMPLATE, twrange, wid, DL, ROOT

def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else "report"
    hunters = sorted(glob.glob(os.path.join(DL, "beephunter_hunter_*.json")))      # chronological
    refs    = sorted(glob.glob(os.path.join(DL, "beephunter_reference_*.json")))
    # number every tone-locked walk that has an overlapping reference, in time order
    numbered = []
    for h in hunters:
        a, b = twrange(h)
        import numpy as np
        fok = np.isfinite(np.array([x.get("f") for x in json.load(open(h, encoding="utf-8"))["rows"]], float)).mean()
        if fok < 0.05: continue
        Rf = next((r for r in refs if twrange(r)[0] < b and twrange(r)[1] > a), None)
        if Rf is None: continue
        numbered.append({"Pf": h, "Rf": Rf})
    for i, w in enumerate(numbered, 1): w["n"] = i
    by_n = {w["n"]: w for w in numbered}
    if 1 not in by_n or 3 not in by_n:
        sys.exit(f"need measurements #1 and #3; found #{sorted(by_n)}")

    # dropdown lists only #1 and #3 (publish set); #2 is withheld
    meta = [{"id": "1", "label": "#1",          "href": "m1.html"},
            {"id": "3", "label": "#3 (latest)", "href": "index.html"}]
    out_dir = os.path.join(ROOT, slug); os.makedirs(out_dir, exist_ok=True)

    def emit(n, current, fname):
        w = by_n[n]
        with contextlib.redirect_stdout(io.StringIO()):           # silence per-walk report
            data = analyze(w["Pf"], w["Rf"])
        data["walks"] = meta; data["current"] = current; data["homeLabel"] = "HOME"
        open(os.path.join(out_dir, fname), "w", encoding="utf-8").write(TEMPLATE.replace("__DATA__", json.dumps(data)))
        return data

    d3 = emit(3, "3", "index.html")
    d1 = emit(1, "1", "m1.html")
    print(f"published anonymised bundle -> {slug}/index.html (#3, default) + {slug}/m1.html (#1); #2 withheld")
    print(f"   #3 fit: R2={d3['R2']}, {d3['dh']} m from home, {d3['verdict']}")
    print(f"   #1 fit: R2={d1['R2']}, {d1['dh']} m from home, {d1['verdict']}")

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    main()
