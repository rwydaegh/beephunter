"""Local config loader — keeps home coordinates / OSM contact OUT of the public repo.

Real values live in desktop/home.local.json (gitignored). If that file is absent
(e.g. a fresh clone), the scripts still run using a non-identifying placeholder.
To use your own location, create desktop/home.local.json:
    {"home": [LAT, LON], "contact": "you@example.com", "dopsrc": [LAT, LON]}
"""
import json, os

_f = os.path.join(os.path.dirname(__file__), "home.local.json")
if os.path.exists(_f):
    _d = json.load(open(_f, encoding="utf-8"))
    HOME = tuple(_d["home"])
    CONTACT = _d.get("contact", "https://github.com/rwydaegh/beephunter")
    DOPSRC = tuple(_d["dopsrc"]) if "dopsrc" in _d else HOME
    WALK = [tuple(p) for p in _d.get("walk", [])]
    HOMELABEL = _d.get("homelabel", "HOME")     # keep any house number out of the public repo
else:
    HOME = (51.05, 3.72)        # placeholder — set yours in desktop/home.local.json
    CONTACT = "https://github.com/rwydaegh/beephunter"
    DOPSRC = (51.05, 3.72)
    WALK = []
    HOMELABEL = "HOME"
