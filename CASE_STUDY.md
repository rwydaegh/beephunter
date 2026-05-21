# BeepHunter — the day we found the beep

A persistent **"beep… beep… beep…"** had been sounding in our street **day and night**.
A pure tone near **3042 Hz**, gated at about **2.9 beeps/second**, with no obvious source.
By ear it was almost impossible to place — the whole street stood outside for an hour
arguing about which house it came from. It turned out to be a **smoke-detector
low-battery chirp** in a house whose occupants were **away on holiday**.

This is what the tool did, what it couldn't do, and the role it actually played.

---

## What it was up against

The street is a **canyon** — narrow, hard façades on both sides. Acoustically and for
GPS, that's the worst case:

- **Acoustic multipath.** The tone arrives from every wall at once. Loudness stops
  meaning "closer" — we measured that **the loud spots were the *dirtiest*, not the
  nearest** (band level correlated *negatively* with lock confidence). The naive
  "hotter / colder" instinct actively misleads you here.
- **GPS multipath.** Position in the canyon jitters by metres; one walk even appeared
  to cut straight through a building.

So the honest ceiling was set on day one: **you cannot point-localise a pure tone to
one metre in a canyon from the street.** Three independent methods agreed on that.

---

## The features that were built

**Capture — a two-phone-style web app (`index.html` + `app.js`)**

- **Two roles, one source.** A walking **Hunter** (phone, GPS) and a stationary
  **Reference** (laptop). Subtracting the reference's frequency removes the source's
  own oscillator drift, so what's left is **Doppler** — motion-induced pitch shift.
- **Tone tracking.** A matched-filter fine-frequency estimator with a **confidence**
  score, plus **cadence detection (~2.9/s)** to confirm "this really is the beep" and
  reject steady hums.
- **Live Hunt meter (AirTag-style).** A glanceable hot/cold readout that **auto-zeroes
  its rest frequency** while you stand still, then shows ▲ *warmer* / ▼ *colder* as you
  move — Doppler-based, so it reflects whether you're closing on the source, not just
  how loud it is.

**Analysis — a desktop pipeline (`desktop/`)**

- **`overview.html`** — one interactive map with toggleable layers: Doppler, lock
  confidence, band level, raw input level, cadence, frequency, speed; click-through
  popups; walk-direction arrows that densify as you zoom; raw vs. smoothed GPS track;
  and an in-page **measurement selector**.
- **Doppler localisation, three ways:**
  - a **point-source fit** (where would a single source best explain all the Doppler?),
  - **abeam lines-of-position** (at each pitch flip the source is exactly perpendicular
    to your path),
  - a **per-cell bearing field** + **per-sample Doppler arrows** (each arrow points the
    way the Doppler says the source lies).
- **A data-quality report** that grades every walk — trustworthy %, drift, clock-sync
  sanity, the level-vs-confidence correlation — so we never trusted dirty data.
- **Tight-cropped map/PNG renders** shrink-wrapped exactly to the data.

---

## The role it actually played

The tool **did not** drop a pin on the exact house — the physics didn't allow it. What
it **did** was decisive in two ways:

1. **It narrowed the search.** The Doppler analysis and the live hunt pointed
   consistently at a **small cluster of houses near ours**, not the whole street. That
   turned "somewhere on the block" into a **shortlist of two**.
2. **It carried credibility.** When the fire brigade arrived, showing them the maps and
   the reasoning is **why they investigated those specific houses**. They ruled out the
   top-ranked candidate by going inside, then found it in the **second-ranked
   candidate** — the holiday house with the dying detector.

> The win wasn't a magic GPS pin. It was **converting an un-pinnable signal into a
> ranked shortlist plus evidence a decision-maker would act on.**

---

## What we learned (and the one lever we didn't pull)

- **Multipath, not the source, was the enemy.** Every street-level method bottoms out on
  it. The cleanest data came from the *quiet* spots, the opposite of intuition.
- **GPS was the co-limiting constraint** — clean (pure-GNSS) walks localised far better
  than network-assisted ones.
- **Elevation beats everything.** Above the rooflines there *is* no canyon multipath:
  the direct path dominates and a single bearing sweep would likely just point at it.
  A mic at an upper window, on a balcony, or on a pole is the biggest unrealised win —
  the fire truck's aerial platform was the dream version of exactly this idea.
- **Next time:** a two-mic array on **one clock** (true stereo), gating each beep's
  **first arrival** (the one part of the signal multipath can't corrupt) to read a
  direct-path bearing, then triangulating from a few quiet spots.

---

*Interactive map of the hunt: `report/` (measurement #3 by default; #1 selectable).
Built with a Doppler-first capture app and a Python analysis pipeline.*
