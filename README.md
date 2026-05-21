# BeepHunter 🔊🔎

Hunt down a mystery beep ("beep beep beep" at a constant pitch) using your
phone and laptop as a two-microphone team. Runs entirely in the browser — no
install, no backend — and is hosted on GitHub Pages.

**Live app:** see the repo's GitHub Pages URL (Settings → Pages).
- **Capture** (`index.html`) — runs on phone *or* laptop; tracks the tone with
  sub-Hz precision, shows a Doppler "sharp/flat" needle, logs level + cadence +
  GPS, and downloads a JSON recording.
- **Analyze** (`analyze.html`) — load the two recordings; it auto-syncs their
  clocks, removes the source's own behaviour differentially, and maps the result.

## Why two devices (the whole trick)

A single moving mic can't tell **"I got closer"** from **"the source just got
louder"**, and slow walking gives only a ~7 Hz Doppler shift on a ~3 kHz tone.
A **stationary reference** device fixes both:

| Differential | Removes | Leaves |
|---|---|---|
| `hunterLevel − referenceLevel` | the source's amplitude modulation (temporal "hotspots") | spatial proximity |
| `hunterFreq − referenceFreq` | the source's intrinsic pitch drift | **pure Doppler** |

The reference *is* the live rest-frequency and the loudness baseline.

## How the frequency precision works

We don't fight the FFT bin width. The tone is **complex-demodulated** to a
narrow baseband around ~3050 Hz, then a fine matched-filter search ("zoom-DFT")
finds the sub-Hz offset. A single 170 ms beep only resolves ~6 Hz, but
**averaging N beeps → 6/√N**, so a couple seconds of walking gives ~1 Hz —
enough to see the 7–14 Hz Doppler swing. We track the **zero-crossing trend**
(sharp→flat as you pass abeam), not an absolute value.

## Measurement protocol

1. **Stability check (60 s):** both devices side-by-side, still, recording.
   Confirms they read the same f₀, how much it drifts, and that Doppler is viable.
2. **Place the reference:** laptop stationary at an open spot with decent
   signal, recording the whole time (role = *Reference*).
3. **Walk transects:** phone (role = *Hunter*), record while walking **straight
   lines at a brisk, steady pace** through/near the suspected zone. The pitch
   rises approaching → flips through rest **abeam** → drops receding. Do **≥2
   straight passes from different directions**.
4. Keep the phone steady in hand (mic unobstructed), avoid wind, don't curve
   mid-transect. Grant **location** permission on the phone.
5. Stop + **Download** on both devices. Open `analyze.html`, load both files.

### Localization principle
At closest approach the radial velocity (Doppler) is zero → the source is
**abeam** (perpendicular to your heading). Each zero-crossing draws a line; the
analyzer least-squares-intersects lines from multiple transects to estimate the
source — immune to the loudness artifacts (open doors, reflections) that fooled
the amplitude-only method.

## Notes
- Requires **HTTPS** (mic + GPS) — GitHub Pages provides it.
- The app disables `autoGainControl`/`noiseSuppression`/`echoCancellation`;
  AGC especially would otherwise ruin level measurements.
- The analyzer is **v1/experimental** — field data is messy; expect iteration.

## `desktop/` — original single-laptop tool
The earlier Python app (`beephunter.py` + `dashboard.html`) and the GPS
log-mapping tools (`analyze_log.py`, `make_map.py`) live in `desktop/`. See
`desktop/` for the local-server version. Run: `py -3.14 desktop/beephunter.py`.
