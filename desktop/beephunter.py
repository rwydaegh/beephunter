#!/usr/bin/env python3
"""
BeepHunter - live microphone beep locator.

Captures audio from the laptop microphone, auto-locks onto the strongest
narrowband tone in a search band ("the beep's pitch"), confirms it is really
a beep by detecting its repetition cadence (~2-4 beeps/sec), and reports a
robust peak-hold signal-strength meter you can use to walk toward the source.

Serves a fullscreen web dashboard on http://localhost:8000 (no GUI deps).

Run:
    py -3.14 beephunter.py
    py -3.14 beephunter.py --device 1 --port 8000
"""

import argparse
import json
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, hilbert

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tunables (sensible defaults; most are adjustable live from the dashboard)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100          # Hz
BLOCKSIZE = 1024             # frames per audio callback
RING_SECONDS = 4.0           # length of the rolling capture buffer

SPEC_FFT = 8192              # FFT size for the displayed spectrum / peak pick
SPEC_AVG_S = 1.5             # seconds averaged (Welch) so gated beeps survive
SPEC_BINS = 360              # downsampled spectrum points sent to the browser

ENV_WINDOW_S = 3.0           # seconds of audio used for envelope/cadence
ENV_FS = 250                 # Hz, envelope resample rate for cadence autocorr
STRENGTH_WINDOW_S = 0.30     # recent slice used for the instantaneous level

DSP_HZ = 20                  # DSP / dashboard update rate
EPS = 1e-12

# Lock criteria
PROMINENCE_DB = 8.0          # carrier peak must stand this far above band median
CADENCE_MIN_HZ = 1.0
CADENCE_MAX_HZ = 6.0
CADENCE_CONF_MIN = 0.25      # normalized autocorr peak needed to call it "beeping"

# Meter behaviour
PEAK_DECAY_DB_PER_S = 6.0    # how fast the peak-hold marker bleeds down
SMOOTH_TAU_S = 0.4           # EMA time constant for the smoothed level
TREND_LAG_S = 3.0            # compare against the level this long ago


class Config:
    """Live, dashboard-adjustable settings, guarded by `lock`."""
    def __init__(self):
        self.lock = threading.Lock()
        self.search_lo = 2500.0
        self.search_hi = 4000.0
        self.bandwidth = 60.0           # half-width of the locked band (Hz)
        self.manual_freq = None         # None => auto-track strongest peak
        self.logging = False

    def snapshot(self):
        with self.lock:
            return (self.search_lo, self.search_hi, self.bandwidth,
                    self.manual_freq, self.logging)


class AudioEngine:
    """Owns the input stream and a mono rolling ring buffer."""
    def __init__(self, device, samplerate=SAMPLE_RATE):
        self.device = device
        self.samplerate = samplerate
        self.n = int(RING_SECONDS * samplerate)
        self.buf = np.zeros(self.n, dtype=np.float32)
        self.widx = 0
        self.lock = threading.Lock()
        self.stream = None
        self.input_rms = 0.0
        self.overflows = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            if status.input_overflow:
                self.overflows += 1
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:]
        self.input_rms = float(np.sqrt(np.mean(mono**2) + EPS))
        with self.lock:
            end = self.widx + frames
            if end <= self.n:
                self.buf[self.widx:end] = mono
            else:
                first = self.n - self.widx
                self.buf[self.widx:] = mono[:first]
                self.buf[:frames - first] = mono[first:]
            self.widx = end % self.n

    def latest(self, count):
        """Return the most recent `count` samples in chronological order."""
        count = min(count, self.n)
        with self.lock:
            end = self.widx
            idx = (np.arange(end - count, end) % self.n)
            return self.buf[idx].copy()

    def start(self):
        self.stream = sd.InputStream(
            device=self.device, channels=1, samplerate=self.samplerate,
            blocksize=BLOCKSIZE, dtype="float32", callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def restart(self, device):
        self.stop()
        self.device = device
        self.buf[:] = 0.0
        self.widx = 0
        self.start()


def list_input_devices():
    out = []
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"],
                        "channels": d["max_input_channels"],
                        "default": (i == default_in)})
    return out


class DSP:
    """Pulls audio from the engine, analyses it, publishes a state dict."""
    def __init__(self, engine: AudioEngine, cfg: Config):
        self.engine = engine
        self.cfg = cfg
        self.fs = engine.samplerate
        self.window = np.hanning(SPEC_FFT).astype(np.float32)
        self.freqs = np.fft.rfftfreq(SPEC_FFT, 1.0 / self.fs)

        # meter state
        self.smoothed_db = -120.0
        self.peakhold_db = -120.0
        self.trend_hist = deque()        # (t, smoothed_db)
        self.strength_hist = deque()     # (t, smoothed_db) for the 60s strip

        # publishing
        self.state = {}
        self.state_lock = threading.Lock()
        self.running = True

        # logging
        self.log_fp = None
        self.log_path = None
        self.last_log_t = 0.0

    # -- spectrum + peak picking ------------------------------------------
    def _spectrum_db(self):
        """Welch-averaged power spectrum (dB) over the recent SPEC_AVG_S.

        A single short FFT is shorter than one beep, so it often lands in a
        gap and sees only noise. Averaging several overlapping frames lets the
        gated tone integrate well above the noise floor for a stable lock.
        """
        n = int(SPEC_AVG_S * self.fs)
        seg = self.engine.latest(n)
        if seg.shape[0] < SPEC_FFT:
            seg = np.pad(seg, (SPEC_FFT - seg.shape[0], 0))
        hop = SPEC_FFT // 2
        nseg = 1 + (seg.shape[0] - SPEC_FFT) // hop
        acc = np.zeros(SPEC_FFT // 2 + 1)
        cnt = 0
        for i in range(nseg):
            s = seg[i * hop:i * hop + SPEC_FFT]
            if s.shape[0] < SPEC_FFT:
                break
            mag = np.abs(np.fft.rfft(s * self.window)) / (SPEC_FFT / 2)
            acc += mag ** 2
            cnt += 1
        psd = acc / max(cnt, 1)
        return 10.0 * np.log10(psd + EPS)

    def _pick_peak(self, mag_db, lo, hi):
        band = (self.freqs >= lo) & (self.freqs <= hi)
        if not np.any(band):
            return None, 0.0
        idx = np.where(band)[0]
        local = mag_db[idx]
        floor = np.median(local)
        k = int(np.argmax(local))
        peak_db = float(local[k])
        return float(self.freqs[idx[k]]), peak_db - float(floor)

    @staticmethod
    def _downsample_spec(freqs, mag_db, bins):
        """Max-pool into `bins` log-spaced cells so sharp peaks survive."""
        lo = max(freqs[1], 20.0)
        hi = freqs[-1]
        edges = np.geomspace(lo, hi, bins + 1)
        out_f, out_db = [], []
        for j in range(bins):
            m = (freqs >= edges[j]) & (freqs < edges[j + 1])
            if np.any(m):
                out_f.append(float(np.sqrt(edges[j] * edges[j + 1])))
                out_db.append(float(np.max(mag_db[m])))
        return out_f, out_db

    # -- envelope + cadence ------------------------------------------------
    def _band_envelope(self, freq, bw):
        n_env = int(ENV_WINDOW_S * self.fs)
        sig = self.engine.latest(n_env)
        lo = max(20.0, freq - bw)
        hi = min(self.fs / 2 - 100.0, freq + bw)
        if hi <= lo:
            return None, None
        sos = butter(4, [lo, hi], btype="bandpass", fs=self.fs, output="sos")
        try:
            filt = sosfiltfilt(sos, sig)
        except ValueError:
            return None, None
        env = np.abs(hilbert(filt))
        # resample envelope to ENV_FS for cadence analysis
        step = max(1, int(self.fs / ENV_FS))
        env_ds = env[::step]
        return filt, env_ds

    def _cadence(self, env_ds):
        if env_ds is None or env_ds.size < 32:
            return 0.0, 0.0
        x = env_ds - env_ds.mean()
        if np.allclose(x, 0):
            return 0.0, 0.0
        ac = np.correlate(x, x, mode="full")[x.size - 1:]
        ac0 = ac[0] + EPS
        lag_min = int(ENV_FS / CADENCE_MAX_HZ)
        lag_max = int(ENV_FS / CADENCE_MIN_HZ)
        lag_max = min(lag_max, ac.size - 1)
        if lag_max <= lag_min:
            return 0.0, 0.0
        seg = ac[lag_min:lag_max]
        k = int(np.argmax(seg)) + lag_min
        conf = float(ac[k] / ac0)
        cadence_hz = ENV_FS / k
        return cadence_hz, max(0.0, conf)

    # -- main loop ---------------------------------------------------------
    def run(self):
        period = 1.0 / DSP_HZ
        n_strength = int(STRENGTH_WINDOW_S * self.fs)
        while self.running:
            t0 = time.time()
            lo, hi, bw, manual, logging_on = self.cfg.snapshot()

            mag_db = self._spectrum_db()
            if manual is not None:
                target = float(manual)
                # prominence of the manually chosen freq for the badge
                bidx = np.argmin(np.abs(self.freqs - target))
                bandmask = (self.freqs >= lo) & (self.freqs <= hi)
                floor = np.median(mag_db[bandmask]) if np.any(bandmask) else -120.0
                prominence = float(mag_db[bidx] - floor)
            else:
                target, prominence = self._pick_peak(mag_db, lo, hi)

            cadence_hz, cadence_conf = 0.0, 0.0
            strength_db = -120.0
            if target is not None:
                filt, env_ds = self._band_envelope(target, bw)
                if filt is not None:
                    recent = filt[-n_strength:]
                    strength_db = 10.0 * np.log10(np.mean(recent**2) + EPS)
                    cadence_hz, cadence_conf = self._cadence(env_ds)

            now = time.time()
            # smoothed (EMA) level
            alpha = 1.0 - np.exp(-period / SMOOTH_TAU_S)
            if self.smoothed_db < -119:
                self.smoothed_db = strength_db
            else:
                self.smoothed_db += alpha * (strength_db - self.smoothed_db)
            # peak-hold with decay
            self.peakhold_db -= PEAK_DECAY_DB_PER_S * period
            if strength_db > self.peakhold_db:
                self.peakhold_db = strength_db
            # trend vs ~TREND_LAG_S ago: drop entries older than the lag, then
            # the oldest remaining sample is our reference point.
            self.trend_hist.append((now, self.smoothed_db))
            while len(self.trend_hist) > 1 and now - self.trend_hist[0][0] > TREND_LAG_S:
                self.trend_hist.popleft()
            trend_ref = self.trend_hist[0][1]
            trend = self.smoothed_db - trend_ref
            # 60s history strip
            self.strength_hist.append((now, self.smoothed_db))
            while self.strength_hist and now - self.strength_hist[0][0] > 60.0:
                self.strength_hist.popleft()

            locked = (target is not None and prominence >= PROMINENCE_DB and
                      cadence_conf >= CADENCE_CONF_MIN and
                      CADENCE_MIN_HZ <= cadence_hz <= CADENCE_MAX_HZ)

            sf, sdb = self._downsample_spec(self.freqs, mag_db, SPEC_BINS)
            hist = [round(v, 2) for _, v in self.strength_hist]

            state = {
                "t": now,
                "locked": bool(locked),
                "manual": manual is not None,
                "freq": round(target, 1) if target else 0.0,
                "prominence": round(prominence, 1),
                "search_lo": lo, "search_hi": hi, "bandwidth": bw,
                "strength_db": round(float(strength_db), 2),
                "smoothed_db": round(float(self.smoothed_db), 2),
                "peakhold_db": round(float(self.peakhold_db), 2),
                "trend": round(float(trend), 2),
                "cadence_hz": round(float(cadence_hz), 2),
                "cadence_bpm": round(float(cadence_hz * 60), 0),
                "cadence_conf": round(float(cadence_conf), 2),
                "input_rms": round(float(self.engine.input_rms), 5),
                "input_db": round(float(20*np.log10(self.engine.input_rms + EPS)), 1),
                "overflows": self.engine.overflows,
                "logging": logging_on,
                "log_path": str(self.log_path) if self.log_path else "",
                "spectrum_freqs": [round(f, 1) for f in sf],
                "spectrum_db": [round(v, 1) for v in sdb],
                "history": hist,
                "device": self.engine.device,
            }
            with self.state_lock:
                self.state = state

            self._handle_logging(logging_on, now, state)

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    def get_state(self):
        with self.state_lock:
            return dict(self.state)

    def reset_peak(self):
        self.peakhold_db = -120.0

    def _handle_logging(self, on, now, state):
        if on and self.log_fp is None:
            name = "beephunter_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            self.log_path = HERE / name
            self.log_fp = open(self.log_path, "w", encoding="utf-8")
            self.log_fp.write("iso_time,freq_hz,strength_db,smoothed_db,"
                              "peakhold_db,cadence_hz,cadence_conf,locked\n")
        if not on and self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None
        if on and self.log_fp is not None and now - self.last_log_t >= 0.1:
            self.last_log_t = now
            self.log_fp.write(
                f"{datetime.now().isoformat()},{state['freq']},"
                f"{state['strength_db']},{state['smoothed_db']},"
                f"{state['peakhold_db']},{state['cadence_hz']},"
                f"{state['cadence_conf']},{int(state['locked'])}\n")
            self.log_fp.flush()


def make_handler(dsp: DSP, cfg: Config, engine: AudioEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # quiet

        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (HERE / "dashboard.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
            elif self.path == "/data":
                self._send(200, json.dumps(dsp.get_state()))
            elif self.path == "/devices":
                self._send(200, json.dumps(list_input_devices()))
            else:
                self._send(404, "{}")

        def do_POST(self):
            if self.path != "/control":
                self._send(404, "{}")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, '{"error":"bad json"}')
                return
            action = body.get("action")
            with cfg.lock:
                if action == "set_search":
                    cfg.search_lo = float(body["lo"])
                    cfg.search_hi = float(body["hi"])
                elif action == "set_bw":
                    cfg.bandwidth = max(5.0, float(body["bw"]))
                elif action == "set_freq":
                    cfg.manual_freq = float(body["freq"])
                elif action == "auto":
                    cfg.manual_freq = None
                elif action == "nudge":
                    base = cfg.manual_freq if cfg.manual_freq else dsp.get_state().get("freq", 0)
                    cfg.manual_freq = max(20.0, base + float(body["delta"]))
                elif action == "set_logging":
                    cfg.logging = bool(body["on"])
            if action == "reset_peak":
                dsp.reset_peak()
            if action == "set_device":
                try:
                    engine.restart(int(body["index"]))
                except Exception as e:
                    self._send(500, json.dumps({"error": str(e)}))
                    return
            self._send(200, '{"ok":true}')

    return Handler


def main():
    ap = argparse.ArgumentParser(description="BeepHunter live beep locator")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--samplerate", type=int, default=SAMPLE_RATE)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    device = args.device if args.device is not None else sd.default.device[0]
    print("Input devices:")
    for d in list_input_devices():
        mark = " (default)" if d["default"] else ""
        print(f"  [{d['index']}] {d['name']}{mark}")
    print(f"Using device index: {device}")

    engine = AudioEngine(device, samplerate=args.samplerate)
    engine.start()
    cfg = Config()
    dsp = DSP(engine, cfg)
    threading.Thread(target=dsp.run, daemon=True).start()

    handler = make_handler(dsp, cfg, engine)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"\nBeepHunter dashboard: {url}\n(Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dsp.running = False
        if dsp.log_fp:
            dsp.log_fp.close()
        engine.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
