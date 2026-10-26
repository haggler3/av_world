"""Watch a city drive in your terminal: no window, no GPU, just truecolor text.

    av-world watch oakland                     # the whole area, Ctrl-C to stop
    av-world watch oakland --follow            # a 400 m window that follows one AV
    av-world watch oakland --frames 80 --gif docs/img/watch.gif     # record instead (for docs)

Each character cell is two pixels (the upper-half block with separate foreground and background colours), so the map has square-ish
pixels. Roads come from the same street geometry as the GIFs; vehicles are coloured by speed relative to the limit (green moving,
red stopped), AVs are blue, buses yellow, crashes orange, pedestrians pink, and signal heads show their colour.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

GROUND, ROAD, AV, BUS, WRECK, PED = (13, 17, 23), (48, 54, 61), (47, 129, 247), (242, 204, 96), (240, 136, 62), (247, 120, 186)
LIGHTS = np.array([(255, 77, 77), (255, 176, 32), (61, 220, 132)])


def _speed_colour(rel):
    """rel in [0, 1] (speed / limit) -> red (stopped) .. green (at the limit)."""
    rel = np.clip(rel, 0, 1)[:, None]
    return (np.array([235, 64, 52]) * (1 - rel) + np.array([92, 214, 92]) * rel).astype(int)


class View:
    def __init__(self, area, cols, rows, follow=False, seed=0, p_av=0.25, window_m=250.0):
        from xnet import cities
        from xnet import network as NW
        from xnet import render as R
        from xnet import sim as X
        self.X, self.R = X, R
        self.net, self.geo, self.info = cities.build(area) if area in cities.CITIES else NW.build(area)
        self.sim = X.Sim(self.net, n_envs=1, beh=X.Behaviour(p_av=p_av), seed=seed)
        self.tracks, self.heads = R.Tracks(self.net, self.geo), R.signal_heads(self.net, self.geo)
        self.W, self.H = cols, 2 * rows  # pixels
        self.road = R.Map(self.net, self.geo, self.info).road
        self.follow, self.window_m = follow, window_m
        self.ego = None
        pts = np.concatenate(self.geo.link_pts)
        self.full = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        self.t = 0.0
        self._bg_key, self._bg = None, None

    def _box(self, st):
        if not self.follow:
            x0, x1, y0, y1 = self.full
        else:
            a = np.nonzero(st.alive & st.av)[0]
            if self.ego is None or not st.alive[self.ego] or not st.av[self.ego]:
                self.ego = int(a[0]) if len(a) else None
            if self.ego is None:
                x0, x1, y0, y1 = self.full
            else:
                x, y, _ = self.tracks.pose(st.trk[[self.ego]], st.s[[self.ego]], st.length[[self.ego]])
                x0, x1, y0, y1 = x[0] - self.window_m / 2, x[0] + self.window_m / 2, y[0] - self.window_m / 2, y[0] + self.window_m / 2
        # fit the box to the pixel grid's aspect, then quantise it so the road raster can be cached between frames
        w, h = x1 - x0, y1 - y0
        m = max(w / self.W, h / self.H)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        q = m * 4
        cx, cy = round(cx / q) * q, round(cy / q) * q
        return cx - m * self.W / 2, cy + m * self.H / 2, m  # top-left corner (x, y) and metres per pixel

    def _background(self, box):
        if self._bg_key != box:
            import shapely
            x0, y0, m = box
            xs = x0 + (np.arange(self.W) + 0.5) * m
            ys = y0 - (np.arange(self.H) + 0.5) * m
            X_, Y_ = np.meshgrid(xs, ys)
            on = shapely.contains_xy(self.road, X_.ravel(), Y_.ravel()).reshape(self.H, self.W)
            for p in self.geo.link_pts:  # centrelines too, so streets narrower than a pixel still show at whole-city scale
                p = np.asarray(p)
                seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
                t = np.concatenate([[0], np.cumsum(seg)])
                ss = np.arange(0, t[-1], m / 2)
                px, py = np.interp(ss, t, p[:, 0]), np.interp(ss, t, p[:, 1])
                c, r = ((px - x0) / m).astype(int), ((y0 - py) / m).astype(int)
                ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
                on[r[ok], c[ok]] = True
            bg = np.empty((self.H, self.W, 3), int)
            bg[:] = GROUND
            bg[on] = ROAD
            self._bg_key, self._bg = box, bg
        return self._bg.copy()

    def frame(self, seconds=1.0):
        """Advance the city and return the picture [H, W, 3] plus a status line."""
        X = self.X
        frames = self.sim.run(seconds, record=(0, int(round(seconds / X.DT))))
        st = frames[-1]
        self.t += seconds
        box = self._box(st)
        img = self._background(box)
        x0, y0, m = box

        def put(x, y, col):
            c = ((np.asarray(x) - x0) / m).astype(int)
            r = ((y0 - np.asarray(y)) / m).astype(int)
            ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
            img[r[ok], c[ok]] = np.asarray(col)[ok] if np.ndim(col) == 2 else col

        a = np.nonzero(st.alive)[0]
        x, y, _ = self.tracks.pose(st.trk[a], st.s[a], st.length[a])
        vlim = self.net["link_v"][np.minimum(st.trk[a] // 2, len(self.net["link_v"]) - 1)]
        col = _speed_colour(st.v[a] / np.maximum(vlim, 1.0))
        col[st.kind[a] == X.BUS] = BUS
        col[st.av[a]] = AV
        col[st.wreck_t[a] > 0] = WRECK
        put(x, y, col)
        on = st.ped_t > 0
        if on.any():
            k = np.nonzero(on)[0]
            pa, pb = self.geo.leg_xy[k, 0], self.geo.leg_xy[k, 1]
            pp = pa + (pb - pa) * np.clip(1 - st.ped_t[k] / 11.0, 0.05, 0.95)[:, None]
            put(pp[:, 0], pp[:, 1], PED)
        sig = np.asarray(X.signal_state({k: np.asarray(v) for k, v in self.net.items()}, float(st.t)))
        hx, hy, li = self.heads
        put(hx, hy, LIGHTS[sig[li]])
        d = dict(zip(X.METRICS, np.asarray(self.sim.st.m)[0]))
        crashes = int(sum(d[k] for k in X.CRASH_NAMES))
        status = (f" t {self.t:6.0f} s   vehicles {len(a):4d}   AVs {int(st.av[a].sum()):3d}   mean speed {3.6 * float(st.v[a].mean() if len(a) else 0):4.1f} km/h"
                  f"   crashes {crashes}   {m:.1f} m/pixel{'   following AV ' + str(self.ego) if self.follow and self.ego is not None else ''}")
        return img, status


def to_ansi(img):
    """[2R, W, 3] -> R lines of upper-half blocks (foreground = top pixel, background = bottom pixel)."""
    out = []
    for r in range(0, img.shape[0] - 1, 2):
        top, bot = img[r], img[r + 1]
        out.append("".join(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀" for a, b in zip(top, bot)) + "\x1b[0m")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="av-world watch")
    ap.add_argument("area", nargs="?", default="oakland")
    ap.add_argument("--follow", action="store_true", help="follow one AV")
    ap.add_argument("--window", type=float, default=400.0, help="metres across the window when following")
    ap.add_argument("--speed", type=float, default=2.0, help="simulated seconds per frame")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--warm", type=float, default=90.0, help="simulated seconds before the first frame")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until Ctrl-C)")
    ap.add_argument("--gif", help="write the frames to a GIF instead of the terminal")
    ap.add_argument("--size", default=None, help="COLSxROWS (default: the terminal's size)")
    a = ap.parse_args(argv)
    cols, rows = (int(v) for v in a.size.split("x")) if a.size else (lambda s: (s.columns, s.lines - 2))(shutil.get_terminal_size((120, 40)))
    v = View(a.area, cols, rows, follow=a.follow, window_m=a.window)
    if a.warm:
        v.sim.run(a.warm)
        v.t = a.warm
    if a.gif:
        from PIL import Image, ImageDraw
        ims = []
        for _ in range(a.frames or 60):
            img, status = v.frame(a.speed)
            big = Image.fromarray(img.astype(np.uint8)).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST)
            canvas = Image.new("RGB", (big.width, big.height + 22), GROUND)
            canvas.paste(big, (0, 22))
            ImageDraw.Draw(canvas).text((6, 5), status.strip(), fill=(230, 237, 243))
            ims.append(canvas)
        ims[0].save(a.gif, save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
        print(a.gif)
        return 0
    sys.stdout.write("\x1b[?25l\x1b[2J")  # hide the cursor, clear the screen
    try:
        n = 0
        while not a.frames or n < a.frames:
            t0 = time.time()
            img, status = v.frame(a.speed)
            sys.stdout.write("\x1b[H" + to_ansi(img) + "\n\x1b[0m" + status.ljust(cols)[:cols] + "\n")
            sys.stdout.flush()
            n += 1
            time.sleep(max(0.0, 1.0 / a.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
