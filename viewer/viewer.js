// av_world viewer: replays whole-area xnet simulations of Pittsburgh recorded by `python -m xnet.export_web`.
(() => {
  "use strict";
  const DATA = "data/";
  const $ = (s) => document.querySelector(s);
  const canvas = $("#map"), ctx = canvas.getContext("2d");
  const EV = {
    crash: { color: "#f0883e", name: "Crashes" }, red: { color: "#ff6b6b", name: "Red-light runs" },
    breakdown: { color: "#d2a8ff", name: "Breakdowns" }, bus: { color: "#f2cc60", name: "Bus stops" },
    van: { color: "#c9d1d9", name: "Double-parking" },
  };
  const HUMAN = ["#8b949e", "#a8b0ba", "#9c8f80", "#b5a58f", "#767f8a", "#c4c9cf", "#8a7f74"];  // muted, so the AVs stand out
  // the recording: raw bytes (.bin), or base64 text (.b64.txt) where a host does not serve binary files
  async function loadFrames(run) {
    const r = await fetch(DATA + run + ".bin");
    if (r.ok) return r.arrayBuffer();
    const t = (await (await fetch(DATA + run + ".b64.txt")).text()).trim();
    const bin = atob(t), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }
  const COL = { av: "#2f81f7", bus: "#f2cc60", van: "#e6edf3", wreck: "#f0883e", bike: "#3ddc84" };
  const LIGHT = ["#ff4d4d", "#ffb020", "#3ddc84"];
  const F = { AV: 1, BUS: 2, VAN: 4, WRECK: 8, STOP: 16, BROKEN: 32, RED: 64, DIST: 128 };

  const S = {
    index: [], run: null, map: null, frames: [], dt: 0.4, t: 0, playing: false, speed: 4,
    cam: { x: 0, y: 0, z: 0.25 }, view: "area", follow: -1, color: "type",
    filters: new Set(Object.keys(EV)), focus: null, paths: null, trail: [],
  };

  // ------------------------------------------------------------------ data
  async function boot() {
    S.index = await (await fetch(DATA + "index.json")).json();
    const sel = $("#run");
    for (const r of S.index) {
      const o = document.createElement("option");
      o.value = r.run; o.textContent = r.title; sel.appendChild(o);
    }
    sel.onchange = () => load(sel.value);
    const want = new URLSearchParams(location.search).get("run");
    await load(want && S.index.some((r) => r.run === want) ? want : S.index[0].run);
  }

  const mapCache = {};
  async function load(run) {
    $("#loading").style.display = "grid";
    S.playing = false; setPlay();
    const meta = await (await fetch(DATA + run + ".json")).json();
    if (!mapCache[meta.area]) mapCache[meta.area] = await (await fetch(DATA + meta.area + "/map.json")).json();
    const buf = await loadFrames(run);
    S.run = meta; S.map = mapCache[meta.area]; S.dt = meta.dt;
    S.frames = parse(buf);
    S.paths = buildPaths(S.map);
    S.t = 0; S.follow = -1; S.focus = null; S.trail = [];
    $("#blurb").textContent = meta.blurb;
    $("#run").value = run;
    const st = S.map.stats;
    $("#mapstats").textContent = `This area: ${st.junctions} junctions, ${st.signals} signals, ${st.stop_approaches} stop-sign approaches, ${st.crosswalks} crosswalks, ${st.km_of_lanes} km of lanes. Recording: ${(meta.frames * meta.dt / 60).toFixed(0)} simulated minutes after a warm-up.`;
    $("#scrub").max = S.frames.length - 1;
    setView("area");
    buildEvents();
    $("#loading").style.display = "none";
    S.playing = true; setPlay();
  }

  function parse(buf) {
    const dv = new DataView(buf), out = [];
    let o = 0;
    while (o < buf.byteLength) {
      const nv = dv.getUint16(o, true), ns = dv.getUint16(o + 2, true), np = dv.getUint16(o + 4, true);
      o += 6;
      const x = new Float32Array(nv), y = new Float32Array(nv), h = new Float32Array(nv), v = new Float32Array(nv);
      const f = new Uint8Array(nv), id = new Uint16Array(nv), k = new Uint8Array(nv);  // k: 0 car, 1 bus, 2 van, 3 bicycle
      for (let i = 0; i < nv; i++, o += 10) {
        x[i] = dv.getInt16(o, true) / 10; y[i] = dv.getInt16(o + 2, true) / 10;
        h[i] = dv.getUint8(o + 4) / 256 * 2 * Math.PI; v[i] = dv.getUint8(o + 5) / 4;
        f[i] = dv.getUint8(o + 6); k[i] = dv.getUint8(o + 7); id[i] = dv.getUint16(o + 8, true);
      }
      const sig = new Uint8Array(buf.slice(o, o + ns)); o += ns;
      const leg = new Uint16Array(np), p = new Float32Array(np);
      for (let i = 0; i < np; i++, o += 4) { leg[i] = dv.getUint16(o, true); p[i] = dv.getUint8(o + 2) / 255; }
      const byId = new Map();
      for (let i = 0; i < nv; i++) byId.set(id[i], i);
      out.push({ x, y, h, v, f, k, id, sig, leg, p, byId });
    }
    return out;
  }

  function buildPaths(m) {
    const road = new Path2D();
    for (const r of m.roads) {
      ring(road, r.outer);
      for (const hole of r.holes) ring(road, hole);
    }
    const centre = new Path2D(), div = new Path2D();
    for (const l of m.centre_lines) line(centre, l);
    for (const l of m.dividers) line(div, l);
    const stopLines = new Path2D();
    for (const s of m.stops.concat(m.signals)) line(stopLines, s.line);
    return { road, centre, div, stopLines };
  }
  function ring(p, a) { p.moveTo(a[0], -a[1]); for (let i = 2; i < a.length; i += 2) p.lineTo(a[i], -a[i + 1]); p.closePath(); }
  function line(p, a) { p.moveTo(a[0], -a[1]); for (let i = 2; i < a.length; i += 2) p.lineTo(a[i], -a[i + 1]); }

  // ------------------------------------------------------------------ camera
  function size() {
    const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    canvas.width = Math.round(r.width * d); canvas.height = Math.round(r.height * d);
    return { w: r.width, h: r.height, d };
  }
  function fitArea() {
    const b = S.map.bounds, r = canvas.getBoundingClientRect();
    S.cam = { x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2, z: Math.min(r.width / (b[2] - b[0]), r.height / (b[3] - b[1])) * 0.98 };
  }
  function setView(v) {
    S.view = v;
    document.querySelectorAll(".views button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
    if (v === "area") { S.follow = -1; fitArea(); }
    if (v === "street") { S.follow = -1; S.cam.z = 4.5; if (!S.focus) centreOnBusiest(); }
    if (v === "follow") {
      const fr = frame();
      const avs = [];
      for (let i = 0; i < fr.f.length; i++) if (fr.f[i] & F.AV && fr.v[i] > 3) avs.push(fr.id[i]);
      const pool = avs.length ? avs : Array.from(fr.id);
      S.follow = pool[Math.floor(Math.random() * pool.length)]; S.trail = []; S.cam.z = 5;
    }
  }
  function centreOnBusiest() {
    const fr = frame(), g = new Map();
    for (let i = 0; i < fr.x.length; i++) {
      const k = Math.round(fr.x[i] / 120) + "," + Math.round(fr.y[i] / 120);
      g.set(k, (g.get(k) || 0) + 1);
    }
    let best = null, n = -1;
    for (const [k, c] of g) if (c > n) { n = c; best = k; }
    const [i, j] = best.split(",").map(Number);
    S.cam.x = i * 120; S.cam.y = j * 120;
  }
  const frame = () => S.frames[Math.min(Math.floor(S.t), S.frames.length - 1)];

  // ------------------------------------------------------------------ draw
  function draw() {
    if (!S.frames.length) return;
    const { w, h, d } = size();
    const k = Math.floor(S.t), a = S.t - k;
    const f0 = S.frames[k], f1 = S.frames[Math.min(k + 1, S.frames.length - 1)];
    if (S.follow >= 0) {
      const i = f0.byId.get(S.follow);
      if (i === undefined) { S.follow = -1; } else {
        const p = lerpPos(f0, f1, i, a);
        S.cam.x += (p[0] - S.cam.x) * 0.25; S.cam.y += (p[1] - S.cam.y) * 0.25;
        if (!S.trail.length || Math.hypot(S.trail.at(-1)[0] - p[0], S.trail.at(-1)[1] - p[1]) > 2) S.trail.push(p);
        if (S.trail.length > 400) S.trail.shift();
      }
    }
    const z = S.cam.z;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.fillStyle = "#10151b"; ctx.fillRect(0, 0, w, h);
    ctx.setTransform(d * z, 0, 0, d * z, d * (w / 2 - S.cam.x * z), d * (h / 2 + S.cam.y * z));
    ctx.fillStyle = "#2b3038"; ctx.fill(S.paths.road, "evenodd");
    if (z > 0.9) {
      ctx.lineWidth = 0.3; ctx.strokeStyle = "rgba(227,179,65,0.85)"; ctx.stroke(S.paths.centre);
      ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(201,209,217,0.55)"; ctx.stroke(S.paths.div); ctx.setLineDash([]);
      ctx.lineWidth = 0.5; ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.stroke(S.paths.stopLines);
      crosswalks(f0);
      for (const s of S.map.stops) octagon(s.xy[0], -s.xy[1], 1.1);
    }
    signals(f0, z);
    if (S.trail.length > 1) {
      ctx.beginPath(); ctx.moveTo(S.trail[0][0], -S.trail[0][1]);
      for (const p of S.trail) ctx.lineTo(p[0], -p[1]);
      ctx.strokeStyle = "rgba(47,129,247,0.55)"; ctx.lineWidth = Math.max(0.8, 2 / z); ctx.stroke();
    }
    vehicles(f0, f1, a, z);
    if (z > 2.2) peds(f0);
    focusRing(z);
    labels(z, w, h);
    hud(f0);
  }

  function lerpPos(f0, f1, i, a) {
    const j = f1.byId.get(f0.id[i]);
    if (j === undefined) return [f0.x[i], f0.y[i]];
    const dx = f1.x[j] - f0.x[i], dy = f1.y[j] - f0.y[i];
    if (dx * dx + dy * dy > 400) return [f0.x[i], f0.y[i]];
    return [f0.x[i] + dx * a, f0.y[i] + dy * a];
  }

  function vehicles(f0, f1, a, z) {
    const big = z > 1.6;
    for (let i = 0; i < f0.x.length; i++) {
      const fl = f0.f[i];
      const [x, y] = lerpPos(f0, f1, i, a);
      let c;
      const bike = f0.k[i] === 3;
      if (S.color === "speed") c = speedColour(f0.v[i]);
      else c = fl & F.WRECK ? COL.wreck : fl & F.AV ? COL.av : fl & F.BUS ? COL.bus : fl & F.VAN ? COL.van : bike ? COL.bike : HUMAN[f0.id[i] % HUMAN.length];
      if (fl & F.WRECK) { ctx.fillStyle = "rgba(240,136,62,0.22)"; ctx.beginPath(); ctx.arc(x, -y, 7, 0, 7); ctx.fill(); c = COL.wreck; }
      if (!big) {
        ctx.fillStyle = c; ctx.beginPath();
        const r = Math.max(1.6 / z, fl & F.BUS ? 4 : bike ? 1.4 : 2.4);
        ctx.arc(x, -y, r, 0, 7); ctx.fill();
        continue;
      }
      const L = fl & F.BUS ? 12 : fl & F.VAN ? 7 : bike ? 1.8 : 4.6, W = fl & F.BUS ? 2.5 : fl & F.VAN ? 2.1 : bike ? 0.7 : 1.85;
      ctx.save(); ctx.translate(x, -y); ctx.rotate(-f0.h[i]);
      ctx.fillStyle = c; roundRect(-L / 2, -W / 2, L, W, 0.5); ctx.fill();
      ctx.fillStyle = "rgba(13,27,42,0.85)"; ctx.fillRect(L * 0.12, -W * 0.4, L * 0.16, W * 0.8);
      if (fl & (F.STOP | F.DIST | F.BROKEN | F.RED) || f0.id[i] === S.follow) {
        ctx.lineWidth = 0.35;
        ctx.strokeStyle = f0.id[i] === S.follow ? "#ffffff" : fl & (F.BROKEN | F.WRECK) ? "#ff4d4d" : fl & F.RED ? "#ff6b6b" : "#ffb020";
        roundRect(-L / 2 - 0.3, -W / 2 - 0.3, L + 0.6, W + 0.6, 0.6); ctx.stroke();
      }
      ctx.fillStyle = "#fff3b0"; ctx.fillRect(L / 2 - 0.25, -W * 0.38, 0.22, 0.28); ctx.fillRect(L / 2 - 0.25, W * 0.1, 0.22, 0.28);
      ctx.restore();
    }
  }
  function roundRect(x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function speedColour(v) {
    const t = Math.min(v / 15, 1); // stopped red -> amber -> green -> cyan
    const stops = [[255, 77, 77], [255, 176, 32], [61, 220, 132], [121, 192, 255]];
    const s = t * 3, i = Math.min(Math.floor(s), 2), u = s - i;
    const c = stops[i].map((q, j) => Math.round(q + (stops[i + 1][j] - q) * u));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function signals(fr, z) {
    const r = z > 1.6 ? 0.8 : Math.max(1.2 / z * 1.8, 2.5);
    S.map.signals.forEach((s, i) => {
      ctx.fillStyle = LIGHT[fr.sig[i] ?? 2]; ctx.beginPath(); ctx.arc(s.xy[0], -s.xy[1], r, 0, 7); ctx.fill();
    });
  }
  function octagon(x, y, r) {
    ctx.beginPath();
    for (let k = 0; k < 8; k++) { const t = Math.PI / 8 + k * Math.PI / 4; ctx.lineTo(x + r * Math.cos(t), y + r * Math.sin(t)); }
    ctx.closePath(); ctx.fillStyle = "#d1242f"; ctx.fill();
  }
  function crosswalks(fr) {
    ctx.strokeStyle = "rgba(201,209,217,0.32)"; ctx.lineWidth = 2.4; ctx.setLineDash([0.6, 0.7]);
    ctx.beginPath();
    for (const c of S.map.crosswalks) { if (!c[0] && !c[2]) continue; ctx.moveTo(c[0], -c[1]); ctx.lineTo(c[2], -c[3]); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  function peds(fr) {
    ctx.fillStyle = "#f778ba";
    for (let i = 0; i < fr.leg.length; i++) {
      const c = S.map.crosswalks[fr.leg[i]], p = fr.p[i], flip = fr.leg[i] % 2;
      const ax = flip ? c[2] : c[0], ay = flip ? c[3] : c[1], bx = flip ? c[0] : c[2], by = flip ? c[1] : c[3];
      ctx.beginPath(); ctx.arc(ax + (bx - ax) * p, -(ay + (by - ay) * p), 0.55, 0, 7); ctx.fill();
    }
  }
  function focusRing(z) {
    if (!S.focus) return;
    const age = S.t * S.dt - S.focus.t;
    if (age < -8 || age > 12) return;
    const pulse = 1 + 0.25 * Math.sin(performance.now() / 180);
    ctx.strokeStyle = EV[S.focus.type].color; ctx.lineWidth = 2.2 / z;
    ctx.beginPath(); ctx.arc(S.focus.x, -S.focus.y, 12 * pulse, 0, 7); ctx.stroke();
  }
  function labels(z, w, h) {
    if (z < 0.45) return;
    ctx.font = `500 ${Math.min(12, 3 + z * 2.2) / z}px Inter, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const l of S.map.labels) {
      if (z < 1.2 && l.rank < 3) continue;
      let ang = -l.angle; if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
      ctx.save(); ctx.translate(l.xy[0], -l.xy[1]); ctx.rotate(ang);
      ctx.fillStyle = "rgba(13,17,23,0.7)"; const tw = ctx.measureText(l.name).width;
      ctx.fillRect(-tw / 2 - 3 / z, -7 / z, tw + 6 / z, 14 / z);
      ctx.fillStyle = "#e6edf3"; ctx.fillText(l.name, 0, 0); ctx.restore();
    }
  }
  function hud(fr) {
    const tsec = S.t * S.dt;
    let av = 0, sp = 0, wreck = 0;
    for (let i = 0; i < fr.v.length; i++) { sp += fr.v[i]; if (fr.f[i] & F.AV) av++; if (fr.f[i] & F.WRECK) wreck++; }
    const crashes = S.run.events.filter((e) => e.type === "crash" && e.t <= tsec).length;
    $("#hud").innerHTML = [
      ["Vehicles", fr.v.length], ["AVs", av], ["Mean speed", (3.6 * sp / Math.max(fr.v.length, 1)).toFixed(1) + " km/h"],
      ["Pedestrians crossing", fr.leg.length], ["Crashes so far", crashes],
    ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
    $("#clock").textContent = fmt(tsec);
    if (document.activeElement !== $("#scrub")) $("#scrub").value = Math.floor(S.t);
  }
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  // ------------------------------------------------------------------ events
  function buildEvents() {
    const counts = {};
    for (const e of S.run.events) counts[e.type] = (counts[e.type] || 0) + 1;
    $("#filters").innerHTML = Object.entries(EV).map(([k, v]) =>
      `<button class="chip ${S.filters.has(k) ? "on" : ""}" data-k="${k}"><i style="background:${v.color}"></i>${v.name} · ${counts[k] || 0}</button>`).join("");
    $("#filters").querySelectorAll(".chip").forEach((b) => b.onclick = () => {
      const k = b.dataset.k; S.filters.has(k) ? S.filters.delete(k) : S.filters.add(k); buildEvents();
    });
    const list = S.run.events.filter((e) => S.filters.has(e.type));
    $("#events").innerHTML = list.map((e, i) =>
      `<li data-i="${i}"><time>${fmt(e.t)}</time><i style="background:${EV[e.type].color}"></i><span>${e.label}</span></li>`).join("")
      || `<li><span></span><span></span><span class="muted">No events of these kinds in this run.</span></li>`;
    $("#events").querySelectorAll("li[data-i]").forEach((li) => li.onclick = () => {
      $("#events").querySelectorAll("li").forEach((x) => x.classList.remove("on")); li.classList.add("on");
      jump(list[+li.dataset.i]);
    });
    const n = S.frames.length * S.dt;
    $("#ticks").innerHTML = list.map((e) => `<b style="left:${(e.t / n) * 100}%;background:${EV[e.type].color}"></b>`).join("");
  }
  function jump(e) {
    S.focus = e; S.follow = -1; S.view = "street";
    document.querySelectorAll(".views button").forEach((b) => b.classList.toggle("on", b.dataset.view === "street"));
    S.cam = { x: e.x, y: e.y, z: 6 };
    S.t = Math.max(0, (e.t - 6) / S.dt);
    S.playing = true; S.speed = 1; $("#speed").value = "1"; setPlay();
  }

  // ------------------------------------------------------------------ input
  function setPlay() { $("#play").textContent = S.playing ? "❚❚" : "▶"; $("#play").setAttribute("aria-label", S.playing ? "Pause" : "Play"); }
  $("#play").onclick = () => { S.playing = !S.playing; setPlay(); };
  $("#speed").onchange = (e) => { S.speed = +e.target.value; };
  $("#scrub").oninput = (e) => { S.t = +e.target.value; S.trail = []; };
  document.querySelectorAll(".views button").forEach((b) => b.onclick = () => setView(b.dataset.view));
  document.querySelectorAll("#legend .seg button").forEach((b) => b.onclick = () => {
    S.color = b.dataset.color;
    document.querySelectorAll("#legend .seg button").forEach((x) => x.classList.toggle("on", x === b)); legend();
  });
  function legend() {
    const items = S.color === "type"
      ? [["#8b949e", "human driver"], [COL.av, "AV (learned or rule-based)"], [COL.bus, "bus"], [COL.van, "delivery van"], [COL.bike, "cyclist"], [COL.wreck, "crash wreck"], ["#f778ba", "pedestrian"]]
      : [[speedColour(0), "stopped"], [speedColour(5), "18 km/h"], [speedColour(10), "36 km/h"], [speedColour(15), "54 km/h"]];
    $("#legend-items").innerHTML = items.map(([c, n]) => `<li><i style="background:${c}"></i>${n}</li>`).join("");
  }
  legend();
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { S.playing = !S.playing; setPlay(); e.preventDefault(); }
    if (e.code === "ArrowRight") S.t = Math.min(S.t + 10 / S.dt, S.frames.length - 1);
    if (e.code === "ArrowLeft") S.t = Math.max(S.t - 10 / S.dt, 0);
  });
  const ptrs = new Map();
  let pinch0 = null, moved = false;
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); moved = false; canvas.classList.add("drag"); });
  canvas.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) return;
    const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 1) {
      const dx = e.clientX - prev[0], dy = e.clientY - prev[1];
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      S.cam.x -= dx / S.cam.z; S.cam.y += dy / S.cam.z; if (moved) S.follow = -1;
    } else if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()], dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch0) S.cam.z = clampZ(S.cam.z * dist / pinch0); pinch0 = dist; moved = true;
    }
  });
  const up = (e) => {
    ptrs.delete(e.pointerId); pinch0 = null; canvas.classList.remove("drag");
    if (!moved && e.type === "pointerup") pick(e);
  };
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left - r.width / 2, my = e.clientY - r.top - r.height / 2;
    const wx = S.cam.x + mx / S.cam.z, wy = S.cam.y - my / S.cam.z;
    S.cam.z = clampZ(S.cam.z * Math.exp(-e.deltaY * 0.0015));
    S.cam.x = wx - mx / S.cam.z; S.cam.y = wy + my / S.cam.z;
  }, { passive: false });
  const clampZ = (z) => Math.min(Math.max(z, 0.08), 25);
  function pick(e) {
    const r = canvas.getBoundingClientRect();
    const wx = S.cam.x + (e.clientX - r.left - r.width / 2) / S.cam.z, wy = S.cam.y - (e.clientY - r.top - r.height / 2) / S.cam.z;
    const fr = frame();
    let best = -1, bd = (12 / S.cam.z) ** 2 + 9;
    for (let i = 0; i < fr.x.length; i++) { const d = (fr.x[i] - wx) ** 2 + (fr.y[i] - wy) ** 2; if (d < bd) { bd = d; best = i; } }
    if (best >= 0) {
      S.follow = fr.id[best]; S.trail = []; S.view = "follow"; if (S.cam.z < 3) S.cam.z = 5;
      document.querySelectorAll(".views button").forEach((b) => b.classList.toggle("on", b.dataset.view === "follow"));
    }
  }

  // ------------------------------------------------------------------ loop
  let last = performance.now();
  function loop(now) {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    if (S.playing && S.frames.length) {
      S.t += dt * S.speed / S.dt;
      if (S.t >= S.frames.length - 1) { S.t = 0; S.trail = []; }
    }
    draw();
    requestAnimationFrame(loop);
  }
  window.addEventListener("resize", () => { if (S.view === "area" && S.map) fitArea(); });
  boot().then(() => requestAnimationFrame(loop)).catch((err) => { $("#loading").textContent = "Could not load the simulation: " + err; });
})();
