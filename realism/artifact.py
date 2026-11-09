"""Build the published page docs/artifact_realism/index.html (house style of docs/artifact_autotag) from the figures in docs/img/realism."""
import json
import os
import re
import shutil

OUT = "docs/artifact_realism"


def pc(pros, cons):
    li = lambda xs: "".join(f"<li>{x}</li>" for x in xs)
    return f'<div class="pc"><div class="pro"><h4>Pros</h4><ul>{li(pros)}</ul></div><div class="con"><h4>Cons</h4><ul>{li(cons)}</ul></div></div>'


def fig(src, alt, cap, gif=False):
    cls = ' class="gif"' if gif else ""
    return f'<figure{cls}><div class="card"><img src="img/{src}" alt="{alt}" loading="lazy"></div><figcaption>{cap}</figcaption></figure>'


def method(n, title, stage, claim, code, pros, cons, extra=""):
    return f'''<section aria-labelledby="m{n}">
  <div class="head"><span class="idea">{n:02d}</span><h2 id="m{n}">{title}</h2><span class="stage">{stage}</span></div>
  <p class="claim">{claim}</p>
  <pre class="math">{code}</pre>
  {extra}
  {pc(pros, cons)}
</section>'''


def build():
    r = json.load(open("results/realism/scoreboard.json"))
    dec = r["decomposition"]
    sbp = r["scoreboard"]["patched"]["clean"]["tails"]
    disc = r["discriminator"]
    rp = r["replay"]
    pre = r["preset"]
    bank = r["following"]["bank"]
    rows = []
    labels = ["as logged: stored drivers against the real tracker", "sim pushed through the real tracker", "both cleaned (driving only)",
              "+ reweighted to the real scene mix", f"+ real traffic density (demand ×{pre['demand']:.1f} alone)",
              "<b>+ IDM+, headway bank, launches, OU residual, pulses</b>"]
    for i, (d, lab) in enumerate(zip(dec, labels)):
        hi = ' class="hi"' if i == len(dec) - 1 else ""
        g = f"<b>{d['gap']:.2f}</b>" if hi else f"{d['gap']:.2f}"
        rows.append(f'<tr{hi}><td>{lab}</td><td class="n">{g}</td>' + "".join(f'<td class="n">×{d[k]:.2f}</td>' for k in
                                                                              ("hard_brakes_per_h", "tight_share", "ttc2_share", "ttc1_share")) + "</tr>")
    ci = lambda k: f"[{sbp[k]['ratio_ci'][0]:.2f}, {sbp[k]['ratio_ci'][1]:.2f}]"
    tail_fit_gap = r["scoreboard"]["tail_fit"]["clean"]["gap"]
    hist = " → ".join(f"{h['mean_abs_quantile_err_s']:.3f}" for h in bank["history"])
    T = bank["T_p10_50_90"]

    body = f'''<main>
<header>
  <div class="eyebrow">av_world · closing the sim-to-real gap against 400 real Argoverse 2 scenes</div>
  <h1>Closing the Sim-to-Real Gap</h1>
  <p class="lede">Every earlier comparison said the simulator's drivers are too gentle. Before patching anything, this splits the difference into the three places it can come from: the <b>sensor</b> (real speeds come from a tracker), the <b>scene</b> (field of view and traffic density) and the <b>drivers</b>. Only the last is worth patching, and it is patched with mechanisms fitted to the logs.</p>
</header>

<section aria-labelledby="results">
  <h2 id="results">The gap, step by step</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Step</th><th class="n">gap</th><th class="n">hard brakes</th><th class="n">headway &lt; 0.8 s</th><th class="n">TTC &lt; 2 s</th><th class="n">TTC &lt; 1 s</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
  <p>gap = mean |log sim/real| of four tail rates (capped at log 20) + mean Wasserstein-1 of seven distributions in real IQR units; 0 = indistinguishable by these statistics. The final row's 95 % bootstrap CIs are hard brakes {ci("hard_brakes_per_h")}, tight following {ci("tight_share")}, TTC &lt; 2 s {ci("ttc2_share")} and TTC &lt; 1 s {ci("ttc1_share")}. The earlier tail fit (<code>avlab.tail_calibrate</code>) reaches {tail_fit_gap:.2f}.</p>
  {fig("waterfall.png", "Bar chart of the gap at each step, and tail-rate ratios along the steps", "The patch is the only step that moves tight following and TTC under 1 s. The sensor steps mostly move hard braking.")}
  {fig("following.gif", "Animated top-down views of a real scene, a stored-driver scene and a patched scene; cars coloured by time headway", "Cars coloured by time headway (red 0.4 s, green 3 s). The stored drivers keep 2 to 3 s everywhere. Real and patched drivers both have red followers.", gif=True)}
  <p class="note"><b>Still different.</b> A classifier tells real from simulated vehicles at AUROC {disc['patched']['auroc']:.3f} (stored drivers {disc['default']['auroc']:.3f}); its clues moved from traffic density to <b>jerk and the acceleration tail</b>. TTC under 2 s overshoots while TTC under 1 s undershoots. Standstill gaps came out smaller than real. Log replay barely improves: the 5 s error is {rp['default']['5s']['sim_m']:.1f} → {rp['patched']['5s']['sim_m']:.1f} m, against {rp['default']['5s']['const_velocity_m']:.1f} m for constant velocity.</p>
</section>

<section aria-labelledby="fp">
  <h2 id="fp">First principles</h2>
  <ol class="principles">
    <li><b>Three sources of a gap.</b> Sensor, scene, drivers. Patch drivers only after removing the other two, or the fit learns the tracker.</li>
    <li><b>The sensor can be fitted from the logs.</b> With y = v + e, an OU driver acceleration and an AR(1) tracker error, the velocity structure function D(h) = 2 s<sub>a</sub>² τ² (h/τ − 1 + e<sup>−h/τ</sup>) + 2 s<sub>m</sub>² (1 − ρ<sup>k</sup>) separates the smooth driver (∝ h²) from the jumpy sensor. The AV's own track comes from localisation and is the control. It shows almost no sensor term ({r['tracker_noise']['ego']['sm']:.2f} against {r['tracker_noise']['other']['sm']:.2f} m/s).</li>
    <li><b>Parity both ways.</b> Compare clean(real) with clean(corrupt(sim)): identical processing on both sides. An RTS smoother under the fitted model cleans; the same model corrupts.</li>
    <li><b>Car following is structural before it is parametric.</b> IDM's equilibrium gap is s*/√(1 − (v/v0)<sup>4</sup>), which grows as followers near their desired speed. IDM+ uses min(free, interaction), so the gap is exactly s0 + vT.</li>
    <li><b>Deconvolve through the simulator.</b> A driver's T shows in a log only blurred. Use the simulator as the measurement model and shift T's quantiles by real − simulated until they agree.</li>
    <li><b>A smooth residual needs no state.</b> Random Fourier features with Cauchy frequencies have exactly the OU covariance (Bochner). Hashed from the driver id, they keep common random numbers.</li>
    <li><b>Ask a classifier.</b> AUROC of real-vs-sim is the honest single number; permutation importance says what to fix next.</li>
  </ol>
</section>

<section aria-labelledby="dag">
  <h2 id="dag">The engineering DAG</h2>
  <pre class="math">real_logs ─► tracker_noise ─► real_clean ──────────────────────────────────────┐
                   │                 │                                          │
                   │                 └─► <b>headway_bank</b>  (deconvolution: 3 simulations) │
                   │                          │                                 │
                   ├──────────────────────────┴─► <b>refit</b>  (CEM: 4 × 8 city copies)    │
                   │                                  │                         │
                   │                                  └─► sim_patched (+ dense) │
sim_worlds (stored | tail_fit | IDM+) ──────────────────────┐                   │
                   └────────────────────────────────► processed (raw | sensor | clean)
                                                            ├─► mix ─► scoreboard ◄┘
                                                            ├─► following_check
                                                            ├─► discriminator
                      refit, headway_bank ──────────────────► replay ─► report</pre>
  <p>Each stage is a pure function of its inputs and config, cached as <code>results/realism/cache/&lt;stage&gt;-&lt;sha256(name, config, input keys, source)&gt;.pkl</code> by the auto-tagger's runner. Changing the refit's search box reruns the refit and everything below it, and never re-reads the logs.</p>
  {pc(["Every number traces to one cached stage", "Simulations inside stages run once", "Quick mode has its own cache"],
      ["Whole scene lists are pickled (a few hundred MB)", "A module change outside a stage's own function needs a config version bump"])}
</section>

{method(1, "Distribution scoreboard", "stats.py · scoreboard",
        "One extractor runs on real and simulated scenes. It produces four tail rates and seven distributions, with a bootstrap over scenes.",
        """measure(scene): per moving non-ego vehicle-frame: v, a, a1, jerk, headway, 1/TTC; standstill gaps; tail counts
scoreboard(R, S):
    tails  sim/real with CI (resample scenes on both sides)
    dists  W1(real, sim) / IQR(real)
    <b>gap</b> = mean(min(|log ratio|, log 20)) + mean(W1)""",
        ["One number to minimise, one table to read", "The CIs say which differences are real"],
        ["A fixed statistic list: lateral behaviour and gap acceptance are not on it"])}

{method(2, "Where", "stats.where_table", "The same comparison per context cell: speed bin × leader × turning.",
        """for speed in (1-5, 5-10, 10-15, 15+) × leader × turning:
    acceleration W1;  share of headway &lt; 0.8 s, sim/real;  headway W1""",
        ["Shows tight following missing in <i>every</i> bin for the stored drivers"],
        ["No junction or lane-type labels on real scenes; turning stands in"],
        fig("where.png", "Per-context bars of acceleration W1 and tight-following ratio for stored and patched drivers",
            "Stored drivers: tight following at 0.1 to 0.3 of real in every speed bin. Patched: around 1."))}

{method(3, "Measurement parity", "measure.py · tracker_noise, real_clean, processed",
        f"{100 * r['tracker_noise']['other']['spiky_track_share']:.0f} % of real tracks contain impossible speed jumps; the rest carry a correlated {r['tracker_noise']['other']['sm']:.2f} m/s error. Cleaning the logs halves their hard-braking rate.",
        """fit (s_m, ρ, s_a, τ) to D(1..15) of non-spiky tracks
clean(real)  = RTS(Hampel(y))  with  F = [[1, dt, 0], [0, e^(−dt/τ), 0], [0, 0, ρ]],  y = v + e
corrupt(sim) = v + AR1(s_m, ρ) + spikes(rate, empirical sizes);  positions + N(0, jitter)
crop(real)   = road users within 80 m of the ego (as simulated scenes are cut)""",
        ["Removes the part no driver model should chase", "Built-in control: the AV's own track", "Brackets the answer from both directions"],
        ["Stationary, object-independent noise; real noise grows with range and occlusion", "Smoothing also removes some real high-frequency driving"],
        fig("sensor.png", "Structure functions, acceleration histograms and tail ratios by processing mode",
            "Left: the fitted model separates driver from sensor. Middle: raw real accelerations have ±10 m/s² spikes. Right: through the real tracker the stored drivers over-brake, while tight following stays near ×0.1."))}

{method(4, "Scene-mix reweighting", "mix.py · mix",
        "A density ratio over scene context (speed, density, standing share, moving vehicles, turning) reweights simulated scenes toward the real mix.",
        """p = logistic(real vs sim | context)
w = clip(p / (1 − p) · n_sim / n_real, 0.05, 20), normalised;   ESS = (Σw)² / Σw²""",
        ["Separates a different city from different drivers without simulating"],
        [f"Only reweights contexts the simulator produces: ESS {r['mix']['default']['ess']:.0f} of {r['mix']['default']['n_sim']}. Real scenes have {r['mix']['default']['real_context_mean']['moving_vehicles']:.0f} moving vehicles within 80 m, simulated ones {r['mix']['default']['sim_context_mean']['moving_vehicles']:.0f}. That finding led to the demand knob."])}

{method(5, "Classifier two-sample test", "discriminator.py · discriminator",
        f"An MLP is trained to tell real vehicles from simulated ones, split by scene. AUROC {disc['default']['auroc']:.3f} for stored drivers, {disc['patched']['auroc']:.3f} patched.",
        """units = moving vehicles ≥ 5 s, 13 summary features
train on half the scenes (balanced), AUROC on the other half (bootstrap CI)
importance(f) = AUROC − AUROC(f shuffled)""",
        ["No statistic chosen in advance", "Names the next thing to fix: now jerk and the acceleration tail"],
        ["Wins on any difference, including harmless ones"],
        fig("discriminator.png", "AUROC per world and permutation importances",
            f"Density stopped being the main clue after the patch. Jerk (real {disc['patched']['real_mean']['jerk_rms']:.1f} against {disc['patched']['sim_mean']['jerk_rms']:.1f} m/s³) and the 98th-percentile acceleration (real {disc['patched']['real_mean']['a_p98']:.1f} against {disc['patched']['sim_mean']['a_p98']:.1f} m/s²) remain."))}

{method(6, "Car-following fits", "following.py · following_check",
        "Fit IDM per real leader-follower episode. The honest result is negative: the parameters are not identifiable from 4 to 10 s of urban driving.",
        f"""episode: one follower, one leader, ≥ 4 s
least squares of IDM on a = dv/dt  →  (T, s0, amax, b, v0), R²
check on perfect simulated data, where T is known:  Spearman(T_fit, T_true) = {r['following']['episode_fit_check'].get('T_spearman_vs_truth', float('nan')):.2f}""",
        ["Looks straight at the mechanism", "The check against simulator truth caught the problem"],
        ["Not identifiable; parameters pile on their bounds. Hence method 7."])}

{method(7, "Headway bank by deconvolution, plus IDM+", "following.deconvolve · xnet.sim · headway_bank",
        "Find the T distribution whose <i>simulated</i> closest-following matches the real one, with the simulator itself as the measurement model.",
        f"""r = per-episode p10 headway (clean real);   T_q = quantiles(r) − 0.8
repeat:  s = p10 headways of clean(corrupt(sim(IDM+, bank(T_q))));   T_q += quantiles(r) − quantiles(s)
quantile error [s]: {hist};   bank T at the 10/50/90th percentiles = {T[0]:.2f} / {T[1]:.2f} / {T[2]:.1f} s  (stored: 1.0–1.8 s)""",
        ["Every blur (unconstrained gaps, s0/v, the sensor) is inside the loop", "IDM+ removes the structural reason followers hang back"],
        ["Only T is deconvolved; s0, amax, b keep the simulator's spread"],
        fig("distributions.png", "Headway histograms, per-driver closest-following CDFs and 1/TTC histograms",
            "Left: stored drivers peak at 2.3 s, real and patched drivers spread down to 0.3 s. Middle: closest following per driver, the bank's target. Right: patched drivers close on leaders more often than real ones do, but less hard."))}

{method(8, "Smooth acceleration residual", "xnet.sim.ou_residual", "A per-driver residual correlated in time like real throttle modulation. White noise is not.",
        f"""e_i(t) = σ · √(2/K) Σ_k cos(w_ik t + φ_ik),   w ~ Cauchy(0, 1/τ),   φ ~ U(0, 2π),   hashed from the driver id
covariance  σ² exp(−|Δt| / τ)   (Bochner);   K = 8;   fitted: σ = {pre['ou_sigma']:.2f} m/s², τ = {pre['ou_tau']:.1f} s""",
        ["No State field", "Common-random-number safe", "Tested: unit variance, lag correlation e<sup>−1/τ</sup>"],
        ["Not exactly Gaussian at K = 8", "Ignores the scene: it does not grow near leaders"])}

{method(9, "Refit", "refit.py · refit", "A cross-entropy search over 8 knobs. Every candidate drives its own copy of the city in one vmapped run with common random numbers.",
        f"""knobs: demand, v0_scale, amax_scale, ou_σ, ou_τ, pulse_rate, bank share, s0_scale;  idm_plus = 1
4 generations × 8 candidates;  loss = scoreboard gap of clean(corrupt(sim)) against clean(real)
→ demand {pre['demand']:.1f}, v0 ×{pre['v0_scale']:.2f}, amax ×{pre['amax_scale']:.2f}, σ {pre['ou_sigma']:.2f}, τ {pre['ou_tau']:.1f}, pulses {pre['pulse_rate']:.4f}/s, bank {pre['bank']:.2f}, s0 ×{pre['s0_scale']:.2f}""",
        ["Fitted on exactly the scoreboard the report uses", "Candidates differ by knobs, not dice"],
        [f"Noisy at this budget: the winner scored {r['refit_loss']:.2f} on its own run and {dec[-1]['gap']:.2f} on a fresh one",
         "Version 1, without standstill gaps, bought tight following with s0 ×0.3; the statistic now rules that out, but s0 still drifts low"])}

{method(10, "Realism gates and log replay", "tests/test_realism.py · replay.py", "CI fails if the patched world stops beating the stored one. Replay checks prediction, not only statistics.",
        f"""gates: patched gap &lt; stored;  tight and TTC &lt; 1 s closer to real;  hard brakes within 2×;  AUROC drops
replay: seed {rp['scenes']} real scenes, 5 s rollouts with bank-drawn traits  →  {rp['default']['5s']['sim_m']:.1f} m stored, {rp['patched']['5s']['sim_m']:.1f} m patched, {rp['default']['5s']['const_velocity_m']:.1f} m constant velocity""",
        ["Regressions are caught; 18 tests in all", "Every new knob defaults off: stored results reproduce (all counts identical)"],
        ["The gates guard the last run's file, not a new run", "Replay is dominated by routes and signals, not following"])}

<section aria-labelledby="use">
  <h2 id="use">Use it</h2>
  <pre class="cmd">av-world realism run            # the cached DAG (~45 min the first time on 4 cores; --quick ~10 min)
av-world realism report         # the gap, step by step
av-world realism explain        # which stages are fresh</pre>
  <pre class="cmd">from av_world import easy
beh = easy.realism.preset()     # patched human drivers as an xnet Behaviour (also loads the trait bank)</pre>
  <footer>Code in <code>realism/</code> (stats, measure, mix, following, discriminator, refit, replay, simrec, pipeline, figures, api) and new opt-in knobs in <code>xnet/sim.py</code> (<code>idm_plus</code>, <code>bank</code>, <code>s0_scale</code>, <code>v0_scale</code>, <code>amax_scale</code>, <code>ou_sigma</code>, <code>ou_tau</code>). Write-up: <code>docs/SIM2REAL.md</code>. Tests: <code>tests/test_realism.py</code>.</footer>
</section>
</main>'''
    t = open("docs/artifact_autotag/index.html").read()
    style = re.search(r"<style>.*?</style>", t, re.S).group(0)
    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:ital,wght@0,500;0,600;1,400&display=swap">')
    html = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Closing the Sim-to-Real Gap</title>\n" + fonts + "\n" + style + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    os.makedirs(f"{OUT}/img", exist_ok=True)
    open(f"{OUT}/index.html", "w").write(html)
    for f in os.listdir("docs/img/realism"):
        shutil.copy(f"docs/img/realism/{f}", f"{OUT}/img/{f}")
    return len(html)


if __name__ == "__main__":
    print(build())
