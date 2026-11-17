---
name: daily-research
description: The daily av_world run. Makes one foundational change to the repository, tests it with one pre-registered experiment, writes it up in plain English and math, and merges it into main automatically when the experiment succeeds and every check passes. Use when the scheduled daily routine fires or when asked to "run the daily".
---

# Daily research run for av_world

You have about 75 minutes of wall-clock time on 4 CPU cores. You work alone and nobody is watching.
Each day has one job: **make the repository more capable, then prove the change works with one experiment.**
A day that only re-runs an old analysis on new random seeds is a failed day.

## 0. Set up (5 min)

1. Clone `main` fresh. Install with `pip install -q -e ".[dev,easy,notebooks,perception,web]"`.
2. Run `av-world doctor --smoke`. It must print `READY`. If not, run `av-world doctor --fix --extras easy,notebooks,perception` once.
   If it still fails, stop, write down exactly what failed, and send the notification.
3. Use the interpreter the doctor names (often `python3.13`) for every command.
4. Read `README.md`, `docs/ROADMAP.md`, the top of `docs/DAILY.md` (the "Foundations backlog" and "What to try next" lists),
   and `git log --oneline -15`.
5. Start the health check in the background:
   `python3.13 -m pytest -q -x tests/test_ideas.py tests/test_easy.py tests/test_examples.py > health.log 2>&1 &`.
   If a test that passed before now fails, fixing it is today's change.

## 1. Pick today's foundational change (10 min)

A foundational change adds something the repository could not do yesterday. It is used by later days. Good kinds:

| kind | examples |
|---|---|
| simulator feature | friction that varies by street, rain and night visibility, a new road user (scooter, bus), occlusion, a sensor-noise model, lane-level merges, signal timing plans |
| speed | a faster step, fewer recompiles, a cheaper crash check, batched scenarios |
| new testing tool | a scenario generator, a falsification search, a new AVBench scenario with a fixed pass bar, a replay or diff tool |
| learning | a new baseline policy, a training recipe that is cheaper or more stable, an evaluation that ranks policies with intervals |
| data and maps | a new city, an import path for a public dataset, calibration against a new crash source |
| usability | a one-line API for a common task, a notebook that teaches one idea, clearer errors |

Write 8 to 12 candidates. Do a 1-2 query web search for recent methods on the best few and cite what you use.
For each candidate write: one sentence, the files it touches, build time in minutes $b_i$, experiment time $e_i$,
and your prior $p_i \in [0,1]$ that it works and gives a clear result.

Score each candidate:

$$Q_i = \frac{U_i \cdot R_i \cdot p_i}{\sqrt{b_i + e_i}}$$

where $U_i \in [1,5]$ is how much later work it unlocks, and $R_i \in [1,5]$ is how much a reader would care.
Choose with three rounds of
$$i^* = \arg\max_i \; Q_i + 1.2 \, \tilde p_i \, \frac{\sqrt{N}}{1 + n_i},$$
where $\tilde p_i = p_i / \sum_j p_j$, $n_i$ is the number of cheap checks already spent on candidate $i$, and $N = \sum_j n_j$.
A cheap check is a grep that the hook exists, a 5-line script, or a 60-second quick run. Update $p_i$ after each check.

Drop any candidate that needs a GPU, needs more than 40 minutes in total ($b_i + e_i > 40$), or that a check shows does nothing.
Keep **one** foundational change. Save candidates, scores and checks to `results/daily/<YYYY-MM-DD>/search.json`.

## 2. Build it (25 min)

- Work on a branch: `git checkout -b daily/<YYYY-MM-DD>`.
- Put new code where it belongs (`xnet/` for the simulator, `avrl/` for learning, `avlab/` for analysis, `av_world/` for the API).
- **Changes to `xnet/` are allowed** only under these rules:
  1. The new behaviour is off by default (a field in `Behaviour`, an argument, or a flag).
  2. With it off, results do not change: add a regression test showing the per-copy metrics for a fixed seed are identical
     before and after the change.
  3. Speed does not get worse: time 200 steps of 16 Oakland copies before and after. Let $t_0$ and $t_1$ be the median of 5 runs.
     Require $t_1 / t_0 \le 1.05$.
- Add fast tests (under a minute) for every new function. Add a dependency only if needed, and then also to `pyproject.toml`,
  `requirements-lock.txt` and `av_world/doctor.py`'s `PACKAGES`.
- Use `--quick` modes first. Run at most 2 simulations at once, and none longer than 12 minutes.

## 3. Test it with one experiment (20 min)

The experiment shows the new capability doing something real. Before running it, write in
`results/daily/<date>/predictions.md`:

- **H1**, the one primary hypothesis, as a number with a direction. Example: "with patchy ice, the AV crash rate ratio
  $\mathrm{RR} = \lambda_{\text{new}} / \lambda_{\text{old}}$ is below 1".
- The test and its bar. For crash counts use the exact conditional Poisson test: given $n = k_1 + k_2$ crashes in exposures
  $t_1, t_2$, under $\mathrm{RR} = 1$ we have $k_1 \sim \mathrm{Binomial}(n,\, t_1/(t_1+t_2))$. The bar is two-sided $p < 0.05$.
- The sample size, chosen so the power is at least 0.8 for the effect you expect. For a rate ratio $r$ with about $k$ crashes
  per arm, the standard error of $\log r$ is about $\sqrt{2/k}$, so you need $k \gtrsim 2\,(1.96 + 0.84)^2 / (\log r)^2$.
  If that does not fit in the time, pick a larger expected effect or a different question. Do not run an underpowered test.
- Secondary checks, if any, with a Holm correction: sort the $m$ p-values $p_{(1)} \le \dots \le p_{(m)}$ and reject
  $p_{(j)}$ only if $p_{(i)} \le 0.05/(m - i + 1)$ for every $i \le j$.

Run it in the background with a log file. Report every interval as a 95 % interval. Check every number in the prose against
the saved results file.

**The experiment is successful when** H1 meets its bar as written in advance, the result file reproduces the numbers in the
write-up, and no check in step 2 failed. Anything else is not successful. Say so plainly and keep the result.

## 4. Show it (10 min)

- One GIF (under 1.5 MB, 10 fps, title and legend) that shows the new capability, using `av_world.watch`,
  `xnet.render.animate` or a zoomed `avlab.blackbox` animation.
- One or two figures: background `#0d1117`, one colour-blind-safe palette, labels on the lines instead of a legend box,
  no clipped text. Open every image with the Read tool before committing.
- Add a dated entry at the top of `docs/DAILY.md` (format in section 6). Update the "Foundations backlog" and
  "What to try next" lists there.
- Update `README.md` where today's change belongs: the guide table, the "What we found" table, or the API section.
  Remove any claim the change makes stale.
- Add a card to `docs/index.html` and check with a Playwright screenshot that a 360 px wide page has no sideways scroll.

## 5. Check, merge, publish (5 min)

1. Run the test files you touched plus `tests/test_xnet.py` and `tests/test_env.py`.
2. Commit with a message that says what was added and what the experiment found. No model names.
3. Push the branch `daily/<date>` and open a pull request into `main`. The PR body is the DAILY entry.
4. **Merge rule.** Squash-merge the pull request yourself, with the GitHub merge tool, if and only if all of these hold:
   - every test you ran passes, and the health check from step 0 still passes;
   - the default-off regression test and the speed check passed (when `xnet/` changed);
   - the experiment is successful as defined in step 3.
   Otherwise leave the pull request open, add the label `needs-review` if possible, and say in the notification
   which condition failed.
5. Update the one living artifact page "av_world daily" (find it with the Artifact list action; update its URL in place and
   keep the dated history). Put the newest day at the top.
6. Send one notification (see section 7).

If time runs short, skip to step 5 with what you have. Never merge something unfinished.

## 6. How to write

Write for a smart reader who has not seen the code. Use plain English and math.

- Short sentences. One idea per sentence. Active voice.
- Define every symbol the first time you use it. Write formulas in math, for example
  $\lambda = k / t$ (crashes per vehicle-hour), not in words.
- Give numbers with units and 95 % intervals: "0.81 (0.58 to 1.12)".
- Say what happened, then what it means, then what it does not show.
- Do not use: "honest", "notably", "it is worth noting", "importantly", "delve", "robust" (unless it is a statistical term you
  define), "leverage", "showcase", dashes used as asides, "not X but Y" sentences, or a closing summary that repeats the text.
- Pseudocode must describe what the code actually does.

The DAILY entry has exactly these parts:

```
## <date> - <what was added, in under 10 words>
What changed: 2-4 sentences. Files added or changed.
Why: 1-2 sentences, with the citation.
How it works: pseudocode or equations.
Experiment: H1, the test, the sample size and power, written before the run.
Result: a table, then 2-3 sentences. Success or not, and why.
Limits: 1-3 sentences.
Next: 1-3 items.
```

## 7. The notification and the final reply

One notification, plain text, in this order:

```
<date>: added <capability>. Experiment <succeeded|failed>: <H1 result with interval and p>.
Merged into main | PR #<n> left open because <condition>.
Tests: <passed>/<run>. Page: <artifact link>
```

The final reply in the session uses the same lines, plus at most three bullet points on what went wrong.
Do not add a sign-off, a summary of the summary, or offers of more help.

## Always

- Report what happened. If a test or an experiment fails, say so and show the output.
- Never write credentials to files. Never start a GPU job. Do not delete files under `results/` that a doc links to.
- Do not slow the simulator down, and do not change its default behaviour.
