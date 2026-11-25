# 2026-11-25 pre-registration (written before the full run)

Capability: `avlab.crashlog` names the other vehicle of every rear-end (partner slot, its AV flag, kind, speed, and whether it
was already a wreck). Experiment: `python -m avlab.rearpartner --run`; Oakland, 96 copies (3 batches of 32, keys from stream 51),
half AVs (IDM policy, envelope off), friction 0.4 everywhere, 120 s warm-up then 600 s.

**H1.** Among rear-end pairs with exactly one AV (the struck car was not already a wreck), the share $q$ in which the AV is the
*struck* car is above 0.5. Test: exact two-sided binomial test of $q = 0.5$; bar $p < 0.05$ with $q > 0.5$.
Expected $q \approx 0.8$ (the AV keeps a 1.3 s headway and brakes at the friction cap, a human behind it may not; real-world
AV collision reports show the same pattern). Sample size: the quick probe gave ~4 rear-end rows per 40 copy-minutes, so about
90 rear-ends and roughly 30-40 mixed pairs are expected; with $n = 30$ the power for $q = 0.8$ is about 0.97 and for $q = 0.7$
about 0.5. If $n < 20$ the test is declared underpowered and not successful.

**Secondary (Holm, m = 2).**
- S1: among rear-ends with a human striker, the share of AV leaders differs from the AV share of live vehicles (sampled at the
  end of each batch). Exact binomial test against that share.
- S2: for human strikers, the closing speed (striker speed minus struck speed, before contact) differs between AV leaders and
  human leaders (Mann-Whitney).

Validity check (not a hypothesis): every rear-end striker row names a partner (`rear_paired == rear_rows`).
