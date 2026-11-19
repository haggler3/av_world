/-
# When is the AV's braking barrier sound?  (machine-checked)

The AVs' safety filter in `xnet/sim.py` (`cbf_bound`, line 118, `SHIELD = "cbf"`) uses the barrier

    h = g - d0 - τ v - (v² - vl²) / (2 b)

with the gains `CBF = dict(alpha=1.0, tau=0.5, d0=2.0, b=6.0, ...)`, and keeps `ḣ + α h ≥ 0` by capping the
acceleration at

    a ≤ ((vl - v) + vl al / b + α h) / (τ + v / b).

`b = 6 m/s²` is an *assumed* braking ability, for the AV and for the obstacle ahead. The road allows at most
`B = 9 μ m/s²` (`a_min = -9 * friction`, `xnet/sim.py:247`). This file proves, over the reals:

* `bound_feasible`      : if B ≥ b, the cap is always above -B whenever h ≥ 0 (the filter can always be obeyed);
* `bound_infeasible`    : if B < b, every speed v > B b τ / (b - B) has a state on the barrier's edge (h = 0, obstacle
                          stopped) whose cap is below -B: the filter asks for braking the road cannot give;
* `stop_gap_sound`      : if the AV brakes at B_f ≥ b and the leader at B_l ≤ b, then h ≥ 0 implies the gap after both
                          stop is at least d0 (a stopping-distance certificate);
* `stop_gap_unsound`    : if B_f < b there is a state with h = 0 that ends in a collision;
* `friction_threshold`  : with b = 6, τ = 1/2, B = 9 μ: the cap is feasible on every state with h ≥ 0 iff μ ≥ 2/3;
* `vfeas_*`             : the speeds above which the filter is infeasible at μ = 0.6, 0.5, 0.4.
-/
import Mathlib

namespace AvFormal

/-- The barrier value of `xnet.sim.cbf_bound`. -/
noncomputable def barrier (d0 τ b g v vl : ℝ) : ℝ := g - d0 - τ * v - (v ^ 2 - vl ^ 2) / (2 * b)

/-- The acceleration cap of `xnet.sim.cbf_bound`, written in terms of the barrier value `h`. -/
noncomputable def cap (α τ b v vl al h : ℝ) : ℝ := ((vl - v) + vl * al / b + α * h) / (τ + v / b)

/-- If the road allows braking `B ≥ b`, the cap never asks for more than the road allows while h ≥ 0
(and the obstacle brakes no harder than b). -/
theorem bound_feasible {α τ b B v vl al h : ℝ} (hb : 0 < b) (hτ : 0 < τ) (hα : 0 ≤ α) (hv : 0 ≤ v)
    (hvl : 0 ≤ vl) (hal : -b ≤ al) (hh : 0 ≤ h) (hB : b ≤ B) : -B < cap α τ b v vl al h := by
  unfold cap
  have hD : 0 < τ + v / b := by positivity
  rw [lt_div_iff₀ hD]
  have h1 : -vl ≤ vl * al / b := by
    rw [le_div_iff₀ hb]; nlinarith
  have h2 : v ≤ B * (v / b) := by
    rw [mul_div_assoc']; rw [le_div_iff₀ hb]; nlinarith
  have h3 : 0 < B * τ := by nlinarith
  nlinarith [mul_nonneg hα hh]

/-- If the road allows only `B < b`, then above the speed `B b τ / (b - B)` the cap on the barrier's edge
(h = 0, obstacle stopped) is below -B: the filter demands braking the road cannot give. -/
theorem bound_infeasible {α τ b B v : ℝ} (hB0 : 0 < B) (hBb : B < b) (hτ : 0 < τ)
    (hv : B * b * τ / (b - B) < v) : cap α τ b v 0 0 0 < -B := by
  unfold cap
  have hb : 0 < b := by linarith
  have hbB : 0 < b - B := by linarith
  have hv0 : 0 < v := lt_trans (div_pos (by positivity) hbB) hv
  have hD : 0 < τ + v / b := by positivity
  rw [div_lt_iff₀ hD]
  rw [div_lt_iff₀ hbB] at hv
  have this : B * (v / b) * b = B * v := by field_simp
  have key : B * τ * b < b * v - B * v := by nlinarith
  have goal : (0 - v + 0 * 0 / b + α * 0) * b < (-B * (τ + v / b)) * b := by
    have e : (-B * (τ + v / b)) * b = -(B * τ * b) - B * v := by
      rw [show (-B * (τ + v / b)) * b = -(B * τ * b) - B * (v / b) * b by ring, this]
    have e2 : (0 - v + 0 * 0 / b + α * 0) * b = -(b * v) := by ring
    rw [e, e2]; linarith
  exact lt_of_mul_lt_mul_right goal hb.le

/-- Stopping-distance certificate: if the AV can brake at `Bf ≥ b` and the leader brakes at most `Bl ≤ b`,
then h ≥ 0 implies the gap left after both have stopped is at least d0. -/
theorem stop_gap_sound {d0 τ b Bf Bl g v vl : ℝ} (hBl : 0 < Bl) (hlb : Bl ≤ b) (hbf : b ≤ Bf) (hτ : 0 ≤ τ)
    (hv : 0 ≤ v) (hh : 0 ≤ barrier d0 τ b g v vl) :
    d0 ≤ g - v ^ 2 / (2 * Bf) + vl ^ 2 / (2 * Bl) := by
  unfold barrier at hh
  have hb : 0 < b := lt_of_lt_of_le hBl hlb
  have e1 : v ^ 2 / (2 * Bf) ≤ v ^ 2 / (2 * b) :=
    div_le_div_of_nonneg_left (by positivity) (by positivity) (by linarith)
  have e2 : vl ^ 2 / (2 * b) ≤ vl ^ 2 / (2 * Bl) :=
    div_le_div_of_nonneg_left (by positivity) (by positivity) (by linarith)
  have e3 : (v ^ 2 - vl ^ 2) / (2 * b) = v ^ 2 / (2 * b) - vl ^ 2 / (2 * b) := by ring
  nlinarith [mul_nonneg hτ hv]

/-- If the AV can only brake at `Bf < b`, some state on the barrier's edge (h = 0, leader stopped) ends in a
collision: the gap after stopping is negative. -/
theorem stop_gap_unsound {d0 τ b Bf : ℝ} (hBf : 0 < Bf) (hfb : Bf < b) (hd0 : 0 ≤ d0) (hτ : 0 ≤ τ) :
    ∃ g v : ℝ, 0 ≤ v ∧ barrier d0 τ b g v 0 = 0 ∧ g - v ^ 2 / (2 * Bf) + 0 ^ 2 / (2 * b) < 0 := by
  have hb : 0 < b := by linarith
  set k : ℝ := 1 / (2 * Bf) - 1 / (2 * b) with hk
  have kpos : 0 < k := by
    rw [hk, sub_pos]; exact one_div_lt_one_div_of_lt (by positivity) (by linarith)
  set v : ℝ := (d0 + τ + 1) / k + 1 with hvdef
  have hv1 : 1 ≤ v := by rw [hvdef]; have : 0 ≤ (d0 + τ + 1) / k := by positivity
                         linarith
  have hkv : d0 + τ + 1 ≤ k * v := by
    rw [hvdef, mul_add, mul_div_cancel₀ _ kpos.ne']; linarith
  refine ⟨d0 + τ * v + v ^ 2 / (2 * b), v, by linarith, ?_, ?_⟩
  · unfold barrier; ring
  · have e : d0 + τ * v + v ^ 2 / (2 * b) - v ^ 2 / (2 * Bf) + 0 ^ 2 / (2 * b) = d0 + τ * v - k * v ^ 2 := by
      rw [hk]; ring
    rw [e]
    have : k * v ^ 2 = (k * v) * v := by ring
    rw [this]
    nlinarith

/-- The simulator's numbers: b = 6, τ = 1/2, road braking 9 μ. The cap is feasible on every state with h ≥ 0
(any α ≥ 0, any obstacle braking no harder than b) iff μ ≥ 2/3. -/
theorem friction_threshold {μ : ℝ} (hμ : 0 < μ) :
    (∀ α v vl al h : ℝ, 0 ≤ α → 0 ≤ v → 0 ≤ vl → -6 ≤ al → 0 ≤ h → -(9 * μ) < cap α (1 / 2) 6 v vl al h)
      ↔ 2 / 3 ≤ μ := by
  constructor
  · intro H
    by_contra hlt
    push_neg at hlt
    have hB : 9 * μ < 6 := by linarith
    have hv := lt_add_one (9 * μ * 6 * (1 / 2) / (6 - 9 * μ))
    have := bound_infeasible (α := 0) (by linarith) hB (by norm_num : (0 : ℝ) < 1 / 2) hv
    have v0 : 0 ≤ 9 * μ * 6 * (1 / 2) / (6 - 9 * μ) := div_nonneg (by positivity) (by linarith)
    have := H 0 (9 * μ * 6 * (1 / 2) / (6 - 9 * μ) + 1) 0 0 0 le_rfl (by linarith) le_rfl (by norm_num) le_rfl
    linarith
  · intro hμ23 α v vl al h hα hv hvl hal hh
    exact bound_feasible (by norm_num) (by norm_num) hα hv hvl hal hh (by linarith)

/-- Critical speed of the filter, v_feas(μ) = 9μ · b · τ / (b - 9μ), at three frictions (m/s). -/
theorem vfeas_06 : (9 * (0.6 : ℝ)) * 6 * (1 / 2) / (6 - 9 * 0.6) = 27 := by norm_num
theorem vfeas_05 : (9 * (0.5 : ℝ)) * 6 * (1 / 2) / (6 - 9 * 0.5) = 9 := by norm_num
theorem vfeas_04 : (9 * (0.4 : ℝ)) * 6 * (1 / 2) / (6 - 9 * 0.4) = 4.5 := by norm_num

end AvFormal
