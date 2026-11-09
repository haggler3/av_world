# Prediction for avlab/frictionsense.py (follow-up, written before the run)

Half-AV Oakland, 8 copies, friction patches in time (1.0 -> 0.45 -> 0.7 -> 0.3 -> 1.0, 50 s ramps). An estimator pools the fleet's hardest
braking over 30 s and reads mu = decel / 9 when 3 or more events sit within 3 % of the maximum.
* R1: on slick road (mu < 0.6) the estimate is identified > 80 % of the time and within 0.1 of the truth in > 80 % of those seconds.
* R2: on dry road (mu >= 0.9) it is identified < 20 % of the time (cars rarely need 9 m/s^2), so the AV only has a lower bound there.
* R3: one AV alone identifies the floor in < 30 % of slick seconds; ~30 AVs are needed for > 80 %.
