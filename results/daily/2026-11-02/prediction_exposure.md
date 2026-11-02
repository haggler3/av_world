Prediction (written after a 300-step quick run showed step-15 RMSE TF 0.748 vs SF 0.682, before the full 1000-step x 3-seed run):
  teacher forcing (TF) has the lowest one-step RMSE; self-forcing (SF) has a step-15 RMSE at least 10 % below TF's, with the seed ranges not overlapping;
  TF+noise does not close that gap; SF-stop (no gradient through time) lands between TF and SF or equal to SF.
