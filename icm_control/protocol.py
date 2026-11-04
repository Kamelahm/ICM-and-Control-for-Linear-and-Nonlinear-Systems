"""
Experimental protocol shared by all experiments (revision, version (b)).

* Independent one-step experiments (Assumptions 4 and 9): states are drawn
  uniformly from a box fixed before the data are seen, not along a trajectory.
* The operating region X in (12g) is that same fixed box, as the scenario
  lemma requires.
* Regularized objective J_M + REG * ||vartheta||^2 (scenario lemma).
* Linear plant: input bound u_max imposed in the LP, (31f).
"""
X_RADIUS_LIN = 1.0      # linear plant: x_k ~ U[-1, 1]^3  (= the safe set)
X_RADIUS_PEN = 0.5      # pendulum:     x_k ~ U[-0.5, 0.5]^2
REG = 1e-6              # epsilon_r of the scenario lemma (anchored at the prior centers)
U_MAX_LIN = 1.0         # linear plant input bound, twice the excitation amplitude
U_MAX_PEN = 5.0         # pendulum torque bound (N m) used for Fig. 2
