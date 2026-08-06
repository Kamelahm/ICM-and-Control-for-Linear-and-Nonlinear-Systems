"""
Benchmark systems and data generation.

Three plants, matching Section V of the paper:
  * first-order linear   (Sec. V-A)
  * third-order linear   (Sec. V-B, matrices from [42])
  * inverted pendulum    (Sec. V-C, Euler discretization as in [43])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .sets import Polytope, Zonotope


# --------------------------------------------------------------------------- #
#  Linear plants
# --------------------------------------------------------------------------- #
@dataclass
class LinearSystem:
    A: np.ndarray
    B: np.ndarray
    name: str = "linear"

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def m(self):
        return self.B.shape[1]

    def step(self, x, u, w):
        return self.A @ x + self.B @ u + w


def first_order_system():
    """Sec. V-A: unstable scalar plant."""
    return LinearSystem(np.array([[1.021]]), np.array([[0.041]]), name="scalar")


def third_order_system():
    """Sec. V-B: third-order plant from [42]."""
    A = np.array([[0.8, 0.1, 0.0],
                  [-0.2, 0.9, 0.1],
                  [0.1, 0.0, 0.7]])
    B = np.eye(3)
    return LinearSystem(A, B, name="third_order")


def safe_set_box(n, radius=1.0):
    """S_s = {x : H x <= h} with H = [I; -I]."""
    H = np.vstack([np.eye(n), -np.eye(n)])
    h = radius * np.ones(2 * n)
    return Polytope(H, h)


# --------------------------------------------------------------------------- #
#  Inverted pendulum (input-affine nonlinear, Sec. V-C)
# --------------------------------------------------------------------------- #
@dataclass
class Pendulum:
    Ts: float = 0.1
    m: float = 1.0
    l: float = 1.0
    g: float = 9.8
    mu: float = 0.01
    name: str = "pendulum"

    n: int = field(default=2, init=False)
    m_in: int = field(default=1, init=False)

    def step(self, x, u, w):
        x1, x2 = float(x[0]), float(x[1])
        u = float(np.atleast_1d(u)[0])
        x1p = x1 + self.Ts * x2
        x2p = (self.Ts * self.g / self.l) * np.sin(x1) \
            + (1.0 - self.Ts * self.mu / (self.m * self.l ** 2)) * x2 \
            + (self.Ts / (self.m * self.l ** 2)) * u
        return np.array([x1p, x2p]) + np.asarray(w, float).reshape(-1)

    # -- basis functions ----------------------------------------------------
    @staticmethod
    def basis_full(x):
        """Z(x) = [x1, x2, sin(x1) - x1, 1 - cos(x1)]^T  (paper, Sec. V-C)."""
        x1, x2 = float(x[0]), float(x[1])
        return np.array([x1, x2, np.sin(x1) - x1, 1.0 - np.cos(x1)])

    @staticmethod
    def basis_drop_cos(x):
        """Misspecified basis: the (1 - cos x1) term is removed (R5.6)."""
        x1, x2 = float(x[0]), float(x[1])
        return np.array([x1, x2, np.sin(x1) - x1])

    @staticmethod
    def basis_drop_sin(x):
        """Misspecified basis: the (sin x1 - x1) term is removed (R5.6)."""
        x1, x2 = float(x[0]), float(x[1])
        return np.array([x1, x2, 1.0 - np.cos(x1)])

    @staticmethod
    def basis_linear_only(x):
        """Severely misspecified basis: purely linear regressor."""
        return np.array([float(x[0]), float(x[1])])

    def true_matrices(self, basis: str = "full"):
        """Ground-truth (A_tr, B_tr) in the chosen basis, when exactly representable."""
        a = self.Ts * self.g / self.l
        d = 1.0 - self.Ts * self.mu / (self.m * self.l ** 2)
        b = self.Ts / (self.m * self.l ** 2)
        if basis == "full":
            # row 1: x1 + Ts x2 ;  row 2: a*(sin x1 - x1) + a*x1 + d*x2
            A = np.array([[1.0, self.Ts, 0.0, 0.0],
                          [a, d, a, 0.0]])
        elif basis == "drop_cos":
            A = np.array([[1.0, self.Ts, 0.0],
                          [a, d, a]])
        else:
            raise ValueError(f"no exact representation for basis '{basis}'")
        B = np.array([[0.0], [b]])
        return A, B


# --------------------------------------------------------------------------- #
#  Data collection
# --------------------------------------------------------------------------- #
def collect_data(system,
                 N: int,
                 x0: np.ndarray,
                 rng: np.random.Generator,
                 u_range: float = 0.5,
                 w_gen: Optional[np.ndarray] = None,
                 w_sampler: Optional[Callable] = None,
                 basis: Optional[Callable] = None,
                 reset_radius: Optional[float] = None):
    """
    Roll the plant forward with i.i.d. uniform inputs.

    Returns
    -------
    Phi : (p, N) regressor  (states, or Z(x) when `basis` is given)
    U   : (m, N) inputs
    Xp  : (n, N) successors
    Xs  : (n, N) raw states (always the physical state, even if basis is used)
    """
    x = np.asarray(x0, float).reshape(-1)
    n = x.shape[0]
    m = system.B.shape[1] if hasattr(system, "B") else system.m_in

    Xs, U, Xp = [], [], []
    for _ in range(N):
        u = rng.uniform(-u_range, u_range, size=m)
        if w_sampler is not None:
            w = w_sampler(rng)
        elif w_gen is not None:
            w = w_gen @ rng.uniform(-1, 1, size=w_gen.shape[1])
        else:
            w = np.zeros(n)
        xp = system.step(x, u, w)
        Xs.append(x.copy())
        U.append(u)
        Xp.append(xp)
        x = xp
        if reset_radius is not None and np.linalg.norm(x) > reset_radius:
            x = np.asarray(x0, float).reshape(-1)

    Xs = np.array(Xs).T
    U = np.array(U).T
    Xp = np.array(Xp).T
    Phi = Xs if basis is None else np.array([basis(Xs[:, k]) for k in range(N)]).T
    return Phi, U, Xp, Xs


# --------------------------------------------------------------------------- #
#  Physically motivated side information
# --------------------------------------------------------------------------- #
def linear_side_info(A_tr, B_tr, rel_err=0.15, bias=0.0, floor=0.01,
                     rng=None, bias_dir=None):
    """
    Prior sets A_side, B_side for a linear plant.

    rel_err : relative half-width of the prior around the (biased) centre
    bias    : centre offset as a fraction of |A_tr| -- with bias > rel_err the
              true system is NOT contained, which is the misspecified-prior
              case requested in R5.3 / R1.2
    """
    from .sets import structured_prior

    A_tr = np.atleast_2d(A_tr)
    B_tr = np.atleast_2d(B_tr)
    if bias_dir is None:
        bias_dir = np.ones_like(A_tr), np.ones_like(B_tr)
    dA = bias * (np.abs(A_tr) + floor) * bias_dir[0]
    dB = bias * (np.abs(B_tr) + floor) * bias_dir[1]
    RA = rel_err * (np.abs(A_tr) + floor)
    RB = rel_err * (np.abs(B_tr) + floor)
    A_side, GA = structured_prior(A_tr + dA, RA)
    B_side, GB = structured_prior(B_tr + dB, RB)
    return A_side, GA, B_side, GB


def pendulum_side_info(pen: Pendulum, basis: str = "full", rel_err=0.20,
                       bias=0.0, keep_structure=True):
    """
    Prior for the pendulum built from *physical* knowledge rather than from the
    answer: the model structure (Euler discretization) fixes the structural
    zeros and the unit entry of the first row, while the physical parameters
    (g/l, mu/(m l^2), Ts/(m l^2)) are only known to within `rel_err`.

    Setting keep_structure=False removes the structural zeros, which is used in
    the ablation to show what the structural part of the prior is worth.
    """
    from .sets import structured_prior

    A_tr, B_tr = pen.true_matrices(basis)
    RA = rel_err * np.abs(A_tr)
    RB = rel_err * np.abs(B_tr)
    if keep_structure:
        RA[np.abs(A_tr) < 1e-12] = 0.0      # structural zeros stay zero
        RA[0, 0] = 0.0                      # x1 integrator entry is exact
        RB[np.abs(B_tr) < 1e-12] = 0.0
    else:
        RA = np.maximum(RA, rel_err * 0.1)
        RB = np.maximum(RB, rel_err * 0.1)
    CA = A_tr + bias * np.abs(A_tr)
    CB = B_tr + bias * np.abs(B_tr)
    A_side, GA = structured_prior(CA, RA)
    B_side, GB = structured_prior(CB, RB)
    return A_side, GA, B_side, GB


def operating_region_from_data(Xs, margin=1.2):
    """Box operating region X = <H_X, h_X>_P covering the data with a margin."""
    n = Xs.shape[0]
    r = margin * np.max(np.abs(Xs), axis=1)
    r = np.maximum(r, 1e-3)
    H = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([r, r])
    return Polytope(H, h)


def disturbance_zonotope(n, scale):
    return Zonotope(np.zeros(n), scale * np.eye(n))
