"""Probabilistic client value: BG/NBD for visits and Gamma-Gamma for spend per visit.

Implemented from the original papers so every step can be explained:
- Fader, Hardie and Lee (2005), "Counting Your Customers the Easy Way: An
  Alternative to the Pareto/NBD Model", Marketing Science.
- Fader, Hardie and Lee (2005), "RFM and CLV: Using Iso-Value Curves for
  Customer Base Analysis", Journal of Marketing Research (Gamma-Gamma).

Inputs per client, measured at a cut-off date in weeks:
``frequency`` repeat visits (visits after the first), ``recency`` weeks from
first to last visit, ``tenure`` weeks from first visit to the cut-off, and
``monetary`` mean value of repeat visits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1, roots_jacobi


@dataclass(frozen=True)
class BGNBD:
    r: float
    alpha: float
    a: float
    b: float

    def log_likelihood(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        return _bgnbd_ll(np.array([self.r, self.alpha, self.a, self.b]), x, t_x, T)

    def p_alive(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        ratio = np.where(
            x > 0,
            (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x),
            0.0,
        )
        return 1.0 / (1.0 + ratio)

    def expected_visits_closed_form(self, t: float, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Fader-Hardie-Lee equation (10); valid only when a > 1."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        if a <= 1:
            raise ValueError("The closed form requires a > 1; use expected_visits")
        z = t / (alpha + T + t)
        head = (a + b + x - 1) / (a - 1)
        tail = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, z)
        denom = 1 + np.where(x > 0, (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return head * tail / denom

    def expected_visits(
        self, t: float, x: np.ndarray, t_x: np.ndarray, T: np.ndarray, nodes: int = 48
    ) -> np.ndarray:
        """Expected visits in the next ``t`` weeks given each client's history.

        E[Y(t) | history] = P(active) x E[(1 - ((alpha+T)/(alpha+T+p t))^(r+x)) / p],
        with the churn probability p averaged over its updated Beta(a, b + x)
        distribution by Gauss-Jacobi quadrature. Valid for any a > 0.
        """
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        x, T = np.asarray(x, dtype=float), np.asarray(T, dtype=float)
        active_mean = np.empty_like(x)
        for value in np.unique(x):
            rows = x == value
            u, w = roots_jacobi(nodes, b + value - 1, a - 1)
            p = (1 + u) / 2
            base = (alpha + T[rows])[:, None]
            integrand = (1 - (base / (base + p * t)) ** (r + value)) / p
            active_mean[rows] = integrand @ w / w.sum()
        return self.p_alive(x, np.asarray(t_x, dtype=float), T) * active_mean


def _bgnbd_ll(params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
    r, alpha, a, b = params
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    ln_a3 = -(r + x) * np.log(alpha + T)
    repeat = x > 0
    ln_a4 = np.full_like(ln_a3, -np.inf)
    ln_a4[repeat] = (
        np.log(a) - np.log(b + x[repeat] - 1) - (r + x[repeat]) * np.log(alpha + t_x[repeat])
    )
    return ln_a1 + ln_a2 + np.logaddexp(ln_a3, ln_a4)


def fit_bgnbd(x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> BGNBD:
    x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
    if (t_x > T + 1e-9).any() or (t_x < 0).any():
        raise ValueError("Recency must lie between 0 and tenure")

    def loss(log_params: np.ndarray) -> float:
        return -_bgnbd_ll(np.exp(log_params), x, t_x, T).mean()

    result = minimize(loss, x0=np.log([0.5, 5.0, 1.5, 3.0]), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"BG/NBD fit did not converge: {result.message}")
    return BGNBD(*np.exp(result.x))


@dataclass(frozen=True)
class GammaGamma:
    p: float
    q: float
    gamma: float

    def expected_spend_per_visit(self, x: np.ndarray, monetary: np.ndarray) -> np.ndarray:
        """Shrunk mean spend per visit; clients with no repeat visits get the population mean."""
        p, q, g = self.p, self.q, self.gamma
        population = p * g / (q - 1)
        individual = p * (g + x * monetary) / (p * x + q - 1)
        return np.where(x > 0, individual, population)


def fit_gamma_gamma(x: np.ndarray, monetary: np.ndarray) -> GammaGamma:
    x, m = np.asarray(x, dtype=float), np.asarray(monetary, dtype=float)
    keep = (x > 0) & (m > 0)
    x, m = x[keep], m[keep]
    scale = m.mean()
    m = m / scale  # fit on a unit scale for numerical stability, then rescale gamma

    def loss(log_params: np.ndarray) -> float:
        p, q, g = np.exp(log_params)
        ll = (
            gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g)
            + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(g + m * x)
        )
        return -ll.mean()

    result = minimize(loss, x0=np.log([2.0, 3.0, 1.0]), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Gamma-Gamma fit did not converge: {result.message}")
    p, q, g = np.exp(result.x)
    if q <= 1:
        raise RuntimeError("Gamma-Gamma fit gave q <= 1; expected spend is undefined")
    return GammaGamma(p, q, g * scale)
