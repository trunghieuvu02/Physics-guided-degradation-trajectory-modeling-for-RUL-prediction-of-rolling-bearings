import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.spatial import cKDTree

@dataclass
class PSRConfig:
    tau_max: int = 60
    mi_bins: int = 64
    m_max: int = 20
    e1_range: Tuple[float, float] = (0.99, 1.01)
    eps: float = 1e-12

def mutual_information_tau(x: np.ndarray, cfg: Optional[PSRConfig] = None) -> int:
    """
    Mutual information (histogram estimator), pick FIRST local minimum of I(xt, xt+τ).
    """
    if cfg is None:
        cfg = PSRConfig()
    x = np.asarray(x, dtype=np.float64).ravel()

    # normalize for stable binning
    x = (x - np.mean(x)) / (np.std(x) + cfg.eps)

    I = np.zeros((cfg.tau_max,), dtype=np.float64)
    for tau in range(1, cfg.tau_max + 1):
        a = x[:-tau]
        b = x[tau:]

        # joint histogram
        H, _, _ = np.histogram2d(a, b, bins=cfg.mi_bins, density=False)
        Pxy = H / (np.sum(H) + cfg.eps)
        Px = np.sum(Pxy, axis=1, keepdims=True)
        Py = np.sum(Pxy, axis=0, keepdims=True)

        nz = Pxy > 0
        I[tau - 1] = float(np.sum(Pxy[nz] * np.log((Pxy[nz] + cfg.eps) / (Px @ Py + cfg.eps)[nz])))

    # first local minimum
    for i in range(1, len(I) - 1):
        if I[i - 1] > I[i] and I[i] < I[i + 1]:
            return i + 1  # because tau starts at 1
    return int(np.argmin(I) + 1)

def _delay_embed(x: np.ndarray, tau: int, d: int) -> np.ndarray:
    """
    Build Yi(d) = [x_i, x_{i+tau}, ..., x_{i+(d-1)tau}] (Appendix B).
    Returns array of shape [n_points, d]
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x) - (d - 1) * tau
    if n <= 2:
        return np.zeros((0, d), dtype=np.float64)
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        out[:, j] = x[j * tau : j * tau + n]
    return out

def cao_embedding_dimension(x: np.ndarray, tau: int, cfg: Optional[PSRConfig] = None) -> int:
    """
    Cao’s method using E1(d)=E(d+1)/E(d); pick minimum d with E1(d) in [0.99,1.01].
    """
    if cfg is None:
        cfg = PSRConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    x = (x - np.mean(x)) / (np.std(x) + cfg.eps)

    E = []
    # compute E(d) for d=1..m_max+1 (since E1(d) needs E(d+1))
    for d in range(1, cfg.m_max + 2):
        Yd = _delay_embed(x, tau, d)
        if Yd.shape[0] < 5:
            E.append(np.nan)
            continue

        # build d+1 embedding
        Yd1 = _delay_embed(x, tau, d + 1)
        n1 = Yd1.shape[0]

        if n1 < 2:
            E.append(np.nan)
            continue

        # Compute NN in d-space for denominator (Appendix B)
        tree_d = cKDTree(Yd)
        dist_d, _ = tree_d.query(Yd[:n1], k=2)  # query first n1 points
        dist_d = dist_d[:, 1] + cfg.eps  # exclude self (k=1)

        # Compute NN in (d+1)-space for numerator (Appendix B)
        # Y_NN,i(d+1) is the NN of Y_i(d+1) in (d+1)-space
        tree_d1 = cKDTree(Yd1)
        dist_d1, _ = tree_d1.query(Yd1, k=2)  # first is itself
        dist_d1 = dist_d1[:, 1] + cfg.eps  # exclude self

        # a2(i,d) = ||Yi(d+1) - YNN_i(d+1)|| / ||Yi(d) - YNN_i(d)||
        a2 = dist_d1 / dist_d
        E.append(float(np.mean(a2)))

    # E1(d)=E(d+1)/E(d)
    lo, hi = cfg.e1_range
    for d in range(1, cfg.m_max + 1):
        if not np.isfinite(E[d - 1]) or not np.isfinite(E[d]):
            continue
        E1 = E[d] / (E[d - 1] + cfg.eps)
        if lo <= E1 <= hi:
            return d
        
    # fallback
    best_d = 2
    best_err = 1e9
    for d in range(1, cfg.m_max + 1):
        if not np.isfinite(E[d - 1]) or not np.isfinite(E[d]):
            continue
        E1 = E[d] / (E[d - 1] + cfg.eps)
        err = abs(E1 - 1.0)
        if err < best_err:
            best_err = err
            best_d = d
    return int(best_d)