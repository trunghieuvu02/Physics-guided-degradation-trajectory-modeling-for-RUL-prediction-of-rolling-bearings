import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA

from .psr import PSRConfig, mutual_information_tau, cao_embedding_dimension
from .denoise import denoise_modified_ienemd_atd, IENEMDATDConfig

def _crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    n = len(x)
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2
        return x[start : start + target_len]
    # reflect pad
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(x, (left, right), mode="reflect")

def phase_space_matrix(x: np.ndarray, tau: int, m: int, points: int = 2560) -> np.ndarray:
    """
    Eq. (6): R is [m x (N-(m-1)tau)]. We force (N-(m-1)tau)=points=2560 (paper). 
    """
    tau = int(max(1, tau))
    m = int(max(2, m))
    N_req = points + (m - 1) * tau
    x = _crop_or_pad(x, N_req)

    R = np.empty((m, points), dtype=np.float32)
    for k in range(m):
        R[k] = x[k * tau : k * tau + points]
    return R

def pca_reduce_trajectory(R: np.ndarray, D: int = 1) -> np.ndarray:
    """
    PCA(R)^D in Eq. (9): treat each column as a point in m-dim, reduce to D.
    """
    R = np.asarray(R, dtype=np.float32)
    pts = R.T  # [points, m]
    pca = PCA(n_components=D)
    z = pca.fit_transform(pts)  # [points, D]
    return z.T.astype(np.float32)  # [D, points]

def delta_R_from_adjacent(
    x_prev: np.ndarray,
    x_cur: np.ndarray,
    *,
    points: int = 2560,
    D: int = 1,
    do_denoise: bool = True,
    psr_cfg: Optional[PSRConfig] = None,
    den_cfg: Optional[IENEMDATDConfig] = None,
    normalize: bool = True,
    eps: float = 1e-8,
    verbose: bool = False,
) -> np.ndarray:
    """
    Algorithm 1 Steps 1-4 + normalize input (Step 5 says "normalized").
    """
    import sys
    if psr_cfg is None:
        psr_cfg = PSRConfig()
    if den_cfg is None:
        den_cfg = IENEMDATDConfig()

    xp = x_prev.astype(np.float32).ravel()
    xc = x_cur.astype(np.float32).ravel()

    # Step 1: Denoising (slowest step)
    if do_denoise:
        if verbose:
            print("  [1/4] Denoising prev signal...", end='', flush=True)
        xp = denoise_modified_ienemd_atd(xp, den_cfg)
        if verbose:
            print(" done", flush=True)
            print("  [2/4] Denoising curr signal...", end='', flush=True)
        xc = denoise_modified_ienemd_atd(xc, den_cfg)
        if verbose:
            print(" done", flush=True)

    # Step 2: PSR parameters
    import time
    t0 = time.time()
    if verbose:
        print("  [2/4] Computing mutual information...", end='', flush=True)
    tau_p = mutual_information_tau(xp, psr_cfg)
    tau_c = mutual_information_tau(xc, psr_cfg)
    t1 = time.time()
    if verbose:
        print(f" done in {t1-t0:.1f}s (τ_p={tau_p}, τ_c={tau_c})", flush=True)
        print("  [3/4] Computing Cao embedding dimension...", end='', flush=True)

    m_p = cao_embedding_dimension(xp, tau_p, psr_cfg)
    m_c = cao_embedding_dimension(xc, tau_c, psr_cfg)
    t2 = time.time()
    if verbose:
        print(f" done in {t2-t1:.1f}s (m_p={m_p}, m_c={m_c}, m_max={psr_cfg.m_max})", flush=True)

    # Step 3: Phase space reconstruction
    if verbose:
        print("  [4/4] Phase space reconstruction & PCA...", end='', flush=True)
    Rp = phase_space_matrix(xp, tau_p, m_p, points=points)
    Rc = phase_space_matrix(xc, tau_c, m_c, points=points)

    # Step 4:
    RpD = pca_reduce_trajectory(Rp, D=D)
    RcD = pca_reduce_trajectory(Rc, D=D)

    dR = (RcD - RpD).astype(np.float32)

    if normalize:
        mean = dR.mean(axis=1, keepdims=True)
        std = dR.std(axis=1, keepdims=True) + eps
        dR = (dR - mean) / std

    if verbose:
        print(" done", flush=True)

    return dR  # [D, points]