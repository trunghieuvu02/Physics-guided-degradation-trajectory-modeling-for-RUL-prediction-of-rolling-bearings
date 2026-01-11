# modified IENEMD-ATD, Algorithm I
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from PyEMD import EMD  # provided by EMD-signal or PyEMD (depends on package)
except ImportError:
    EMD = None

from scipy.stats import kurtosis

@dataclass
class IENEMDATDConfig:
    beta: float = 0.719
    rho95: float = 2.449
    rho99: float = 1.919
    alpha: float = 0.0  # margin for confidence bounds; higher = more IMFs selected as noise
                        # Recommended: 1.0-3.0 depending on noise level
                        # α=0.0 works for very noisy signals (tight selection)
                        # α=2.0 works for moderately noisy signals (default)
    J: int = 10         # number of noise-assisted decompositions in Step 6
    top_imfs: int = 5   # paper uses top five IMFs (Step 7)
    eps: float = 1e-12

def _emd_imfs(x: np.ndarray) -> np.ndarray:
    emd = EMD()
    imfs = emd.emd(x)
    if imfs is None or len(imfs) == 0:
        return np.zeros((0, len(x)), dtype=np.float64)
    return np.asarray(imfs, dtype=np.float64)

def _noise_energy(imf: np.ndarray, eps: float) -> float:
    # Step 2: Ek = (median(|ck|)/0.6745)^2
    med = np.median(np.abs(imf))
    return float((med / 0.6745) ** 2 + eps)

def _extract_inherent_noise(noise_only_imfs: np.ndarray,
                            energies: np.ndarray,
                            signal_length: int,
                            eps: float = 1e-12) -> np.ndarray:
    """
    Step 5 of Algorithm I: extract inherent noise n_hat(t).
    Implements the r_li, lambda, Gamma_l, and piecewise extraction rule.
    """
    if noise_only_imfs.size == 0:
        return np.zeros((signal_length,), dtype=np.float64)

    N = noise_only_imfs.shape[1]
    n_hat = np.zeros((N,), dtype=np.float64)

    for l in range(noise_only_imfs.shape[0]):
        c = noise_only_imfs[l]
        E_l = float(energies[l])

        cs = np.sort(c ** 2) # squared and arranged small -> large
        prefix = np.cumsum(cs)   # cumulative sum of squared values

        # rli = 1/N * (N - 2i + sum_{j=1}^i cs(j) + (N-i)*cs(i))
        # careful with 1-indexing in paper
        rli = np.empty((N,), dtype=np.float64)
        for i0 in range(N):
            i = i0 + 1
            rli[i0] = (N - 2 * i + prefix[i0] + (N - i) * cs[i0]) / N

        lam = int(np.argmin(rli))

        Gamma = float(np.sqrt(E_l * cs[lam] + eps))

        abs_c = np.abs(c)
        out = np.zeros_like(c)

        # piecewise rule
        m1 = abs_c <= Gamma
        m2 = (abs_c > Gamma) & (abs_c <= 2 * Gamma)
        # m3 otherwise -> 0

        out[m1] = c[m1]
        out[m2] = np.sign(c[m2]) * (2 * Gamma - abs_c[m2])

        n_hat += out  # the final noise estimation

    return n_hat


def denoise_modified_ienemd_atd(x: np.ndarray,
                                cfg: Optional[IENEMDATDConfig] = None) -> np.ndarray:
    """
    Modified IENEMD-ATD (Algorithm I, Appendix A).
    If PyEMD is missing, returns the input signal (so pipeline still runs).

    Goal: denoise signal x by modified IENEMD-ATD.
    - phase-space trajectory “đúng topology” hơn (ít bị noise làm rung rinh)
    - phản ánh thay đổi hình dạng/quỹ đạo do degradation thay vì do nhiễu
    - mô hình trở nên robust (paper còn làm ablation cho thấy bỏ denoise là tụt mạnh)
    """
    if cfg is None:
        cfg = IENEMDATDConfig()

    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)

    if EMD is None:
        # keep pipeline runnable; but note: you won’t match paper-level denoising without EMD
        return x.astype(np.float32)
    
    # Step 1: Decompose xt by EMD and obtain ck(t)(1 ≤ k ≤ I)
    ck = _emd_imfs(x)
    if ck.shape[0] == 0:
        return x.astype(np.float32)
    
    # Step 2: noise energies Ek
    Ek = np.array([_noise_energy(imf, cfg.eps) for imf in ck], dtype=np.float64)

    # Step 3: confidence intervals based on E1
    E1 = Ek[0]
    k_idx = np.arange(1, len(Ek) + 1, dtype=np.float64)  # 1..I
    Ehat95 = E1 * (cfg.rho95 ** (-k_idx / cfg.beta))
    Ehat99 = E1 * (cfg.rho99 ** (-k_idx / cfg.beta))

    # Step 4: select noise-only IMFs by energy bounds (log2), with alpha margin
    log2 = lambda a: np.log(a + cfg.eps) / np.log(2.0)
    noise_mask = (log2(Ek) >= (log2(Ehat95) - cfg.alpha)) & (log2(Ek) <= (log2(Ehat99) + cfg.alpha))
    noise_only = ck[noise_mask]
    noise_only_E = Ek[noise_mask]

    # Step 5: inherent noise n_hat(t) from the noise-only IMFs
    n_hat = _extract_inherent_noise(noise_only, noise_only_E, N, cfg.eps)

    # Step 6: noise-assisted decomposition J times, then average IMFs
    imf_list = []
    max_imfs = 0
    for j in range(cfg.J):
        sign = 1.0 if (j % 2 == 0) else -1.0
        xj = x + sign * n_hat
        imfs_j = _emd_imfs(xj)
        imf_list.append(imfs_j)
        max_imfs = max(max_imfs, imfs_j.shape[0])

    # average (pad with zeros if different IMF counts)
    ci = np.zeros((max_imfs, N), dtype=np.float64)
    for imfs_j in imf_list:
        if imfs_j.shape[0] < max_imfs:
            pad = np.zeros((max_imfs - imfs_j.shape[0], N), dtype=np.float64)
            imfs_j = np.vstack([imfs_j, pad])
        ci += imfs_j
    ci /= float(cfg.J)

    # Step 7: select top five IMFs and hard-threshold with adaptive Ti
    K = min(cfg.top_imfs, ci.shape[0])
    den_imfs = np.zeros((K, N), dtype=np.float64)

    for i in range(K):
        c = ci[i]
        mu = np.mean(np.abs(c))
        krt = float(np.abs(kurtosis(c, fisher=False)) + cfg.eps)

        # Ti (paper formula shown in Algorithm I)
        Ti = (mu / krt) * (np.sqrt(2.0 * np.log(N)) / 0.6745)

        den_imfs[i] = c * (np.abs(c) >= Ti)

    # Step 8: Reconstruct the final denoised signals
    x_tilde = np.sum(den_imfs, axis=0)
    return x_tilde.astype(np.float32)