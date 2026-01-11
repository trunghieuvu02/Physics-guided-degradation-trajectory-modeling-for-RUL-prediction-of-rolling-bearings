# (build (ΔR, Δy) pairs using FPT/EOF + piecewise y)
import os
import glob
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import torch
from torch.utils.data import Dataset

from .features import delta_R_from_adjacent
from .psr import PSRConfig
from .denoise import IENEMDATDConfig

# Table 4 from the paper (file numbers).
FPT_EOF_TABLE4: Dict[str, Tuple[int, int]] = {
    "X-bearing 1-1": (71, 123),
    "X-bearing 1-2": (30, 161),
    "X-bearing 1-3": (58, 158),
    "X-bearing 1-4": (26, 122),
    "X-bearing 1-5": (28, 52),
    "X-bearing 2-1": (451, 491),
    "X-bearing 2-2": (42, 161),
    "X-bearing 2-3": (302, 533),
    "X-bearing 2-4": (30, 42),
    "X-bearing 2-5": (119, 339),
    "X-bearing 3-1": (2387, 2528),
    "X-bearing 3-3": (339, 372),
    "X-bearing 3-4": (1417, 1516),
    "U-bearing": (912, 1034),
    # PHM examples (add remaining if you use PHM folder names)
    "P-bearing 1-1": (1490, 2740),
    "P-bearing 1-2": (827, 871),
    "P-bearing 1-3": (1433, 2289),
    "P-bearing 1-4": (1084, 1191),
    "P-bearing 1-5": (2409, 2453),
    "P-bearing 1-6": (2408, 2448),
    "P-bearing 1-7": (2204, 2259),
    "P-bearing 2-1": (874, 911),
    "P-bearing 2-2": (745, 797),
    "P-bearing 2-3": (1945, 1955),
    "P-bearing 2-4": (739, 751),
    "P-bearing 2-6": (683, 701),
    "P-bearing 2-7": (220, 230),
    "P-bearing 3-2": (1585, 1637),
}

def default_file_reader(filepath: str) -> np.ndarray:
    """
    Read CSV/TXT file with 2 channels: Horizontal, Vertical vibration signals.
    Returns [2, N] array.
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
    # data is [N, 2] -> transpose to [2, N]
    return data.T

def piecewise_y(file_no: int, fpt: int, eof: int) -> float:
    """
    Eq. (5): y=1 for t<=FPT else (EOL-t)/(EOL-FPT). Paper uses EOF as the file-number end.
    """
    if file_no <= fpt:
        return 1.0
    if file_no >= eof:
        return 0.0
    return float((eof - file_no) / (eof - fpt))


@dataclass
class PairDatasetConfig:
    points: int = 2560       # paper uses 2560 points
    D: int = 1               # paper compresses to 1-D
    use_channels: Tuple[int, ...] = (0,)  # e.g. (0,1) for vertical+horizontal
    do_denoise: bool = True
    start_from_fpt_plus_one: bool = True  # ensures both t-1 and t are within [FPT,EOF]
    cache_dir: Optional[str] = None

class BearingPairDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        bearings: Sequence[str],
        fpt_eof: Optional[Dict[str, Tuple[int, int]]] = None,
        *,
        file_reader=default_file_reader,
        cfg: Optional[PairDatasetConfig] = None,
        psr_cfg: Optional[PSRConfig] = None,
        den_cfg: Optional[IENEMDATDConfig] = None,
        return_meta: bool = False,
    ):
        self.root_dir = root_dir
        self.bearings = list(bearings)
        self.fpt_eof = fpt_eof if fpt_eof is not None else dict(FPT_EOF_TABLE4)
        self.file_reader = file_reader
        self.cfg = cfg if cfg is not None else PairDatasetConfig()
        self.psr_cfg = psr_cfg
        self.den_cfg = den_cfg
        self.return_meta = return_meta

        if self.cfg.cache_dir is not None:
            os.makedirs(self.cfg.cache_dir, exist_ok=True)

        # build index: list of (bearing, cur_idx0based)
        self.samples: List[Tuple[str, int]] = []
        self.files: Dict[str, List[str]] = {}

        for b in self.bearings:
            bdir = os.path.join(self.root_dir, b)
            flist = sorted(
                glob.glob(os.path.join(bdir, "*.csv")) +
                glob.glob(os.path.join(bdir, "*.txt"))
            )
            if len(flist) < 3:
                raise RuntimeError(f"Not enough files for bearing {b} in {bdir}")
            self.files[b] = flist

            if b not in self.fpt_eof:
                raise RuntimeError(f"Missing FPT/EOF for bearing '{b}'. Add it to fpt_eof dict.")

            fpt, eof = self.fpt_eof[b]
            # file_no is 1-based; idx is 0-based
            start_file = fpt + 1 if self.cfg.start_from_fpt_plus_one else fpt
            start_idx = max(1, start_file - 1)  # need prev
            end_idx = min(len(flist) - 1, eof - 1)

            for cur_idx in range(start_idx, end_idx + 1):
                # only keep pairs where both file numbers are <= eof
                self.samples.append((b, cur_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, bearing: str, cur_idx: int, ch: int) -> Optional[str]:
        if self.cfg.cache_dir is None:
            return None
        # Replace path separators and spaces to create flat cache structure
        bearing_safe = bearing.replace('/', '_').replace('..', '').replace(' ', '_')
        return os.path.join(self.cfg.cache_dir, f"{bearing_safe}_t{cur_idx:05d}_ch{ch}.npy")

    def __getitem__(self, i: int):
        bearing, cur_idx = self.samples[i]
        prev_idx = cur_idx - 1

        cur_file_no = cur_idx + 1
        prev_file_no = prev_idx + 1
        fpt, eof = self.fpt_eof[bearing]

        # label Δy = y_{t-1} - y_t (Eq. 3), where y is piecewise (Eq. 5). 
        y_prev = piecewise_y(prev_file_no, fpt, eof)
        y_cur = piecewise_y(cur_file_no, fpt, eof)
        dy = float(y_prev - y_cur)

        # load raw signals
        x_prev_all = self.file_reader(self.files[bearing][prev_idx])  # [C, N]
        x_cur_all = self.file_reader(self.files[bearing][cur_idx])    # [C, N]

        feats = []
        for ch in self.cfg.use_channels:
            cache_p = self._cache_path(bearing, cur_idx, ch)
            if cache_p is not None and os.path.exists(cache_p):
                dR = np.load(cache_p).astype(np.float32)
            else:
                dR = delta_R_from_adjacent(
                    x_prev_all[ch],
                    x_cur_all[ch],
                    points=self.cfg.points,
                    D=self.cfg.D,
                    do_denoise=self.cfg.do_denoise,
                    psr_cfg=self.psr_cfg,
                    den_cfg=self.den_cfg,
                    normalize=True,
                )
                if cache_p is not None:
                    np.save(cache_p, dR.astype(np.float32))
            feats.append(dR)  # [D, points]

        x = np.concatenate(feats, axis=0)  # [D * n_channels, points]
        x_t = torch.from_numpy(x).float()
        y_t = torch.tensor([dy], dtype=torch.float32)

        if self.return_meta:
            meta = {"bearing": bearing, "cur_file_no": cur_file_no, "prev_file_no": prev_file_no}
            return x_t, y_t, meta
        return x_t, y_t