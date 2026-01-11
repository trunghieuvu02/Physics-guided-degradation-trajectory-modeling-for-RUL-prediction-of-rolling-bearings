#!/usr/bin/env python3
"""
Pre-cache all features before training to avoid slow first epoch.
This script processes all training samples and saves them to cache.
"""
import os
import sys
import time
from tqdm import tqdm

# Add parent directory to path to import pgdtm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgdtm.dataset import BearingPairDataset, PairDatasetConfig, FPT_EOF_TABLE4

def main():
    # Configuration
    root_dir = "./datasets/XJTU-SY/35Hz12kN"
    cache_dir = "./cache/35Hz12kN"
    os.makedirs(cache_dir, exist_ok=True)

    # Training bearings
    train_bearings = [
        "X-bearing 1-2",
        "X-bearing 1-3",
        "X-bearing 1-4",
        "X-bearing 1-5",
        "X-bearing 2-1",
        "X-bearing 2-2"
    ]

    # Map to folder names
    bearing_name_map = {
        "X-bearing 1-1": "Bearing1_1",
        "X-bearing 1-2": "Bearing1_2",
        "X-bearing 1-3": "Bearing1_3",
        "X-bearing 1-4": "Bearing1_4",
        "X-bearing 1-5": "Bearing1_5",
        "X-bearing 2-1": "../37.5Hz11kN/Bearing2_1",
        "X-bearing 2-2": "../37.5Hz11kN/Bearing2_2",
    }

    # Dataset configuration
    ds_cfg = PairDatasetConfig(
        points=2560,
        D=1,
        use_channels=(0,),
        do_denoise=False,
        start_from_fpt_plus_one=True,
        cache_dir=cache_dir
    )

    # Optimized PSR settings to speed up processing
    from pgdtm.psr import PSRConfig
    psr_cfg = PSRConfig(m_max=10)  # Reduced from default 20 to speed up cao_embedding_dimension

    # Create FPT/EOF mapping
    fpt_eof_mapped = {}
    for paper_name, folder_name in bearing_name_map.items():
        if paper_name in FPT_EOF_TABLE4:
            fpt_eof_mapped[folder_name] = FPT_EOF_TABLE4[paper_name]

    print("=" * 70)
    print("FEATURE PRE-CACHING SCRIPT")
    print("=" * 70)
    print(f"\nThis will pre-process all {len(train_bearings)} training bearings")
    print(f"Cache directory: {cache_dir}")
    print(f"\nProcessing configuration:")
    print(f"  - Denoising: {ds_cfg.do_denoise}")
    print(f"  - Points: {ds_cfg.points}")
    print(f"  - Channels: {ds_cfg.use_channels}")
    print(f"  - PSR m_max: {psr_cfg.m_max} (reduced from 20 for speed)")
    print(f"\nNote: Each sample takes ~60-80 seconds to process (PSR is slow)")
    print("=" * 70)

    # Create dataset
    print("\nCreating dataset...")
    train_ds = BearingPairDataset(
        root_dir=root_dir,
        bearings=[bearing_name_map.get(b, b) for b in train_bearings],
        fpt_eof=fpt_eof_mapped,
        cfg=ds_cfg,
        psr_cfg=psr_cfg
    )

    print(f"Total samples to process: {len(train_ds)}")

    # Check how many are already cached
    n_cached = 0
    for i in range(len(train_ds)):
        bearing, cur_idx = train_ds.samples[i]
        cache_path = train_ds._cache_path(bearing, cur_idx, 0)
        if cache_path and os.path.exists(cache_path):
            n_cached += 1

    print(f"Already cached: {n_cached}/{len(train_ds)}")
    print(f"Need to process: {len(train_ds) - n_cached}/{len(train_ds)}")

    if n_cached == len(train_ds):
        print("\n✓ All features are already cached! You can start training.")
        return

    # Process all samples
    print("\n" + "=" * 70)
    print("Starting feature extraction...")
    print("=" * 70 + "\n")

    start_time = time.time()

    for i in tqdm(range(len(train_ds)), desc="Processing samples"):
        bearing, cur_idx = train_ds.samples[i]
        cache_path = train_ds._cache_path(bearing, cur_idx, 0)

        # Skip if already cached
        if cache_path and os.path.exists(cache_path):
            continue

        # Process sample (this will cache it)
        sample_start = time.time()
        x, y = train_ds[i]
        sample_time = time.time() - sample_start

        tqdm.write(f"  Sample {i+1}/{len(train_ds)}: {bearing} file {cur_idx+1} - {sample_time:.1f}s")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("CACHING COMPLETED!")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per sample: {total_time/len(train_ds):.1f} seconds")
    print(f"\nAll features cached in: {cache_dir}")
    print("\nYou can now run train_pddtm.py - it will be much faster!")
    print("=" * 70)

if __name__ == "__main__":
    main()
