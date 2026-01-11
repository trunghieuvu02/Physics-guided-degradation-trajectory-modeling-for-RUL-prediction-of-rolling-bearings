import os
import math
import numpy as np
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pgdtm.dataset import BearingPairDataset, PairDatasetConfig, FPT_EOF_TABLE4, piecewise_y
from pgdtm.model import Compact1DCNN


def eval_iterative(model, root_dir: str, bearings: List[str], fpt_eof: Dict[str, Tuple[int, int]],
                   ds_cfg: PairDatasetConfig, device: str, psr_cfg=None, den_cfg=None) -> Dict[str, float]:
    """
    Evaluate by iterating y_t = y_{t-1} - Δy_t (Algorithm 1). 
    """
    model.eval()
    ds = BearingPairDataset(root_dir, bearings, fpt_eof, cfg=ds_cfg, return_meta=True, psr_cfg=psr_cfg, den_cfg=den_cfg)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # group per bearing
    pred = {b: {} for b in bearings}
    true = {b: {} for b in bearings}

    # initialize y_0=1 (healthy start in paper’s surrogate)
    y_hat = {b: 1.0 for b in bearings}

    with torch.no_grad():
        for x, dy_true, meta in dl:
            b = meta["bearing"][0]
            cur_file = int(meta["cur_file_no"].item())
            prev_file = int(meta["prev_file_no"].item())
            fpt, eof = fpt_eof[b]

            x = x.to(device)
            dy_pred = float(model(x).cpu().item())

            # iterate
            y_hat[b] = max(0.0, min(1.0, y_hat[b] - dy_pred))

            # store
            pred[b][cur_file] = y_hat[b]
            true[b][cur_file] = piecewise_y(cur_file, fpt, eof)

    # compute RMSE + MAPE over all points
    ys_p, ys_t = [], []
    for b in bearings:
        keys = sorted(set(true[b].keys()) & set(pred[b].keys()))
        for k in keys:
            ys_p.append(pred[b][k])
            ys_t.append(true[b][k])

    ys_p = np.array(ys_p, dtype=np.float64)
    ys_t = np.array(ys_t, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((ys_p - ys_t) ** 2)))

    # Paper Eq. (16): MAPE = (1/N) * Σ|y_real - y_pred| / y_real
    # Exclude points where y_true == 0 (or <= 1e-8) to avoid division instability
    mask = ys_t > 1e-8
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((ys_p[mask] - ys_t[mask]) / ys_t[mask])))
    else:
        mape = 0.0  # fallback if all points are zero

    return {"RMSE": rmse, "MAPE": mape}


def main():
    # Configuration from paper (Table 3)
    # Set GPU device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (GPU 1)")

    # Dataset paths - modify as needed
    root_dir = "./datasets/XJTU-SY/35Hz12kN"
    cache_dir = "./cache/35Hz12kN"
    os.makedirs(cache_dir, exist_ok=True)

    # Training and testing bearings as specified
    train_bearings = [
        "X-bearing 1-2",
        "X-bearing 1-3",
        "X-bearing 1-4",
        "X-bearing 1-5",
        "X-bearing 2-1",
        "X-bearing 2-2"
    ]

    test_bearings = ["X-bearing 1-1"]

    # Map XJTU folder names to paper notation
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
        use_channels=(1,),  # Use vertical channel (index 1 since reader is [Horizontal, Vertical])
        do_denoise=True,    # Match paper's preprocessing
        start_from_fpt_plus_one=True,
        cache_dir=cache_dir
    )

    # Optimized PSR settings to speed up processing
    from pgdtm.psr import PSRConfig
    psr_cfg = PSRConfig(m_max=20)  # Don't reduce for reproduction

    # Hyperparameters from Table 3
    batch_size = 20
    learning_rate = 1e-4
    num_epochs = 100
    eta = 0.05  # NBF parameter
    dropout = 0.3

    print(f"\n=== Training Configuration ===")
    print(f"Train bearings: {train_bearings}")
    print(f"Test bearings: {test_bearings}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")

    # Create datasets
    print("\n=== Creating datasets ===")

    # Create custom FPT/EOF dict with folder names as keys
    fpt_eof_mapped = {}
    for paper_name, folder_name in bearing_name_map.items():
        if paper_name in FPT_EOF_TABLE4:
            fpt_eof_mapped[folder_name] = FPT_EOF_TABLE4[paper_name]

    train_ds = BearingPairDataset(
        root_dir=root_dir,
        bearings=[bearing_name_map.get(b, b) for b in train_bearings],
        fpt_eof=fpt_eof_mapped,
        cfg=ds_cfg,
        psr_cfg=psr_cfg
    )

    print(f"Training samples: {len(train_ds)}")

    # Create dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )

    # Initialize model
    in_channels = ds_cfg.D * len(ds_cfg.use_channels)  # 1 * 1 = 1
    model = Compact1DCNN(
        in_channels=in_channels,
        points=ds_cfg.points,
        eta=eta,
        dropout=dropout
    ).to(device)

    print(f"\n=== Model Architecture ===")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Training loop
    print(f"\n=== Starting Training ===")
    best_train_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'best_model.pth')
            print(f"  --> Saved best model (loss: {avg_loss:.6f})")

        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n=== Evaluating on Test Set (Epoch {epoch+1}) ===")
            test_metrics = eval_iterative(
                model=model,
                root_dir=root_dir,
                bearings=[bearing_name_map.get(b, b) for b in test_bearings],
                fpt_eof=fpt_eof_mapped,
                ds_cfg=ds_cfg,
                device=device,
                psr_cfg=psr_cfg
            )
            print(f"Test RMSE: {test_metrics['RMSE']:.6f}")
            print(f"Test MAPE: {test_metrics['MAPE']:.6f}")

    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    print("Loading best model...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nEvaluating on test bearings:")
    test_metrics = eval_iterative(
        model=model,
        root_dir=root_dir,
        bearings=[bearing_name_map.get(b, b) for b in test_bearings],
        fpt_eof=fpt_eof_mapped,
        ds_cfg=ds_cfg,
        device=device,
        psr_cfg=psr_cfg
    )
    print(f"Final Test RMSE: {test_metrics['RMSE']:.6f}")
    print(f"Final Test MAPE: {test_metrics['MAPE']:.6f}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()