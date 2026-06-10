"""
Train the pre-refinement base Dual-Dimensional PatchTST Transformer (the
"before-ITT" recipe described in Sec. VII.A "Five bottlenecks identified in
the base recipe") for three independent seeds, capture per-seed test
metrics, and write the CSV that the post-hoc t-test script consumes.

The pre-refinement recipe has ALL five bottlenecks present:
  1. LOSS MISALIGNMENT    -- nn.MSELoss (symmetric)
  2. NO LR SCHEDULE       -- fixed lr = 1e-4 (no cosine warm restarts)
  3. UNDER-CAPACITY       -- d_model=64, n_layers=2, d_ff=128
  4. SHORT WINDOW         -- whatever window the upstream data-prep used
                             (typically 40 in MainSingleEng_*_Improved.py)
  5. NO NOISE AUGMENTATION-- noise_std = 0.0

The driver pairs with the ITT 3-seed runs from the patched run_all_scenarios
because (a) the same three seeds [42, 137, 271] are used, and (b) the same
train_test_split(random_state=seed) call is made, so seed S maps to the same
80/20 split in both arms.

HOW TO USE
----------
In a Jupyter cell INSIDE MainSingleEng_FD001_F2_FD003_F4_Final_Improved.py,
*after* the data-prep cells have populated X_train_sw, y_train_sw, X_testf,
y_test, features, eng_type, device:

    from _train_baseline_3seeds import train_baseline_3seeds
    train_baseline_3seeds(
        X_train_sw=X_train_sw, y_train_sw=y_train_sw,
        X_testf=X_testf,       y_test=y_test,
        features=features,     eng_type=eng_type,
        device=device,
    )

Compute cost: 3 seeds * ~3-3.5 h per seed = roughly 10 hours on a single GPU
at the small-architecture config.  Output: per_seed_metrics/Baseline_<engtype>.csv.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from improve_transformer import (
    TrainConfig,
    fit_improved,
    make_loaders_augmented,
    evaluate_improved,
    score_nasa,
)


BASELINE_SEEDS  = (42, 137, 271)         # same seeds as the ITT ensemble
BASELINE_WINDOW = 40                     # bottleneck #4 -- the pre-ITT window


def _make_baseline_cfg(features, window_size: int) -> TrainConfig:
    """Construct the pre-refinement baseline config (all 5 bottlenecks present)."""
    return TrainConfig(
        feature_cols = list(features),
        C            = len(features),
        L            = window_size,
        patch_len_t  = 10, stride_t = 5,
        patch_len_c  = 3,  stride_c = 1,
        # Bottleneck #3 -- small architecture
        d_model_t = 64, n_heads_t = 8, n_layers_t = 2, d_ff_t = 128, dropout_t = 0.1,
        d_model_c = 64, n_heads_c = 8, n_layers_c = 2, d_ff_c = 128, dropout_c = 0.1,
        head_hidden = 128, head_dropout = 0.1,
        # Bottleneck #2 -- fixed lr (no scheduler set by caller)
        batch_size = 40, epochs = 150, lr = 1e-4,
        weight_decay = 1e-4, grad_clip = 1.0, patience = 20,
        device     = "cuda" if torch.cuda.is_available() else "cpu",
        model_path = "baseline_pre_refinement.pt",
    )


class _SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):           return len(self.X)
    def __getitem__(self, i):    return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def train_baseline_3seeds(
    X_train_sw, y_train_sw, X_testf, y_test,
    features, eng_type, device,
    seeds=BASELINE_SEEDS,
    window_size: int = BASELINE_WINDOW,
    verbose: bool = True,
):
    """
    Train the pre-refinement baseline for ``seeds`` and write per-seed
    test-set metrics to ``per_seed_metrics/Baseline_<eng_type>.csv``.

    Parameters
    ----------
    X_train_sw, y_train_sw : numpy arrays
        The windowed training sequences from create_training_sequences_sw.
    X_testf, y_test        : numpy arrays
        The windowed test sequences from create_testing_sequences_sw.
        ``X_testf`` may be in either ``(N, L, C)`` or ``(N, C, L)`` layout;
        we transpose if needed.
    features               : list of feature names (the filtered sensor cols).
    eng_type               : "FD001" / "FD002" / "FD003" / "FD004" string;
                             used for the output filename.
    device                 : torch.device.
    seeds                  : iterable of ints (default [42, 137, 271] - paired
                             with the patched ITT ensemble seeds).
    window_size            : the temporal window length the model expects;
                             usually 40 to match the pre-refinement bottleneck.
    """
    print("=" * 70)
    print(f"  Pre-refinement BASELINE 3-seed run on {eng_type}")
    print(f"  Seeds       : {list(seeds)}")
    print(f"  Window size : {window_size}")
    print(f"  Recipe      : MSE loss, no LR scheduler, d_model=64, n_layers=2,")
    print(f"                d_ff=128, noise_std=0.0  (Sec. VII.A bottlenecks 1-5)")
    print("=" * 70)

    # Build a test loader in the (N, C, L) layout the model expects.
    Xtf = np.asarray(X_testf, dtype=np.float32)
    if Xtf.shape[1] >= Xtf.shape[2]:
        Xtf = Xtf.transpose(0, 2, 1)
    te_loader = DataLoader(
        _SimpleDS(Xtf, np.asarray(y_test, dtype=np.float32).reshape(-1)),
        batch_size=64, shuffle=False,
    )

    cfg = _make_baseline_cfg(features, window_size)

    per_seed_rows = []
    for seed in seeds:
        print(f"\n--- Baseline seed = {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Same 80/20 split call as the ITT ensemble loop in run_all_scenarios
        # -- ensures the two arms are paired on seed.
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_train_sw, y_train_sw, test_size=0.2, random_state=seed)

        tr_ld, vl_ld, (C, L) = make_loaders_augmented(
            X_tr, X_vl, y_tr, y_vl,
            batch_size=cfg.batch_size,
            num_workers=getattr(cfg, "num_workers", 0),
            noise_std=0.0,                       # bottleneck #5
            use_cuda=str(device).startswith("cuda"),
        )
        cfg.C = C
        cfg.L = L

        # Pre-refinement training: MSE loss, no scheduler.
        model, _, _ = fit_improved(
            tr_ld, vl_ld, features, cfg, device,
            loss_fn=nn.MSELoss(),                # bottleneck #1
            use_scheduler=False,                 # bottleneck #2
            accumulation_steps=1,
            verbose=verbose,
        )
        _, mets, yt, yp = evaluate_improved(model, te_loader, device, nn.MSELoss())
        nasa = float(score_nasa(yp - yt))

        rec = {
            "variant":  "Baseline",
            "seed":     seed,
            "eng_type": eng_type,
            "RMSE":     mets["RMSE"],
            "MAE":      mets["MAE"],
            "R2":       mets["R2"],
            "NASA":     nasa,
        }
        per_seed_rows.append(rec)
        print(f"  seed={seed}   RMSE={mets['RMSE']:.4f}   "
              f"MAE={mets['MAE']:.4f}   R2={mets['R2']:.4f}   "
              f"NASA={nasa:.1f}")

        # Persist after every seed so a crash never costs more than one run.
        os.makedirs("per_seed_metrics", exist_ok=True)
        out_path = f"per_seed_metrics/Baseline_{eng_type}.csv"
        pd.DataFrame(per_seed_rows).to_csv(out_path, index=False)

    print(f"\nBaseline 3-seed run complete.")
    print(f"Saved -> {out_path}")
    return pd.DataFrame(per_seed_rows)


if __name__ == "__main__":
    # If run as a script via "%run _train_baseline_3seeds.py" inside Jupyter,
    # the variables below come from the calling kernel's globals.
    train_baseline_3seeds(
        X_train_sw=X_train_sw, y_train_sw=y_train_sw,
        X_testf=X_testf,       y_test=y_test,
        features=features,     eng_type=eng_type,
        device=device,
    )
