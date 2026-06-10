# %%
"""
Post-hoc paired t-tests across the per-seed metrics produced by the patched
ensemble loops (ITT, CrossAttn, RoPE, BERT) and the standalone Baseline
driver.

Reads every CSV in per_seed_metrics/<Variant>_<EngType>.csv, pairs runs across
variants by (seed, eng_type), and computes:
  - paired t-statistic and two-tailed p-value via scipy.stats.ttest_rel
  - Cohen's d for paired samples
  - sign-consistency  ('K/N seeds improved')

Comparisons reported:
  - Every refinement vs Baseline    (the manuscript's Sec. VII.E claim)
  - Every refinement vs ITT         (does the additional refinement improve
                                     on top of the basic Improved Trained
                                     Transformer?)

Usage:
  As a script:
      python _run_paired_ttests.py              # all engines, all comparisons
      python _run_paired_ttests.py FD001        # only one engine

  As a module:
      from _run_paired_ttests import run
      df_results = run(only_eng="FD001")

Outputs:
  per_seed_metrics/paired_ttests.csv
  (and a formatted table printed to stdout)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["RMSE", "MAE", "R2", "NASA"]
LOWER_IS_BETTER = {"RMSE": True, "MAE": True, "R2": False, "NASA": True}

# Reference variants to compare every other variant against.  The order
# matters only for the output table; comparisons are computed for every
# (ref, comp) pair where ref != comp.
REFERENCE_VARIANTS = ("Baseline", "ITT")


def _cohen_d_paired(diff):
    """Cohen's d for paired samples = mean(diff) / sd(diff, ddof=1)."""
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(diff.mean() / sd)


def _improvement_count(diff, lower_is_better: bool) -> int:
    """Number of seeds where the comparison variant beats the reference."""
    diff = np.asarray(diff)
    if lower_is_better:
        return int((diff < 0).sum())   # comp - ref < 0  =>  comp lower (better)
    return int((diff > 0).sum())       # comp - ref > 0  =>  comp higher (better)


def load_all(folder: str = "per_seed_metrics") -> pd.DataFrame:
    """Concatenate every per-seed CSV into one DataFrame."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    files = [f for f in files if "paired_ttests" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}/. Run a patched run_*_scenarios "
            "first, or the baseline driver _train_baseline_3seeds.py."
        )
    return pd.concat([pd.read_csv(fp) for fp in files], ignore_index=True)


def paired_comparison(df: pd.DataFrame, ref: str, comp: str, eng: str):
    """Return a list of metric rows comparing ``comp`` against ``ref`` on
    seeds shared between the two variants for engine ``eng``."""
    a = df[(df.variant == ref)  & (df.eng_type == eng)].sort_values("seed")
    b = df[(df.variant == comp) & (df.eng_type == eng)].sort_values("seed")
    common = sorted(set(a.seed) & set(b.seed))
    if len(common) < 2:
        return []

    a = a[a.seed.isin(common)].sort_values("seed")
    b = b[b.seed.isin(common)].sort_values("seed")

    rows = []
    for m in METRICS:
        ai = a[m].to_numpy()
        bi = b[m].to_numpy()
        diff = bi - ai
        t_stat, p_val = stats.ttest_rel(bi, ai)
        d = _cohen_d_paired(diff)
        improved = _improvement_count(diff, LOWER_IS_BETTER[m])
        rows.append({
            "eng_type":     eng,
            "ref_variant":  ref,
            "comp_variant": comp,
            "metric":       m,
            "n_seeds":      len(ai),
            "ref_mean":     float(ai.mean()),
            "ref_std":      float(ai.std(ddof=1)),
            "comp_mean":    float(bi.mean()),
            "comp_std":     float(bi.std(ddof=1)),
            "delta_mean":   float(diff.mean()),
            "cohens_d":     d,
            "t":            float(t_stat),
            "p_value":      float(p_val),
            "sign_K_of_N":  f"{improved}/{len(ai)}",
        })
    return rows


def run(folder: str = "per_seed_metrics",
        out_csv: str | None = None,
        only_eng: str | None = None) -> pd.DataFrame:
    """Run all pairwise comparisons; print table; save CSV; return DataFrame."""
    df = load_all(folder)
    eng_present = sorted(df.eng_type.unique())
    print(f"Loaded {len(df)} per-seed rows from {folder}/")
    print(f"  variants  : {sorted(df.variant.unique())}")
    print(f"  eng_types : {eng_present}")

    # ---- DEBUG: show cwd / file list / dtypes so we can see what's loaded ----
    print(f"\n[debug] cwd                 = {os.getcwd()!r}")
    print(f"[debug] folder arg          = {folder!r}")
    try:
        print(f"[debug] files in folder     = {sorted(os.listdir(folder))}")
    except Exception as _e:
        print(f"[debug] listdir error       : {_e!r}")
    print(f"[debug] df.dtypes           :\n{df.dtypes}")
    print(f"[debug] seed sample values  = {df.seed.head(6).tolist()}")
    # ---- end debug ----------------------------------------------------------

    eng_types = [only_eng] if only_eng else eng_present
    all_rows = []
    for eng in eng_types:
        variants = sorted(df[df.eng_type == eng].variant.unique())
        # Choose reference variants: prefer Baseline and ITT when available.
        refs = [r for r in REFERENCE_VARIANTS if r in variants]
        if not refs:
            # Fallback: take the lexicographically first variant as the ref.
            refs = variants[:1]
        # DEBUG line per engine
        print(f"[debug] eng={eng!r}  variants={variants}  refs={refs}")
        for ref in refs:
            for comp in variants:
                if comp == ref:
                    continue
                rows = paired_comparison(df, ref, comp, eng)
                # DEBUG: show what each pairwise call returns
                if not rows:
                    a_seeds = sorted(df[(df.variant==ref )&(df.eng_type==eng)].seed.tolist())
                    b_seeds = sorted(df[(df.variant==comp)&(df.eng_type==eng)].seed.tolist())
                    common = sorted(set(a_seeds) & set(b_seeds))
                    print(f"[debug]   {ref:10} vs {comp:10} @ {eng}: "
                          f"a_seeds={a_seeds}  b_seeds={b_seeds}  "
                          f"common={common}  -> 0 rows")
                else:
                    print(f"[debug]   {ref:10} vs {comp:10} @ {eng}: "
                          f"returned {len(rows)} rows")
                all_rows.extend(rows)

    if not all_rows:
        print(
            "\nNo comparisons could be made.  Each engine type needs at least "
            "two variants with overlapping seeds (>=2 paired observations).\n"
            "Tip: run the baseline driver `_train_baseline_3seeds.py` so that "
            "you have a Baseline_<eng>.csv to compare every refinement against."
        )
        return pd.DataFrame()

    res = pd.DataFrame(all_rows)

    # Pretty-printed table
    print(f"\n{'=' * 106}")
    print(f"  PAIRED t-TESTS  (variant vs reference;  * = p<0.05;  K/N = seeds improving)")
    print(f"{'=' * 106}")
    print(f"{'eng':6} {'ref':10} {'comp':10} {'metric':6} "
          f"{'ref_mean':>10} {'comp_mean':>10} {'delta':>9} "
          f"{'d':>7} {'t':>7} {'p':>9} {'K/N':>6}")
    print("-" * 106)
    for _, r in res.iterrows():
        star = "*" if r.p_value < 0.05 else " "
        print(f"{r.eng_type:6} {r.ref_variant:10} {r.comp_variant:10} {r.metric:6} "
              f"{r.ref_mean:10.4f} {r.comp_mean:10.4f} {r.delta_mean:9.4f} "
              f"{r.cohens_d:7.3f} {r.t:7.3f} {r.p_value:8.4f}{star} {r.sign_K_of_N:>5}")
    print(f"{'=' * 106}")

    # Brief interpretation footnote
    n_min = res.n_seeds.min()
    n_max = res.n_seeds.max()
    print(
        f"\nNote: n_seeds = {n_min}-{n_max} paired observations per test. "
        f"At n=3 the paired t-test has df=2 and requires |Cohen's d| > ~2.5 "
        f"to reach p<0.05 (two-tailed).  The 'K/N' column reports the number "
        f"of seeds where the comparison variant beat the reference -- a "
        f"K=N result with consistent sign is a stronger small-n indicator "
        f"than the p-value alone."
    )

    out_csv = out_csv or os.path.join(folder, "paired_ttests.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")
    return res


if __name__ == "__main__":
    arg_eng = sys.argv[1] if len(sys.argv) > 1 else None
    run(only_eng=arg_eng)
# %%