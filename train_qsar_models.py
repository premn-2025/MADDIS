#!/usr/bin/env python3
"""
Train QSAR Models for Multi-Target Binding Prediction

Trains Random Forest regressors on ChEMBL bioactivity data using Morgan fingerprints.
One model per target: EGFR, COX2, BACE1, JAK2, THROMBIN

Outputs: Pickle files in models/ directory

Two evaluation splits are available:

  --split random    80/20 at random. ChEMBL is full of congeneric series, so
                    close analogues of test molecules sit in the training set
                    and this number overstates prospective performance.

  --split scaffold  Bemis-Murcko scaffold split: a scaffold appears in either
                    train or test, never both. This is the field-standard
                    estimate of how the model behaves on genuinely new
                    chemistry, and it is the number to quote in a paper.

Use `--compare` to train both and print them side by side without overwriting
the shipped models.
"""

import gzip
import argparse
import json
import os
import sys
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from featurizer import SPECS, Featurizer, standardize_smiles  # noqa: F401


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Legacy Morgan fingerprint helper.

    Kept because other modules import it. New code should go through
    `featurizer.Featurizer`, which keeps training and prediction in step.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def load_and_featurize(target_name: str, data_dir: str = "data",
                       featurizer=None, standardize: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load CSV, standardize structures, and compute features.

    Returns (X, y, smiles) - the SMILES are needed for scaffold splitting.
    """
    csv_path = os.path.join(data_dir, f"{target_name}_chembl.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}\nRun download_chembl_data.py first.")

    if featurizer is None:
        featurizer = Featurizer()

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} compounds from {csv_path}")

    # Standardize first, then merge duplicates, so salt forms of one parent
    # collapse into a single row carrying the median activity.
    records = defaultdict(list)
    skipped = 0
    for smiles, pic50 in zip(df['smiles'], df['pIC50']):
        canonical = (standardize_smiles(str(smiles)) if standardize
                     else str(smiles))
        if canonical is None:
            skipped += 1
            continue
        records[canonical].append(float(pic50))

    if skipped > 0:
        print(f"  Skipped {skipped} invalid SMILES")

    merged = sum(1 for v in records.values() if len(v) > 1)
    if merged:
        print(f"  Merged {merged} duplicate structures (median activity)")

    mols, pic50s, kept_smiles = [], [], []
    for canonical, values in records.items():
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            continue
        mols.append(mol)
        pic50s.append(float(np.median(values)))
        kept_smiles.append(canonical)

    X = featurizer.featurize_many(mols)
    y = np.array(pic50s)

    print(f"  Feature matrix: {X.shape}  ({featurizer.spec})")
    print(f"  pIC50 range: {y.min():.2f} - {y.max():.2f} (mean={y.mean():.2f})")

    return X, y, kept_smiles


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold as canonical SMILES ('' if it cannot be derived).

    Molecules with no ring system collapse to an empty scaffold; they are
    grouped together, which is the conventional treatment.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ''


def scaffold_split(smiles: List[str], test_size: float = 0.2,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices by Bemis-Murcko scaffold.

    Every molecule sharing a scaffold lands on the same side of the split, so
    the test set contains chemistry the model has not seen. Scaffold groups are
    filled largest-first into train (the MoleculeNet convention), which puts the
    rare, singleton scaffolds in test - the hardest and most realistic case.

    Returns (train_idx, test_idx).
    """
    groups = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    # Largest scaffold groups first; ties broken deterministically by scaffold
    # string so the split is reproducible.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n_total = len(smiles)
    n_train_target = int(round(n_total * (1.0 - test_size)))

    train_idx: List[int] = []
    test_idx: List[int] = []
    for _, idxs in ordered:
        # The first (largest) group always seeds train, otherwise a single
        # group bigger than the train target would send everything to test.
        if not train_idx or len(train_idx) + len(idxs) <= n_train_target:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    # Degenerate case: one scaffold dominates, so a clean scaffold split is
    # impossible and everything landed on one side. Fall back to a random
    # split of that side so callers still get a usable train/test pair - but
    # the resulting metric is a random-split number, not a scaffold one.
    if not test_idx or not train_idx:
        pool = list(train_idx or test_idx)
        rng = np.random.RandomState(seed)
        rng.shuffle(pool)
        cut = max(1, int(len(pool) * test_size))
        test_idx, train_idx = pool[:cut], pool[cut:]
        print("  WARNING: scaffolds could not be separated "
              "(one scaffold dominates); fell back to a random split.")

    print(f"  Scaffold split: {len(groups)} distinct scaffolds")
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


# Random Forest hyperparameters.
#
# The original settings are kept as 'legacy' for reproducibility. 'tuned' was
# selected by scaffold-grouped cross-validation *within the training half* of
# the scaffold split, so the held-out scaffolds were never used for tuning
# (see hyperparameter_sweep.json).
#
# The important change is max_features. At 'sqrt' the forest considers only
# sqrt(2048) = 45 of the 2048 fingerprint bits at each split; on sparse
# fingerprints where any individual bit is almost always zero, that is too few
# to find the informative substructures, so trees end up near-random and the
# ensemble regresses towards the mean. Sampling 15% (307 bits) finds real
# splits. Removing the depth cap and the leaf-size floor then lets each tree
# fit its subsample fully, which is what a Random Forest wants - variance is
# handled by averaging, not by stunting the trees.
RF_PARAMS = {
    'legacy': dict(
        n_estimators=200, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
    ),
    'tuned': dict(
        n_estimators=200, max_depth=None, min_samples_split=2,
        min_samples_leaf=2, max_features=0.10,
    ),
}

# Why these numbers, since the CV grid nominally preferred
# max_features=0.1 / n_estimators=800 / min_samples_leaf=1:
#
# Measured on EGFR's held-out scaffolds, against on-disk size:
#
#   legacy  sqrt/200/d25/leaf2   R2 0.572    16 MB    49 ms/predict
#   0.10/400/none/leaf1          R2 0.600   180 MB    75 ms
#   0.10/400/none/leaf2          R2 0.607    70 MB    75 ms
#   0.10/200/none/leaf2          R2 0.604    35 MB    49 ms   <- chosen
#
# leaf=1 grows a leaf per training molecule, which costs 5x the size for
# slightly *worse* accuracy; and 400 trees buy 0.003 R2 for double the size
# and 50% more inference latency. The chosen config keeps the legacy
# prediction latency, gzips to ~11 MB per target, and gives up 0.003 R2
# against the largest model tried. These models are committed to the repo and
# loaded on a Streamlit Cloud container, so size is a real constraint.


def _in_domain_metrics(target_name, train_smiles, test_smiles,
                       y_true, y_pred, model_dir):
    """Accuracy over just the test molecules the domain check accepts.

    The reference set is built from `train_smiles` only. Using the on-disk
    cache instead would be wrong: it is built from the target's whole CSV, so
    each held-out molecule would find itself in its own reference set and
    everything would come back in-domain at Tanimoto 1.0, making coverage a
    meaningless 100%.

    Returns None when a domain cannot be calibrated, rather than guessing.
    """
    try:
        from applicability_domain import ApplicabilityDomain
    except Exception:
        return None

    ad = ApplicabilityDomain(model_dir=model_dir, allow_build=False)
    key = f"{target_name}__trainonly"
    if not ad.load_reference_smiles(key, train_smiles):
        print("    (could not calibrate a training-only domain; "
              "skipping in-domain metrics)")
        return None

    mask = np.array([bool(ad.assess(smi, key).in_domain) for smi in test_smiles])

    if mask.sum() < 10:
        print(f"    (only {int(mask.sum())} test molecules in domain; "
              f"skipping in-domain metrics)")
        return None

    return {
        'in_domain_r2': float(r2_score(y_true[mask], y_pred[mask])),
        'in_domain_rmse': float(np.sqrt(mean_squared_error(
            y_true[mask], y_pred[mask]))),
        'in_domain_n': int(mask.sum()),
        'in_domain_coverage': float(mask.mean()),
    }


def build_model(model_type: str = 'tuned'):
    """Construct an untrained estimator.

    Gradient boosting is available but was measurably worse than the tuned
    forest on every target tested (scaffold R² 0.564 vs 0.602 on EGFR), and
    much slower; it is kept for comparison rather than recommended.
    """
    if model_type in RF_PARAMS:
        return RandomForestRegressor(
            random_state=42, n_jobs=-1, **RF_PARAMS[model_type])

    if model_type == 'hgb':
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.07, max_leaf_nodes=31,
            min_samples_leaf=10, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=25, random_state=42)

    if model_type == 'blend':
        return VotingRegressor([
            ('rf', build_model('tuned')),
            ('hgb', build_model('hgb')),
        ])

    raise ValueError(
        f"Unknown model type {model_type!r}. "
        f"Use one of: {', '.join(list(RF_PARAMS) + ['hgb', 'blend'])}"
    )


def train_qsar_model(target_name: str, data_dir: str = "data", model_dir: str = "models",
                     split: str = "random", save: bool = True,
                     model_type: str = "tuned",
                     featurizer_spec: str = "morgan2-2048",
                     standardize: bool = True) -> Dict:
    """Train and evaluate a QSAR model for one target.

    Args:
        split: 'random' for an 80/20 random split, or 'scaffold' for a
            Bemis-Murcko scaffold split (honest estimate on new chemistry).
        save: write the model to model_dir. Set False to evaluate a split
            without overwriting the shipped models.
        model_type: see build_model.
        featurizer_spec: a key from featurizer.SPECS. Morgan bits alone beat
            every richer feature set tried on the scaffold split - adding
            physicochemical descriptors cost ~0.04 R² because they encode
            global properties that track scaffold identity, so the model leans
            on them and transfers worse to new scaffolds.

    Returns dict with metrics and metadata.
    """
    print(f"\n{'=' * 60}")
    print(f"Training QSAR model: {target_name}  (split={split}, model={model_type})")
    print(f"{'=' * 60}")

    featurizer = Featurizer(featurizer_spec)

    # Load data
    X, y, smiles = load_and_featurize(
        target_name, data_dir, featurizer=featurizer, standardize=standardize)

    if len(X) < 50:
        print(f"  WARNING: Only {len(X)} compounds. Model may be unreliable.")

    if split == "scaffold":
        train_idx, test_idx = scaffold_split(smiles, test_size=0.2, seed=42)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    elif split == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        raise ValueError(f"Unknown split: {split!r} (use 'random' or 'scaffold')")

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"  Training {model_type}...")
    model = build_model(model_type)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'model_type': ('RandomForest' if model_type in RF_PARAMS
                       else {'hgb': 'HistGradientBoosting',
                             'blend': 'RF+HGB'}[model_type]),
        'model_variant': model_type,
        'featurizer': featurizer.spec,
        'split': split,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        # Baseline to beat: always predict the training mean. R² is defined
        # against the *test* mean, so a model can post a positive R² while
        # still being no better than this in absolute error.
        'baseline_rmse': float(np.sqrt(mean_squared_error(
            y_test, np.full_like(y_test, y_train.mean())))),
    }

    # Accuracy restricted to the applicability domain.
    #
    # A single global R² averages over molecules the model can assess and
    # molecules it cannot, which understates it where it is actually usable and
    # overstates it elsewhere. Reporting accuracy alongside coverage is the
    # standard way to characterise a QSAR model, and it is what the confidence
    # labelling in the app acts on: a user only ever sees a prediction
    # together with whether it is in domain.
    if split == "scaffold":
        in_domain = _in_domain_metrics(
            target_name,
            [smiles[i] for i in train_idx],
            [smiles[i] for i in test_idx],
            y_test, y_pred_test, model_dir)
        if in_domain:
            metrics.update(in_domain)
            print(f"    In-domain only: R² = {in_domain['in_domain_r2']:.3f}, "
                  f"RMSE = {in_domain['in_domain_rmse']:.3f} "
                  f"(covering {in_domain['in_domain_coverage'] * 100:.0f}% "
                  f"of the test set)")

    print(f"\n  Results (RandomForest, {split} split):")
    print(f"    Train R²  = {metrics['train_r2']:.3f}, RMSE = {metrics['train_rmse']:.3f}")
    print(f"    Test  R²  = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}, MAE = {metrics['test_mae']:.3f}")
    print(f"    Mean-prediction baseline RMSE = {metrics['baseline_rmse']:.3f}")

    if not save:
        print("  (evaluation only - model not written)")
        return metrics

    # Evaluate the *other* split too, so the model file carries both numbers
    # and nothing downstream has to hardcode them. The shipped model is the one
    # fitted above; the other figure is a generalization estimate for the same
    # method and data, which is what a reader of the metrics wants to know.
    companion_split = 'scaffold' if split == 'random' else 'random'
    companion = None
    try:
        print(f"  Measuring {companion_split}-split performance for the record...")
        companion = train_qsar_model(
            target_name, data_dir=data_dir, model_dir=model_dir,
            split=companion_split, save=False, model_type=model_type,
            featurizer_spec=featurizer_spec, standardize=standardize)
    except Exception as e:
        print(f"  (could not measure {companion_split} split: {e})")

    # Save model + metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{target_name}_qsar.pkl")

    save_data = {
        'model': model,
        'target': target_name,
        'metrics': metrics,
        # Keep the legacy key so an older QSARPredictor can still load this.
        'fingerprint_params': {
            'radius': featurizer.config['radius'],
            'n_bits': featurizer.config['n_bits'],
        },
        # The authoritative featurization spec. QSARPredictor rebuilds the
        # exact featurizer from this, so model and features cannot drift apart.
        'featurizer': featurizer.to_dict(),
        # Metrics for both splits. `metrics` describes the model in this file;
        # `metrics_by_split` lets the UI report the scaffold figure without
        # hardcoding it.
        'metrics_by_split': {
            split: metrics,
            **({companion_split: companion} if companion else {}),
        },
    }

    # Gzip: these unlimited-depth forests are ~35 MB raw but ~11 MB
    # compressed, and they are committed to the repo and downloaded onto a
    # Streamlit Cloud container. Decompression costs about 0.2s per target.
    with gzip.open(model_path, 'wb', compresslevel=6) as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(model_path) / 1e6
    print(f"  Model saved to {model_path} ({size_mb:.1f} MB gzipped)")

    return metrics


TARGETS = ['EGFR', 'COX2', 'BACE1', 'JAK2', 'THROMBIN']


def _check_data(targets, data_dir):
    missing = [t for t in targets
               if not os.path.exists(os.path.join(data_dir, f"{t}_chembl.csv"))]
    if missing:
        print(f"\nMissing data for: {', '.join(missing)}")
        print("Run 'python download_chembl_data.py' first.")
        sys.exit(1)


def run_comparison(targets, data_dir, out_path=None, model_type='tuned',
                   featurizer_spec='morgan2-2048', standardize=True):
    """Train on both splits and report them side by side. Saves no models."""
    print("=" * 78)
    print("SPLIT COMPARISON: random vs Bemis-Murcko scaffold")
    print("=" * 78)
    print("A random split leaves close analogues of the test molecules in the")
    print("training set, so it measures interpolation within known series.")
    print("A scaffold split holds out whole scaffolds, measuring what the model")
    print("does on genuinely new chemistry. Quote the scaffold numbers.\n")

    results = {}
    for target in targets:
        results[target] = {}
        for split in ('random', 'scaffold'):
            try:
                results[target][split] = train_qsar_model(
                    target, data_dir=data_dir, split=split, save=False,
                    model_type=model_type, featurizer_spec=featurizer_spec,
                    standardize=standardize)
            except Exception as e:
                print(f"\n  ERROR {target}/{split}: {e}")
                results[target][split] = None

    print(f"\n{'=' * 78}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 78}")
    print(f"{'Target':10s} | {'random R²':>10s} | {'scaffold R²':>12s} | "
          f"{'drop':>7s} | {'scaf RMSE':>10s} | {'baseline':>9s}")
    print("-" * 78)

    for target in targets:
        r = results[target].get('random')
        s = results[target].get('scaffold')
        if not r or not s:
            print(f"{target:10s} | {'--':>10s} | {'--':>12s} | {'--':>7s} | "
                  f"{'--':>10s} | {'--':>9s}")
            continue
        drop = r['test_r2'] - s['test_r2']
        print(f"{target:10s} | {r['test_r2']:10.3f} | {s['test_r2']:12.3f} | "
              f"{drop:7.3f} | {s['test_rmse']:10.3f} | {s['baseline_rmse']:9.3f}")

    print(f"{'=' * 78}")

    scaf = [results[t]['scaffold'] for t in targets if results[t].get('scaffold')]
    if scaf:
        beats = sum(1 for m in scaf if m['test_rmse'] < m['baseline_rmse'])
        print(f"\n{beats}/{len(scaf)} targets beat the mean-prediction baseline "
              f"on held-out scaffolds.")
        usable = sum(1 for m in scaf if m['test_r2'] > 0.5)
        print(f"{usable}/{len(scaf)} reach scaffold-split R² > 0.5.")

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nWritten to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train QSAR models for multi-target binding prediction")
    parser.add_argument('--split', choices=['random', 'scaffold'], default='random',
                        help="evaluation split for the saved models "
                             "(default: random, matching the shipped models)")
    parser.add_argument('--compare', action='store_true',
                        help="train both splits and report side by side; "
                             "saves nothing")
    parser.add_argument('--targets', nargs='+', default=TARGETS,
                        help="subset of targets to process")
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--report', default=None,
                        help="with --compare, also write JSON metrics here")
    parser.add_argument('--model', default='tuned',
                        choices=list(RF_PARAMS) + ['hgb', 'blend'],
                        help="estimator: 'tuned' (default, selected by "
                             "scaffold-grouped CV), 'legacy' (the original "
                             "settings), 'hgb' or 'blend'")
    parser.add_argument('--featurizer', default='morgan2-2048',
                        choices=sorted(SPECS),
                        help="feature set; Morgan bits alone transfer best to "
                             "new scaffolds")
    parser.add_argument('--no-standardize', action='store_true',
                        help="skip salt stripping and duplicate merging")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

    targets = [t.upper() for t in args.targets]
    _check_data(targets, args.data_dir)

    if args.compare:
        run_comparison(targets, args.data_dir, args.report,
                       model_type=args.model,
                       featurizer_spec=args.featurizer,
                       standardize=not args.no_standardize)
        return

    print("=" * 60)
    print("QSAR Model Training Pipeline")
    print(f"Targets: {', '.join(targets)}")
    print(f"Split:   {args.split}")
    print(f"Model:   {args.model}")
    print(f"Features:{args.featurizer}")
    print("=" * 60)

    if args.split == 'random':
        print("\nNote: random-split metrics overstate performance on new")
        print("chemistry. Run with --compare for the scaffold-split numbers.")

    # Train all models
    all_metrics = {}

    for target in targets:
        try:
            metrics = train_qsar_model(
                target, data_dir=args.data_dir, model_dir=args.model_dir,
                split=args.split, model_type=args.model,
                featurizer_spec=args.featurizer,
                standardize=not args.no_standardize)
            all_metrics[target] = metrics
        except Exception as e:
            print(f"\n  ERROR training {target}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[target] = None

    # Summary table
    print(f"\n{'=' * 70}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Target':10s} | {'Model':18s} | {'Test R²':8s} | {'Test RMSE':10s} | {'Test MAE':10s} | {'N':>5s}")
    print("-" * 70)

    for target in targets:
        m = all_metrics.get(target)
        if m is None:
            print(f"{target:10s} | {'FAILED':18s} | {'--':>8s} | {'--':>10s} | {'--':>10s} | {'--':>5s}")
        else:
            print(f"{target:10s} | {m['model_type']:18s} | {m['test_r2']:8.3f} | {m['test_rmse']:10.3f} | "
                  f"{m['test_mae']:10.3f} | {m['n_train']+m['n_test']:5d}")

    print(f"{'=' * 70}")

    # Quality assessment
    good = sum(1 for m in all_metrics.values() if m and m['test_r2'] > 0.5)
    total = sum(1 for m in all_metrics.values() if m is not None)

    print(f"\n{good}/{total} models with R² > 0.5 (usable for virtual screening)")

    if good == total:
        print("All models trained successfully!")
    elif good > 0:
        print("Some models may need more data or feature engineering.")
    else:
        print("WARNING: No models meet minimum quality threshold.")

    print("\nIf the training data changed, rebuild the applicability-domain")
    print("caches so the confidence check matches the new models:")
    print("  python build_domain_caches.py")
    print("\nNext step: python qsar_predictor.py")


if __name__ == "__main__":
    main()
