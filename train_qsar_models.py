#!/usr/bin/env python3
"""
Train QSAR Models for Multi-Target Binding Prediction

Trains Random Forest regressors on ChEMBL bioactivity data using Morgan fingerprints.
One model per target: EGFR, COX2, BACE1, JAK2, THROMBIN

Outputs: Pickle files in models/ directory
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def load_and_featurize(target_name: str, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV and convert SMILES to fingerprints."""
    csv_path = os.path.join(data_dir, f"{target_name}_chembl.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}\nRun download_chembl_data.py first.")
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} compounds from {csv_path}")
    
    fps = []
    pic50s = []
    skipped = 0
    
    for smiles, pic50 in zip(df['smiles'], df['pIC50']):
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
            pic50s.append(pic50)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped} invalid SMILES")
    
    X = np.array(fps)
    y = np.array(pic50s)
    
    print(f"  Feature matrix: {X.shape}")
    print(f"  pIC50 range: {y.min():.2f} - {y.max():.2f} (mean={y.mean():.2f})")
    
    return X, y


def train_qsar_model(target_name: str, data_dir: str = "data", model_dir: str = "models") -> Dict:
    """Train and evaluate a QSAR model for one target.
    
    Returns dict with model, metrics, and metadata.
    """
    print(f"\n{'=' * 60}")
    print(f"Training QSAR model: {target_name}")
    print(f"{'=' * 60}")
    
    # Load data
    X, y = load_and_featurize(target_name, data_dir)
    
    if len(X) < 50:
        print(f"  WARNING: Only {len(X)} compounds. Model may be unreliable.")
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Random Forest (fast, parallelizable, good for fingerprints)
    print("  Training Random Forest (200 trees)...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'model_type': 'RandomForest',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"\n  Results (RandomForest):")
    print(f"    Train R²  = {metrics['train_r2']:.3f}, RMSE = {metrics['train_rmse']:.3f}")
    print(f"    Test  R²  = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}, MAE = {metrics['test_mae']:.3f}")
    
    # Save model + metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{target_name}_qsar.pkl")
    
    save_data = {
        'model': model,
        'target': target_name,
        'metrics': metrics,
        'fingerprint_params': {'radius': 2, 'n_bits': 2048},
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"  Model saved to {model_path}")
    
    return metrics


def main():
    """Train QSAR models for all 5 targets."""
    
    targets = ['EGFR', 'COX2', 'BACE1', 'JAK2', 'THROMBIN']
    
    print("=" * 60)
    print("QSAR Model Training Pipeline")
    print(f"Targets: {', '.join(targets)}")
    print("=" * 60)
    
    # Check data availability
    data_dir = "data"
    missing = []
    for t in targets:
        if not os.path.exists(os.path.join(data_dir, f"{t}_chembl.csv")):
            missing.append(t)
    
    if missing:
        print(f"\nMissing data for: {', '.join(missing)}")
        print("Run 'python download_chembl_data.py' first.")
        sys.exit(1)
    
    # Train all models
    all_metrics = {}
    
    for target in targets:
        try:
            metrics = train_qsar_model(target)
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
    
    print("\nNext step: python qsar_predictor.py")


if __name__ == "__main__":
    main()
