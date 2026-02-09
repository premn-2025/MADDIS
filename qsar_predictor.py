#!/usr/bin/env python3
"""
QSAR Predictor - Real Binding Affinity Prediction

Uses trained Random Forest / Gradient Boosting models on ChEMBL data
to predict pIC50 and binding affinity for 5 targets:
  EGFR, COX2, BACE1, JAK2, THROMBIN

This replaces the fake descriptor-based "docking" with real ML predictions
trained on experimental bioactivity data.
"""

import os
import pickle
import logging
import numpy as np
from typing import Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import QED

logger = logging.getLogger(__name__)


class QSARPredictor:
    """
    Multi-target QSAR predictor using trained ML models.
    
    Usage:
        predictor = QSARPredictor()
        results = predictor.predict_all_targets("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        
        # Per-target prediction
        pic50 = predictor.predict_pic50("CC(=O)Oc1ccccc1C(=O)O", "EGFR")
        affinity = predictor.predict_binding_affinity("CC(=O)Oc1ccccc1C(=O)O", "EGFR")
    """
    
    TARGETS = ['EGFR', 'COX2', 'BACE1', 'JAK2', 'THROMBIN']
    
    # Fingerprint parameters (must match training)
    FP_RADIUS = 2
    FP_NBITS = 2048
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[str, object] = {}
        self.metadata: Dict[str, dict] = {}
        self._prediction_cache: Dict[str, dict] = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load all available QSAR models from disk."""
        loaded = 0
        for target in self.TARGETS:
            model_path = os.path.join(self.model_dir, f"{target}_qsar.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Handle both old format (just model) and new format (dict)
                    if isinstance(data, dict):
                        self.models[target] = data['model']
                        self.metadata[target] = data.get('metrics', {})
                    else:
                        self.models[target] = data
                        self.metadata[target] = {}
                    
                    r2 = self.metadata.get(target, {}).get('test_r2', 'N/A')
                    logger.info(f"Loaded {target} QSAR model (R²={r2})")
                    loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load {target} model: {e}")
            else:
                logger.warning(f"{target} model not found at {model_path}")
        
        if loaded == 0:
            logger.warning(
                "No QSAR models loaded! Run these commands first:\n"
                "  python download_chembl_data.py\n"
                "  python train_qsar_models.py"
            )
        else:
            logger.info(f"QSAR Predictor ready: {loaded}/{len(self.TARGETS)} models loaded")
    
    @property
    def available_targets(self) -> list:
        """Return list of targets that have loaded models."""
        return list(self.models.keys())
    
    @property
    def is_available(self) -> bool:
        """Return True if at least one model is loaded."""
        return len(self.models) > 0
    
    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.FP_RADIUS, nBits=self.FP_NBITS
        )
        return np.array(fp)
    
    def predict_pic50(self, smiles: str, target: str) -> Optional[float]:
        """Predict pIC50 for a molecule against a specific target.
        
        Args:
            smiles: SMILES string
            target: Target protein name (e.g., 'EGFR')
            
        Returns:
            Predicted pIC50 value, or None if prediction fails.
            Higher pIC50 = stronger binding.
            - pIC50 > 8: excellent (IC50 < 10 nM)
            - pIC50 > 7: good (IC50 < 100 nM)
            - pIC50 > 6: moderate (IC50 < 1 µM)
            - pIC50 < 5: weak (IC50 > 10 µM)
        """
        if target not in self.models:
            logger.warning(f"No model for target {target}")
            return None
        
        # Check cache
        cache_key = f"{smiles}_{target}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key].get('pIC50')
        
        fp = self.smiles_to_fingerprint(smiles)
        if fp is None:
            return None
        
        try:
            fp_2d = fp.reshape(1, -1)
            pic50 = float(self.models[target].predict(fp_2d)[0])
            
            # Clamp to reasonable range (pIC50 typically 3-12)
            pic50 = max(3.0, min(12.0, pic50))
            
            # Cache result
            self._prediction_cache[cache_key] = {'pIC50': pic50}
            
            return pic50
        except Exception as e:
            logger.error(f"Prediction failed for {smiles} -> {target}: {e}")
            return None
    
    def predict_binding_affinity(self, smiles: str, target: str) -> Optional[float]:
        """Predict approximate binding affinity in kcal/mol.
        
        Uses the conversion: ΔG ≈ -RT * ln(Ki) ≈ -1.36 * pIC50
        (at 298K, assuming IC50 ≈ Ki)
        
        Returns:
            Binding affinity in kcal/mol (negative = stronger binding).
            E.g., -10.5 kcal/mol is strong binding.
        """
        pic50 = self.predict_pic50(smiles, target)
        if pic50 is None:
            return None
        return -1.36 * pic50
    
    def predict_ic50_nm(self, smiles: str, target: str) -> Optional[float]:
        """Predict IC50 in nanomolar."""
        pic50 = self.predict_pic50(smiles, target)
        if pic50 is None:
            return None
        return 10 ** (-pic50) * 1e9
    
    def predict_all_targets(self, smiles: str) -> Dict[str, dict]:
        """Predict binding against all available targets.
        
        Returns:
            Dict mapping target name to prediction dict with keys:
            'pIC50', 'IC50_nM', 'binding_affinity_kcal', 'classification'
        """
        results = {}
        for target in self.models.keys():
            pic50 = self.predict_pic50(smiles, target)
            if pic50 is not None:
                ic50_nm = 10 ** (-pic50) * 1e9
                
                # Classify binding strength
                if pic50 >= 8:
                    classification = "excellent"
                elif pic50 >= 7:
                    classification = "good"
                elif pic50 >= 6:
                    classification = "moderate"
                elif pic50 >= 5:
                    classification = "weak"
                else:
                    classification = "inactive"
                
                results[target] = {
                    'pIC50': round(pic50, 3),
                    'IC50_nM': round(ic50_nm, 2),
                    'binding_affinity_kcal': round(-1.36 * pic50, 2),
                    'classification': classification,
                }
        return results
    
    def get_binding_reward(self, smiles: str, target: str) -> float:
        """Get normalized binding reward (0-1 scale) for RL training.
        
        Maps pIC50 to reward:
            pIC50 <= 4  -> 0.0 (inactive)
            pIC50 = 7   -> 0.5 (good)
            pIC50 >= 10  -> 1.0 (excellent)
        
        This replaces the fake simulated binding reward.
        """
        pic50 = self.predict_pic50(smiles, target)
        if pic50 is None:
            return 0.0
        
        # Linear mapping: pIC50 4-10 -> reward 0-1
        reward = np.clip((pic50 - 4.0) / 6.0, 0.0, 1.0)
        return float(reward)
    
    def get_multi_target_binding_reward(self, smiles: str, targets: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Get weighted multi-target binding reward.
        
        Args:
            smiles: SMILES string
            targets: Dict of {target_name: weight} (weights should sum to 1.0)
            
        Returns:
            (total_reward, per_target_rewards)
        """
        per_target = {}
        total_reward = 0.0
        total_weight = 0.0
        
        for target, weight in targets.items():
            reward = self.get_binding_reward(smiles, target)
            per_target[target] = reward
            total_reward += weight * reward
            total_weight += weight
        
        if total_weight > 0:
            total_reward /= total_weight
        
        return total_reward, per_target
    
    def compare_molecules(self, smiles_list: list, target: str) -> list:
        """Compare multiple molecules against a target. Returns sorted list."""
        results = []
        for smiles in smiles_list:
            pic50 = self.predict_pic50(smiles, target)
            if pic50 is not None:
                results.append({
                    'smiles': smiles,
                    'pIC50': pic50,
                    'IC50_nM': 10 ** (-pic50) * 1e9,
                    'binding_affinity': -1.36 * pic50,
                })
        
        # Sort by pIC50 descending (best binder first)
        results.sort(key=lambda x: x['pIC50'], reverse=True)
        return results
    
    def clear_cache(self):
        """Clear prediction cache."""
        self._prediction_cache.clear()
    
    def get_model_info(self) -> Dict[str, dict]:
        """Get metadata about loaded models."""
        info = {}
        for target in self.TARGETS:
            if target in self.models:
                meta = self.metadata.get(target, {})
                info[target] = {
                    'loaded': True,
                    'model_type': meta.get('model_type', 'Unknown'),
                    'test_r2': meta.get('test_r2', 'N/A'),
                    'test_rmse': meta.get('test_rmse', 'N/A'),
                    'n_training': meta.get('n_train', 'N/A'),
                }
            else:
                info[target] = {'loaded': False}
        return info


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = QSARPredictor()
    
    if not predictor.is_available:
        print("\nNo models found! Train them first:")
        print("  python download_chembl_data.py")
        print("  python train_qsar_models.py")
        exit(1)
    
    # Model info
    print("\n" + "=" * 70)
    print("LOADED MODELS")
    print("=" * 70)
    for target, info in predictor.get_model_info().items():
        if info['loaded']:
            print(f"  {target:10s}: {info['model_type']:18s} R²={info['test_r2']}")
        else:
            print(f"  {target:10s}: NOT LOADED")
    
    # Test molecules
    test_molecules = {
        'Aspirin (COX inhibitor)':       'CC(=O)Oc1ccccc1C(=O)O',
        'Ibuprofen (COX inhibitor)':     'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
        'Gefitinib (EGFR inhibitor)':    'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
        'Erlotinib (EGFR inhibitor)':    'C=Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1',
        'Celecoxib (COX2 inhibitor)':    'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1',
        'Ruxolitinib (JAK2 inhibitor)':  'N#Cc1ccnc(NC2CC2)c1c1cccc2[nH]ccc12',
        'Verubecestat (BACE1 inhibitor)':'CC(C)(C)c1nc2cc(F)ccc2c(=O)n1-c1ccc(C#N)cc1',
        'Dabigatran (Thrombin inhib)':   'Cn1c(=O)c(CC(=O)O)c2cc(CNc3ccc(C(=N)N)cc3)ccc21',
        'Naphthalene (no activity)':     'c1ccc2ccccc2c1',
        'Caffeine':                      'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    }
    
    print("\n" + "=" * 70)
    print("MULTI-TARGET PREDICTIONS")
    print("=" * 70)
    
    for name, smiles in test_molecules.items():
        print(f"\n  {name}")
        print(f"  SMILES: {smiles}")
        
        results = predictor.predict_all_targets(smiles)
        
        if not results:
            print("  No predictions available")
            continue
        
        for target, metrics in sorted(results.items()):
            bar = "█" * int(metrics['pIC50']) + "░" * (10 - int(metrics['pIC50']))
            print(f"    {target:10s}: pIC50={metrics['pIC50']:6.2f}  "
                  f"IC50={metrics['IC50_nM']:>10.1f} nM  "
                  f"ΔG≈{metrics['binding_affinity_kcal']:6.2f} kcal/mol  "
                  f"[{metrics['classification']:>10s}]  {bar}")
    
    # Reward comparison
    print("\n" + "=" * 70)
    print("RL REWARD COMPARISON (EGFR)")
    print("=" * 70)
    
    if 'EGFR' in predictor.available_targets:
        for name, smiles in test_molecules.items():
            reward = predictor.get_binding_reward(smiles, 'EGFR')
            bar = "█" * int(reward * 20) + "░" * (20 - int(reward * 20))
            print(f"  {name:35s}: reward={reward:.3f}  {bar}")
    else:
        print("  EGFR model not available")
    
    print("\nDone!")
