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
import gzip
import pickle
import logging
import sys
import numpy as np
from typing import Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import QED

from featurizer import Featurizer

try:
    from applicability_domain import ApplicabilityDomain, DomainReport
    DOMAIN_CHECK_AVAILABLE = True
except Exception:  # pragma: no cover - optional at import time
    ApplicabilityDomain = None
    DomainReport = None
    DOMAIN_CHECK_AVAILABLE = False

logger = logging.getLogger(__name__)

# gzip's magic number. Model files are written compressed (the forests are
# ~35 MB raw, ~11 MB gzipped, and they ship in the repo), but models trained
# before that change are plain pickles, so sniff rather than assume.
_GZIP_MAGIC = b'\x1f\x8b'


def _load_pickle(path: str):
    """Load a model file, compressed or not."""
    with open(path, 'rb') as f:
        head = f.read(2)
    opener = gzip.open if head == _GZIP_MAGIC else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


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
    
    def __init__(self, model_dir: str = None, check_domain: bool = True,
                 allow_domain_build: bool = None):
        # Resolve model directory relative to this file's location (robust for deployment)
        if model_dir is None:
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        else:
            self.model_dir = model_dir
        self.models: Dict[str, object] = {}
        self.metadata: Dict[str, dict] = {}
        # Per-target featurizer, rebuilt from the spec stored in each model
        # file so features always match the model that consumes them.
        self.featurizers: Dict[str, object] = {}
        # Metrics for each evaluation split, so callers can report the honest
        # scaffold-split figure without hardcoding it.
        self.metrics_by_split: Dict[str, dict] = {}
        self._prediction_cache: Dict[str, dict] = {}
        self._domain_cache: Dict[str, object] = {}

        # Applicability domain: tells us whether a prediction is trustworthy or
        # just the forest returning its training mean for an unfamiliar molecule.
        self.domain: Optional[object] = None
        if check_domain and DOMAIN_CHECK_AVAILABLE:
            try:
                self.domain = ApplicabilityDomain(
                    model_dir=self.model_dir,
                    allow_build=allow_domain_build,
                )
            except Exception as e:
                logger.warning(f"Applicability domain unavailable: {e}")

        self._load_models()
    
    def _load_models(self):
        """Load all available QSAR models from disk."""
        loaded = 0
        for target in self.TARGETS:
            model_path = os.path.join(self.model_dir, f"{target}_qsar.pkl")
            if os.path.exists(model_path):
                try:
                    data = _load_pickle(model_path)
                    
                    # Handle both old format (just model) and new format (dict)
                    if isinstance(data, dict):
                        self.models[target] = data['model']
                        self.metadata[target] = data.get('metrics', {})
                        # Rebuild the exact featurizer this model was trained
                        # with. A model file predating featurizer.py carries
                        # no spec and used the legacy Morgan featurization.
                        self.featurizers[target] = Featurizer.from_dict(
                            data.get('featurizer')
                            or data.get('fingerprint_params')
                        )
                        self.metrics_by_split[target] = data.get(
                            'metrics_by_split', {})
                    else:
                        self.models[target] = data
                        self.metadata[target] = {}
                        self.featurizers[target] = Featurizer()
                    
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
    
    def smiles_to_fingerprint(self, smiles: str, target: str = None) -> Optional[np.ndarray]:
        """Feature vector for a molecule, as the target's model expects it.

        `target` picks the featurizer recorded in that model's file. Omitting
        it uses the legacy Morgan featurization, which is what every shipped
        model uses; pass it whenever a target could have been retrained with a
        different feature set.
        """
        featurizer = self.featurizers.get(target) if target else None
        if featurizer is None:
            featurizer = Featurizer()
        return featurizer.featurize_smiles(smiles)
    
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
        
        fp = self.smiles_to_fingerprint(smiles, target)
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
    
    def assess_domain(self, smiles: str, target: str):
        """Judge whether a prediction for this molecule/target is trustworthy.

        A Random Forest returns its training mean for molecules unlike anything
        it has seen, and that output is indistinguishable from a real
        prediction. This compares the molecule against the target's training
        set and reports a confidence tier plus a 0-1 `reliability` scalar.

        Returns:
            A DomainReport, or None if the domain check is unavailable.
        """
        if self.domain is None or target not in self.models:
            return None

        cache_key = f"{smiles}_{target}"
        if cache_key in self._domain_cache:
            return self._domain_cache[cache_key]

        test_rmse = self.metadata.get(target, {}).get('test_rmse')
        try:
            report = self.domain.assess(
                smiles, target,
                model=self.models[target],
                test_rmse=test_rmse if isinstance(test_rmse, (int, float)) else None,
                model_features=self.smiles_to_fingerprint(smiles, target),
            )
        except Exception as e:
            logger.error(f"Domain check failed for {smiles} -> {target}: {e}")
            return None

        self._domain_cache[cache_key] = report
        return report

    def predict_all_targets(self, smiles: str, include_confidence: bool = True) -> Dict[str, dict]:
        """Predict binding against all available targets.

        Returns:
            Dict mapping target name to prediction dict with keys:
            'pIC50', 'IC50_nM', 'binding_affinity_kcal', 'classification',
            and - when the applicability domain is available and
            include_confidence is set - 'confidence', 'reliability',
            'in_domain', 'nn_similarity' and 'confidence_reason'.
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
                
                entry = {
                    'pIC50': round(pic50, 3),
                    'IC50_nM': round(ic50_nm, 2),
                    'binding_affinity_kcal': round(-1.36 * pic50, 2),
                    'classification': classification,
                }

                if include_confidence:
                    report = self.assess_domain(smiles, target)
                    if report is not None:
                        entry.update({
                            'confidence': report.confidence,
                            'reliability': round(report.reliability, 3),
                            'in_domain': report.in_domain,
                            'nn_similarity': report.nn_similarity,
                            'confidence_reason': report.reason,
                        })
                        # An out-of-domain number is not a binding estimate.
                        if not report.in_domain:
                            entry['classification'] = 'unreliable'

                results[target] = entry
        return results
    
    def get_binding_reward(self, smiles: str, target: str,
                           apply_domain_gate: bool = True) -> float:
        """Get normalized binding reward (0-1 scale) for RL training.

        Maps pIC50 to reward:
            pIC50 <= 4  -> 0.0 (inactive)
            pIC50 = 7   -> 0.5 (good)
            pIC50 >= 10  -> 1.0 (excellent)

        This replaces the fake simulated binding reward.

        The raw reward is then scaled by the applicability-domain reliability.
        Without that gate the forest hands back its training mean for molecules
        it has never seen anything like, and the generator gets paid for
        chemical nonsense: methane scores pIC50 6.74 against JAK2, which maps to
        a raw reward of 0.46 before gating and ~0.0 after.

        Args:
            apply_domain_gate: set False to recover the ungated pre-gate
                behaviour (useful for reproducing earlier training runs).
        """
        pic50 = self.predict_pic50(smiles, target)
        if pic50 is None:
            return 0.0

        # Linear mapping: pIC50 4-10 -> reward 0-1
        reward = np.clip((pic50 - 4.0) / 6.0, 0.0, 1.0)

        if apply_domain_gate:
            report = self.assess_domain(smiles, target)
            if report is not None:
                reward = reward * report.reliability

        return float(reward)

    def get_multi_target_binding_reward(self, smiles: str, targets: Dict[str, float],
                                        apply_domain_gate: bool = True) -> Tuple[float, Dict[str, float]]:
        """Get weighted multi-target binding reward.

        Args:
            smiles: SMILES string
            targets: Dict of {target_name: weight} (weights should sum to 1.0)
            apply_domain_gate: scale each target's reward by its
                applicability-domain reliability (see get_binding_reward)

        Returns:
            (total_reward, per_target_rewards)
        """
        per_target = {}
        total_reward = 0.0
        total_weight = 0.0

        for target, weight in targets.items():
            reward = self.get_binding_reward(
                smiles, target, apply_domain_gate=apply_domain_gate)
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
        self._domain_cache.clear()
    
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
                # Scaffold-split performance: the honest estimate on new
                # chemistry, and the number worth quoting. A model file
                # predating this carries only the random-split metrics.
                scaffold = (self.metrics_by_split.get(target, {})
                            .get('scaffold', {}))
                if scaffold:
                    info[target]['scaffold_r2'] = scaffold.get('test_r2')
                    info[target]['scaffold_rmse'] = scaffold.get('test_rmse')
                    info[target]['baseline_rmse'] = scaffold.get('baseline_rmse')
                    if scaffold.get('in_domain_r2') is not None:
                        info[target]['in_domain_r2'] = scaffold['in_domain_r2']
                        info[target]['in_domain_coverage'] = scaffold.get(
                            'in_domain_coverage')

                if self.domain is not None:
                    dom = self.domain.domain_info(target)
                    if dom:
                        info[target]['applicability_domain'] = dom
            else:
                info[target] = {'loaded': False}
        return info


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # This demo prints ΔG and R², which crash the default cp1252 Windows
    # console with UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

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
    
    # Test molecules.
    # Every SMILES here is checked against its expected molecular formula by
    # tests/test_qsar_predictor.py::test_demo_smiles_are_the_drugs_they_claim,
    # because four of the original entries in this dict were the wrong molecule
    # (see git history) and one did not even parse.
    test_molecules = {
        'Aspirin (COX inhibitor)':       'CC(=O)Oc1ccccc1C(=O)O',
        'Ibuprofen (COX inhibitor)':     'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
        'Gefitinib (EGFR inhibitor)':    'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
        'Erlotinib (EGFR inhibitor)':    'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1',
        'Celecoxib (COX2 inhibitor)':    'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1',
        'Ruxolitinib (JAK2 inhibitor)':  'N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1',
        # CHEMBL BACE1 reference, measured pIC50 10.96 in data/BACE1_chembl.csv
        'BACE1 inhibitor (pIC50 10.96)': 'C[C@@]1(c2cc(NC(=O)c3ccc(C#N)cn3)ccc2F)CSCC(N)=N1',
        'Dabigatran (Thrombin inhib)':   'CN1c2ccc(cc2nc1CNc1ccc(cc1)C(N)=N)C(=O)N(CCC(=O)O)c1ccccn1',
        'Naphthalene (no activity)':     'c1ccc2ccccc2c1',
        'Caffeine':                      'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
        'Methane (out of domain)':       'C',
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
            conf = metrics.get('confidence', 'n/a')
            sim = metrics.get('nn_similarity', float('nan'))
            print(f"    {target:10s}: pIC50={metrics['pIC50']:6.2f}  "
                  f"IC50={metrics['IC50_nM']:>10.1f} nM  "
                  f"ΔG≈{metrics['binding_affinity_kcal']:6.2f} kcal/mol  "
                  f"[{metrics['classification']:>10s}]  "
                  f"conf={conf:<13s} nnSim={sim:.2f}")

    # Reward comparison - shows what the applicability-domain gate changes
    print("\n" + "=" * 70)
    print("RL REWARD COMPARISON (EGFR): ungated vs domain-gated")
    print("=" * 70)

    if 'EGFR' in predictor.available_targets:
        print(f"  {'molecule':35s} {'raw':>7s} {'gated':>7s} {'nnSim':>7s}  confidence")
        for name, smiles in test_molecules.items():
            raw = predictor.get_binding_reward(smiles, 'EGFR', apply_domain_gate=False)
            gated = predictor.get_binding_reward(smiles, 'EGFR')
            rep = predictor.assess_domain(smiles, 'EGFR')
            conf = rep.confidence if rep else 'n/a'
            sim = rep.nn_similarity if rep else float('nan')
            print(f"  {name:35s} {raw:7.3f} {gated:7.3f} {sim:7.2f}  {conf}")
    else:
        print("  EGFR model not available")

    print("\nDone!")
