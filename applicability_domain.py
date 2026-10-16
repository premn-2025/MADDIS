#!/usr/bin/env python3
"""
Applicability Domain for the QSAR models.

A Random Forest always returns a number. For a molecule unlike anything it was
trained on, that number is just the training-set mean, and it looks exactly as
confident as a real prediction. Measured on the shipped models:

    methane ("C") -> JAK2 pIC50 6.74    (JAK2 training mean: 7.13)
    ruxolitinib   -> JAK2 pIC50 8.08    (a real JAK2 inhibitor)

Those two are 1.3 log units apart while methane is not a kinase inhibitor at all.
Because `QSARPredictor.get_binding_reward` feeds pIC50 straight into the RL
reward, a generator can be rewarded for producing chemically meaningless output.

This module answers "should we believe this prediction?" using two signals:

1. **Nearest-neighbour Tanimoto similarity** to the target's training set.
   This is the discriminating signal. Calibrated on the shipped data, in-domain
   molecules have a median nearest-neighbour similarity of 0.80-0.83, while
   methane/ethanol/benzene land at 0.02-0.23.

2. **Per-tree prediction spread** across the forest's estimators, compared with
   the model's own held-out RMSE. A secondary signal: it catches molecules that
   are similar enough to the training set but land in a region where the trees
   disagree.

Thresholds are not hand-picked constants. For each target they are derived from
the leave-one-out nearest-neighbour similarity distribution of that target's own
training set, so a sparse target (THROMBIN, n=1801) gets a looser domain than a
dense one (JAK2, n=7228).

Usage:
    from applicability_domain import ApplicabilityDomain

    ad = ApplicabilityDomain()
    report = ad.assess("C", "JAK2", model)
    report.confidence    # 'out_of_domain'
    report.reliability   # 0.0  -> use to gate an RL reward
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from featurizer import standardize_smiles

logger = logging.getLogger(__name__)

# Fingerprint parameters - must match QSARPredictor / train_qsar_models.py
FP_RADIUS = 2
FP_NBITS = 2048

# Percentiles of the in-domain nearest-neighbour similarity distribution that
# define the confidence tiers. p1 is the outer edge of the domain: 99% of real
# training molecules have a neighbour at least this similar.
_TIER_PERCENTILES = (1, 5, 50)

# Number of training molecules sampled when calibrating the leave-one-out
# similarity distribution. 400 is enough to pin the percentiles we need and
# keeps cache construction to a few seconds per target.
_CALIBRATION_SAMPLE = 400

CONFIDENCE_ORDER = ['out_of_domain', 'low', 'medium', 'high']


@dataclass
class DomainReport:
    """The verdict on one (molecule, target) prediction."""

    target: str
    in_domain: bool
    confidence: str  # one of CONFIDENCE_ORDER
    reliability: float  # 0.0-1.0, suitable as a reward multiplier
    nn_similarity: float  # max Tanimoto to the training set
    nn_similarity_top5: float  # mean Tanimoto over the 5 nearest neighbours
    tree_std: Optional[float]  # spread across forest estimators
    reason: str  # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


def _popcount(packed: np.ndarray) -> np.ndarray:
    """Count set bits along the last axis, on numpy 1.x or 2.x."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(packed).sum(axis=-1)
    # numpy < 2.0: byte-wise lookup table
    table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)
    return table[packed].sum(axis=-1)


def smiles_to_bits(smiles: str) -> Optional[np.ndarray]:
    """Morgan fingerprint as a uint8 array of 0/1, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp, dtype=np.uint8)


class ApplicabilityDomain:
    """Similarity-to-training-set domain check for the per-target QSAR models.

    Training fingerprints are bit-packed and cached to
    ``models/<TARGET>_ad.npz`` (about 0.25 KB per molecule), so the check costs
    one vectorised pass over a few thousand rows.
    """

    def __init__(self, model_dir: str = None, data_dir: str = None,
                 allow_build: bool = None):
        """
        Args:
            allow_build: whether a missing cache may be built on demand.
                Building fingerprints a target's whole training set, which
                takes roughly 40 seconds across the five shipped targets - far
                too long to do inside a web request, so a server should pass
                False and ship prebuilt caches. Defaults to the
                MADDIS_BUILD_DOMAIN_CACHE environment variable, else True.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(here, 'models')
        self.data_dir = data_dir or os.path.join(here, 'data')
        if allow_build is None:
            allow_build = os.getenv('MADDIS_BUILD_DOMAIN_CACHE', '1') != '0'
        self.allow_build = allow_build
        self._cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Cache construction / loading
    # ------------------------------------------------------------------

    def _cache_path(self, target: str) -> str:
        return os.path.join(self.model_dir, f'{target}_ad.npz')

    def _training_csv(self, target: str) -> str:
        return os.path.join(self.data_dir, f'{target}_chembl.csv')

    def build_cache(self, target: str, force: bool = False) -> bool:
        """Build the packed-fingerprint cache for one target.

        Returns True if a usable cache exists afterwards.
        """
        path = self._cache_path(target)
        if os.path.exists(path) and not force:
            return True

        if not self.allow_build and not force:
            logger.warning(
                f"Applicability-domain cache for {target} is missing and "
                f"on-demand building is disabled. Run build_domain_caches.py "
                f"and ship models/{target}_ad.npz; confidence will be "
                f"unavailable until then."
            )
            return False

        csv_path = self._training_csv(target)
        if not os.path.exists(csv_path):
            logger.warning(
                f'Cannot build applicability domain for {target}: '
                f'{csv_path} not found'
            )
            return False

        import pandas as pd

        df = pd.read_csv(csv_path)
        if 'smiles' not in df.columns:
            logger.warning(f'{csv_path} has no "smiles" column')
            return False

        # Standardize exactly as training does, so the reference set the
        # domain check compares against is the set the model learned from.
        # A salt form's fingerprint carries counter-ion bits, which would
        # depress its similarity to the parent for no chemical reason.
        seen = set()
        bits = []
        for smi in df['smiles']:
            canonical = standardize_smiles(str(smi))
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            b = smiles_to_bits(canonical)
            if b is not None:
                bits.append(b)

        if not bits:
            logger.warning(f'No parseable molecules in {csv_path}')
            return False

        packed = np.packbits(np.vstack(bits), axis=-1)
        pop = _popcount(packed).astype(np.int32)

        tiers = self._calibrate(packed, pop)

        np.savez_compressed(
            path,
            packed=packed,
            pop=pop,
            tiers=np.asarray(tiers, dtype=np.float64),
            n_train=np.asarray(len(packed)),
        )
        logger.info(
            f'Built applicability domain for {target}: n={len(packed)}, '
            f'tiers={[round(t, 3) for t in tiers]}'
        )
        return True

    def _calibrate(self, packed: np.ndarray, pop: np.ndarray) -> List[float]:
        """Leave-one-out nearest-neighbour similarity percentiles."""
        n = len(packed)
        rng = np.random.RandomState(0)
        idx = rng.choice(n, size=min(_CALIBRATION_SAMPLE, n), replace=False)

        loo = []
        for i in idx:
            sims = self._tanimoto_all(packed, pop, packed[i])
            sims[i] = -1.0  # exclude the molecule itself
            loo.append(float(sims.max()))

        return [float(np.percentile(loo, p)) for p in _TIER_PERCENTILES]

    def load_reference_smiles(self, key: str, smiles: Sequence[str]) -> bool:
        """Register an in-memory reference set under `key`, bypassing disk.

        Needed to measure accuracy-within-domain honestly. The on-disk cache is
        built from a target's whole CSV, so evaluating held-out molecules
        against it finds each test molecule in its own reference set: every
        prediction comes back "in domain" at Tanimoto 1.0 and coverage is a
        meaningless 100%. Passing only the training half gives a reference set
        that has genuinely not seen the test molecules.
        """
        seen = set()
        bits = []
        for smi in smiles:
            canonical = standardize_smiles(str(smi))
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            b = smiles_to_bits(canonical)
            if b is not None:
                bits.append(b)

        if len(bits) < 20:
            logger.warning(
                f"Only {len(bits)} usable reference molecules for {key!r}; "
                f"too few to calibrate a domain."
            )
            self._cache[key] = None
            return False

        packed = np.packbits(np.vstack(bits), axis=-1)
        pop = _popcount(packed).astype(np.int32)
        self._cache[key] = {
            'packed': packed,
            'pop': pop,
            'tiers': self._calibrate(packed, pop),
            'n_train': len(packed),
        }
        return True

    def _load(self, target: str) -> Optional[dict]:
        """Load (building if needed) the cache for one target."""
        if target in self._cache:
            return self._cache[target]

        # An in-memory reference registered by load_reference_smiles has no
        # on-disk counterpart; never try to build one for it.
        if '__' in target:
            return None

        if not self.build_cache(target):
            self._cache[target] = None
            return None

        try:
            with np.load(self._cache_path(target)) as z:
                entry = {
                    'packed': z['packed'],
                    'pop': z['pop'].astype(np.int32),
                    'tiers': z['tiers'].tolist(),
                    'n_train': int(z['n_train']),
                }
        except Exception as e:
            logger.error(f'Failed to load applicability domain for {target}: {e}')
            entry = None

        self._cache[target] = entry
        return entry

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _tanimoto_all(packed: np.ndarray, pop: np.ndarray,
                      query_packed: np.ndarray) -> np.ndarray:
        """Tanimoto of one packed fingerprint against every row of `packed`."""
        inter = _popcount(np.bitwise_and(packed, query_packed)).astype(np.int32)
        qpop = int(_popcount(query_packed))
        union = pop + qpop - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.where(union > 0, inter / union, 0.0)
        return np.asarray(sims, dtype=np.float64)

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(self, smiles: str, target: str, model=None,
               test_rmse: float = None,
               model_features: np.ndarray = None) -> DomainReport:
        """Judge whether a prediction for `smiles` against `target` is trustworthy.

        Args:
            smiles: the molecule
            model: the fitted estimator; if it exposes ``estimators_`` the
                per-tree spread is used as a secondary signal
            test_rmse: the model's held-out RMSE, used to scale the tree spread
            model_features: the feature vector this model expects. Similarity
                is always measured on Morgan bits (that is a question about
                chemistry, not about the model), but the per-tree spread has to
                be computed with the model's own features - passing the Morgan
                bits to a model trained on something else would raise on shape
                and silently lose the signal.

        Returns:
            A DomainReport. ``reliability`` is 0.0-1.0 and is intended to
            multiply an RL reward.
        """
        bits = smiles_to_bits(smiles)
        if bits is None:
            return DomainReport(
                target=target, in_domain=False, confidence='out_of_domain',
                reliability=0.0, nn_similarity=0.0, nn_similarity_top5=0.0,
                tree_std=None, reason='SMILES could not be parsed',
            )

        entry = self._load(target)
        if entry is None:
            # No training data to compare against: report honestly rather than
            # silently claiming the prediction is fine.
            return DomainReport(
                target=target, in_domain=False, confidence='low',
                reliability=0.5, nn_similarity=float('nan'),
                nn_similarity_top5=float('nan'), tree_std=None,
                reason='No training-set reference available for this target; '
                       'domain could not be checked',
            )

        query = np.packbits(bits)
        sims = self._tanimoto_all(entry['packed'], entry['pop'], query)

        nn = float(sims.max())
        k = min(5, len(sims))
        top5 = float(np.sort(sims)[-k:].mean())

        p1, p5, p50 = entry['tiers']

        # Primary verdict from similarity
        if nn >= p50:
            confidence, base = 'high', 1.0
        elif nn >= p5:
            confidence = 'medium'
            base = 0.6 + 0.4 * (nn - p5) / max(p50 - p5, 1e-9)
        elif nn >= p1:
            confidence = 'low'
            base = 0.25 + 0.35 * (nn - p1) / max(p5 - p1, 1e-9)
        else:
            confidence = 'out_of_domain'
            # Ramp to zero below the domain edge instead of a hard cliff
            base = 0.25 * max(nn / max(p1, 1e-9), 0.0)

        in_domain = nn >= p1

        # Secondary signal: forest disagreement relative to the model's own RMSE
        tree_std = None
        if model is not None and hasattr(model, 'estimators_'):
            try:
                features = model_features if model_features is not None else bits
                X = np.asarray(features, dtype=np.float32).reshape(1, -1)
                per_tree = np.array(
                    [est.predict(X)[0] for est in model.estimators_])
                tree_std = float(per_tree.std())
            except Exception as e:
                logger.debug(f'Per-tree spread unavailable for {target}: {e}')

        if tree_std is not None and test_rmse:
            # Spread well above the model's RMSE means the forest is
            # extrapolating; damp reliability, but never boost it.
            ratio = tree_std / max(float(test_rmse), 1e-9)
            if ratio > 1.0:
                base *= float(np.clip(1.0 / ratio, 0.3, 1.0))
                if confidence == 'high':
                    confidence = 'medium'
                elif confidence == 'medium':
                    confidence = 'low'

        reliability = float(np.clip(base, 0.0, 1.0))

        if confidence == 'out_of_domain':
            reason = (
                f'Nearest training molecule is only {nn:.2f} Tanimoto-similar '
                f'(domain edge {p1:.2f}, n={entry["n_train"]}). The model has '
                f'seen nothing like this, so the prediction will sit near the '
                f'training mean regardless of the true activity.'
            )
        elif confidence == 'low':
            reason = (
                f'Sparse neighbourhood: nearest training molecule {nn:.2f} '
                f'Tanimoto (typical in-domain {p50:.2f}). Treat as a rough '
                f'estimate only.'
            )
        elif confidence == 'medium':
            reason = (
                f'Nearest training molecule {nn:.2f} Tanimoto, inside the '
                f'domain but below the typical {p50:.2f}.'
            )
        else:
            reason = (
                f'Well inside the applicability domain: nearest training '
                f'molecule {nn:.2f} Tanimoto.'
            )

        return DomainReport(
            target=target, in_domain=in_domain, confidence=confidence,
            reliability=reliability, nn_similarity=round(nn, 4),
            nn_similarity_top5=round(top5, 4),
            tree_std=round(tree_std, 4) if tree_std is not None else None,
            reason=reason,
        )

    def domain_info(self, target: str) -> Optional[dict]:
        """Calibration summary for one target, or None if unavailable."""
        entry = self._load(target)
        if entry is None:
            return None
        p1, p5, p50 = entry['tiers']
        return {
            'n_train': entry['n_train'],
            'domain_edge_p1': round(p1, 4),
            'similarity_p5': round(p5, 4),
            'similarity_median': round(p50, 4),
        }
