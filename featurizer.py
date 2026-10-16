#!/usr/bin/env python3
"""
Molecular featurization for the QSAR models.

Training and prediction must compute *identical* features. Previously both
sides independently hard-coded `GetMorganFingerprintAsBitVect(mol, 2, 2048)` —
`train_qsar_models.smiles_to_fingerprint` and
`QSARPredictor.smiles_to_fingerprint`. That works only as long as nobody edits
one of them, and a silent mismatch would not raise: the model would simply
return meaningless numbers for every molecule.

This module makes the featurization a named, versioned spec that is stored
inside each model file. `QSARPredictor` reads the spec back out and rebuilds the
exact featurizer the model was trained with, so a model and its features cannot
be separated.

Specs:
    morgan2-2048            Morgan radius 2, 2048 bits, binary.
                            What the originally shipped models used.
    morgan2-2048-counts     As above but bit counts rather than presence.
    morgan2-2048+desc       Morgan bits plus curated physicochemical
                            descriptors.
    morgan2-2048+maccs+desc Morgan bits, MACCS keys, and descriptors.

Usage:
    from featurizer import Featurizer

    f = Featurizer('morgan2-2048+desc')
    X = f.featurize_many(mols)          # (n_molecules, f.n_features)
    spec = f.to_dict()                  # store alongside the model
    f2 = Featurizer.from_dict(spec)     # rebuild at predict time
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# Structure standardizers, built once (stateless but not free to construct).
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_smiles(smiles: str) -> Optional[str]:
    """Strip salts/solvents to the largest fragment and neutralise charges.

    ChEMBL rows include salt forms, so one parent compound can appear several
    times (a hydrobromide and its free base, say) with the counter-ion
    contributing fingerprint bits that say nothing about binding. Under 3% of
    rows per target are affected, so this is a modest cleanup rather than a fix
    for a major problem - but it is free and removes noise.

    This lives here, next to featurization, because the training set, the
    applicability-domain reference set and anything scored at predict time all
    have to be standardized the same way or their fingerprints are not
    comparable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return Chem.MolToSmiles(mol)

# Curated physicochemical descriptors.
#
# Deliberately not `Descriptors._descList` in full: that includes Ipc, which
# overflows to inf on medium-sized molecules, and the BCUT2D family, which
# raises when Gasteiger charges cannot be assigned. Both would silently become
# zeros. These are the fast, robust, interpretable ones.
_DESCRIPTOR_NAMES = [
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'MolLogP', 'MolMR', 'TPSA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'NumHeteroatoms',
    'HeavyAtomCount', 'NHOHCount', 'NOCount', 'RingCount',
    'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
    'NumAromaticHeterocycles', 'NumAliphaticHeterocycles',
    'NumSaturatedHeterocycles', 'NumAromaticCarbocycles',
    'NumAliphaticCarbocycles', 'NumSaturatedCarbocycles',
    'FractionCSP3', 'BalabanJ', 'BertzCT', 'HallKierAlpha',
    'Kappa1', 'Kappa2', 'Kappa3',
    'Chi0', 'Chi1', 'Chi0n', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n',
    'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
    'LabuteASA', 'NumValenceElectrons', 'NumRadicalElectrons',
    'MaxPartialCharge', 'MinPartialCharge',
    'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'qed',
]
_DESCRIPTOR_NAMES += [f'SlogP_VSA{i}' for i in range(1, 13)]
_DESCRIPTOR_NAMES += [f'SMR_VSA{i}' for i in range(1, 11)]
_DESCRIPTOR_NAMES += [f'PEOE_VSA{i}' for i in range(1, 15)]
_DESCRIPTOR_NAMES += [f'EState_VSA{i}' for i in range(1, 12)]

# Resolve once; a name missing from this RDKit build is dropped with a warning
# rather than crashing, but the resolved list is what defines n_features.
_DESCRIPTOR_FUNCS = []
for _name in _DESCRIPTOR_NAMES:
    _fn = getattr(Descriptors, _name, None)
    if _fn is None:
        logger.debug(f"Descriptor {_name} unavailable in this RDKit build")
        continue
    _DESCRIPTOR_FUNCS.append((_name, _fn))

DESCRIPTOR_COUNT = len(_DESCRIPTOR_FUNCS)
MACCS_COUNT = 167

# Descriptors span many orders of magnitude (BertzCT in the thousands,
# FractionCSP3 in [0, 1]). Tree models are scale-invariant so no scaling is
# applied, but values are clipped to keep a pathological descriptor from
# dominating a split threshold.
_DESCRIPTOR_CLIP = 1e6

SPECS: Dict[str, dict] = {
    'morgan2-2048': {
        'radius': 2, 'n_bits': 2048, 'counts': False,
        'maccs': False, 'descriptors': False,
    },
    'morgan2-2048-counts': {
        'radius': 2, 'n_bits': 2048, 'counts': True,
        'maccs': False, 'descriptors': False,
    },
    'morgan3-2048': {
        'radius': 3, 'n_bits': 2048, 'counts': False,
        'maccs': False, 'descriptors': False,
    },
    'morgan2-2048+desc': {
        'radius': 2, 'n_bits': 2048, 'counts': False,
        'maccs': False, 'descriptors': True,
    },
    'morgan2-2048+maccs+desc': {
        'radius': 2, 'n_bits': 2048, 'counts': False,
        'maccs': True, 'descriptors': True,
    },
    'morgan2-2048-counts+maccs+desc': {
        'radius': 2, 'n_bits': 2048, 'counts': True,
        'maccs': True, 'descriptors': True,
    },
}

# The spec the originally shipped models were trained with.
LEGACY_SPEC = 'morgan2-2048'

# Bumped if the meaning of a spec ever changes, so an old model file is
# rejected rather than silently mis-featurized.
FEATURIZER_VERSION = 1


class Featurizer:
    """Turns molecules into the feature vector a given model expects."""

    def __init__(self, spec: str = LEGACY_SPEC):
        if spec not in SPECS:
            raise ValueError(
                f"Unknown featurizer spec {spec!r}. "
                f"Known: {', '.join(sorted(SPECS))}"
            )
        self.spec = spec
        self.config = dict(SPECS[spec])

    # ------------------------------------------------------------------
    # Introspection / persistence
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        n = self.config['n_bits']
        if self.config['maccs']:
            n += MACCS_COUNT
        if self.config['descriptors']:
            n += DESCRIPTOR_COUNT
        return n

    def to_dict(self) -> dict:
        """Serialisable spec to store next to a trained model."""
        return {
            'spec': self.spec,
            'version': FEATURIZER_VERSION,
            'n_features': self.n_features,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> 'Featurizer':
        """Rebuild from a stored spec.

        A model file without a spec predates this module and used the legacy
        Morgan featurization.
        """
        if not data:
            return cls(LEGACY_SPEC)

        # Pre-featurizer model files stored {'radius': 2, 'n_bits': 2048}
        if 'spec' not in data:
            radius = data.get('radius', 2)
            n_bits = data.get('n_bits', 2048)
            if radius == 2 and n_bits == 2048:
                return cls(LEGACY_SPEC)
            raise ValueError(
                f"Model file uses Morgan radius={radius} n_bits={n_bits}, "
                f"which has no named spec. Retrain with train_qsar_models.py."
            )

        version = data.get('version', FEATURIZER_VERSION)
        if version != FEATURIZER_VERSION:
            raise ValueError(
                f"Model file was built with featurizer version {version}, this "
                f"is version {FEATURIZER_VERSION}. The features would not "
                f"match - retrain with train_qsar_models.py."
            )

        f = cls(data['spec'])

        stored = data.get('n_features')
        if stored is not None and stored != f.n_features:
            raise ValueError(
                f"Featurizer {f.spec!r} produces {f.n_features} features but "
                f"the model expects {stored}. This RDKit build may be missing "
                f"descriptors - retrain with train_qsar_models.py."
            )
        return f

    # ------------------------------------------------------------------
    # Featurization
    # ------------------------------------------------------------------

    def _morgan(self, mol) -> np.ndarray:
        radius = self.config['radius']
        n_bits = self.config['n_bits']

        if not self.config['counts']:
            return np.array(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits),
                dtype=np.float32,
            )

        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx] = count
        return arr

    @staticmethod
    def _descriptors(mol) -> np.ndarray:
        out = np.empty(DESCRIPTOR_COUNT, dtype=np.float32)
        for i, (_, fn) in enumerate(_DESCRIPTOR_FUNCS):
            try:
                out[i] = fn(mol)
            except Exception:
                # A descriptor that cannot be computed for this molecule
                # contributes nothing rather than killing the whole vector.
                out[i] = 0.0
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -_DESCRIPTOR_CLIP, _DESCRIPTOR_CLIP)

    def featurize(self, mol) -> Optional[np.ndarray]:
        """Feature vector for one RDKit molecule, or None if mol is None."""
        if mol is None:
            return None

        parts = [self._morgan(mol)]
        if self.config['maccs']:
            parts.append(
                np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
        if self.config['descriptors']:
            parts.append(self._descriptors(mol))

        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def featurize_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """Feature vector for one SMILES, or None if it cannot be parsed."""
        return self.featurize(Chem.MolFromSmiles(smiles))

    def featurize_many(self, mols: Sequence) -> np.ndarray:
        """Stack features for a list of molecules, skipping unparseable ones."""
        rows: List[np.ndarray] = []
        for mol in mols:
            vec = self.featurize(mol)
            if vec is not None:
                rows.append(vec)
        if not rows:
            return np.empty((0, self.n_features), dtype=np.float32)
        return np.vstack(rows)

    def __repr__(self) -> str:
        return f"Featurizer({self.spec!r}, n_features={self.n_features})"


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

    print(f"Curated descriptors resolved: {DESCRIPTOR_COUNT}")
    print(f"Featurizer version: {FEATURIZER_VERSION}\n")

    mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
    for spec in SPECS:
        f = Featurizer(spec)
        vec = f.featurize(mol)
        assert vec is not None and len(vec) == f.n_features
        print(f"  {spec:32s} {f.n_features:5d} features  "
              f"(nonzero: {int((vec != 0).sum())})")

    # Round-trip
    f = Featurizer('morgan2-2048+desc')
    assert Featurizer.from_dict(f.to_dict()).spec == f.spec
    # Legacy model files carry no spec
    assert Featurizer.from_dict(None).spec == LEGACY_SPEC
    assert Featurizer.from_dict({'radius': 2, 'n_bits': 2048}).spec == LEGACY_SPEC
    print("\nRound-trip and legacy fallback OK.")
