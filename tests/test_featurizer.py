"""Tests for the shared featurizer.

Training and prediction used to each hard-code
`GetMorganFingerprintAsBitVect(mol, 2, 2048)` in their own module. A mismatch
between the two would not raise — the model would just return meaningless
numbers for every molecule — so these tests cover the spec round-trip and the
guards that make such a mismatch loud instead of silent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("rdkit", reason="rdkit is required")

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from featurizer import (  # noqa: E402
    DESCRIPTOR_COUNT,
    FEATURIZER_VERSION,
    LEGACY_SPEC,
    MACCS_COUNT,
    SPECS,
    Featurizer,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


@pytest.mark.parametrize("spec", sorted(SPECS))
def test_every_spec_produces_its_declared_width(spec):
    f = Featurizer(spec)
    vec = f.featurize_smiles(ASPIRIN)
    assert vec is not None
    assert vec.shape == (f.n_features,)


@pytest.mark.parametrize("spec", sorted(SPECS))
def test_features_are_finite(spec):
    """Descriptors can overflow to inf or produce NaN; neither may escape."""
    f = Featurizer(spec)
    for smiles in (ASPIRIN, GEFITINIB, "C", "c1ccccc1", "[Na+].[Cl-]"):
        vec = f.featurize_smiles(smiles)
        if vec is None:
            continue
        assert np.all(np.isfinite(vec)), f"{spec} produced non-finite values"


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError, match="Unknown featurizer spec"):
        Featurizer("not-a-spec")


def test_invalid_smiles_returns_none():
    assert Featurizer().featurize_smiles("not_a_smiles") is None
    assert Featurizer().featurize(None) is None


def test_legacy_spec_matches_the_original_hardcoded_fingerprint():
    """The shipped models were trained on this exact fingerprint."""
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(GEFITINIB)
    expected = np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
        dtype=np.float32,
    )
    actual = Featurizer(LEGACY_SPEC).featurize(mol)
    assert np.array_equal(actual, expected)


def test_legacy_spec_matches_training_helper():
    """featurizer and train_qsar_models must agree, or models break silently."""
    from train_qsar_models import smiles_to_fingerprint

    a = Featurizer(LEGACY_SPEC).featurize_smiles(GEFITINIB)
    b = smiles_to_fingerprint(GEFITINIB)
    assert np.array_equal(a, b.astype(np.float32))


def test_binary_spec_is_binary_and_counts_spec_is_not():
    mol = Chem.MolFromSmiles(GEFITINIB)
    binary = Featurizer("morgan2-2048").featurize(mol)
    counts = Featurizer("morgan2-2048-counts").featurize(mol)

    assert set(np.unique(binary)) <= {0.0, 1.0}
    # Same molecule, same bits set, but at least one bit seen more than once
    assert counts.max() > 1.0
    assert np.array_equal(binary != 0, counts != 0)


def test_descriptor_and_maccs_widths_add_up():
    base = Featurizer("morgan2-2048").n_features
    assert Featurizer("morgan2-2048+desc").n_features == base + DESCRIPTOR_COUNT
    assert (Featurizer("morgan2-2048+maccs+desc").n_features
            == base + MACCS_COUNT + DESCRIPTOR_COUNT)


def test_featurize_many_stacks_and_skips_unparseable():
    mols = [Chem.MolFromSmiles(s) for s in (ASPIRIN, "not_a_smiles", GEFITINIB)]
    f = Featurizer()
    X = f.featurize_many(mols)
    assert X.shape == (2, f.n_features)


def test_featurize_many_of_nothing_keeps_the_width():
    f = Featurizer()
    X = f.featurize_many([])
    assert X.shape == (0, f.n_features)


# ----------------------------------------------------------------------
# Spec persistence - the part that keeps a model and its features together
# ----------------------------------------------------------------------

@pytest.mark.parametrize("spec", sorted(SPECS))
def test_spec_round_trips(spec):
    f = Featurizer(spec)
    rebuilt = Featurizer.from_dict(f.to_dict())
    assert rebuilt.spec == spec
    assert rebuilt.n_features == f.n_features
    assert np.array_equal(
        rebuilt.featurize_smiles(ASPIRIN), f.featurize_smiles(ASPIRIN))


def test_missing_spec_falls_back_to_legacy():
    """Model files written before this module carry no spec."""
    assert Featurizer.from_dict(None).spec == LEGACY_SPEC
    assert Featurizer.from_dict({}).spec == LEGACY_SPEC


def test_old_fingerprint_params_are_understood():
    """The pre-featurizer model files stored radius/n_bits instead."""
    f = Featurizer.from_dict({"radius": 2, "n_bits": 2048})
    assert f.spec == LEGACY_SPEC


def test_unmappable_fingerprint_params_are_rejected():
    """Better to refuse than to featurize a model's input differently."""
    with pytest.raises(ValueError, match="no named spec"):
        Featurizer.from_dict({"radius": 4, "n_bits": 1024})


def test_version_mismatch_is_rejected():
    """A future spec version must not be silently mis-featurized."""
    with pytest.raises(ValueError, match="featurizer version"):
        Featurizer.from_dict({
            "spec": LEGACY_SPEC,
            "version": FEATURIZER_VERSION + 1,
        })


def test_feature_width_mismatch_is_rejected():
    """Guards an RDKit build that resolves a different descriptor set."""
    with pytest.raises(ValueError, match="features but"):
        Featurizer.from_dict({
            "spec": "morgan2-2048+desc",
            "version": FEATURIZER_VERSION,
            "n_features": 99999,
        })


# ----------------------------------------------------------------------
# The shipped models must still load and agree with their stored spec
# ----------------------------------------------------------------------

def test_shipped_models_use_a_loadable_featurizer():
    from qsar_predictor import QSARPredictor

    p = QSARPredictor()
    if not p.is_available:
        pytest.skip("no QSAR models present")

    for target in p.available_targets:
        f = p.featurizers.get(target)
        assert f is not None, f"{target} has no featurizer"
        vec = f.featurize_smiles(GEFITINIB)
        assert vec is not None
        # The feature width must match what the fitted model expects, or
        # every prediction for this target is nonsense.
        model = p.models[target]
        expected = getattr(model, 'n_features_in_', None)
        if expected is not None:
            assert vec.shape[0] == expected, (
                f"{target}: featurizer gives {vec.shape[0]} features, "
                f"model expects {expected}"
            )
