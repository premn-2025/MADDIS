"""Tests for the QSAR predictor and its applicability domain.

These cover the two things that were actually wrong in this code:

1. The models return their training mean for molecules they have never seen
   anything like, and that number was feeding the RL reward unchecked.
2. Four of the "known drug" demo SMILES were the wrong molecule and one did
   not parse, so the demo appeared to validate the models against drugs it was
   not actually testing.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

rdkit = pytest.importorskip("rdkit", reason="rdkit is required for QSAR tests")

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from qsar_predictor import QSARPredictor  # noqa: E402


@pytest.fixture(scope="module")
def predictor():
    p = QSARPredictor()
    if not p.is_available:
        pytest.skip(
            "No QSAR models in models/. Run download_chembl_data.py then "
            "train_qsar_models.py."
        )
    return p


# ----------------------------------------------------------------------
# Basic prediction behaviour
# ----------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


def test_models_load(predictor):
    assert predictor.is_available
    assert set(predictor.available_targets) <= set(QSARPredictor.TARGETS)


def test_predicts_plausible_pic50(predictor):
    for target in predictor.available_targets:
        pic50 = predictor.predict_pic50(ASPIRIN, target)
        assert pic50 is not None
        # The predictor clamps to this range; anything outside means the
        # clamping or the units broke.
        assert 3.0 <= pic50 <= 12.0


def test_invalid_smiles_returns_none(predictor):
    target = predictor.available_targets[0]
    assert predictor.predict_pic50("not_a_smiles", target) is None
    assert predictor.predict_binding_affinity("not_a_smiles", target) is None
    assert predictor.get_binding_reward("not_a_smiles", target) == 0.0


def test_unknown_target_returns_none(predictor):
    assert predictor.predict_pic50(ASPIRIN, "NOT_A_TARGET") is None


def test_binding_affinity_follows_pic50(predictor):
    """ΔG ≈ -1.36 * pIC50, and stronger binding must be more negative."""
    target = predictor.available_targets[0]
    pic50 = predictor.predict_pic50(ASPIRIN, target)
    dg = predictor.predict_binding_affinity(ASPIRIN, target)
    assert dg == pytest.approx(-1.36 * pic50, rel=1e-6)
    assert dg < 0


def test_ic50_roundtrips_from_pic50(predictor):
    target = predictor.available_targets[0]
    pic50 = predictor.predict_pic50(ASPIRIN, target)
    ic50_nm = predictor.predict_ic50_nm(ASPIRIN, target)
    assert ic50_nm == pytest.approx(10 ** (-pic50) * 1e9, rel=1e-6)


def test_model_metrics_are_reported(predictor):
    """Guard the headline claim: these are real models with real held-out scores."""
    info = predictor.get_model_info()
    for target in predictor.available_targets:
        meta = info[target]
        assert meta["loaded"] is True
        assert isinstance(meta["test_r2"], float)
        # Anything at or below zero would mean the model is worse than
        # predicting the mean, i.e. not usable as a reward signal.
        assert meta["test_r2"] > 0.0
        assert meta["n_training"] > 0


# ----------------------------------------------------------------------
# Applicability domain - the real bug this guards
# ----------------------------------------------------------------------

# Small, chemically trivial molecules that no kinase/protease model was
# trained on. The forest still returns a confident-looking number for each.
OUT_OF_DOMAIN = ["C", "CCO", "CO", "c1ccccc1"]


def test_domain_check_is_wired_up(predictor):
    if predictor.domain is None:
        pytest.skip("applicability_domain module unavailable")
    report = predictor.assess_domain(ASPIRIN, predictor.available_targets[0])
    assert report is not None
    assert 0.0 <= report.reliability <= 1.0
    assert 0.0 <= report.nn_similarity <= 1.0
    assert report.reason


def test_trivial_molecules_are_flagged_out_of_domain(predictor):
    """The core regression: methane must not look like a real prediction."""
    if predictor.domain is None:
        pytest.skip("applicability_domain module unavailable")

    for target in predictor.available_targets:
        for smiles in OUT_OF_DOMAIN:
            report = predictor.assess_domain(smiles, target)
            assert report is not None
            assert not report.in_domain, (
                f"{smiles} was accepted as in-domain for {target} "
                f"(similarity {report.nn_similarity})"
            )
            assert report.confidence == "out_of_domain"


def test_domain_gate_suppresses_out_of_domain_reward(predictor):
    """Methane scores ~0.46 raw on JAK2. Gated it must be near zero.

    This is what stops the RL generator being paid for chemical nonsense.
    """
    if predictor.domain is None:
        pytest.skip("applicability_domain module unavailable")

    for target in predictor.available_targets:
        for smiles in OUT_OF_DOMAIN:
            raw = predictor.get_binding_reward(
                smiles, target, apply_domain_gate=False)
            gated = predictor.get_binding_reward(smiles, target)
            assert gated <= raw
            assert gated < 0.1, (
                f"{smiles}/{target}: gated reward {gated:.3f} is still high "
                f"(raw {raw:.3f})"
            )


def test_training_set_members_are_in_domain(predictor):
    """A molecule drawn from a target's own training set must pass the check."""
    if predictor.domain is None:
        pytest.skip("applicability_domain module unavailable")

    pd = pytest.importorskip("pandas")

    for target in predictor.available_targets:
        csv = ROOT / "data" / f"{target}_chembl.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        # Take a handful spread through the file rather than just the head
        sample = df["smiles"].iloc[:: max(1, len(df) // 5)].head(5)
        for smiles in sample:
            report = predictor.assess_domain(str(smiles), target)
            assert report is not None
            assert report.in_domain, (
                f"training molecule rejected from {target} domain: {smiles}"
            )
            assert report.confidence in ("high", "medium", "low")


def test_domain_gate_preserves_in_domain_reward(predictor):
    """Gating must not penalise a molecule the model genuinely knows."""
    if predictor.domain is None or "EGFR" not in predictor.available_targets:
        pytest.skip("EGFR model or domain check unavailable")

    report = predictor.assess_domain(GEFITINIB, "EGFR")
    assert report.confidence == "high"
    raw = predictor.get_binding_reward(GEFITINIB, "EGFR", apply_domain_gate=False)
    gated = predictor.get_binding_reward(GEFITINIB, "EGFR")
    assert gated == pytest.approx(raw, rel=1e-9)


def test_domain_separates_same_molecule_across_targets(predictor):
    """Aspirin is a COX inhibitor, not an EGFR one.

    The same molecule should be in-domain for the target whose training set
    contains its chemistry and out of it elsewhere. This is the check that
    proves the domain measure tracks chemistry rather than molecule size.
    """
    if predictor.domain is None:
        pytest.skip("applicability_domain module unavailable")
    if not {"COX2", "EGFR"} <= set(predictor.available_targets):
        pytest.skip("needs both COX2 and EGFR models")

    cox2 = predictor.assess_domain(ASPIRIN, "COX2")
    egfr = predictor.assess_domain(ASPIRIN, "EGFR")
    assert cox2.nn_similarity > egfr.nn_similarity
    assert cox2.confidence == "high"


def test_predict_all_targets_marks_unreliable_entries(predictor):
    if predictor.domain is None:
        pytest.skip("applicability_domain module unavailable")

    results = predictor.predict_all_targets("C")
    assert results
    for target, entry in results.items():
        assert entry["classification"] == "unreliable"
        assert entry["in_domain"] is False
        assert entry["confidence"] == "out_of_domain"


def test_confidence_can_be_switched_off(predictor):
    results = predictor.predict_all_targets(ASPIRIN, include_confidence=False)
    assert results
    for entry in results.values():
        assert "confidence" not in entry
        assert "pIC50" in entry


def test_multi_target_reward_is_weighted_mean(predictor):
    targets = {t: 1.0 for t in predictor.available_targets[:2]}
    if len(targets) < 2:
        pytest.skip("needs at least two models")
    total, per_target = predictor.get_multi_target_binding_reward(
        GEFITINIB, targets)
    assert set(per_target) == set(targets)
    assert total == pytest.approx(sum(per_target.values()) / len(per_target))
    assert 0.0 <= total <= 1.0


def test_cache_clear_resets_both_caches(predictor):
    target = predictor.available_targets[0]
    predictor.predict_pic50(ASPIRIN, target)
    predictor.assess_domain(ASPIRIN, target)
    predictor.clear_cache()
    assert not predictor._prediction_cache
    assert not predictor._domain_cache


# ----------------------------------------------------------------------
# Demo data integrity - four of these used to be the wrong molecule
# ----------------------------------------------------------------------

# name fragment -> (molecular formula, monoisotopic-ish average MW)
EXPECTED_DEMO_IDENTITY = {
    "Aspirin": ("C9H8O4", 180.16),
    "Ibuprofen": ("C13H18O2", 206.28),
    "Gefitinib": ("C22H24ClFN4O3", 446.91),
    "Erlotinib": ("C22H23N3O4", 393.44),
    "Celecoxib": ("C17H14F3N3O2S", 381.38),
    "Ruxolitinib": ("C17H18N6", 306.37),
    "Dabigatran": ("C25H25N7O3", 471.52),
    "Caffeine": ("C8H10N4O2", 194.19),
    "Naphthalene": ("C10H8", 128.17),
}


def _demo_molecules():
    """Parse the test_molecules dict out of qsar_predictor's __main__ block."""
    src = (ROOT / "qsar_predictor.py").read_text(encoding="utf-8")
    block = src.split("test_molecules = {")[1].split("}")[0]
    return dict(re.findall(r"'([^']+)':\s*'([^']+)'", block))


def test_every_demo_smiles_parses():
    """One of these used to be unparseable, so the demo silently skipped it."""
    for name, smiles in _demo_molecules().items():
        assert Chem.MolFromSmiles(smiles) is not None, (
            f"demo molecule {name!r} has an invalid SMILES: {smiles}"
        )


def test_demo_smiles_are_the_drugs_they_claim():
    """Each named drug must match that drug's real molecular formula."""
    demo = _demo_molecules()
    checked = 0

    for name, smiles in demo.items():
        for drug, (formula, mw) in EXPECTED_DEMO_IDENTITY.items():
            if not name.startswith(drug):
                continue
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"{name}: unparseable"
            assert rdMolDescriptors.CalcMolFormula(mol) == formula, (
                f"{name} is not {drug}: formula is "
                f"{rdMolDescriptors.CalcMolFormula(mol)}, expected {formula}"
            )
            assert Descriptors.MolWt(mol) == pytest.approx(mw, abs=0.05)
            checked += 1

    assert checked == len(EXPECTED_DEMO_IDENTITY), (
        f"only verified {checked}/{len(EXPECTED_DEMO_IDENTITY)} demo drugs"
    )
