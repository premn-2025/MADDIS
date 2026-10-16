"""Tests for novelty scoring, selectivity profiling and synthetic accessibility.

Each of these guards a specific gap that existed in the code:

- Novelty was advertised in two places but never computed, so a memorised
  training compound was reported as a discovery.
- The five QSAR models were never compared against each other, so a
  promiscuous binder scored the same as a selective one.
- The app's synthetic-accessibility term fell back to `num_atoms / 8.0`
  because it imported `sascorer` without putting RDKit's contrib directory on
  the path.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("rdkit", reason="rdkit is required")

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from qsar_predictor import QSARPredictor  # noqa: E402

# Molecules known to be in the shipped reference sets
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


@pytest.fixture(scope="module")
def predictor():
    p = QSARPredictor()
    if not p.is_available:
        pytest.skip("No QSAR models in models/")
    return p


# ======================================================================
# Synthetic accessibility
# ======================================================================

class TestSyntheticAccessibility:

    def test_sascorer_is_actually_importable(self):
        """The whole point: a bare `import sascorer` fails, this must not."""
        from synthetic_accessibility import SA_AVAILABLE
        assert SA_AVAILABLE, (
            "RDKit's contrib SAScore could not be loaded. Without it the "
            "reward term silently falls back to a guess."
        )

    def test_unavailable_score_is_none_not_a_guess(self):
        """An unparseable molecule must yield None, never a plausible number."""
        from synthetic_accessibility import sa_score
        assert sa_score("not_a_smiles") is None

    def test_reward_is_neutral_when_score_unavailable(self):
        from synthetic_accessibility import sa_reward
        assert sa_reward("not_a_smiles") == 0.5

    def test_score_is_in_native_range(self):
        from synthetic_accessibility import sa_score
        for smi in (GEFITINIB, ASPIRIN, CAFFEINE, "CCO"):
            raw = sa_score(smi)
            assert raw is not None
            assert 1.0 <= raw <= 10.0

    def test_distinguishes_complexity_at_equal_size(self):
        """The regression the atom-count proxy could never catch.

        n-decane and twistane both have ten carbons. The old fallback scored
        them identically; a real SAScore must not.
        """
        from synthetic_accessibility import sa_score

        linear = sa_score("CCCCCCCCCC")          # n-decane
        bridged = sa_score("C1C2CC3CC1CC(C2)C3")  # twistane

        assert linear is not None and bridged is not None
        assert bridged > linear + 1.0, (
            f"bridged polycycle ({bridged:.2f}) should be clearly harder than "
            f"the linear alkane ({linear:.2f})"
        )

    def test_reward_decreases_as_difficulty_increases(self):
        from synthetic_accessibility import sa_reward
        assert sa_reward("CCCCCCCCCC") > sa_reward("C1C2CC3CC1CC(C2)C3")

    def test_hard_molecule_scores_zero_reward(self):
        from synthetic_accessibility import sa_reward, sa_score
        morphine = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        # Morphine sits near the ceiling; the reward must be heavily penalised
        assert sa_score(morphine) > 4.5
        assert sa_reward(morphine) < 0.3

    def test_labels_are_ordered(self):
        from synthetic_accessibility import sa_label
        assert sa_label("CCO") in ("easy", "moderate")
        assert sa_label("not_a_smiles") == "unknown"


# ======================================================================
# Novelty
# ======================================================================

@pytest.fixture(scope="module")
def novelty_scorer():
    from novelty_scorer import NoveltyScorer
    s = NoveltyScorer()
    if not s.is_available:
        pytest.skip("No reference caches; run build_domain_caches.py")
    return s


@pytest.fixture(scope="module")
def selectivity_scorer(predictor):
    from selectivity_scorer import SelectivityScorer
    s = SelectivityScorer(predictor=predictor)
    if not s.is_available:
        pytest.skip("Need at least two QSAR models")
    return s


class TestNovelty:

    def test_training_compound_is_detected_as_not_novel(self, novelty_scorer):
        """Gefitinib is in the EGFR reference set. It is not a discovery."""
        r = novelty_scorer.score(GEFITINIB)
        assert r.is_exact_match
        assert r.is_duplicate
        assert r.novelty == pytest.approx(0.0, abs=0.02)
        assert r.nearest_target == "EGFR"

    def test_aspirin_is_detected_in_the_cox2_set(self, novelty_scorer):
        r = novelty_scorer.score(ASPIRIN)
        assert r.is_exact_match
        assert r.nearest_target == "COX2"

    def test_memorised_compound_earns_no_novelty_reward(self, novelty_scorer):
        """The reward consequence: copying must not pay."""
        assert novelty_scorer.get_novelty_reward(GEFITINIB) < 0.05
        assert novelty_scorer.get_novelty_reward(ERLOTINIB) < 0.05

    def test_unrelated_molecule_is_novel(self, novelty_scorer):
        r = novelty_scorer.score(CAFFEINE)
        assert not r.is_exact_match
        assert r.novelty > 0.5

    def test_invalid_smiles_is_not_credited_as_novel(self, novelty_scorer):
        """Garbage must not score as maximally novel."""
        r = novelty_scorer.score("not_a_smiles")
        assert r.novelty == 0.0
        assert not r.is_duplicate
        assert "parse" in r.reason.lower()

    def test_novelty_reward_tapers_for_wholly_unrelated_molecules(self, novelty_scorer):
        """Methane is maximally dissimilar but worthless.

        Raw novelty is near 1.0; the reward must not be, or the generator
        learns that the way to score is to emit tiny nonsense.
        """
        r = novelty_scorer.score("C")
        assert r.novelty > 0.9
        assert novelty_scorer.get_novelty_reward("C") < 0.6

    def test_per_target_similarity_covers_every_reference_set(self, novelty_scorer):
        r = novelty_scorer.score(CAFFEINE)
        assert set(r.per_target_similarity) == set(novelty_scorer.available_targets)
        for sim in r.per_target_similarity.values():
            assert 0.0 <= sim <= 1.0

    def test_max_similarity_matches_the_nearest_target(self, novelty_scorer):
        r = novelty_scorer.score(CAFFEINE)
        assert r.per_target_similarity[r.nearest_target] == pytest.approx(
            r.max_similarity, abs=1e-6)

    def test_batch_detects_mode_collapse(self, novelty_scorer):
        """A generator emitting near-copies must be caught.

        Every molecule here looks novel against ChEMBL, so novelty alone would
        not flag it - internal diversity has to.
        """
        collapsed = novelty_scorer.score_batch(
            ['CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO'])
        diverse = novelty_scorer.score_batch([GEFITINIB, ASPIRIN, CAFFEINE, 'C1CCCCC1'])

        assert collapsed.internal_diversity < diverse.internal_diversity
        assert collapsed.scaffold_diversity < diverse.scaffold_diversity

    def test_batch_counts_reference_duplicates(self, novelty_scorer):
        batch = novelty_scorer.score_batch([GEFITINIB, ASPIRIN, CAFFEINE])
        assert batch.n_molecules == 3
        assert batch.n_duplicates_of_reference >= 2  # gefitinib + aspirin

    def test_batch_counts_unique_molecules(self, novelty_scorer):
        batch = novelty_scorer.score_batch([CAFFEINE, CAFFEINE, ASPIRIN])
        assert batch.n_molecules == 3
        assert batch.n_unique == 2

    def test_combined_quality_punishes_either_failure(self, novelty_scorer):
        """Novel-but-unpredictable and predictable-but-copied both score low."""
        # Memorised compound: high reliability, zero novelty
        memorised = novelty_scorer.combined_quality(GEFITINIB, domain_reliability=1.0)
        # Exotic molecule: some novelty, no trustworthy prediction
        exotic = novelty_scorer.combined_quality('C', domain_reliability=0.0)
        # The useful band: novel and predictable
        useful = novelty_scorer.combined_quality(CAFFEINE, domain_reliability=0.8)

        assert memorised < useful
        assert exotic < useful


# ======================================================================
# Selectivity
# ======================================================================

class TestSelectivity:

    def test_unknown_target_is_reported_not_guessed(self, selectivity_scorer):
        p = selectivity_scorer.profile(GEFITINIB, "NOT_A_TARGET")
        assert p.verdict == "unavailable"
        assert not p.reliable
        assert p.selectivity_log_units is None

    def test_invalid_smiles_is_handled(self, selectivity_scorer):
        p = selectivity_scorer.profile("not_a_smiles", "EGFR")
        assert p.verdict == "unavailable"
        assert not p.reliable

    def test_egfr_drug_prefers_egfr(self, selectivity_scorer):
        """Erlotinib is an EGFR-selective drug; the profile must say so."""
        if "EGFR" not in selectivity_scorer.predictor.available_targets:
            pytest.skip("EGFR model unavailable")
        p = selectivity_scorer.profile(ERLOTINIB, "EGFR")
        assert p.reliable
        assert p.selectivity_log_units > 0, (
            "an EGFR inhibitor should not be predicted to prefer another target"
        )
        assert p.verdict in ("selective", "modestly_selective", "weakly_selective")

    def test_wrong_target_assignment_is_called_out(self, selectivity_scorer):
        """Aspirin is a COX inhibitor. Assigned to EGFR it must not look good."""
        if not {"EGFR", "COX2"} <= set(selectivity_scorer.predictor.available_targets):
            pytest.skip("need EGFR and COX2")
        p = selectivity_scorer.profile(ASPIRIN, "EGFR")
        assert p.selectivity_log_units is not None
        assert p.selectivity_log_units < 1.0
        assert p.verdict in ("promiscuous", "weakly_selective")

    def test_fold_selectivity_matches_log_units(self, selectivity_scorer):
        p = selectivity_scorer.profile(ERLOTINIB, "EGFR")
        if p.selectivity_log_units is None:
            pytest.skip("no window established")
        assert p.fold_selectivity == pytest.approx(
            10 ** p.selectivity_log_units, rel=0.01)

    def test_out_of_domain_molecule_gets_no_selectivity_claim(self, selectivity_scorer):
        """Methane must not be awarded a selectivity window."""
        p = selectivity_scorer.profile("C", "EGFR")
        assert not p.reliable
        assert p.verdict in ("unknown", "unreliable")

    def test_out_of_domain_off_targets_are_excluded_and_reported(self, selectivity_scorer):
        """Excluding them is right; hiding the exclusion is not."""
        p = selectivity_scorer.profile(ASPIRIN, "COX2")
        # Aspirin is far from the kinase training sets, so some off-targets
        # must have been excluded, and the reason must say which.
        assert p.excluded_off_targets
        for t in p.excluded_off_targets:
            assert t in p.reason

    def test_excluded_off_targets_are_not_used_as_the_comparator(self, selectivity_scorer):
        p = selectivity_scorer.profile(ASPIRIN, "COX2")
        assert p.best_off_target not in p.excluded_off_targets

    def test_per_target_pic50_includes_on_target(self, selectivity_scorer):
        p = selectivity_scorer.profile(GEFITINIB, "EGFR")
        assert "EGFR" in p.per_target_pic50
        assert p.per_target_pic50["EGFR"] == p.on_target_pic50

    def test_reward_is_neutral_when_unestablished(self, selectivity_scorer):
        assert selectivity_scorer.get_selectivity_reward("C", "EGFR") == 0.5

    def test_reward_is_bounded(self, selectivity_scorer):
        for smi in (GEFITINIB, ERLOTINIB, ASPIRIN, CAFFEINE):
            r = selectivity_scorer.get_selectivity_reward(smi, "EGFR")
            assert 0.0 <= r <= 1.0

    def test_ranking_puts_unreliable_last(self, selectivity_scorer):
        profiles = selectivity_scorer.rank_by_selectivity(
            [ERLOTINIB, "C", GEFITINIB], "EGFR")
        assert len(profiles) == 3
        assert not profiles[-1].reliable, (
            "the out-of-domain molecule should rank last"
        )
        # Reliable entries must be in descending selectivity order
        reliable = [p.selectivity_log_units for p in profiles if p.reliable]
        assert reliable == sorted(reliable, reverse=True)


# ======================================================================
# Scaffold splitting
# ======================================================================

class TestScaffoldSplit:

    def test_murcko_scaffold_strips_substituents(self):
        from train_qsar_models import murcko_scaffold
        # Two differently-substituted benzenes share one scaffold
        a = murcko_scaffold("c1ccccc1CCO")
        b = murcko_scaffold("c1ccccc1CCCN")
        assert a == b == "c1ccccc1"

    def test_acyclic_molecule_has_empty_scaffold(self):
        from train_qsar_models import murcko_scaffold
        assert murcko_scaffold("CCO") == ""

    def test_invalid_smiles_gives_empty_scaffold(self):
        from train_qsar_models import murcko_scaffold
        assert murcko_scaffold("not_a_smiles") == ""

    def test_split_is_disjoint_and_complete(self):
        from train_qsar_models import scaffold_split
        smiles = [
            "c1ccccc1CCO", "c1ccccc1CCCN", "c1ccccc1Cl",
            "c1ccncc1CCO", "c1ccncc1Br",
            "C1CCCCC1C", "C1CCCCC1CC",
            "CCO", "CCCO",
        ]
        train, test = scaffold_split(smiles, test_size=0.3)
        assert set(train) & set(test) == set()
        assert len(train) + len(test) == len(smiles)
        assert len(test) > 0

    def test_no_scaffold_appears_on_both_sides(self):
        """The property that makes the split meaningful."""
        from train_qsar_models import murcko_scaffold, scaffold_split
        smiles = [
            "c1ccccc1CCO", "c1ccccc1CCCN", "c1ccccc1Cl", "c1ccccc1F",
            "c1ccncc1CCO", "c1ccncc1Br", "c1ccncc1C",
            "C1CCCCC1C", "C1CCCCC1CC", "C1CCCCC1CCC",
            "C1CCNCC1", "C1CCNCC1C",
        ]
        train, test = scaffold_split(smiles, test_size=0.3)
        train_scaffolds = {murcko_scaffold(smiles[i]) for i in train}
        test_scaffolds = {murcko_scaffold(smiles[i]) for i in test}
        assert train_scaffolds & test_scaffolds == set()

    def test_split_is_reproducible(self):
        from train_qsar_models import scaffold_split
        smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccncc1", "C1CCCCC1", "CCO"]
        a_train, a_test = scaffold_split(smiles, test_size=0.4)
        b_train, b_test = scaffold_split(smiles, test_size=0.4)
        assert list(a_train) == list(b_train)
        assert list(a_test) == list(b_test)

    def test_single_scaffold_still_yields_a_test_set(self):
        """Degenerate input must not produce an empty test set."""
        from train_qsar_models import scaffold_split
        smiles = ["c1ccccc1C"] * 10
        train, test = scaffold_split(smiles, test_size=0.2)
        assert len(test) > 0
        assert len(train) > 0


class TestInDomainReference:
    """The in-domain accuracy metric is only meaningful against a reference
    set that excludes the test molecules."""

    def test_in_memory_reference_excludes_unseen_molecules(self, predictor):
        """A molecule absent from the reference set must not be in-domain.

        The first version of the in-domain metric compared held-out molecules
        against the on-disk cache, which is built from the whole CSV - so each
        test molecule found itself at Tanimoto 1.0 and coverage was a
        meaningless 100%.
        """
        from applicability_domain import ApplicabilityDomain

        ad = ApplicabilityDomain(allow_build=False)

        # A reference set of simple aliphatic alcohols, nothing kinase-like.
        reference = [
            'CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO', 'CC(C)O',
            'CC(C)CO', 'CCC(C)O', 'OCCO', 'OCCCO', 'CCCCCCCO',
            'CC(O)CC', 'CCCC(C)O', 'CC(C)(C)O', 'OCC(O)CO',
            'CCCCCCCCO', 'CCCCCCCCCO', 'CC(C)CCO', 'CCC(O)CC',
            'OCCCCO', 'CCCCCCCCCCO', 'CC(C)C(C)O',
        ]
        assert ad.load_reference_smiles('probe', reference)

        # Gefitinib shares none of that chemistry
        out = ad.assess(GEFITINIB, 'probe')
        assert not out.in_domain, (
            f"gefitinib accepted into an alcohols-only domain "
            f"(similarity {out.nn_similarity})"
        )

        # A member of the reference set must be in-domain
        inside = ad.assess('CCCO', 'probe')
        assert inside.in_domain
        assert inside.nn_similarity == pytest.approx(1.0, abs=1e-6)

    def test_too_small_a_reference_is_refused(self):
        """Better to report nothing than to calibrate on a handful."""
        from applicability_domain import ApplicabilityDomain
        ad = ApplicabilityDomain(allow_build=False)
        assert not ad.load_reference_smiles('tiny', ['CCO', 'CCCO'])
        assert ad.assess('CCO', 'tiny').confidence == 'low'

    def test_in_memory_key_never_triggers_a_disk_build(self):
        """An in-memory key must not be mistaken for a target needing a cache."""
        from applicability_domain import ApplicabilityDomain
        ad = ApplicabilityDomain(allow_build=False)
        assert ad._load('EGFR__trainonly') is None

    def test_stored_in_domain_coverage_is_not_degenerate(self, predictor):
        """If a model file reports in-domain metrics, coverage must be < 100%.

        Exactly 100% means the reference set contained the test molecules.
        """
        info = predictor.get_model_info()
        checked = 0
        for target, meta in info.items():
            cov = meta.get('in_domain_coverage')
            if cov is None:
                continue
            checked += 1
            assert 0.0 < cov < 1.0, (
                f"{target}: in-domain coverage is {cov:.2f}; a value of 1.0 "
                f"means the domain reference included the held-out molecules"
            )
        if checked == 0:
            pytest.skip("no model file reports in-domain metrics yet")
