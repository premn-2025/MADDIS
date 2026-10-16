#!/usr/bin/env python3
"""
Novelty scoring for generated molecules.

A generative model can score beautifully by memorising its training data. The
platform advertised a novelty term — `gemini3_orchestrator.py` lists
`"novelty_score": minimum_tanimoto_difference` as a success criterion, and
`chatbot/interpreters/rl_interpreter.py` tells users the reward includes
novelty — but nothing ever computed one. Generated molecules were never checked
against the ChEMBL sets the QSAR models were trained on, so a straight
regurgitation of a known inhibitor would be reported as a discovery.

This module closes that gap. It reuses the bit-packed training fingerprints
built for the applicability domain (`models/<TARGET>_ad.npz`), so checking
novelty costs one vectorised pass and needs no extra data.

Two distinct notions, deliberately kept apart:

- **Database novelty**: is this molecule new relative to the reference sets
  (ChEMBL training data)? A Tanimoto of 1.0 to a reference molecule means it is
  that molecule.
- **Internal diversity**: is this molecule new relative to the *other molecules
  in the same generated batch*? A generator can also collapse onto one scaffold
  and emit near-copies of itself.

Note the deliberate tension with the applicability domain: that check rewards
similarity to the training set, this one rewards distance from it. A useful
candidate sits in between — close enough that the model's prediction means
something, far enough to be new. `combined_quality` scores that band.

Usage:
    from novelty_scorer import NoveltyScorer

    scorer = NoveltyScorer()
    report = scorer.score("CC(=O)Oc1ccccc1C(=O)O")
    report.is_duplicate      # True - aspirin is in the COX2 reference set
    report.novelty           # 0.0

    batch = scorer.score_batch(generated_smiles)
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from applicability_domain import (
    ApplicabilityDomain,
    _popcount,
    smiles_to_bits,
)

logger = logging.getLogger(__name__)

# A molecule at or above this Tanimoto to a reference compound is treated as
# the same chemical matter rather than a new discovery. 0.85 is the long-standing
# convention for Morgan/ECFP4 "same series"; 1.0 is an exact fingerprint match.
DUPLICATE_THRESHOLD = 0.85
EXACT_THRESHOLD = 0.99


@dataclass
class NoveltyReport:
    """Novelty verdict for one molecule against the reference sets."""

    smiles: str
    novelty: float  # 1 - max similarity to any reference molecule, 0-1
    max_similarity: float  # highest Tanimoto found
    nearest_target: Optional[str]  # which reference set it matched
    is_duplicate: bool  # at/above DUPLICATE_THRESHOLD
    is_exact_match: bool  # fingerprint-identical to a reference molecule
    per_target_similarity: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchNoveltyReport:
    """Novelty and diversity summary for a generated batch."""

    n_molecules: int
    n_unique: int
    n_duplicates_of_reference: int
    mean_novelty: float
    internal_diversity: float  # mean pairwise Tanimoto distance within the batch
    scaffold_diversity: float  # distinct Murcko scaffolds / molecules
    per_molecule: List[NoveltyReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['per_molecule'] = [r.to_dict() for r in self.per_molecule]
        return d


class NoveltyScorer:
    """Scores molecules for novelty against the QSAR training sets."""

    def __init__(self, targets: Sequence[str] = None, model_dir: str = None,
                 data_dir: str = None, allow_build: bool = None):
        """
        Args:
            allow_build: whether a missing reference cache may be built on
                demand. Pass False in a server so a cold start cannot block
                for ~40s fingerprinting the training sets; ship the prebuilt
                models/*_ad.npz instead.
        """
        self.domain = ApplicabilityDomain(
            model_dir=model_dir, data_dir=data_dir, allow_build=allow_build)
        if targets is None:
            from qsar_predictor import QSARPredictor
            targets = QSARPredictor.TARGETS
        self.targets = list(targets)
        self._available: Optional[List[str]] = None

    @property
    def available_targets(self) -> List[str]:
        """Reference sets that actually have a usable fingerprint cache."""
        if self._available is None:
            self._available = [
                t for t in self.targets if self.domain._load(t) is not None
            ]
            if not self._available:
                logger.warning(
                    "No reference fingerprint caches available; novelty cannot "
                    "be computed. Run build_domain_caches.py."
                )
        return self._available

    @property
    def is_available(self) -> bool:
        return bool(self.available_targets)

    # ------------------------------------------------------------------
    # Single molecule
    # ------------------------------------------------------------------

    def score(self, smiles: str) -> NoveltyReport:
        """Novelty of one molecule against every available reference set."""
        bits = smiles_to_bits(smiles)
        if bits is None:
            return NoveltyReport(
                smiles=smiles, novelty=0.0, max_similarity=0.0,
                nearest_target=None, is_duplicate=False, is_exact_match=False,
                reason="SMILES could not be parsed",
            )

        if not self.is_available:
            return NoveltyReport(
                smiles=smiles, novelty=float('nan'), max_similarity=float('nan'),
                nearest_target=None, is_duplicate=False, is_exact_match=False,
                reason="No reference sets available; novelty not checked",
            )

        query = np.packbits(bits)
        per_target: Dict[str, float] = {}

        for target in self.available_targets:
            entry = self.domain._load(target)
            sims = self.domain._tanimoto_all(
                entry['packed'], entry['pop'], query)
            per_target[target] = float(sims.max())

        nearest_target = max(per_target, key=per_target.get)
        max_sim = per_target[nearest_target]

        is_exact = max_sim >= EXACT_THRESHOLD
        is_dup = max_sim >= DUPLICATE_THRESHOLD

        if is_exact:
            reason = (
                f"Already present in the {nearest_target} reference set "
                f"(Tanimoto {max_sim:.2f}). This is a known compound, not a "
                f"new one."
            )
        elif is_dup:
            reason = (
                f"Same chemical series as a known {nearest_target} compound "
                f"(Tanimoto {max_sim:.2f}). An analogue, not a new scaffold."
            )
        elif max_sim >= 0.5:
            reason = (
                f"Related to known {nearest_target} chemistry "
                f"(Tanimoto {max_sim:.2f}) but distinct from any single compound."
            )
        else:
            reason = (
                f"No close relative in any reference set (best {max_sim:.2f}, "
                f"{nearest_target}). Structurally novel - but check the "
                f"applicability domain before trusting its predicted activity."
            )

        return NoveltyReport(
            smiles=smiles,
            novelty=round(1.0 - max_sim, 4),
            max_similarity=round(max_sim, 4),
            nearest_target=nearest_target,
            is_duplicate=is_dup,
            is_exact_match=is_exact,
            per_target_similarity={k: round(v, 4) for k, v in per_target.items()},
            reason=reason,
        )

    def get_novelty_reward(self, smiles: str) -> float:
        """Novelty as a 0-1 RL reward term.

        Rewards distance from known compounds, but does not pay extra for being
        wildly dissimilar to everything - a molecule with no relatives anywhere
        is usually either trivial or unsynthesisable, and the applicability
        domain will refuse to predict for it anyway. The curve peaks in the
        "new scaffold, recognisable chemistry" band around Tanimoto 0.3-0.5.
        """
        report = self.score(smiles)
        sim = report.max_similarity
        if not np.isfinite(sim):
            return 0.5  # no reference data: stay neutral

        if sim >= DUPLICATE_THRESHOLD:
            # Known compound or close analogue: little credit.
            return float(np.clip((1.0 - sim) / (1.0 - DUPLICATE_THRESHOLD) * 0.2,
                                 0.0, 0.2))
        if sim >= 0.5:
            # Recognisable but distinct: ramp 0.2 -> 1.0
            return float(0.2 + 0.8 * (DUPLICATE_THRESHOLD - sim) /
                         (DUPLICATE_THRESHOLD - 0.5))
        if sim >= 0.25:
            # The sweet spot: new scaffold, still drug-like chemistry.
            return 1.0
        # Unrelated to anything known: taper, since predictions here are
        # untrustworthy regardless.
        return float(np.clip(0.4 + 2.4 * sim, 0.4, 1.0))

    def combined_quality(self, smiles: str, domain_reliability: float) -> float:
        """Score the useful band: novel enough to matter, familiar enough to predict.

        Args:
            domain_reliability: the applicability-domain reliability for the
                target of interest (see QSARPredictor.assess_domain).

        A molecule scores well only when both hold. A memorised training
        compound has high reliability but no novelty; an exotic structure has
        novelty but no trustworthy prediction. The geometric mean punishes
        either failure.
        """
        novelty = self.get_novelty_reward(smiles)
        return float(np.sqrt(max(novelty, 0.0) * max(domain_reliability, 0.0)))

    # ------------------------------------------------------------------
    # Batches
    # ------------------------------------------------------------------

    def score_batch(self, smiles_list: Sequence[str]) -> BatchNoveltyReport:
        """Novelty plus internal diversity for a generated batch."""
        reports = [self.score(s) for s in smiles_list]

        valid_bits = []
        canonical = []
        for s in smiles_list:
            bits = smiles_to_bits(s)
            if bits is not None:
                valid_bits.append(bits)
                canonical.append(s)

        novelties = [r.novelty for r in reports if np.isfinite(r.novelty)]
        n_dup = sum(1 for r in reports if r.is_duplicate)

        return BatchNoveltyReport(
            n_molecules=len(smiles_list),
            n_unique=len(set(canonical)),
            n_duplicates_of_reference=n_dup,
            mean_novelty=round(float(np.mean(novelties)), 4) if novelties else 0.0,
            internal_diversity=self._internal_diversity(valid_bits),
            scaffold_diversity=self._scaffold_diversity(canonical),
            per_molecule=reports,
        )

    @staticmethod
    def _internal_diversity(bits_list: List[np.ndarray]) -> float:
        """Mean pairwise Tanimoto *distance* within a batch (1 = all distinct).

        Detects mode collapse: a generator emitting near-copies of one molecule
        scores near 0 here even if every molecule looks novel against ChEMBL.
        """
        if len(bits_list) < 2:
            return 0.0

        packed = np.packbits(np.vstack(bits_list), axis=-1)
        pop = _popcount(packed).astype(np.int64)

        n = len(packed)
        total, count = 0.0, 0
        for i in range(n - 1):
            inter = _popcount(
                np.bitwise_and(packed[i + 1:], packed[i])).astype(np.int64)
            union = pop[i + 1:] + pop[i] - inter
            with np.errstate(divide='ignore', invalid='ignore'):
                sims = np.where(union > 0, inter / union, 0.0)
            total += float(sims.sum())
            count += len(sims)

        return round(1.0 - (total / count if count else 0.0), 4)

    @staticmethod
    def _scaffold_diversity(smiles_list: List[str]) -> float:
        """Distinct Bemis-Murcko scaffolds divided by molecule count."""
        if not smiles_list:
            return 0.0
        try:
            from train_qsar_models import murcko_scaffold
        except Exception:
            return 0.0
        scaffolds = {murcko_scaffold(s) for s in smiles_list}
        return round(len(scaffolds) / len(smiles_list), 4)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

    scorer = NoveltyScorer()
    if not scorer.is_available:
        print("No reference caches. Run: python build_domain_caches.py")
        sys.exit(1)

    print(f"Reference sets: {', '.join(scorer.available_targets)}\n")

    probes = {
        'Aspirin (in COX2 set)': 'CC(=O)Oc1ccccc1C(=O)O',
        'Gefitinib (in EGFR set)': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
        'Caffeine': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
        'Methane': 'C',
        'Random peptide-ish': 'CC(N)C(=O)NC(Cc1ccccc1)C(=O)NC(C)C(=O)O',
    }

    print(f"{'molecule':28s} {'novelty':>8s} {'maxSim':>7s} {'reward':>7s}  verdict")
    print("-" * 92)
    for name, smi in probes.items():
        r = scorer.score(smi)
        reward = scorer.get_novelty_reward(smi)
        verdict = ('EXACT MATCH' if r.is_exact_match
                   else 'analogue' if r.is_duplicate else 'novel')
        print(f"{name:28s} {r.novelty:8.3f} {r.max_similarity:7.3f} "
              f"{reward:7.3f}  {verdict} ({r.nearest_target})")

    print("\n--- Batch diversity ---")
    collapsed = ['CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO']
    diverse = list(probes.values())
    for label, batch in [('near-identical batch', collapsed),
                         ('diverse batch', diverse)]:
        b = scorer.score_batch(batch)
        print(f"{label:22s} internal diversity={b.internal_diversity:.3f}  "
              f"scaffold diversity={b.scaffold_diversity:.3f}  "
              f"mean novelty={b.mean_novelty:.3f}  "
              f"dups={b.n_duplicates_of_reference}")
