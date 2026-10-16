#!/usr/bin/env python3
"""
Cross-target selectivity profiling.

Potency alone is a poor goal. A compound that hits EGFR at 10 nM and also hits
JAK2, BACE1, COX2 and thrombin at 10 nM is a promiscuity liability, not a lead;
one that hits EGFR at 10 nM and everything else at 10 µM is a drug candidate.
The platform predicted all five targets independently but never compared them,
so the RL generator had no reason to prefer the second over the first.

This module turns the five independent QSAR models into a selectivity profile:

    selectivity = on-target pIC50 - best off-target pIC50

expressed in log units, so +2.0 means a 100-fold preference for the intended
target.

Two things make this honest rather than decorative:

- **Off-targets outside their applicability domain are excluded.** Claiming
  selectivity over a target whose model cannot assess the molecule is
  meaningless — the forest would just return that target's training mean, which
  for JAK2 is pIC50 7.13 and would manufacture apparent non-selectivity for
  anything. Excluded targets are reported, not silently dropped.
- **Selectivity is only reported when the on-target prediction itself is
  trustworthy.** A confident-looking selectivity window built on an
  out-of-domain on-target number is worse than no number.

Usage:
    from selectivity_scorer import SelectivityScorer

    scorer = SelectivityScorer()
    profile = scorer.profile("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "EGFR")
    profile.selectivity_log_units
    profile.verdict
"""

import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Selectivity bands in log units (1 log unit = 10-fold).
SELECTIVE_STRONG = 2.0   # >=100-fold: a genuinely selective compound
SELECTIVE_MODEST = 1.0   # >=10-fold: worth optimising
PROMISCUOUS = 0.0        # off-target >= on-target: a liability


@dataclass
class SelectivityProfile:
    """On-target vs off-target comparison for one molecule."""

    smiles: str
    on_target: str
    on_target_pic50: Optional[float]
    on_target_confidence: Optional[str]
    best_off_target: Optional[str]
    best_off_target_pic50: Optional[float]
    selectivity_log_units: Optional[float]
    fold_selectivity: Optional[float]
    verdict: str
    reliable: bool
    per_target_pic50: Dict[str, float] = field(default_factory=dict)
    excluded_off_targets: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SelectivityScorer:
    """Builds selectivity profiles from the multi-target QSAR models."""

    def __init__(self, predictor=None):
        if predictor is None:
            from qsar_predictor import QSARPredictor
            predictor = QSARPredictor()
        self.predictor = predictor

    @property
    def is_available(self) -> bool:
        return self.predictor.is_available and len(self.predictor.available_targets) >= 2

    def profile(self, smiles: str, on_target: str,
                require_in_domain: bool = True) -> SelectivityProfile:
        """Compare predicted potency at `on_target` against the other targets.

        Args:
            require_in_domain: exclude off-targets whose applicability domain
                rejects this molecule. Leave True unless you specifically want
                the raw cross-target numbers.
        """
        targets = self.predictor.available_targets

        if on_target not in targets:
            return SelectivityProfile(
                smiles=smiles, on_target=on_target, on_target_pic50=None,
                on_target_confidence=None, best_off_target=None,
                best_off_target_pic50=None, selectivity_log_units=None,
                fold_selectivity=None, verdict="unavailable", reliable=False,
                reason=f"No model for {on_target}. Available: {', '.join(targets)}",
            )

        on_pic50 = self.predictor.predict_pic50(smiles, on_target)
        if on_pic50 is None:
            return SelectivityProfile(
                smiles=smiles, on_target=on_target, on_target_pic50=None,
                on_target_confidence=None, best_off_target=None,
                best_off_target_pic50=None, selectivity_log_units=None,
                fold_selectivity=None, verdict="unavailable", reliable=False,
                reason="Molecule could not be scored (invalid SMILES?)",
            )

        on_report = self.predictor.assess_domain(smiles, on_target)
        on_conf = on_report.confidence if on_report else None
        on_in_domain = on_report.in_domain if on_report else True

        per_target: Dict[str, float] = {on_target: round(on_pic50, 3)}
        off_scores: Dict[str, float] = {}
        excluded: List[str] = []

        for target in targets:
            if target == on_target:
                continue
            pic50 = self.predictor.predict_pic50(smiles, target)
            if pic50 is None:
                continue
            per_target[target] = round(pic50, 3)

            if require_in_domain:
                rep = self.predictor.assess_domain(smiles, target)
                if rep is not None and not rep.in_domain:
                    # The model cannot speak to this molecule; its prediction is
                    # the training mean, so comparing against it is noise.
                    excluded.append(target)
                    continue
            off_scores[target] = pic50

        if not off_scores:
            return SelectivityProfile(
                smiles=smiles, on_target=on_target,
                on_target_pic50=round(on_pic50, 3), on_target_confidence=on_conf,
                best_off_target=None, best_off_target_pic50=None,
                selectivity_log_units=None, fold_selectivity=None,
                verdict="unknown", reliable=False,
                per_target_pic50=per_target, excluded_off_targets=excluded,
                reason=(
                    "Every off-target model rejected this molecule as outside "
                    "its applicability domain, so no selectivity window can be "
                    "established. This is a limit of the reference data, not "
                    "evidence of selectivity."
                ),
            )

        best_off = max(off_scores, key=off_scores.get)
        best_off_pic50 = off_scores[best_off]
        delta = on_pic50 - best_off_pic50

        if not on_in_domain:
            verdict, reliable = "unreliable", False
            reason = (
                f"The on-target prediction for {on_target} is itself out of "
                f"domain, so the apparent {delta:+.2f} log-unit window is not "
                f"meaningful."
            )
        else:
            reliable = True
            if delta >= SELECTIVE_STRONG:
                verdict = "selective"
                reason = (
                    f"{10 ** delta:.0f}-fold predicted preference for "
                    f"{on_target} over {best_off}."
                )
            elif delta >= SELECTIVE_MODEST:
                verdict = "modestly_selective"
                reason = (
                    f"{10 ** delta:.0f}-fold predicted preference for "
                    f"{on_target}; worth optimising for a wider window."
                )
            elif delta > PROMISCUOUS:
                verdict = "weakly_selective"
                reason = (
                    f"Only {10 ** delta:.1f}-fold over {best_off} - too narrow "
                    f"to count as selective."
                )
            else:
                verdict = "promiscuous"
                reason = (
                    f"Predicted to bind {best_off} at least as well as "
                    f"{on_target} ({delta:+.2f} log units). A promiscuity "
                    f"liability rather than a lead."
                )

        if excluded:
            reason += (
                f" Excluded as out-of-domain: {', '.join(excluded)}."
            )

        return SelectivityProfile(
            smiles=smiles, on_target=on_target,
            on_target_pic50=round(on_pic50, 3), on_target_confidence=on_conf,
            best_off_target=best_off,
            best_off_target_pic50=round(best_off_pic50, 3),
            selectivity_log_units=round(delta, 3),
            fold_selectivity=round(float(10 ** delta), 1),
            verdict=verdict, reliable=reliable,
            per_target_pic50=per_target, excluded_off_targets=excluded,
            reason=reason,
        )

    def get_selectivity_reward(self, smiles: str, on_target: str) -> float:
        """Selectivity as a 0-1 RL reward term.

        0.5 when selectivity cannot be established, so the generator is neither
        rewarded nor punished for something the reference data cannot assess.
        """
        p = self.profile(smiles, on_target)
        if p.selectivity_log_units is None or not p.reliable:
            return 0.5
        # -1 log unit -> 0.0, 0 -> 0.33, +2 -> 1.0
        return float(np.clip((p.selectivity_log_units + 1.0) / 3.0, 0.0, 1.0))

    def rank_by_selectivity(self, smiles_list: List[str],
                            on_target: str) -> List[SelectivityProfile]:
        """Profile a list and sort most-selective first (unreliable last)."""
        profiles = [self.profile(s, on_target) for s in smiles_list]
        profiles.sort(
            key=lambda p: (
                p.reliable,
                p.selectivity_log_units if p.selectivity_log_units is not None else -99,
            ),
            reverse=True,
        )
        return profiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

    scorer = SelectivityScorer()
    if not scorer.is_available:
        print("Need at least two QSAR models. Run train_qsar_models.py.")
        sys.exit(1)

    cases = [
        ('Gefitinib', 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1', 'EGFR'),
        ('Erlotinib', 'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1', 'EGFR'),
        ('Celecoxib', 'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1', 'COX2'),
        ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O', 'COX2'),
        ('Aspirin (wrong target)', 'CC(=O)Oc1ccccc1C(=O)O', 'EGFR'),
        ('Methane', 'C', 'EGFR'),
    ]

    for name, smi, target in cases:
        p = scorer.profile(smi, target)
        print(f"\n{name}  ->  {target}")
        if p.on_target_pic50 is not None:
            print(f"  on-target pIC50 = {p.on_target_pic50} ({p.on_target_confidence})")
        if p.best_off_target:
            print(f"  best off-target = {p.best_off_target} "
                  f"pIC50 {p.best_off_target_pic50}")
            print(f"  selectivity     = {p.selectivity_log_units:+.2f} log units "
                  f"({p.fold_selectivity}x)")
        print(f"  verdict         = {p.verdict}  (reliable={p.reliable})")
        print(f"  {p.reason}")
