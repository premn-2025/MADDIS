#!/usr/bin/env python3
"""
Synthetic accessibility scoring (Ertl & Schuffenhauer SAScore).

RDKit ships the reference implementation in its contrib tree, but it is not on
the import path by default, so a bare `import sascorer` fails. Two places in
this project tried it:

- `improved_rl_generator.py` appends the contrib directory first, so it works.
- `integrated_multiagent_platform.py` did not, so its `except ImportError`
  branch always ran and the reward term fell back to
  `min(6, max(1, num_atoms / 8.0))` — i.e. "a bigger molecule is harder to
  make". That is close to noise: it cannot tell a linear alkane from a
  bridged polycyclic of the same size, and it was feeding a 10% weight in the
  RL reward.

This module centralises the import so both paths get the real score.

SAScore runs 1 (trivial) to 10 (intractable), combining a fragment-contribution
term learned from PubChem against a complexity penalty for rings, stereocentres,
spiro and bridged systems.

Usage:
    from synthetic_accessibility import sa_score, sa_reward, SA_AVAILABLE

    sa_score("CCO")                      # ~1.4, trivial
    sa_reward("CCO")                     # ~0.92 on a 0-1 scale
"""

import logging
import os
import sys
from typing import Optional, Union

from rdkit import Chem, RDConfig

logger = logging.getLogger(__name__)

# SAScore's native range.
SA_MIN = 1.0
SA_MAX = 10.0

# Above this, a molecule is treated as impractical for a screening campaign.
# 6 is the conventional cut and matches what improved_rl_generator.py used.
SA_REWARD_CEILING = 6.0

_sascorer = None
SA_AVAILABLE = False

try:
    _sa_dir = os.path.join(RDConfig.RDContribDir, 'SA_Score')
    if _sa_dir not in sys.path:
        sys.path.append(_sa_dir)
    import sascorer as _sascorer  # noqa: E402
    SA_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on the RDKit build
    logger.warning(
        f"RDKit SAScore unavailable ({e}); synthetic accessibility will be "
        f"reported as unknown rather than estimated."
    )


def _as_mol(molecule: Union[str, 'Chem.Mol']) -> Optional['Chem.Mol']:
    if isinstance(molecule, str):
        return Chem.MolFromSmiles(molecule)
    return molecule


def sa_score(molecule: Union[str, 'Chem.Mol']) -> Optional[float]:
    """Raw SAScore, 1 (easy) to 10 (hard).

    Returns None when the molecule cannot be parsed or SAScore is unavailable —
    deliberately not a number, so callers cannot mistake a fallback guess for a
    real measurement.
    """
    if not SA_AVAILABLE:
        return None

    mol = _as_mol(molecule)
    if mol is None:
        return None

    try:
        return float(_sascorer.calculateScore(mol))
    except Exception as e:
        logger.debug(f"SAScore failed: {e}")
        return None


def sa_reward(molecule: Union[str, 'Chem.Mol'],
              ceiling: float = SA_REWARD_CEILING,
              default: float = 0.5) -> float:
    """SAScore normalised to a 0-1 reward (higher = easier to synthesise).

    Linear from 1.0 -> reward 1.0 down to `ceiling` -> reward 0.0, and 0 above
    the ceiling. `default` is returned when the score is unavailable, so an
    environment without SAScore stays neutral rather than silently penalising
    or rewarding every molecule.
    """
    raw = sa_score(molecule)
    if raw is None:
        return default

    if raw >= ceiling:
        return 0.0
    return float(max(0.0, min(1.0, (ceiling - raw) / (ceiling - SA_MIN))))


def sa_label(molecule: Union[str, 'Chem.Mol']) -> str:
    """Human-readable synthesis difficulty for UI display."""
    raw = sa_score(molecule)
    if raw is None:
        return "unknown"
    if raw < 2.5:
        return "easy"
    if raw < 4.0:
        return "moderate"
    if raw < 6.0:
        return "hard"
    return "very hard"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

    print(f"SAScore available: {SA_AVAILABLE}\n")
    if not SA_AVAILABLE:
        sys.exit(1)

    # The last two pairs are the point: same heavy-atom count, wildly different
    # synthesis difficulty. The old atom-count proxy scored them identically.
    probes = {
        'ethanol': 'CCO',
        'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'gefitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
        'n-decane (10 C, linear)': 'CCCCCCCCCC',
        'twistane (10 C, bridged)': 'C1C2CC3CC1CC(C2)C3',
        'morphine': 'CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5',
    }

    print(f"{'molecule':28s} {'SAScore':>8s} {'reward':>7s}  difficulty")
    print("-" * 62)
    for name, smi in probes.items():
        raw = sa_score(smi)
        print(f"{name:28s} {raw:8.2f} {sa_reward(smi):7.3f}  {sa_label(smi)}")
