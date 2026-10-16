# Legacy / quarantined modules

The files in this directory **do not parse**. They are kept for reference only and
are excluded from the syntax gate in CI.

## What happened

All five were committed already broken in the initial commit (`e063331`) and were
never executed. An aggressive auto-formatter corrupted their indentation into a
"staircase": every block that should have closed never dedents, so each successive
`def`/`class` ended up nested one level deeper inside the previous one. For example,
in `advanced_ai_trainer.py` the entire class body — including every method
definition — sits inside an `if torch.cuda.is_available():` block, and the `except`
clause no longer matches its `try`.

There is no clean version anywhere in git history to recover, so repairing them
means reconstructing the intent by hand rather than reverting.

## Why they were quarantined rather than repaired

None of them was reachable from working code:

| File | Referenced by |
|---|---|
| `advanced_ai_trainer.py` | nothing |
| `max_accuracy_trainer.py` | nothing |
| `ai_drug_compatibility_checker.py` | nothing |
| `advanced_stability_analyzer.py` | `web/api_server.py`, inside a `try/except` that only set a capability flag — the class was never instantiated |
| `enhanced_multi_target_trainer.py` | `sequential_training_plan.py`, which also imports two modules that do not exist (`test_related_targets`, `comprehensive_validation`) |

Their functionality also already exists in working form elsewhere:

- Stability / ADMET / Lipinski-Veber-Egan / PAINS+BRENK / reactive groups →
  [`admet_predictor.py`](../admet_predictor.py), [`chemical_safety_filter.py`](../chemical_safety_filter.py),
  and the `_assess_stability` / `_compute_mol_profile` helpers in
  [`integrated_multiagent_platform.py`](../integrated_multiagent_platform.py)
- Drug–drug compatibility → `_assess_interaction_risk` in `integrated_multiagent_platform.py`
- Multi-target RL training → [`multi_target_rl_generator.py`](../multi_target_rl_generator.py),
  [`real_rl_trainer.py`](../real_rl_trainer.py)

## If you want to revive one

1. Read the original alongside the corrupted copy:
   `git show e063331:<filename> > /tmp/orig.py`
   (it is the same corrupted content — the staircase is the original state)
2. Reconstruct the intended nesting. The inline comments are a useful guide: they
   kept closer to their original indentation than the code did, so they mark where
   sequential sections at the same level began.
3. Confirm with `python -m py_compile <file>`, then move it back to the repo root
   and add a test before wiring it into anything.
