#!/usr/bin/env python3
"""
Build the applicability-domain caches for the QSAR models.

Each cache (models/<TARGET>_ad.npz) holds the bit-packed Morgan fingerprints of
that target's training set plus the leave-one-out nearest-neighbour similarity
percentiles used as confidence thresholds. The caches are built automatically on
first use, but run this directly after retraining or after refreshing the
ChEMBL data so the domain check matches the models.

Usage:
    python build_domain_caches.py              # build any that are missing
    python build_domain_caches.py --force      # rebuild all
    python build_domain_caches.py --targets EGFR JAK2
"""

import argparse
import logging
import sys

from applicability_domain import ApplicabilityDomain
from qsar_predictor import QSARPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help="rebuild caches that already exist")
    parser.add_argument('--targets', nargs='+', default=QSARPredictor.TARGETS,
                        help="subset of targets to build")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

    ad = ApplicabilityDomain()
    failures = []

    print(f"{'Target':10s} {'Molecules':>10s} {'Domain edge':>12s} {'Median sim':>11s}")
    print("-" * 47)

    for target in (t.upper() for t in args.targets):
        if not ad.build_cache(target, force=args.force):
            failures.append(target)
            print(f"{target:10s} {'FAILED - no training CSV?':>35s}")
            continue

        info = ad.domain_info(target)
        if info is None:
            failures.append(target)
            print(f"{target:10s} {'FAILED - cache unreadable':>35s}")
            continue

        print(f"{target:10s} {info['n_train']:10d} "
              f"{info['domain_edge_p1']:12.3f} {info['similarity_median']:11.3f}")

    if failures:
        print(f"\nFailed: {', '.join(failures)}")
        print("Run 'python download_chembl_data.py' to fetch the training data.")
        return 1

    print("\nAll applicability-domain caches ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
