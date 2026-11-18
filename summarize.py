"""Verify and summarize the four-dataset, five-seed paired pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiment import DATASETS, METHODS


COHORTS = ("random_test", "no_overlap", "guard_10")
SEEDS = range(5)


def load_results(directory):
    results = {}
    configs = set()
    for dataset in DATASETS:
        runs = []
        for seed in SEEDS:
            run = json.loads((directory / f"{dataset}_{seed}.json").read_text(encoding="utf-8"))
            if (run["dataset"], run["seed"]) != (dataset, seed) or set(run["metrics"]) != set(METHODS):
                raise ValueError(f"Inconsistent result for {dataset}/{seed}")
            configs.add(tuple(run[k] for k in ("steps", "shot", "reserve", "patch", "support")))
            for cohort in COHORTS:
                sizes = {run["metrics"][m][cohort]["n"] for m in METHODS}
                if len(sizes) != 1 or min(sizes) <= 0:
                    raise ValueError(f"Invalid paired cohort for {dataset}/{seed}/{cohort}")
            runs.append(run)
        if len({r["split_sha256"] for r in runs}) != len(runs):
            raise ValueError(f"Duplicate splits in {dataset}")
        results[dataset] = runs
    if len(configs) != 1:
        raise ValueError("Run configurations differ")
    return results


def metric(runs, method, cohort, key):
    return np.array([r["metrics"][method][cohort][key] for r in runs], dtype=float)


def mean_sd(values, scale):
    return f"{values.mean() * scale:.2f} ± {values.std(ddof=1) * scale:.2f}"


def interval(values):
    half_width = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return f"{values.mean():+.2f} [{values.mean()-half_width:+.2f}, {values.mean()+half_width:+.2f}]"


def summary(results):
    lines = [
        "| Dataset | Cohort | Method | OA (%) | AA (%) | Kappa | Test n (min–max) | Classes (min–max) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, runs in results.items():
        for cohort in COHORTS:
            for method in METHODS:
                scores = [mean_sd(metric(runs, method, cohort, key), scale)
                          for key, scale in (("oa", 100), ("aa", 100), ("kappa", 1))]
                counts = metric(runs, method, cohort, "n")
                coverage = [sum(n > 0 for n in r["metrics"][method][cohort]["class_counts"]) for r in runs]
                lines.append(f"| {dataset} | {cohort} | {method} | {' | '.join(scores)} | "
                             f"{int(min(counts))}–{int(max(counts))} | {min(coverage)}–{max(coverage)} |")
    lines.extend([
        "",
        "| Dataset | No-overlap vs random retention (%) | Spatial − uniform OA (pp) | Spatial − shuffled OA (pp) |",
        "| --- | ---: | ---: | ---: |",
    ])
    spatial_uniform, spatial_shuffled = [], []
    for dataset, runs in results.items():
        weighted = metric(runs, "spatial", "no_overlap", "oa")
        uniform = 100 * (weighted - metric(runs, "uniform", "no_overlap", "oa"))
        shuffled = 100 * (weighted - metric(runs, "shuffled", "no_overlap", "oa"))
        spatial_uniform.append(uniform)
        spatial_shuffled.append(shuffled)
        coverage = metric(runs, "ce", "no_overlap", "n") / metric(runs, "ce", "random_test", "n")
        lines.append(f"| {dataset} | {coverage.mean() * 100:.2f} | {interval(uniform)} | {interval(shuffled)} |")
    lines.append(f"| Dataset macro-average | | {interval(np.mean(spatial_uniform, axis=0))} | "
                 f"{interval(np.mean(spatial_shuffled, axis=0))} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    print(summary(load_results(args.results)))
