"""Evaluate whether reducing redundant nearby supports improves class prototypes."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiment import (
    PATCH, build_split, load_scene, PatchReader, proto_logits, scores,
    train_model, training_inputs,
)
from summarize import load_results, interval


def prototype_weights(train, width, seed):
    coordinates = np.stack(np.divmod(train, width), axis=-1)
    distance = np.abs(coordinates[:, :, None] - coordinates[:, None, :]).max(axis=-1)
    # A support counts itself. Neighbors with overlapping patches share its weight.
    density = (distance < PATCH).sum(axis=-1)
    inverse = 1 / density
    inverse /= inverse.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed + 30_000)
    shuffled = np.take_along_axis(inverse, rng.random(inverse.shape).argsort(axis=1), axis=1)
    return {"mean": np.full_like(inverse, 1 / train.shape[1]), "shuffled": shuffled, "density": inverse}


def run(reader, gt, main):
    seed = main["seed"]
    train, reserved, test, labels, _, safe = build_split(gt, seed)
    if train.tolist() != main["train_indices"] or reserved.tolist() != main["reserved_indices"]:
        raise ValueError("Cannot reproduce the original split")
    x, mean, std = training_inputs(reader, train)
    model, _, _ = train_model("uniform", x, train, reader.width, seed, main["steps"])
    weights = prototype_weights(train, reader.width, seed)
    affected = np.count_nonzero(np.ptp(weights["density"], axis=1) > 0)
    with torch.inference_mode():
        supports = model(x.flatten(0, 1)).reshape(len(train), train.shape[1], -1)
        prototypes = {
            name: supports.mean(dim=1) if name == "mean" else
            (supports * torch.as_tensor(weight, dtype=x.dtype, device=x.device)[..., None]).sum(dim=1)
            for name, weight in weights.items()
        }
        batches = {name: [] for name in prototypes}
        indices = test[safe]
        for start in range(0, len(indices), main["eval_batch_size"]):
            block = reader.read(indices[start:start + main["eval_batch_size"]])
            features = model((block - mean) / std)
            for name, prototype in prototypes.items():
                batches[name].append(proto_logits(features, prototype).argmax(dim=1).cpu().numpy())
    metrics = {name: scores(labels[safe], np.concatenate(pred), len(train)) for name, pred in batches.items()}
    expected_oa = main["metrics"]["uniform"]["no_overlap"]["oa"]
    # Chunk sizes differ from the full-image pass, so allow a handful of FP32 boundary flips.
    if abs(metrics["mean"]["oa"] - expected_oa) > 0.0005:
        raise RuntimeError("Uniform prototype rerun does not reproduce the original result")
    return {"dataset": main["dataset"], "seed": seed, "split_sha256": main["split_sha256"],
            "affected_classes": int(affected), "class_count": len(train), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main_results = load_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    print("| Dataset | Classes with changed support weights (min–max) | Density − mean OA (pp) | Density − shuffled OA (pp) |")
    print("| --- | ---: | ---: | ---: |")
    for dataset, runs in main_results.items():
        cube, gt = load_scene(args.data_root, dataset)
        reader = PatchReader(cube, runs[0]["device"])
        del cube
        summaries = []
        for original in runs:
            result = run(reader, gt, original)
            path = args.output / f"{dataset}_{original['seed']}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            summaries.append(result)
        del reader
        affected = [r["affected_classes"] for r in summaries]
        deltas = [100 * np.array([r["metrics"]["density"]["oa"] - r["metrics"][m]["oa"]
                                 for r in summaries]) for m in ("mean", "shuffled")]
        print(f"| {dataset} | {min(affected)}–{max(affected)} / {summaries[0]['class_count']} | "
              f"{interval(deltas[0])} | {interval(deltas[1])} |", flush=True)


if __name__ == "__main__":
    main()
