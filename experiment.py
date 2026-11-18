"""Paired pilot for distance-weighted episodic HSI classification."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import distance_transform_cdt
from torch import nn
from torch.nn import functional as F


DATASETS = {
    "ip": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
    "pu": ("PaviaU.mat", "PaviaU_gt.mat"),
    "houston13": ("GRSS2013.mat", "GRSS2013_gt.mat"),
    "whuhh": ("WHU_Hi_HongHu.mat", "WHU_Hi_HongHu_gt.mat"),
}
METHODS = ("ce", "uniform", "shuffled", "spatial")
SHOT = 5
RESERVED = 5
PATCH = 5
SUPPORT = 2
SCALE = 12.0


def read_mat(path, rank):
    arrays = [v for k, v in loadmat(path).items() if not k.startswith("__") and v.ndim == rank]
    if not arrays:
        raise ValueError(f"No {rank}-D array in {path}")
    return max(arrays, key=lambda array: array.size)


def load_scene(root, name):
    image, labels = (root / name / file for file in DATASETS[name])
    gt = read_mat(labels, 2)
    cube = read_mat(image, 3)
    if cube.shape[:2] != gt.shape and cube.shape[-2:] == gt.shape:
        cube = np.moveaxis(cube, 0, -1)
    if cube.shape[:2] != gt.shape:
        raise ValueError(f"{name}: cube {cube.shape} and labels {gt.shape} do not match")
    return np.array(cube, dtype=np.float32, order="C"), gt


def build_split(gt, seed):
    rng = np.random.default_rng(seed)
    labeled = np.flatnonzero(gt.ravel() > 0)
    classes = np.unique(gt.ravel()[labeled])
    by_class = [rng.permutation(labeled[gt.ravel()[labeled] == label]) for label in classes]
    if any(len(indices) <= SHOT + RESERVED for indices in by_class):
        raise ValueError("Every class needs more than 10 labeled pixels")
    train = np.stack([indices[:SHOT] for indices in by_class])
    reserved = np.stack([indices[SHOT:SHOT + RESERVED] for indices in by_class])
    test = np.sort(np.concatenate([indices[SHOT + RESERVED:] for indices in by_class]))
    mask = np.ones(gt.shape, dtype=np.uint8)
    mask.flat[train.ravel()] = 0
    mask.flat[reserved.ravel()] = 0
    distance = distance_transform_cdt(mask, metric="chessboard").ravel()[test]
    safe = distance >= PATCH
    return train, reserved, test, np.searchsorted(classes, gt.ravel()[test]), distance, safe


class PatchReader:
    def __init__(self, cube, device):
        self.width = cube.shape[1]
        image = torch.as_tensor(cube, device=device).permute(2, 0, 1)
        self.padded = F.pad(image.unsqueeze(0), (PATCH // 2,) * 4, mode="reflect")[0]
        self.offsets = torch.arange(PATCH, device=device)

    def read(self, indices):
        idx = torch.as_tensor(indices, device=self.padded.device)
        rows = idx[:, None, None] // self.width + self.offsets[None, :, None]
        cols = idx[:, None, None] % self.width + self.offsets[None, None, :]
        return self.padded[:, rows, cols].permute(1, 0, 2, 3).contiguous()


class Encoder(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.spectral = nn.Linear(bands, 32)
        self.spatial = nn.Sequential(
            nn.Conv2d(bands, 24, 1, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(64, 32)

    def forward(self, patch):
        center = patch[:, :, PATCH // 2, PATCH // 2]
        fused = torch.cat([F.gelu(self.spectral(center)), self.spatial(patch)], dim=1)
        return F.normalize(self.projection(fused), dim=1)


def proto_logits(query, prototypes):
    # Squared distance, dropping the additive query norm which is constant over classes.
    return SCALE * (2 * query @ prototypes.T - prototypes.square().sum(dim=1))


def episodes(train, width, seed, steps):
    """Same support/query permutations and weight distribution across paired methods."""
    rng = np.random.default_rng(seed)
    noise = np.random.default_rng(seed + 10_000)
    classes = len(train)
    coordinates = np.stack(np.divmod(train, width), axis=-1)
    distances = np.abs(coordinates[:, :, None] - coordinates[:, None, :]).max(axis=-1)
    rows = np.arange(classes)[:, None]
    for _ in range(steps):
        order = rng.random((classes, SHOT)).argsort(axis=-1)
        support, query = order[:, :SUPPORT], order[:, SUPPORT:]
        nearest = distances[rows[:, :, None], query[:, :, None], support[:, None, :]].min(axis=-1)
        # Use relative distance within each class so elongated classes do not dominate.
        relative = nearest / np.maximum(nearest.max(axis=1, keepdims=True), 1)
        weights = 1.0 + relative
        weights /= weights.mean(axis=1, keepdims=True)
        shuffled = np.take_along_axis(weights, noise.random(weights.shape).argsort(axis=1), axis=1)
        yield support, query, weights, shuffled


def train_model(method, x, train, width, seed, steps):
    torch.manual_seed(seed)
    classes, _, bands, _, _ = x.shape
    model = Encoder(bands).to(x.device)
    head = nn.Linear(32, classes, bias=False).to(x.device) if method == "ce" else None
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(params, lr=0.002, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.0002)
    row = torch.arange(classes, device=x.device)[:, None]
    yall = row.ravel().repeat_interleave(SHOT)
    yquery = row.ravel().repeat_interleave(SHOT - SUPPORT)
    for support, query, spatial, shuffled in episodes(train, width, seed, steps):
        optimizer.zero_grad(set_to_none=True)
        embedding = model(x.flatten(0, 1)).reshape(classes, SHOT, -1)
        if head is not None:
            logits = SCALE * F.linear(embedding.flatten(0, 1), F.normalize(head.weight, dim=1))
            loss = F.cross_entropy(logits, yall)
        else:
            supports = embedding[row, torch.as_tensor(support, device=x.device)]
            queries = embedding[row, torch.as_tensor(query, device=x.device)]
            logits = proto_logits(queries.flatten(0, 1), supports.mean(dim=1))
            losses = F.cross_entropy(logits, yquery, reduction="none")
            if method in ("spatial", "shuffled"):
                weights = spatial if method == "spatial" else shuffled
                losses = losses * torch.as_tensor(weights.ravel(), dtype=x.dtype, device=x.device)
            loss = losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {method} training loss")
        loss.backward()
        optimizer.step()
        schedule.step()
    return model.eval(), head, sum(p.numel() for p in params)


def scores(truth, pred, classes):
    cm = np.bincount(truth * classes + pred, minlength=classes ** 2).reshape(classes, classes)
    support = cm.sum(axis=1)
    total = cm.sum()
    if not total:
        return {"n": 0, "oa": None, "aa": None, "kappa": None, "class_counts": support.tolist()}
    oa = np.trace(cm) / total
    expected = np.dot(support, cm.sum(axis=0)) / total ** 2
    present = support > 0
    return {
        "n": int(total),
        "oa": float(oa),
        "aa": float((cm.diagonal()[present] / support[present]).mean()),
        "kappa": float((oa - expected) / (1 - expected)) if expected < 1 else None,
        "class_counts": support.tolist(),
        "confusion": cm.tolist(),
    }


def training_inputs(reader, train):
    # Fit preprocessing on labeled training centers only.
    patches = reader.read(train.ravel())
    centers = patches[:, :, PATCH // 2, PATCH // 2]
    mean = centers.mean(dim=0)[None, :, None, None]
    std = centers.std(dim=0, unbiased=False).clamp_min(1e-5)[None, :, None, None]
    x = ((patches - mean) / std).reshape(len(train), SHOT, *patches.shape[1:])
    return x, mean, std


def run_seed(reader, gt, dataset, seed, steps, batch_size, device):
    start = time.perf_counter()
    train, reserved, test, labels, distance, safe = build_split(gt, seed)
    x, mean, std = training_inputs(reader, train)
    trained = {method: train_model(method, x, train, reader.width, seed, steps) for method in METHODS}
    predictions = {method: [] for method in METHODS}
    with torch.inference_mode():
        prototypes = {
            method: model(x.flatten(0, 1)).reshape(len(train), SHOT, -1).mean(dim=1)
            for method, (model, head, _) in trained.items()
            if head is None
        }
        for indices in np.array_split(test, (len(test) + batch_size - 1) // batch_size):
            block = (reader.read(indices) - mean) / std
            for method, (model, head, _) in trained.items():
                encoded = model(block)
                if head is None:
                    logits = proto_logits(encoded, prototypes[method])
                else:
                    logits = F.linear(encoded, F.normalize(head.weight, dim=1))
                predictions[method].append(logits.argmax(dim=1).cpu().numpy())
    metrics = {}
    for method, batches in predictions.items():
        pred = np.concatenate(batches)
        metrics[method] = {
            "random_test": scores(labels, pred, len(train)),
            "no_overlap": scores(labels[safe], pred[safe], len(train)),
            "guard_10": scores(labels[distance >= 10], pred[distance >= 10], len(train)),
            "parameters": trained[method][2],
        }
    split_key = json.dumps([train.tolist(), reserved.tolist()], separators=(",", ":"))
    return {
        "dataset": dataset, "seed": seed, "steps": steps, "eval_batch_size": batch_size,
        "patch": PATCH, "shot": SHOT, "reserve": RESERVED, "support": SUPPORT,
        "device": device, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "duration_seconds": round(time.perf_counter() - start, 2),
        "train_indices": train.tolist(), "reserved_indices": reserved.tolist(),
        "split_sha256": hashlib.sha256(split_key.encode()).hexdigest(), "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or min(args.seeds) < 0:
        parser.error("steps and batch-size must be positive and seeds nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
    for name in args.datasets:
        paths = {seed: args.output / f"{name}_{seed}.json" for seed in args.seeds}
        seeds = []
        for seed, path in paths.items():
            if not path.exists():
                seeds.append(seed)
                continue
            prior = json.loads(path.read_text(encoding="utf-8"))
            if (prior["steps"], prior["eval_batch_size"], prior["device"]) != (
                args.steps, args.batch_size, args.device
            ):
                raise ValueError(f"Different run configuration at {path}. Choose a new output directory.")
        if not seeds:
            continue
        cube, gt = load_scene(args.data_root, name)
        reader = PatchReader(cube, args.device)
        del cube
        for seed in seeds:
            result = run_seed(reader, gt, name, seed, args.steps, args.batch_size, args.device)
            path = paths[seed]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            values = " ".join(
                f"{method}={result['metrics'][method]['no_overlap']['oa']:.3f}"
                for method in METHODS
            )
            print(f"{name} seed={seed}: {values} ({result['duration_seconds']:.1f}s)", flush=True)
        del reader


if __name__ == "__main__":
    main()
