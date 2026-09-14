"""Fine-tune a lightweight crop ranker from dense human opinion scores."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from focalnet.candidates import letterbox_content
from focalnet.data import sha256_file
from focalnet.imaging import open_image, prepare_image
from focalnet.ranking import (
    HumanCropNet,
    RankConfig,
    initialize_human_model,
    load_human_checkpoint,
)


def _gaic_paths(root: Path, split: str) -> list[tuple[Path, Path]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("GAICD split must be train, val, or test")
    image_root, annotation_root = root / "images" / split, root / "annotations" / split
    images = sorted(image_root.glob("*.jpg"))
    if not images:
        raise ValueError(f"No GAICD images found in {image_root}")
    paths = [(image, annotation_root / f"{image.stem}.txt") for image in images]
    missing = [annotation for _, annotation in paths if not annotation.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GAICD annotation: {missing[0]}")
    return paths


def read_gaic_annotation(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes, scores = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            top, left, bottom, right, score = map(float, line.split())
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: expected top left bottom right score") from exc
        if score == -2:
            continue
        if not np.isfinite([top, left, bottom, right, score]).all():
            raise ValueError(f"{path}:{number}: values must be finite")
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError(f"{path}:{number}: crop is outside the {width}x{height} image")
        boxes.append((left / width, top / height, right / width, bottom / height))
        scores.append(score)
    if len(boxes) < 2:
        raise ValueError(f"{path}: at least two rated crops are required")
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


class GAICDataset(Dataset):
    def __init__(self, root: Path, split: str, *, augment: bool = False):
        self.root = root
        self.split = split
        self.augment = augment
        self.records = []
        for image_path, annotation_path in _gaic_paths(root, split):
            with Image.open(image_path) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f"GAICD coordinates require unrotated pixels: {image_path}")
                boxes, scores = read_gaic_annotation(annotation_path, image.width, image.height)
            self.records.append((image_path, boxes, scores))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        path, boxes, scores = self.records[index]
        image = open_image(path)
        boxes = boxes.copy()
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        tensor, content = prepare_image(image)
        return (
            torch.from_numpy(tensor[0]),
            torch.from_numpy(boxes),
            torch.from_numpy(scores.copy()),
            torch.from_numpy(letterbox_content(content)),
            path.stem,
        )


def collate_crops(batch: list[tuple]) -> tuple:
    images, crop_lists, score_lists, contents, groups = zip(*batch, strict=True)
    maximum = max(len(crops) for crops in crop_lists)
    boxes = torch.zeros(len(batch), maximum, 4, dtype=torch.float32)
    scores = torch.zeros(len(batch), maximum, dtype=torch.float32)
    valid = torch.zeros(len(batch), maximum, dtype=torch.bool)
    for index, (crops, values) in enumerate(zip(crop_lists, score_lists, strict=True)):
        count = len(crops)
        boxes[index, :count] = crops
        scores[index, :count] = values
        valid[index, :count] = True
    return (
        torch.stack(images),
        boxes,
        scores,
        valid,
        torch.stack(contents),
        groups,
    )


def crop_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_difference: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = valid.to(predicted.dtype)
    count = weights.sum(1, keepdim=True).clamp_min(1)
    predicted_centered = predicted - (predicted * weights).sum(1, keepdim=True) / count
    target_centered = target - (target * weights).sum(1, keepdim=True) / count
    regression = (
        F.smooth_l1_loss(predicted_centered, target_centered, reduction="none") * weights
    ).sum() / weights.sum().clamp_min(1)

    target_difference = target[:, :, None] - target[:, None, :]
    predicted_difference = predicted[:, :, None] - predicted[:, None, :]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    pair_mask &= torch.triu(
        torch.ones(target.shape[1], target.shape[1], dtype=torch.bool, device=target.device),
        diagonal=1,
    )
    pair_mask &= target_difference.abs() >= minimum_difference
    direction = target_difference.sign()
    pair_weights = target_difference.abs().clamp(max=2)
    pairwise_values = F.softplus(-direction * predicted_difference) * pair_weights
    pairwise = pairwise_values[pair_mask].mean() if pair_mask.any() else predicted.sum() * 0
    loss = pairwise + 0.25 * regression
    return loss, {"pairwise": pairwise.detach(), "regression": regression.detach()}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first, second = first - first.mean(), second - second.mean()
    denominator = np.sqrt(np.square(first).sum() * np.square(second).sum())
    return float(first @ second / denominator) if denominator > 0 else 0.0


def ranking_metrics(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Ranking metrics require matching nonempty prediction and target lists")
    srcc, pcc, top = [], [], {5: [], 10: []}
    correct_pairs = total_pairs = 0
    for predicted, target in zip(predictions, targets, strict=True):
        if predicted.shape != target.shape or predicted.ndim != 1 or len(predicted) < 2:
            raise ValueError(
                "Each crop score list must be matching, one-dimensional, and nontrivial"
            )
        srcc.append(_correlation(_rankdata(predicted), _rankdata(target)))
        pcc.append(_correlation(predicted, target))
        predicted_order = np.argsort(-predicted, kind="stable")
        target_order = np.argsort(-target, kind="stable")
        for n in top:
            accepted = set(target_order[: min(n, len(target))])
            top[n].append(
                np.mean(
                    [
                        len(set(predicted_order[:k]) & accepted) / k
                        for k in range(1, min(4, len(target)) + 1)
                    ]
                )
            )
        difference = target[:, None] - target[None, :]
        mask = np.triu(np.abs(difference) >= 0.25, 1)
        predicted_difference = predicted[:, None] - predicted[None, :]
        correct_pairs += int(np.count_nonzero((predicted_difference * difference > 0) & mask))
        total_pairs += int(np.count_nonzero(mask))
    return {
        "images": len(predictions),
        "srcc": float(np.mean(srcc)),
        "pcc": float(np.mean(pcc)),
        "acc_5": float(np.mean(top[5])),
        "acc_10": float(np.mean(top[10])),
        "pairwise_accuracy": correct_pairs / total_pairs if total_pairs else 0.0,
        "comparable_pairs": total_pairs,
    }


@dataclass(frozen=True)
class HumanTrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    workers: int = 0
    threads: int = 4
    device: str = "auto"
    amp: bool = False

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.threads) < 1 or self.workers < 0:
            raise ValueError(
                "Epochs, batch size, and threads must be positive; workers nonnegative"
            )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(name)


def gaic_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "annotations").glob("*/*.txt")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_human_model(
    model: HumanCropNet,
    dataset: GAICDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_crops,
        pin_memory=device.type == "cuda",
    )
    predictions, baseline_predictions, targets = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, boxes, scores, valid, content, _ in loader:
            images, boxes, content = images.to(device), boxes.to(device), content.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                importance, output = model(images, boxes, content)
                baseline = model.importance_retention(importance, boxes, content)
            output = output.float().cpu().numpy()
            baseline = baseline.float().cpu().numpy()
            for index, mask in enumerate(valid.numpy()):
                predictions.append(output[index, mask])
                baseline_predictions.append(baseline[index, mask])
                targets.append(scores[index, mask].numpy())
    result = ranking_metrics(predictions, targets)
    result["importance_baseline"] = ranking_metrics(baseline_predictions, targets)
    return result


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_crop_ranker(
    train_set: Dataset,
    val_set: Dataset,
    output: Path,
    config: HumanTrainConfig,
    *,
    dataset_fingerprint: str,
    dataset_name: str,
    base_checkpoint: Path | None = None,
    rank_config: RankConfig | None = None,
    resume: Path | None = None,
    initialize: Path | None = None,
) -> dict:
    if resume:
        model, checkpoint = load_human_checkpoint(resume)
        if output.resolve() != resume.resolve().parent:
            raise ValueError("Resume output must be the checkpoint's run directory")
        if checkpoint["train_config"] != asdict(config):
            raise ValueError("Resume requires the original human-ranking configuration")
        base_checkpoint_sha256 = checkpoint["base_checkpoint_sha256"]
        initialize_checkpoint_sha256 = checkpoint.get("initialize_checkpoint_sha256")
    else:
        if output.exists():
            raise ValueError(f"Run directory already exists: {output}")
        if (base_checkpoint is None) == (initialize is None):
            raise ValueError("Provide exactly one base checkpoint or human-ranking initializer")
        if initialize:
            model, initial_checkpoint = load_human_checkpoint(initialize)
            if rank_config is not None and model.config != rank_config:
                raise ValueError("Initializer rank configuration does not match the requested one")
            base_checkpoint_sha256 = initial_checkpoint["base_checkpoint_sha256"]
            initialize_checkpoint_sha256 = sha256_file(initialize)
        else:
            model, _ = initialize_human_model(base_checkpoint, rank_config)
            base_checkpoint_sha256 = sha256_file(base_checkpoint)
            initialize_checkpoint_sha256 = None
        checkpoint = None
    if checkpoint and checkpoint["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError("Resume dataset annotations differ from the checkpoint")
    if checkpoint and checkpoint.get("dataset_name", dataset_name) != dataset_name:
        raise ValueError("Resume dataset name differs from the checkpoint")

    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    generator = torch.Generator().manual_seed(config.seed + 1)
    model.base.requires_grad_(False)
    model.to(device)
    training_parameters = list(model.rank_projection.parameters()) + list(
        model.rank_head.parameters()
    )
    optimizer = torch.optim.AdamW(
        training_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best, history = 0, float("-inf"), []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, history = (
            checkpoint["epoch"] + 1,
            checkpoint["best_val_srcc"],
            checkpoint["history"],
        )
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        generator.set_state(checkpoint["rng"]["loader"])
        if device.type == "cuda" and "cuda" in checkpoint["rng"]:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        if device.type == "mps" and "mps" in checkpoint["rng"]:
            torch.mps.set_rng_state(checkpoint["rng"]["mps"])
    loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_crops,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "total_parameters": model.parameter_count(),
                "ranking_parameters": model.ranking_parameter_count(),
                "train_images": len(train_set),
                "validation_images": len(val_set),
                "amp": use_amp,
            }
        ),
        flush=True,
    )
    for epoch in range(start, config.epochs):
        model.train()
        model.base.eval()
        total_loss = 0.0
        for images, boxes, scores, valid, content, _ in loader:
            images = images.to(device, non_blocking=use_amp)
            boxes = boxes.to(device, non_blocking=use_amp)
            scores = scores.to(device, non_blocking=use_amp)
            valid = valid.to(device, non_blocking=use_amp)
            content = content.to(device, non_blocking=use_amp)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, predicted = model(images, boxes, content)
                loss, _ = crop_ranking_loss(predicted, scores, valid)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite ranking loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(training_parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
        scheduler.step()
        validation = evaluate_human_model(
            model,
            val_set,
            batch_size=config.batch_size,
            workers=config.workers,
            device=device,
        )
        metrics = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_set),
            **{f"val_{key}": value for key, value in validation.items() if key != "images"},
        }
        improved = validation["srcc"] > best
        best = max(best, validation["srcc"])
        history.append(metrics)
        rng = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "loader": generator.get_state(),
        }
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        if device.type == "mps":
            rng["mps"] = torch.mps.get_rng_state()
        state = {
            "format_version": 2,
            "task": "human-crop-ranking",
            "input_size": 256,
            "map_size": 64,
            "model_config": model.base.config.to_dict(),
            "rank_config": model.config.to_dict(),
            "train_config": asdict(config),
            "dataset_name": dataset_name,
            "dataset_fingerprint": dataset_fingerprint,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "initialize_checkpoint_sha256": initialize_checkpoint_sha256,
            "epoch": epoch,
            "best_val_srcc": best,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng,
            "history": history,
        }
        _save_checkpoint(output / "last.pt", state)
        if improved:
            _save_checkpoint(output / "best.pt", state)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
    return {"checkpoint": str(output / "best.pt"), "best_val_srcc": best}


def train_human_ranker(
    dataset_root: Path,
    base_checkpoint: Path | None,
    output: Path,
    config: HumanTrainConfig,
    rank_config: RankConfig | None = None,
    resume: Path | None = None,
    initialize: Path | None = None,
) -> dict:
    return train_crop_ranker(
        GAICDataset(dataset_root, "train", augment=True),
        GAICDataset(dataset_root, "val"),
        output,
        config,
        dataset_fingerprint=gaic_fingerprint(dataset_root),
        dataset_name="GAICD journal version",
        base_checkpoint=base_checkpoint,
        rank_config=rank_config,
        resume=resume,
        initialize=initialize,
    )
