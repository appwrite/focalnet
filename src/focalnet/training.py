"""Supervised distillation with masked heatmap and spatial-distribution losses."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from focalnet.data import read_manifest, sha256_file
from focalnet.imaging import INPUT_SIZE, MAP_SIZE, open_image, prepare_image, project_map
from focalnet.model import FocalNet, ModelConfig, load_checkpoint


class ImportanceDataset(Dataset):
    def __init__(self, records: list[dict], *, augment: bool = False):
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = open_image(record["image"])
        target = np.load(record["importance"], allow_pickle=False)
        if (
            target.shape != (MAP_SIZE, MAP_SIZE)
            or not np.isfinite(target).all()
            or target.min() < 0
            or target.max() > 1
        ):
            raise ValueError(f"Expected a finite {MAP_SIZE}x{MAP_SIZE} target in [0, 1]: {record}")
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target = np.fliplr(target).copy()
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        tensor, box = prepare_image(image)
        target, valid = project_map(target, box)
        return (
            torch.from_numpy(tensor[0]),
            torch.from_numpy(target[None]),
            torch.from_numpy(valid[None]),
        )


def importance_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    axes = (1, 2, 3)
    weights = valid * (1 + 4 * target)
    bce = (
        (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum(axes)
        / weights.sum(axes).clamp_min(1e-8)
    ).mean()
    target_mass = target * valid
    pred_mass = logits.sigmoid() * valid
    target_total = target_mass.sum(axes, keepdim=True)
    p = target_mass / target_total.clamp_min(1e-8)
    q = pred_mass / pred_mass.sum(axes, keepdim=True).clamp_min(1e-8)
    divergence = (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(axes)
    nonempty = (target_total.flatten() > 0).to(logits.dtype)
    kl = (divergence * nonempty).sum() / nonempty.sum().clamp_min(1)
    loss = bce + 0.25 * kl
    return loss, {"bce": bce.detach(), "kl": kl.detach()}


def ensure_disjoint(train: list[dict], val: list[dict]) -> None:
    for field in ("group", "image"):
        overlap = {r[field] for r in train} & {r[field] for r in val}
        if overlap:
            raise ValueError(f"Training/validation leakage: shared {field}: {next(iter(overlap))}")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    freeze_encoder_epochs: int = 1
    face_sampling_weight: float = 2.0
    seed: int = 42
    workers: int = 0
    threads: int = 4
    device: str = "auto"
    pretrained: bool = True
    amp: bool = False

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.threads) < 1 or self.workers < 0:
            raise ValueError("Epochs, batch size, threads must be positive; workers nonnegative")
        if self.freeze_encoder_epochs < 0:
            raise ValueError("Freeze epochs must be nonnegative")
        for name in ("learning_rate", "face_sampling_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
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


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(
    train_manifest: Path,
    val_manifest: Path,
    output: Path,
    model_config: ModelConfig,
    config: TrainConfig,
    resume: Path | None = None,
    initialize: Path | None = None,
    *,
    validate_files: bool = True,
) -> dict:
    if resume and initialize:
        raise ValueError("Choose either resume or initialize, not both")
    train_records = read_manifest(train_manifest, validate_files=validate_files)
    val_records = read_manifest(val_manifest, validate_files=validate_files)
    ensure_disjoint(train_records, val_records)
    fingerprints = {"train": sha256_file(train_manifest), "validation": sha256_file(val_manifest)}
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    sampler_generator = torch.Generator().manual_seed(config.seed)
    loader_generator = torch.Generator().manual_seed(config.seed + 1)
    checkpoint = None
    initialized_from = None
    if resume:
        model, checkpoint = load_checkpoint(str(resume))
        if checkpoint["manifest_sha256"] != fingerprints:
            raise ValueError("Resume manifests differ from the checkpoint")
        if checkpoint["train_config"] != asdict(config) or model.config != model_config:
            raise ValueError("Resume requires the original training and model configuration")
        if output.resolve() != resume.resolve().parent:
            raise ValueError("Resume output must be the checkpoint's run directory")
    elif initialize:
        model, initial_checkpoint = load_checkpoint(str(initialize))
        if model.config != model_config:
            raise ValueError("Initialization checkpoint model configuration differs")
        initialized_from = {
            "checkpoint": sha256_file(initialize),
            "epoch": initial_checkpoint["epoch"],
        }
        if output.exists():
            raise ValueError(f"Run directory already exists: {output}; use --resume or a new path")
    else:
        if output.exists():
            raise ValueError(f"Run directory already exists: {output}; use --resume or a new path")
        model = FocalNet(model_config, pretrained=config.pretrained)
    model.to(device)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.learning_rate * 0.1},
            {
                "params": list(model.lateral.parameters()) + list(model.refine.parameters()),
                "lr": config.learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    start, best, history = 0, float("inf"), []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best, history = (
            checkpoint["epoch"] + 1,
            checkpoint["best_val_loss"],
            checkpoint["history"],
        )
        initialized_from = checkpoint.get("initialized_from")
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        sampler_generator.set_state(checkpoint["rng"]["sampler"])
        loader_generator.set_state(checkpoint["rng"]["loader"])
        if device.type == "cuda" and "cuda" in checkpoint["rng"]:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        if device.type == "mps" and "mps" in checkpoint["rng"]:
            torch.mps.set_rng_state(checkpoint["rng"]["mps"])
        if use_amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    weights = [
        r["weight"] * (config.face_sampling_weight if r.get("face_count", 0) else 1)
        for r in train_records
    ]
    sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True, generator=sampler_generator
    )
    train_loader = DataLoader(
        ImportanceDataset(train_records, augment=True),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        ImportanceDataset(val_records),
        batch_size=config.batch_size,
        num_workers=config.workers,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": model.parameter_count(),
                "train_images": len(train_records),
                "val_images": len(val_records),
                "amp": use_amp,
            }
        ),
        flush=True,
    )
    for epoch in range(start, config.epochs):
        frozen = epoch < config.freeze_encoder_epochs
        model.train()
        model.encoder.requires_grad_(not frozen)
        if frozen:
            model.encoder.eval()  # Freeze BatchNorm statistics as well as gradients.
        sums = {"train_loss": 0.0, "val_loss": 0.0}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            if phase == "val":
                model.eval()
            for image, target, valid in loader:
                image = image.to(device, non_blocking=use_amp)
                target = target.to(device, non_blocking=use_amp)
                valid = valid.to(device, non_blocking=use_amp)
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_amp
                    ):
                        loss, _ = importance_loss(model(image), target, valid)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite {phase} loss at epoch {epoch + 1}")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                sums[f"{phase}_loss"] += float(loss.detach()) * len(image)
        scheduler.step()
        metrics = {
            "epoch": epoch + 1,
            "train_loss": sums["train_loss"] / len(train_records),
            "val_loss": sums["val_loss"] / len(val_records),
        }
        improved = metrics["val_loss"] < best
        best = min(best, metrics["val_loss"])
        history.append(metrics)
        rng = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "sampler": sampler_generator.get_state(),
            "loader": loader_generator.get_state(),
        }
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        if device.type == "mps":
            rng["mps"] = torch.mps.get_rng_state()
        checkpoint = {
            "format_version": 1,
            "input_size": INPUT_SIZE,
            "map_size": MAP_SIZE,
            "model_config": model.config.to_dict(),
            "train_config": asdict(config),
            "manifest_sha256": fingerprints,
            "initialized_from": initialized_from,
            "epoch": epoch,
            "best_val_loss": best,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "scaler": scaler.state_dict(),
            "rng": rng,
        }
        _save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            _save_checkpoint(output / "best.pt", checkpoint)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
    return {"checkpoint": str(output / "best.pt"), "best_val_loss": best}
