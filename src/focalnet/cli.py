"""Command-line workflow. Training dependencies are imported only when needed."""

import argparse
import json
import math
import sys
from pathlib import Path


def aspect_ratio(value: str) -> float:
    try:
        if ":" in value:
            width, height = value.split(":")
            result = float(width) / float(height)
        else:
            result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError
        return result
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("Use a positive ratio, for example 16:9 or 1.5") from exc


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Train and run a tiny image-importance network")
    commands = root.add_subparsers(dest="command", required=True)

    acquire = commands.add_parser(
        "acquire-open-images", help="Download a balanced, resized Open Images V7 subset"
    )
    acquire.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--count", type=int, default=100_000)
    acquire.add_argument("--max-side", type=int, default=640)
    acquire.add_argument("--workers", type=int, default=32)
    acquire.add_argument("--storage-limit-gib", type=float, default=75)
    acquire.add_argument("--min-free-gib", type=float, default=35)
    acquire.add_argument("--seed", type=int, default=42)
    acquire.add_argument("--jpeg-quality", type=int, default=88)
    acquire.add_argument("--mix", choices=["default", "expanded"], default="default")
    acquire.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Exclude Open Images IDs already present in another images.jsonl manifest",
    )
    acquire.add_argument(
        "--plan-only",
        action="store_true",
        help="Build licensed candidate manifests without downloading image files",
    )

    label = commands.add_parser("label", help="Generate heatmaps with U²-Net + YuNet")
    label.add_argument("--images", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    label.add_argument("--saliency-model", type=Path, required=True)
    label.add_argument("--face-model", type=Path, required=True)
    label.add_argument("--face-weight", type=float, default=3)
    label.add_argument("--face-threshold", type=float, default=0.85)
    label.add_argument("--threads", type=int, default=1)
    label.add_argument("--workers", type=int, default=1)
    label.add_argument("--provider", choices=["cpu", "coreml", "cuda"], default="cpu")
    label.add_argument("--resume", action="store_true")

    split = commands.add_parser("split", help="Create group-disjoint train/validation manifests")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train the importance head and fine-tune its encoder")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--val", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--backbone", choices=["repvit_m0_9", "mobilenetv4_conv_small"], default="repvit_m0_9"
    )
    train.add_argument("--decoder-channels", type=int, default=48)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--freeze-encoder-epochs", type=int, default=1)
    train.add_argument("--face-sampling-weight", type=float, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--amp", action="store_true", help="Use CUDA float16 mixed precision")
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--initialize", type=Path, help="Start a new run from model weights in a checkpoint"
    )

    human_train = commands.add_parser(
        "train-human", help="Train a crop ranker from GAICD human opinion scores"
    )
    human_train.add_argument("--dataset", type=Path, required=True)
    human_train.add_argument("--base-checkpoint", type=Path)
    human_train.add_argument("--output", type=Path, required=True)
    human_train.add_argument("--epochs", type=int, default=20)
    human_train.add_argument("--batch-size", type=int, default=32)
    human_train.add_argument("--learning-rate", type=float, default=3e-4)
    human_train.add_argument("--weight-decay", type=float, default=1e-4)
    human_train.add_argument("--seed", type=int, default=42)
    human_train.add_argument("--workers", type=int, default=0)
    human_train.add_argument("--threads", type=int, default=4)
    human_train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    human_train.add_argument("--amp", action="store_true")
    human_train.add_argument("--resume", type=Path)
    human_train.add_argument(
        "--initialize", type=Path, help="Start a new GAICD run from a CPC ranking checkpoint"
    )
    human_train.add_argument("--rank-feature-channels", type=int, default=16)
    human_train.add_argument("--rank-hidden-channels", type=int, default=64)

    cpc_train = commands.add_parser(
        "train-cpc", help="Pretrain the crop ranker on CPC comparative human ratings"
    )
    cpc_train.add_argument("--dataset", type=Path, required=True)
    cpc_train.add_argument("--base-checkpoint", type=Path, required=True)
    cpc_train.add_argument("--output", type=Path, required=True)
    cpc_train.add_argument("--epochs", type=int, default=20)
    cpc_train.add_argument("--batch-size", type=int, default=64)
    cpc_train.add_argument("--learning-rate", type=float, default=3e-4)
    cpc_train.add_argument("--weight-decay", type=float, default=1e-4)
    cpc_train.add_argument("--seed", type=int, default=42)
    cpc_train.add_argument("--workers", type=int, default=0)
    cpc_train.add_argument("--threads", type=int, default=4)
    cpc_train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    cpc_train.add_argument("--amp", action="store_true")
    cpc_train.add_argument("--resume", type=Path)
    cpc_train.add_argument("--validation-fraction", type=float, default=0.1)
    cpc_train.add_argument("--rank-feature-channels", type=int, default=16)
    cpc_train.add_argument("--rank-hidden-channels", type=int, default=64)

    export = commands.add_parser("export", help="Export a checkpoint and verify ONNX parity")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--output", type=Path, required=True)

    human_export = commands.add_parser(
        "export-human", help="Export and verify a human crop-ranking checkpoint"
    )
    human_export.add_argument("checkpoint", type=Path)
    human_export.add_argument("--output", type=Path, required=True)

    quantize = commands.add_parser("quantize", help="Calibrate static INT8 on training images")
    quantize.add_argument("model", type=Path)
    quantize.add_argument("--manifest", type=Path, required=True)
    quantize.add_argument("--output", type=Path, required=True)
    quantize.add_argument("--samples", type=int, default=128)
    quantize.add_argument("--seed", type=int, default=42)

    predict = commands.add_parser(
        "predict", help="Return a focal point and optional aspect-aware crop"
    )
    predict.add_argument("model", type=Path)
    predict.add_argument("image", type=Path)
    predict.add_argument("--aspect-ratio", type=aspect_ratio)
    predict.add_argument("--heatmap", type=Path, help="Save the oriented-image float32 .npy map")
    predict.add_argument("--threads", type=int, default=1)

    human_predict = commands.add_parser(
        "predict-human", help="Rank variable-position and variable-zoom crops"
    )
    human_predict.add_argument("model", type=Path)
    human_predict.add_argument("image", type=Path)
    human_predict.add_argument("--aspect-ratio", type=aspect_ratio, required=True)
    human_predict.add_argument("--retention-tolerance", type=float, default=0.05)
    human_predict.add_argument("--heatmap", type=Path)
    human_predict.add_argument("--threads", type=int, default=1)

    human_evaluate = commands.add_parser(
        "evaluate-human", help="Evaluate a checkpoint on GAICD human crop rankings"
    )
    human_evaluate.add_argument("checkpoint", type=Path)
    human_evaluate.add_argument("--dataset", type=Path, required=True)
    human_evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    human_evaluate.add_argument("--batch-size", type=int, default=32)
    human_evaluate.add_argument("--workers", type=int, default=0)
    human_evaluate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    human_evaluate.add_argument("--output", type=Path)

    evaluate = commands.add_parser("evaluate", help="Evaluate teacher agreement and crop retention")
    evaluate.add_argument("model", type=Path)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--ratios", type=aspect_ratio, nargs="+", default=[1, 16 / 9, 9 / 16])
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--threads", type=int, default=1)

    benchmark = commands.add_parser(
        "benchmark", help="Measure forward and complete pipeline latency"
    )
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--images", type=Path, required=True)
    benchmark.add_argument("--iterations", type=int, default=50)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--threads", type=int, default=1)
    benchmark.add_argument("--output", type=Path)

    human_benchmark = commands.add_parser(
        "benchmark-human", help="Measure human crop-ranking inference latency"
    )
    human_benchmark.add_argument("model", type=Path)
    human_benchmark.add_argument("--images", type=Path, required=True)
    human_benchmark.add_argument("--aspect-ratio", type=aspect_ratio, default=1.0)
    human_benchmark.add_argument("--iterations", type=int, default=50)
    human_benchmark.add_argument("--warmup", type=int, default=5)
    human_benchmark.add_argument("--threads", type=int, default=1)
    human_benchmark.add_argument("--output", type=Path)
    return root


def run(args: argparse.Namespace) -> dict:
    if args.command == "acquire-open-images":
        from focalnet.openimages import AcquireConfig, acquire_open_images

        config = AcquireConfig(
            count=args.count,
            max_side=args.max_side,
            workers=args.workers,
            storage_limit_gib=args.storage_limit_gib,
            min_free_gib=args.min_free_gib,
            seed=args.seed,
            jpeg_quality=args.jpeg_quality,
            mix=args.mix,
        )
        return acquire_open_images(
            args.output, config, args.exclude_manifest, plan_only=args.plan_only
        )
    if args.command == "label":
        from focalnet.data import label_images
        from focalnet.teacher import TeacherConfig

        return label_images(
            args.images,
            args.output,
            args.saliency_model,
            args.face_model,
            TeacherConfig(face_weight=args.face_weight, score_threshold=args.face_threshold),
            threads=args.threads,
            workers=args.workers,
            provider=args.provider,
            resume=args.resume,
        )
    if args.command == "split":
        from focalnet.data import split_manifest

        return split_manifest(args.manifest, args.output, args.validation_fraction, args.seed)
    if args.command == "train":
        from focalnet.model import ModelConfig
        from focalnet.training import TrainConfig, train

        config = TrainConfig(
            **{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__}
        )
        return train(
            args.train,
            args.val,
            args.output,
            ModelConfig(args.backbone, args.decoder_channels),
            config,
            args.resume,
            args.initialize,
        )
    if args.command == "train-human":
        from focalnet.human_training import HumanTrainConfig, train_human_ranker
        from focalnet.ranking import RankConfig

        config = HumanTrainConfig(
            **{name: getattr(args, name) for name in HumanTrainConfig.__dataclass_fields__}
        )
        return train_human_ranker(
            args.dataset,
            None if args.initialize else args.base_checkpoint,
            args.output,
            config,
            RankConfig(args.rank_feature_channels, args.rank_hidden_channels),
            args.resume,
            args.initialize,
        )
    if args.command == "train-cpc":
        from focalnet.cpc import train_cpc_ranker
        from focalnet.human_training import HumanTrainConfig
        from focalnet.ranking import RankConfig

        config = HumanTrainConfig(
            **{name: getattr(args, name) for name in HumanTrainConfig.__dataclass_fields__}
        )
        return train_cpc_ranker(
            args.dataset,
            args.base_checkpoint,
            args.output,
            config,
            RankConfig(args.rank_feature_channels, args.rank_hidden_channels),
            args.resume,
            validation_fraction=args.validation_fraction,
        )
    if args.command == "export":
        from focalnet.exporting import export_model

        return export_model(args.checkpoint, args.output)
    if args.command == "export-human":
        from focalnet.human_exporting import export_human_model

        return export_human_model(args.checkpoint, args.output)
    if args.command == "quantize":
        from focalnet.exporting import quantize_model

        return quantize_model(
            args.model, args.manifest, args.output, samples=args.samples, seed=args.seed
        )
    if args.command == "predict":
        import numpy as np

        from focalnet.runtime import Predictor

        result, heatmap = Predictor(args.model, args.threads).predict(args.image, args.aspect_ratio)
        if args.heatmap:
            args.heatmap.parent.mkdir(parents=True, exist_ok=True)
            with args.heatmap.open("xb") as handle:
                np.save(handle, heatmap, allow_pickle=False)
        return result
    if args.command == "predict-human":
        import numpy as np

        from focalnet.human_runtime import HumanCropPredictor

        result, heatmap = HumanCropPredictor(args.model, args.threads).predict(
            args.image, args.aspect_ratio, retention_tolerance=args.retention_tolerance
        )
        if args.heatmap:
            args.heatmap.parent.mkdir(parents=True, exist_ok=True)
            with args.heatmap.open("xb") as handle:
                np.save(handle, heatmap, allow_pickle=False)
        return result
    if args.command == "evaluate-human":
        from focalnet.human_training import GAICDataset, choose_device, evaluate_human_model
        from focalnet.ranking import load_human_checkpoint

        device = choose_device(args.device)
        model, _ = load_human_checkpoint(args.checkpoint)
        model.to(device)
        result = evaluate_human_model(
            model,
            GAICDataset(args.dataset, args.split),
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                handle.write(json.dumps(result, indent=2) + "\n")
        return result
    if args.command == "evaluate":
        from focalnet.evaluation import evaluate

        result = evaluate(args.model, args.manifest, args.ratios, args.threads)
    elif args.command == "benchmark-human":
        from focalnet.evaluation import benchmark_human

        result = benchmark_human(
            args.model,
            args.images,
            aspect_ratio=args.aspect_ratio,
            iterations=args.iterations,
            warmup=args.warmup,
            threads=args.threads,
        )
    else:
        from focalnet.evaluation import benchmark

        result = benchmark(
            args.model,
            args.images,
            iterations=args.iterations,
            warmup=args.warmup,
            threads=args.threads,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "per_image"}


def main() -> None:
    args = parser().parse_args()
    try:
        result = run(args)
    except KeyboardInterrupt:
        print("focalnet: interrupted; completed records remain resumable", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"focalnet: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ModuleNotFoundError as exc:
        print(
            f"focalnet: {exc}. Install training tools with: uv sync --extra train", file=sys.stderr
        )
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
