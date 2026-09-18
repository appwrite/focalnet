# FocalNet

FocalNet is a compact vision model for content-aware image cropping. It predicts
a 64×64 importance map that identifies content worth preserving, then ranks crop
candidates using a small composition head trained on human preferences.

The project distills an [Autogravity](https://github.com/appwrite/autogravity)
teacher built from U²-Net saliency and YuNet face detection into a single ONNX
model. The reference RepViT-M0.9 artifact has 4.76 million parameters and is
19.45 MiB in FP32.

> [!IMPORTANT]
> Pretrained ONNX and PyTorch checkpoints are on
> [Hugging Face](https://huggingface.co/appwrite/focalnet) and the
> [2026-09-14-rc1 GitHub release](https://github.com/appwrite/focalnet/releases/tag/2026-09-14-rc1).
> The CPC and GAICD archives reviewed during development do not state explicit
> image or annotation licenses; the published weights do not grant rights to
> those datasets.

## How it works

```text
oriented image → fit into 256×256 RGB
                         ↓
                 compact image encoder
                  strides 8, 16, 32
                         ↓
              lightweight feature pyramid
                         ↓
                64×64 importance map
                         ↓
          candidate pooling + crop geometry
                         ↓
              relative crop preference
```

The model supports two inference modes:

- **Importance inference** returns a heatmap, normalized focal point, and the
  largest crop that retains the most predicted importance.
- **Human-ranked inference** generates crops at several positions and zoom
  levels, then chooses the strongest composition score within a configurable
  importance-retention margin.

| Backbone | Importance model | Human-ranking model |
| --- | ---: | ---: |
| `repvit_m0_9` | 4,750,625 parameters | 4,757,694 parameters |
| `mobilenetv4_conv_small` | 1,316,001 parameters | Supported for experimentation |

The human-ranking head adds 7,069 parameters. It keeps the importance network
frozen, so preference training does not change the heatmap output.

## Results

The reference importance model was trained on 500,000 teacher-labeled Open
Images samples. Its group-disjoint 10,000-image validation split measures
agreement with the U²-Net + YuNet teacher:

| Metric | Result |
| --- | ---: |
| Map MAE | 0.09975 |
| Normalized centroid error | 0.06430 |
| 1:1 teacher importance retained | 91.09% |
| 16:9 teacher importance retained | 90.86% |
| 4:5 teacher importance retained | 85.56% |

The composition head was pretrained on CPC and fine-tuned on GAICD. On GAICD's
500-image test split it improved every recorded metric over training on GAICD
alone:

| Metric | GAICD only | CPC → GAICD |
| --- | ---: | ---: |
| Spearman correlation | 0.75696 | **0.76450** |
| Pearson correlation | 0.78211 | **0.79192** |
| Top-5 accuracy | 42.48% | **43.70%** |
| Top-10 accuracy | 64.45% | **64.54%** |
| Pairwise accuracy | 86.00% | **86.44%** |

See [importance-model results](docs/results-500k.md) and
[human-ranking results](docs/results-cpc-gaic-v2.md) for the evaluation protocol,
latency, artifact hashes, and limitations.

## Installation

Python 3.11–3.13 is supported. The project uses
[uv](https://docs.astral.sh/uv/) for reproducible environments.

```sh
uv sync --frozen --extra train
uv run --extra train focalnet --help
```

Training requires PyTorch and timm. CPU inference requires NumPy, Pillow, and
ONNX Runtime, so a runtime-only environment can use `pip install .`.

## Inference

Download the published checkpoints, then predict an importance map and an
aspect-aware crop:

```sh
mkdir -p artifacts
curl -fsSL \
  "https://huggingface.co/appwrite/focalnet/resolve/main/focalnet.onnx" \
  -o artifacts/focalnet.onnx
# same file: https://github.com/appwrite/focalnet/releases/download/2026-09-14-rc1/focalnet.onnx
uv run focalnet predict artifacts/focalnet.onnx photo.jpg \
  --aspect-ratio 16:9 \
  --heatmap artifacts/photo-importance.npy
```

Rank crop candidates with the human-preference model:

```sh
uv run focalnet predict-human artifacts/focalnet-human.onnx photo.jpg \
  --aspect-ratio 16:9
```

`predict-human` generates up to 125 crops across five positions and five zoom
levels. It scores a padded maximum of 128 candidates in one ONNX call and selects
the highest preference score among crops whose retained importance is within
0.05 of the best candidate. This constraint reduces the chance that a preferred
composition discards a strongly weighted subject.

The returned human score is a relative logit. Compare it only among crops for
the same image; it is not a calibrated confidence value.

### Browser demo

A static crop lab under `web/` runs the same format-v2 ONNX in the browser with
ONNX Runtime Web. Photos stay on-device. The 19.45 MiB weights stay gitignored;
`bun run fetch-model` downloads `focalnet-human.onnx` from
[Hugging Face](https://huggingface.co/appwrite/focalnet) and verifies the
published SHA-256. Bun is the package manager, bundler, test runner, and local
static server:

```sh
cd web
bun install
bun run fetch-model
bun test
bun run dev
```

Override the source with `FOCALNET_ONNX` (local file) or `FOCALNET_ONNX_URL`.
`FOCALNET_ONNX_SHA256` defaults to the published ranking-model hash, including
when the variable is set but empty. Source order is existing
`web/public/focalnet-human.onnx`, then `FOCALNET_ONNX`, then the Hub URL.
`FOCALNET_ALLOW_DUMMY=1` writes a tiny placeholder for UI layout only if a
configured source is missing or fails. The demo compares the human-ranked crop
with the importance-retention crop from the same map.

## Training

### 1. Acquire source images

The acquisition command creates a deterministic, category-balanced subset of
the Open Images V7 training split. It preserves source URLs, attribution,
licenses, checksums, labels, and selection settings in JSON manifests.

```sh
uv run focalnet acquire-open-images \
  --output data/open-images \
  --count 100000 \
  --max-side 640 \
  --workers 32 \
  --storage-limit-gib 75 \
  --min-free-gib 35
```

The downloader is resumable. Open Images records may have different source
rights; retain the generated attribution metadata and verify those rights before
redistributing a dataset copy.

### 2. Generate teacher maps

FocalNet reimplements the relevant Autogravity preprocessing and YuNet decoding
without depending on its Go service. Supply compatible U²-Net and YuNet ONNX
files separately:

```sh
uv run focalnet label \
  --images data/open-images/images \
  --saliency-model models/u2net.onnx \
  --face-model models/face_detection_yunet_2023mar.onnx \
  --output data/teacher \
  --workers 8 \
  --threads 1
```

For each image, the teacher:

1. Runs U²-Net and YuNet on their native input sizes.
2. Removes model-input padding from the saliency output.
3. Adds a normalized Gaussian distribution for each reliable face.
4. Weights faces by detection confidence and square-root area.
5. Combines face and saliency distributions into a 64×64 target map.

With the default face weight, faces receive 75% of total importance when both
signals are present. This is a training policy to validate against the intended
image distribution, rather than a universal definition of the best crop.

The command writes float32 maps and a `labels.jsonl` manifest. Repeating it with
`--resume` retains completed records and skips exact decoded duplicates.

### 3. Split and train the importance model

```sh
uv run focalnet split data/teacher/labels.jsonl \
  --output data/splits \
  --validation-fraction 0.1 \
  --seed 42

uv run --extra train focalnet train \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --output runs/repvit \
  --backbone repvit_m0_9 \
  --epochs 20 \
  --batch-size 16
```

The split keeps each group entirely in training or validation. Assign a shared
group to related images, alternate encodings, crops, or frames before splitting;
exact-image deduplication cannot detect every relationship.

Training uses AdamW, a cosine schedule, a soft-target BCE loss, and spatial KL
divergence. The encoder is frozen for the first epoch and then fine-tuned at one
tenth of the decoder learning rate. Runs write `best.pt`, `last.pt`, and
`history.json`. Add `--resume runs/repvit/last.pt` to resume with the original
configuration and manifests.

### 4. Train the crop-ranking head

Pretrain on Comparative Photo Composition, then fine-tune on GAICD:

```sh
uv run --extra train focalnet train-cpc \
  --dataset data/CPCDataset \
  --base-checkpoint runs/repvit/best.pt \
  --output runs/cpc \
  --epochs 20 \
  --batch-size 64

uv run --extra train focalnet train-human \
  --dataset data/GAIC \
  --initialize runs/cpc/best.pt \
  --output runs/cpc-gaic \
  --epochs 30 \
  --batch-size 64
```

The ranking head pools shared image features and predicted importance inside and
outside each candidate. Crop coordinates, area, center, and aspect ratio provide
additional geometry features. Checkpoints are selected by validation Spearman
correlation.

## Export and evaluation

```sh
uv run --extra train focalnet export runs/repvit/best.pt \
  --output artifacts/focalnet.onnx

uv run --extra train focalnet export-human runs/cpc-gaic/best.pt \
  --output artifacts/focalnet-human.onnx

uv run focalnet evaluate artifacts/focalnet.onnx \
  --manifest data/splits/val.jsonl \
  --ratios 1:1 16:9 9:16 \
  --output artifacts/evaluation.json
```

Export validates the ONNX graph, compares ONNX Runtime with PyTorch, and embeds
the preprocessing contract and checkpoint hashes. Evaluation reports teacher
agreement for the importance model; it does not establish human crop quality.

The reference RepViT checkpoint did not pass the post-training INT8 quality
gate. Its FP32 model remains the evaluated artifact. Quantization-aware training
or another compact backbone may provide a better reduced-precision path.

## ONNX contract

| Field | Contract |
| --- | --- |
| Image input | `image`, float32 `[1,3,256,256]`, RGB NCHW |
| Normalization | `(pixel/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]` |
| Resize | Apply EXIF orientation, then aspect-fit with bilinear interpolation |
| Alpha and padding | Composite alpha on black; normalized padding values are zero |
| Importance output | `importance`, float32 `[1,1,64,64]`, sigmoid applied |
| Ranking inputs | `boxes` `[1,128,4]` and letterbox `content` `[1,4]` |
| Ranking output | `crop_scores`, float32 `[1,128]` |

[imaging.py](src/focalnet/imaging.py) is the preprocessing reference, including
masked bilinear unpadding for odd dimensions and narrow images. Validate ports
to other languages against this implementation.

## Development

```sh
uv run --frozen --extra train ruff check .
uv run --frozen --extra train ruff format --check .
uv run --frozen --extra train pytest -q
(cd web && bun install --frozen-lockfile && bun test)
```

Tests cover coordinate handling, EXIF orientation, alpha composition, padding,
teacher fusion, YuNet decoding and NMS, group-disjoint splits, training resume,
crop optimization, and ONNX parity. They use synthetic inputs and do not require
teacher models or pretrained weights.

`modal_app.py` and the scripts in `scripts/` provide reference workflows for
remote data preparation, GPU training, and Hugging Face publishing. Review
their resource names, storage paths, and expected dataset hashes before running
them in another environment.

## Limitations

- Teacher-agreement metrics inherit the teacher's errors and biases.
- Public-dataset crop scores may not match preferences for a specific product or
  image distribution.
- The GAICD test split was inspected during several development iterations, so
  it should be treated as an engineering benchmark rather than an untouched
  final test.
- CPC and GAICD archives do not state explicit licenses; published checkpoints
  do not grant rights to those datasets.
- Model latency depends on the ONNX Runtime build, processor, thread count, and
  surrounding service workload.

## License

FocalNet's source code and the published checkpoints on
[Hugging Face](https://huggingface.co/appwrite/focalnet) and the
[GitHub release](https://github.com/appwrite/focalnet/releases/tag/2026-09-14-rc1)
are available under the [MIT License](LICENSE). This license does not grant
rights to third-party datasets or teacher model weights; review their
respective terms before redistribution.
