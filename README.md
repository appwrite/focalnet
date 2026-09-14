# FocalNet

Learn **what a crop should keep** with one small vision network. FocalNet distills
U²-Net saliency and YuNet face detections into a 64×64 importance map. The map
supports both a normalized focal point and crops that retain multiple subjects.

This repository contains the training and inference toolkit. A 4.75M-parameter
RepViT-M0.9 model has now been trained on 500,000 teacher-labeled Open Images and
evaluated on a fixed 10,000-image validation set. A 7,069-parameter crop-ranking
head has also been pretrained on CPC comparative human judgments and fine-tuned
on GAICD human opinion scores. See the
[CPC → GAICD crop-ranking results](docs/results-cpc-gaic-v2.md) and
[500k training results](docs/results-500k.md) for quality, size, latency,
limitations, and artifact hashes. The initial mechanics test remains documented
in [verification results](docs/verification.md).

```text
oriented image → aspect-fit into 256×256 RGB
                            ↓
                  RepViT / MobileNetV4
                     strides 8, 16, 32
                            ↓
               1×1 projections to 48 channels
                 + bilinear upsampling + sum
                            ↓
                 64×64 depthwise 3×3 + 1×1
                            ↓
                   sigmoid importance map
                            ↓
              remove padding → focal point / crop
```

| Backbone | Total trainable parameters | Status |
| --- | ---: | --- |
| `repvit_m0_9` | 4,750,625 | Default; pretrained encoder, verified export |
| `mobilenetv4_conv_small` | 1,316,001 | Smaller alternative; verified forward, backward, and export |

The decoder has 48 channels by default. These counts include the encoder and
decoder, and exclude the original classification head. RepViT-M0.6 is available
in the [authors' repository](https://github.com/THU-MIG/RepViT), but is not registered
in the pinned timm release. FocalNet uses timm's maintained
[feature extraction interface](https://huggingface.co/docs/timm/feature_extraction).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for development. Python
3.11–3.13 is supported by the package constraints. The lockfile records the
resolved dependencies.

```sh
uv sync --frozen --extra train
uv run --extra train focalnet --help
```

Training uses PyTorch and timm. Inference and teacher labeling need only NumPy,
Pillow, and ONNX Runtime; an inference environment can use `pip install .`.
The training command downloads ImageNet encoder weights on its first run.
`--no-pretrained --freeze-encoder-epochs 0` supports offline mechanics tests.

## 1. Acquire training images

The acquisition command builds a deterministic, balanced subset of the Open
Images V7 training split. The default 100,000-image mix is 30% people, 15% each
animals, food, and vehicles, 10% scenes, 10% busy multi-object images, and 5%
general images. It streams the official annotation indexes, keeps image-level
license and attribution metadata, downloads the selected originals, applies the
dataset's declared rotation, and stores resized JPEGs with a 640-pixel maximum
side.

```sh
uv run focalnet acquire-open-images \
  --output downloads/open-images-v7-100k \
  --count 100000 --max-side 640 --workers 32 \
  --storage-limit-gib 75 --min-free-gib 35
```

The output includes `images.jsonl`, whose records contain the local image path,
bucket, Open Images labels, original source and landing URLs, license, creator,
title, rotation, checksum, and dimensions. `provenance.json` fixes the selection
seed, source indexes, quotas, and encoding settings; `acquisition-report.json`
records the final counts and disk use. Open Images lists these images under CC BY
2.0; preserve `images.jsonl` with any dataset copy and verify the individual
source rights before redistribution.

The downloader is resumable. Repeating the command retains successful files,
skips their network requests, and fills category quotas from surplus candidates.
Worker count and the two disk guards can change between attempts. Selection and
image-encoding settings cannot; use a different output directory to change those.
It writes every success immediately and retries failed candidates. It stops before
the dataset exceeds 75 GiB or available disk falls below 35 GiB.

## 2. Generate teacher labels

FocalNet independently recreates the relevant
[Autogravity](https://github.com/appwrite/autogravity) preprocessing and YuNet
decoding logic in Python. It reuses **model files**, with no dependency on the Go
library or its HTTP service. Supply the full U²-Net FP32 model (preferred for
labeling) or Autogravity's INT8 model, plus its fixed 640×640 YuNet 2023mar model.

```sh
uv run focalnet label \
  --images /path/to/training-images \
  --saliency-model /path/to/autogravity/models/u2net.onnx \
  --face-model /path/to/autogravity/models/face_detection_yunet_2023mar.onnx \
  --output data/teacher --workers 8 --threads 1 --provider coreml
```

The command searches JPEG, PNG, and WebP files recursively. It applies EXIF
orientation, fits U²-Net input to 320×320 with neutral normalized RGB padding,
and fits YuNet input to 640×640 with raw BGR values and black padding. YuNet uses
score threshold 0.85, NMS threshold 0.30, and top-k 5000. Pillow's bilinear and
alpha rounding can differ slightly from Go's imaging library; byte-for-byte
teacher parity is not claimed.

The teacher target intentionally extends Autogravity's face-first policy:

1. Run both models, including on images with faces.
2. Remove padding from the fused U²-Net saliency map.
3. Create a Gaussian for every reliable face, with a minimum width sufficient
   to represent small faces on the 64×64 grid.
4. Distribute face importance using `confidence × sqrt(face area)`.
5. Mix normalized saliency and face distributions, then scale the peak to one.

With the default `--face-weight 3`, faces receive 75% of total importance when
both signals are present. This controls total mass: simply multiplying Gaussian
peaks could still let a large body dominate a small face. No-face images retain
the saliency distribution. This is a starting policy to evaluate, rather than
a guarantee of the best crop for every scene.

Outputs include `labels.jsonl`, float32 `maps/*.npy`, and `provenance.json` with
teacher model hashes and labeling settings. Exact decoded duplicates are skipped.
After an interruption, repeat the same command with `--resume`; successful
records are retained. `--workers` controls simultaneous images and `--threads`
controls ONNX threads per image; tune both on the labeling machine. `--provider
coreml` accelerates the FP32 teacher on supported Macs; the default `cpu` provider
is portable. The provider is recorded because its numerical results can differ
slightly. New runs require a new output directory.

### Train on Modal

After labeling reaches 100,000 images, upload the teacher maps in resumable
1,000-file transactions. Modal then downloads and verifies the public Open
Images files directly from their source:

```sh
uvx modal setup
uvx modal volume create focalnet-data --version=2
uv run --with modal python scripts/upload-modal-data.py
uvx modal run modal_app.py::download_images
```

Start the 20-epoch L40S training job in detached mode. It writes checkpoints,
FP32 ONNX, and calibrated INT8 ONNX artifacts back to the Volume:

```sh
uvx modal run --detach modal_app.py::train_model
uvx modal app logs appwrite-focalnet -f
```

Download the completed run locally:

```sh
uvx modal volume get focalnet-data \
  /runs/repvit-m0-9-v1 downloads/modal-run
```

### Continue labeling on another Mac

Stop the labeler with Control-C, then copy the repository and the three state
directories. Do not copy `.venv`; `uv` recreates the correct environment for the
other Mac's architecture.

```sh
rsync -a --partial --info=progress2 \
  --exclude .venv \
  /path/to/focalnet/ user@spare-mac:/path/to/focalnet/

ssh user@spare-mac \
  'cd /path/to/focalnet && sh scripts/resume-labeling-macos.sh'
```

The helper verifies that the dataset, FP32 teacher, YuNet model, saved label
provenance, and Core ML execution provider are present before resuming. Keep the
relative `downloads/` layout intact. You can repeat `rsync` safely; its second pass
copies only changes. If the destination Mac does not expose Core ML, start a
separate CPU-provider output rather than mixing backend numerics in one manifest.

## 3. Split independent images

```sh
uv run focalnet split data/teacher/labels.jsonl \
  --output data/splits --validation-fraction 0.1 --seed 42
```

The split keeps each `group` entirely in training or validation. Label generation
defaults groups to a hash of decoded pixels. Assign the same group to related
images, alternate encodings, crops, or frames from one sequence **before splitting**;
automatic exact deduplication cannot identify all such relationships. Maintain a
separate final test set that is never used to choose checkpoints or calibrate INT8.

You can also supply external saliency, fixation, or curated importance maps as
JSONL records. Paths are relative to the manifest:

```json
{"image":"images/photo.jpg","importance":"maps/photo.npy","group":"photo-series-01","face_count":2,"weight":1.0}
```

Each importance file must be a finite float32 64×64 NumPy array in `[0,1]`,
covering the **whole oriented source image**, without letterbox padding. Normalize
external maps to a peak of one, leaving empty maps zero. `weight` adjusts sampling
frequency; `face_count` enables the additional face oversampling multiplier.

## 4. Train

```sh
uv run --extra train focalnet train \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --output runs/repvit-v1 \
  --backbone repvit_m0_9 \
  --epochs 20 --batch-size 16
```

The default device is CUDA, then Apple MPS, then CPU, depending on availability.
Override with `--device cpu`, `mps`, or `cuda`. Training uses:

- A pretrained encoder frozen for the first epoch, including its BatchNorm
  statistics; subsequent epochs fine-tune it at one tenth of the decoder's rate.
- AdamW and a cosine learning-rate schedule.
- Soft-target BCE weighted toward important pixels, plus spatial KL divergence.
  Letterbox padding is excluded, including fractional boundary coverage.
- Horizontal flips with aligned target transforms and modest color augmentation.
- Twice the sampling weight for images with reliable faces by default;
  configure `--face-sampling-weight` to suit the dataset.

Training refuses train/validation overlap by group or image path. Runs save
`best.pt`, `last.pt`, and `history.json`. The best checkpoint minimizes validation
loss. Resume an interrupted job by repeating its original arguments and adding
`--resume runs/repvit-v1/last.pt`. Resume requires the same manifests and training
configuration, including the originally planned number of epochs.

For the smaller encoder, use `--backbone mobilenetv4_conv_small` and a different
run directory. Neither model's crop quality can be inferred from parameter count.

## 5. Export and run

```sh
uv run --extra train focalnet export runs/repvit-v1/best.pt \
  --output artifacts/focalnet.onnx

uv run focalnet predict artifacts/focalnet.onnx /path/to/photo.jpg \
  --aspect-ratio 16:9 --heatmap artifacts/photo-importance.npy
```

Export attempts to fuse RepViT's inference branches and checks the fused model
against the original PyTorch model. It falls back to the unfused graph when the
fusion error exceeds the verification bound. The exporter then validates the
ONNX graph and checks ONNX Runtime outputs against PyTorch on zero and random
inputs. Exports use opset 18, fixed batch size one, and one self-contained model
file. Preprocessing and verification metadata are embedded and saved in an
adjacent `.onnx.json` file.

The prediction result contains:

- `gravity.x/y`: the importance-weighted centroid, normalized to the oriented
  image. It can fall between subjects.
- `importance_peak`: the maximum activation, **not a calibrated probability of
  a correct crop**. There is no confidence or subject head in v1.
- `image.width/height`: dimensions after EXIF orientation.
- `crop`: optional integer `left`, `top`, `width`, `height`, and the fraction of
  predicted importance retained by that crop.

The crop solver chooses the largest rectangle fitting the requested aspect ratio,
then places it to retain the most importance. Because such a rectangle spans one
whole image axis, the optimization reduces to a 1D sliding integral. It handles
fractional heatmap cells and tests the necessary integer offsets without scanning
every source-image pixel. Equal scores prefer the image center. The requested
ratio is approximate to one pixel of dimension rounding.

This solver optimizes importance retention. It does not yet optimize composition,
headroom, zoom, or explicit face-box containment. A single focal point cannot
express all the decisions available from a multi-subject map.

## Human-rated crop ranking

The format-v2 model keeps the 500k importance network frozen and learns a small
composition head from the
[official CPC comparative ratings](https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html),
then fine-tunes it on dense mean-opinion crop scores in the
[official GAICD release](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).
It pools visual features inside and outside every candidate crop, combines them
with crop geometry, and returns relative preference scores. Training evaluates
the original importance-retention rank as a baseline and selects checkpoints by
validation Spearman correlation.

```sh
uv run --extra train focalnet train-cpc \
  --dataset /path/to/CPCDataset \
  --base-checkpoint downloads/modal-run-500k/best.pt \
  --output runs/cpc-v1 --epochs 20 --batch-size 64

uv run --extra train focalnet train-human \
  --dataset /path/to/GAIC \
  --initialize runs/cpc-v1/best.pt \
  --output runs/cpc-gaic-v2 --epochs 30 --batch-size 64

uv run --extra train focalnet export-human runs/cpc-gaic-v2/best.pt \
  --output artifacts/focalnet-human.onnx

uv run focalnet predict-human artifacts/focalnet-human.onnx photo.jpg \
  --aspect-ratio 16:9
```

### Browser demo

A static crop lab under `web/` runs the same format-v2 ONNX in the browser with
ONNX Runtime Web. Photos stay on-device. The 19.45 MiB weights are gitignored;
`scripts/fetch-model.sh` downloads them from the published artifact URL (or a
local path / Modal volume), then you can start Vite or publish the build with
[yeet.page](https://yeet.page/):

```sh
cd web
npm install
bash scripts/fetch-model.sh
npm test
npm run dev
npm run build
npx --yes @dittmann/yeet dist
```

The demo compares the human-ranked crop with the importance-retention crop from
the same map. The yeet URL is unlisted, not private: anyone with the link can
download `focalnet-human.onnx`.

At inference, FocalNet generates up to 125 crops over five positions and five
zoom levels, scores all of them in one ONNX call, and chooses the highest human
preference score among candidates within five percentage points of the best
importance retention. This safety gate prevents the composition head from
discarding a subject the importance model strongly prefers.

The human ONNX contract adds normalized `boxes` `[1,128,4]` and letterbox
`content` `[1,4]` inputs plus `crop_scores` `[1,128]`. Its `importance` output is
the same `[1,1,64,64]` map as the format-v1 model. The inference package does not
import PyTorch or timm. Use `benchmark-human` to measure the whole ranking path.

```sh
uv run focalnet benchmark-human artifacts/focalnet-human.onnx \
  --images /path/to/benchmark-images --aspect-ratio 1:1 --threads 4 \
  --output artifacts/benchmark-human.json
```

The trained result and dataset-rights caveat are recorded in
[results-cpc-gaic-v2.md](docs/results-cpc-gaic-v2.md). Neither downloaded data
archive states an image or annotation license. Confirm CPC and GAICD training and
model redistribution rights before distributing the artifact or using it
commercially.

### Go integration contract

The artifact is suitable for a CPU ONNX Runtime session in your Go service:

| Field | Contract |
| --- | --- |
| Input | `image`, float32 `[1,3,256,256]`, RGB NCHW |
| Normalization | `(pixel/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]` |
| Resize | EXIF orient, aspect-fit, bilinear; round positive dimensions half-up |
| Alpha/padding | Composite alpha on black; unused normalized tensor pixels are zero |
| Output | `importance`, float32 `[1,1,64,64]`, sigmoid already applied |
| Coordinates | Output covers the padded square; remove padding before source coordinates |

Keep the fitted rectangle: its continuous output-grid bounds are the input bounds
divided by four. [imaging.py](src/focalnet/imaging.py) implements masked bilinear
unpadding, including odd dimensions and thin images, and serves as the reference.
FocalNet does not replace or modify Autogravity's Go service in this repository;
validate the Go preprocessor against the Python reference when integrating it.

## 6. Evaluate, quantize, and benchmark

```sh
uv run focalnet evaluate artifacts/focalnet.onnx \
  --manifest data/splits/val.jsonl \
  --ratios 1:1 16:9 9:16 --output artifacts/fp32-evaluation.json

uv run --extra train focalnet quantize artifacts/focalnet.onnx \
  --manifest data/splits/train.jsonl \
  --samples 128 --output artifacts/focalnet-int8.onnx

uv run focalnet evaluate artifacts/focalnet-int8.onnx \
  --manifest data/splits/val.jsonl --output artifacts/int8-evaluation.json

uv run focalnet benchmark artifacts/focalnet-int8.onnx \
  --images /path/to/benchmark-images --threads 1 --iterations 50 \
  --output artifacts/benchmark.json
```

Evaluation reports map MAE, normalized centroid error, teacher importance retained
by each crop, a centered-crop baseline, and regret relative to the best crop on
the teacher map. Results measure **teacher agreement**, not human judgment.
Human-curated difficult scenes and cropping preferences are necessary to establish
improvement beyond the teacher's mistakes.

INT8 uses representative training images with static MinMax calibration,
per-channel S8 weights, and S8 activations in QDQ format. This follows ONNX Runtime's
[static CNN quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
Calibration group IDs and source-model hashes are embedded in the model.
Evaluation rejects overlap with calibration groups. Calibration data should
represent the eventual workload; 128 images is a starting sample, not a quality
guarantee. The 500k RepViT model failed this post-training INT8 quality gate, so
its FP32 ONNX file is the production artifact. A smaller release will require
quantization-aware training or a less precision-sensitive model.

Benchmarking measures sequential batch-one CPU inference and the pipeline including
file reads, decoding, preprocessing, unpadding, focal-point calculation, and a
square crop. Startup and HTTP overhead are excluded. Compare FP32 and INT8 on the
actual x86/ARM target, then measure full-service throughput under its CPU limits.

## Development

```sh
uv run --extra train ruff check .
uv run --extra train ruff format --check .
uv run --extra train pytest -q
```

Tests cover coordinate handling, EXIF, alpha, padding exclusion, face-mass fusion,
YuNet decoding/NMS, group splits, loss gradients, crop optimization against brute
force, and ONNX parity for both backbones. Tests create synthetic data and random
encoders, so they require no teacher weights or pretrained-weight downloads.
