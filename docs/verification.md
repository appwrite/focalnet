# Initial implementation verification

Executed on September 9, 2026, on an Apple M3 Pro with macOS 26.5.2, Python
3.12.11, PyTorch 2.14.0, timm 1.0.29, and ONNX Runtime 1.29.0.

All 30 automated tests passed, including ONNX parity for both encoders and an
interrupted/resumed CPU training run with exactly the same final weights and
history as an uninterrupted run. Lint and formatting checks passed. Label
generation resumed with zero duplicate records, and evaluation rejected use of
INT8 calibration groups. The inference import path loads neither PyTorch nor timm.

## Real teacher and training smoke run

The teacher was checked against Autogravity's local `feat/face-priority` checkout
at commit `69f0c63bbff768a9a3ca50436a2101c12e2810d4`. The public source was also
inspected at [`f1a36f0`](https://github.com/appwrite/autogravity/tree/f1a36f0fb23eb1ed401216d44987e6847106e84c).
Model files were read from that checkout; they are not bundled here.

- Full U²-Net FP32 SHA-256:
  `8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`.
- YuNet SHA-256:
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- 21 input fixtures produced 20 unique decoded-image labels. The identical
  PNG/lossless WebP image was deduplicated.
- The six generated single-face fixtures each produced one face. The generated
  two-person fixture produced two. The generated landscape, back-facing person,
  and blurred-face controls produced no detections. The person-in-room fixture
  produced one face; the puppies fixture produced none.
- Related rose encodings were manually assigned one group before splitting.
  The smoke split contained 16 training and four validation images.
- RepViT-M0.9 used its pretrained encoder, batch size four, two CPU threads,
  and two epochs. Training loss fell from 1.1269 to 1.0749; validation loss fell
  from 1.1115 to 1.1074. Checkpoint loading and the completed-run resume path worked.

This is a mechanics test. Four validation fixtures and two epochs cannot establish
generalization. The FP32 model's square crops retained 0.740 mean teacher
importance, versus 0.863 for centered crops. Its mean map MAE was 0.479. **The
smoke model is not a useful trained focal-point detector.**

## Export and local CPU benchmark

RepViT export's maximum absolute difference from PyTorch was `7.15e-7` on the
export verification inputs. The exported checkpoint was calibrated using the
16 training images to exercise the INT8 path. Both artifacts loaded and predicted
successfully, and evaluation ran on the disjoint four-image validation split.

Sequential batch-one measurements used the CPU provider, one intra-op thread,
five warmup iterations, and 30 timed iterations:

| Artifact | File size | Forward median / p95 | Pipeline median / p95 |
| --- | ---: | ---: | ---: |
| RepViT FP32 | 19,821,220 bytes | 28.59 / 31.40 ms | 33.87 / 37.64 ms |
| RepViT INT8 | 6,214,087 bytes | 14.07 / 14.35 ms | 17.97 / 19.20 ms |

Forward timing reused the first fixture's preprocessed tensor. Pipeline timing
cycled the fixtures and included file reads, decoding, preprocessing, inference,
map restoration, focal point, and a square crop. Model startup and HTTP overhead
were excluded. These are measurements of the initial smoke artifacts, not
production throughput or x86 measurements. Later metadata additions can slightly
change artifact sizes without changing the network.

Quantization changed some selected crop locations substantially on this poorly
trained, nearly flat model. A successful quantized inference and similar map MAE
do not establish crop stability. Evaluate both artifacts after representative
training, especially scenes with competing subjects.

The ignored local artifacts are in `data/smoke-teacher`, `data/smoke-splits`,
`runs/smoke-repvit`, and `artifacts/smoke-*`. They are intentionally excluded from
the source tree. Fresh users should follow the README with their own image paths
and teacher model files.
