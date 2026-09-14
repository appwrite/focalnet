# RepViT-M0.9 500k training result

Completed September 13, 2026. The model was trained from a pretrained
RepViT-M0.9 encoder with a 48-channel FPN decoder. The dataset contains 500,000
Open Images V7 images labeled by the U²-Net + YuNet teacher: 490,000 for training
and a fixed, group-disjoint 10,000-image validation set. Training ran for all 10
planned epochs on a Modal L40S. Epoch 7 produced the best validation loss and is
the exported checkpoint.

| Property | Result |
| --- | ---: |
| Parameters | 4,750,625 |
| FP32 ONNX size | 20,184,361 bytes (19.25 MiB) |
| Best train loss | 0.32967 at epoch 7 |
| Best validation loss | 0.34754 at epoch 7 |
| Modal metered cost through final verification | $31.04 |

## Held-out quality

The production FP32 ONNX model was evaluated on all 10,000 validation images.
The reference is the teacher importance map, rather than a human crop judgment.

| Metric | Result |
| --- | ---: |
| Map MAE | 0.09975 |
| Normalized centroid error | 0.06430 |
| 1:1 teacher importance retained | 91.09% |
| 1:1 oracle / centered crop | 92.79% / 86.73% |
| 16:9 teacher importance retained | 90.86% |
| 16:9 oracle / centered crop | 92.33% / 85.36% |
| 4:5 teacher importance retained | 85.56% |
| 4:5 oracle / centered crop | 87.91% / 79.87% |

On the exact first 1,000 validation records used to evaluate the earlier 100k
model, expanding the training set reduced map MAE by 7.36% and centroid error by
3.97%. Crop retention was effectively flat for 1:1 and increased by 0.19 and
0.09 percentage points for 16:9 and 4:5.

These results show close imitation of the U²-Net + YuNet teacher for the tested
crop ratios. They do not show equivalence to U²-Net segmentation, because
FocalNet predicts a task-specific importance map. They also do not measure human
composition preferences. A curated human-rated cropping set is the next useful
quality gate.

## Mac mini latency

Measured on the spare arm64 Mac mini with ONNX Runtime 1.29.0, batch size one,
50 iterations after five warmups. Full pipeline time includes reading and
decoding the image, preprocessing, inference, focal-point calculation, and a
square crop. It excludes model startup and service overhead.

| CPU threads | Forward median / p95 | Full pipeline median / p95 |
| ---: | ---: | ---: |
| 1 | 33.44 / 33.67 ms | 40.84 / 42.85 ms |
| 4 | 11.35 / 11.39 ms | 17.58 / 19.94 ms |

## Artifact decision

The FP32 ONNX model passed ONNX validation, embedded/sidecar metadata equality,
checkpoint hash verification, and a real-image inference smoke test. RepViT
branch fusion exceeded the strict pointwise parity tolerance for this trained
checkpoint, so the verified export preserves the unfused graph. The runtime
contract uses ONNX Runtime's CPU provider with graph optimizations disabled; its
details and measured export error are embedded in the artifact.

Post-training INT8 was rejected. Static S8/S8 quantization reduced 16:9 crop
retention to 84.90%, below the 85.36% centered baseline. U8/S8 and dynamic INT8
experiments also lost too much quality. The rejected files are kept only under
the run's `experiments/` directory and must not be deployed.

Production artifact SHA-256:

```text
focalnet.onnx  87becceb269a2973c359df789783be49a9f840f47170a015d3776d7c4145a2ce
best.pt        d1942f0652f8ea85f75ffc0cb1bf40d7e70b38e7ad102ab0e6df5c2e07ce52cf
```

The complete evaluation report, training history, metadata sidecar, benchmarks,
and checkpoint live beside the local artifact in `downloads/modal-run-500k/` and
in the Modal Volume at `runs/repvit-m0-9-500k-v2/`.
