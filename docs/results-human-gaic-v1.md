# Human crop-ranking result: GAICD v1

This artifact has been superseded by the
[CPC → GAICD v2 result](results-cpc-gaic-v2.md), which improves every recorded
GAICD test metric with the same runtime architecture.

Completed September 13, 2026. This run adds a 7,069-parameter composition head
to the frozen RepViT-M0.9 500k importance model. It learns from dense human mean
opinion scores in the journal version of
[GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch), then
ranks multiple crop positions and zoom levels in a single forward pass.

| Property | Result |
| --- | ---: |
| Total parameters | 4,757,694 |
| New ranking parameters | 7,069 |
| FP32 ONNX size | 20,398,851 bytes (19.45 MiB) |
| Training | 30 epochs, Modal L4 |
| Train / validation / test images | 2,636 / 200 / 500 |
| Usable rated crops across all splits | 288,042 |
| Best validation SRCC | 0.77334 at epoch 30 |
| Modal metered cost through final verification | $31.29 |
| Modal billed cost after credits and free storage | $0.89 |

The base model remains unchanged: its ONNX importance output differs from the
format-v1 production artifact by at most `1.2e-6` on the real-image verification
input. Only the crop-ranking path was trained.

## Untouched human-rated test set

The final checkpoint was selected on the 200-image validation split and evaluated
once on GAICD's separate 500-image test split. The baseline ranks the same rated
crops only by the fraction of FocalNet importance inside each crop. Higher is
better for every metric.

| Metric | Importance-only baseline | Human ranker | Absolute gain |
| --- | ---: | ---: | ---: |
| Spearman rank correlation (SRCC) | 0.46148 | **0.75696** | +0.29547 |
| Pearson correlation (PCC) | 0.47670 | **0.78211** | +0.30541 |
| Acc5 | 34.65% | **42.48%** | +7.83 pp |
| Acc10 | 53.59% | **64.45%** | +10.86 pp |
| Pairwise accuracy | 71.50% | **86.00%** | +14.50 pp |

Pairwise accuracy covers 1,405,340 crop pairs whose human scores differ by at
least 0.25. SRCC improves by 64.0% relative to the importance-only baseline. The
result shows that the added head learns human composition preferences beyond
subject retention on this dataset.

These are public-dataset ranking results. They do not establish preference among
Appwrite users or guarantee that the runtime-selected crop wins a blind A/B test.
The runtime candidate generator is close to GAICD's grid-anchor task, but its
chosen crops do not have ground-truth ratings. An Appwrite-specific evaluation
set remains the next product quality gate.

## Inference behavior

For a requested aspect ratio, the runtime generates up to 125 exact-ratio
candidates from five positions on each available axis and scales of 100%, 90%,
80%, 70%, and 65%. The model scores a padded maximum of 128 candidates. It chooses
the highest preference score among crops whose retained importance is within
0.05 of the best candidate. This keeps the learned composition choice from
cropping away a strongly weighted subject.

The crop score is a relative logit. It is meaningful only when comparing crops
for the same image and is not a calibrated probability or confidence score.

## Mac mini latency

Measured on the spare arm64 Mac mini with ONNX Runtime 1.29.0, batch size one,
50 iterations after five warmups. The full pipeline includes reading and decoding
an image, preprocessing, candidate generation, one ONNX call, the retention safety
gate, and square-crop selection. Model startup and service overhead are excluded.

| CPU threads | Forward median / p95 | Full pipeline median / p95 |
| ---: | ---: | ---: |
| 1 | 33.90 / 33.98 ms | 42.77 / 44.45 ms |
| 4 | 11.68 / 11.74 ms | 21.08 / 22.97 ms |

Against the format-v1 model on the same machine, the four-thread forward median
increases by 0.33 ms and the complete pipeline median by 3.50 ms. The ONNX file
is 214,490 bytes larger.

## Verification and artifacts

The artifact passed ONNX validation, embedded/sidecar metadata equality,
checkpoint hash verification, PyTorch/ONNX Runtime export comparison, real-image
inference at 1:1, 16:9, and 4:5, and a lightweight-runtime check proving that the
inference import loads neither PyTorch nor timm. All 52 repository tests pass.

```text
focalnet-human.onnx  6c74f1a22368eb3ceca48a20fcee5c2b12e5d37b5c95f1fa2584497205c80449
best.pt               2665e5e2abeff6e8bcfeaf8696cd11c6950dd3ecf65ef9c883983025f58e3a4d
GAIC.zip              b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b
```

The local artifacts are in `downloads/modal-run-human-gaic-v1/`. The durable
Modal copy is in `runs/repvit-m0-9-500k-gaic-v1/` on the `focalnet-data` Volume.
The dataset archive is in `datasets/gaic-v2/GAIC.zip` on Modal and under
`human-crops/gaicd-v2/` on the Mac mini USB drive.

The official code repository is MIT licensed. The downloaded GAICD archive does
not contain an explicit license for its images or annotations. Keep this model
internal until the dataset's training and redistribution rights are confirmed.
