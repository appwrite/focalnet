# Human crop-ranking result: CPC → GAICD v2

Completed September 13, 2026. This run first trains the 7,069-parameter crop
ranking head on the official
[Comparative Photo Composition (CPC) dataset](https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html),
then fine-tunes it on GAICD human opinion scores. The 4.75M-parameter importance
network stays frozen throughout both stages.

| Property | Result |
| --- | ---: |
| Total parameters | 4,757,694 |
| Ranking parameters | 7,069 |
| FP32 ONNX size | 20,398,992 bytes (19.45 MiB) |
| CPC train / validation images | 9,717 / 1,080 |
| CPC rated crops | 259,128 |
| GAICD train / validation / test images | 2,636 / 200 / 500 |
| Training hardware | Modal L4 |
| CPC pretraining | 20 epochs, best epoch 19 |
| GAICD fine-tuning | 30 epochs, best epoch 28 |

The reviewed CPC archive contains 10,797 annotated source images with 24 crop
views per image. Every score is the mean of six comparative human judgments. A
seed-42 source-image split keeps all views of an image together. Exact file-hash
comparison found no CPC image overlap with the GAICD validation or test images.

## CPC pretraining

The CPC checkpoint is selected by mean per-image Spearman correlation on the
1,080-image CPC validation split. Higher is better for all metrics.

| Metric | Frozen importance baseline | CPC ranker | Absolute gain |
| --- | ---: | ---: | ---: |
| SRCC | 0.50287 | **0.59176** | +0.08888 |
| PCC | 0.52397 | **0.67158** | +0.14761 |
| Acc5 | 72.20% | **75.78%** | +3.58 pp |
| Acc10 | 87.18% | **90.78%** | +3.61 pp |
| Pairwise accuracy | 74.23% | **78.56%** | +4.33 pp |

After GAICD fine-tuning, CPC validation SRCC is 0.56157 and pairwise accuracy is
76.97%. Some CPC-specific ranking ability is forgotten, but both measurements
remain above the frozen importance baseline.

## GAICD result

The CPC initializer is fine-tuned at `3e-4`. The checkpoint is selected only by
GAICD validation SRCC, where it reaches 0.78059 versus 0.77334 for GAICD v1.
The test split is excluded from gradient updates and checkpoint selection.

| GAICD test metric | GAICD v1 | CPC → GAICD v2 | Absolute gain |
| --- | ---: | ---: | ---: |
| SRCC | 0.75696 | **0.76450** | +0.00754 |
| PCC | 0.78211 | **0.79192** | +0.00981 |
| Acc5 | 42.48% | **43.70%** | +1.22 pp |
| Acc10 | 64.45% | **64.54%** | +0.09 pp |
| Pairwise accuracy | 86.00% | **86.44%** | +0.44 pp |

Pairwise accuracy covers 1,405,340 crop pairs whose human scores differ by at
least 0.25. The improvement is consistent across every recorded GAICD test
metric, so this checkpoint replaces GAICD v1 as the recommended human-ranking
artifact.

GAICD test results have now been inspected for GAICD v1 and two CPC-transfer
runs. They remain useful engineering benchmarks, but are no longer a never-seen
research test. A blind, Appwrite-specific human preference set is the appropriate
next quality gate.

## Mac mini latency

Measured on the spare arm64 Mac mini with ONNX Runtime 1.29.0, batch size one,
50 iterations after five warmups. The full pipeline includes file read, decode,
preprocessing, candidate generation, inference, the importance-retention gate,
and crop selection.

| CPU threads | Forward median / p95 | Full pipeline median / p95 |
| ---: | ---: | ---: |
| 1 | 33.86 / 33.99 ms | 44.09 / 46.23 ms |
| 4 | 11.69 / 11.72 ms | 20.96 / 22.95 ms |

The graph and parameter count are unchanged from GAICD v1. The 141-byte ONNX
size difference is metadata. Measured latency is effectively unchanged.

## Verification and artifacts

The artifact passed ONNX validation, embedded/sidecar metadata equality,
checkpoint-chain hash verification, real-image inference at 1:1, 16:9, and 4:5,
and the lightweight-runtime import check. The importance output is bit-identical
to GAICD v1 on the deterministic verification input; only crop scores changed.
All 53 repository tests pass.

```text
focalnet-human.onnx  59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439
GAICD best.pt         82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a
CPC best.pt           16cf17814cbeec88180d158fd6ba018355e6ca80aecc7f861d7325e295b9c91b
CPCDataset.tar.gz     dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281
GAIC.zip              b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b
```

The final local artifacts are in `downloads/modal-run-cpc-gaic-lr3e4/`; the CPC
initializer is in `downloads/modal-run-cpc-pretrain/`. Durable Modal copies are
in `runs/repvit-m0-9-500k-cpc-gaic-lr3e4-candidate/` and
`runs/repvit-m0-9-500k-cpc-v1/` on the `focalnet-data` Volume. The complete run
can be recreated with `scripts/run-cpc-gaic-training-macos.sh`.

Neither reviewed dataset archive states an explicit license for its images or
annotations. Keep this model internal until CPC and GAICD training and model
redistribution rights are confirmed.
