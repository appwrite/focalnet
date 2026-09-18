# Reference crop-ranking results

The reference crop ranker was pretrained on the official
[Comparative Photo Composition (CPC) dataset](https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html)
and fine-tuned on human opinion scores from
[GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).
Both stages update only the 7,069-parameter ranking head; the RepViT-M0.9
importance model remains frozen.

| Property | Result |
| --- | ---: |
| Total parameters | 4,757,694 |
| Ranking parameters | 7,069 |
| FP32 ONNX size | 20,398,992 bytes (19.45 MiB) |
| CPC training / validation images | 9,717 / 1,080 |
| CPC rated crops | 259,128 |
| GAICD training / validation / test images | 2,636 / 200 / 500 |
| Training accelerator | NVIDIA L4 |
| CPC pretraining | 20 epochs, best epoch 19 |
| GAICD fine-tuning | 30 epochs, best epoch 28 |

The reviewed CPC archive contains 10,797 annotated source images with 24 crop
views per image. Each crop score averages six comparative human judgments. A
seed-42 source-image split keeps all views from one image together. File-hash
comparison found no exact CPC image overlap with the GAICD validation or test
splits.

## CPC pretraining

The CPC checkpoint was selected by mean per-image Spearman correlation on the
1,080-image validation split. Higher is better for every metric.

| Metric | Frozen importance baseline | CPC ranker | Absolute gain |
| --- | ---: | ---: | ---: |
| Spearman correlation | 0.50287 | **0.59176** | +0.08888 |
| Pearson correlation | 0.52397 | **0.67158** | +0.14761 |
| Top-5 accuracy | 72.20% | **75.78%** | +3.58 pp |
| Top-10 accuracy | 87.18% | **90.78%** | +3.61 pp |
| Pairwise accuracy | 74.23% | **78.56%** | +4.33 pp |

After GAICD fine-tuning, CPC validation Spearman correlation is 0.56157 and
pairwise accuracy is 76.97%. The model loses some CPC-specific ranking ability,
but both measurements remain above the frozen importance baseline.

## GAICD evaluation

The CPC initializer was fine-tuned with a `3e-4` learning rate. Checkpoint
selection used only GAICD validation Spearman correlation, which reached 0.78059
compared with 0.77334 for the GAICD-only model. The test split was excluded from
gradient updates and checkpoint selection.

| Test metric | GAICD only | CPC → GAICD | Absolute gain |
| --- | ---: | ---: | ---: |
| Spearman correlation | 0.75696 | **0.76450** | +0.00754 |
| Pearson correlation | 0.78211 | **0.79192** | +0.00981 |
| Top-5 accuracy | 42.48% | **43.70%** | +1.22 pp |
| Top-10 accuracy | 64.45% | **64.54%** | +0.09 pp |
| Pairwise accuracy | 86.00% | **86.44%** | +0.44 pp |

Pairwise accuracy covers 1,405,340 crop pairs whose human scores differ by at
least 0.25. CPC initialization improves every recorded GAICD metric while
keeping the architecture and inference cost unchanged.

The GAICD test split was inspected during multiple development iterations.
These values are useful engineering benchmarks, but they should not be presented
as results from a never-seen final test. A new, application-specific blind human
preference study is the appropriate next quality gate.

## CPU latency

Sequential batch-one latency was measured on an Apple Silicon arm64 host with
ONNX Runtime 1.29.0. Each result covers 50 iterations after five warmups. The
pipeline measurement includes image read and decode, preprocessing, candidate
generation, inference, the importance-retention constraint, and crop selection.
Model startup and service overhead are excluded.

| CPU threads | Forward median / p95 | Full pipeline median / p95 |
| ---: | ---: | ---: |
| 1 | 33.86 / 33.99 ms | 44.09 / 46.23 ms |
| 4 | 11.69 / 11.72 ms | 20.96 / 22.95 ms |

The graph and parameter count match the GAICD-only model. Its 141-byte ONNX size
difference comes from metadata, and measured latency is effectively unchanged.

## Verification

The artifact passed ONNX validation, embedded/sidecar metadata comparison,
checkpoint-chain hash verification, real-image inference at 1:1, 16:9, and 4:5,
and a runtime import check that loads neither PyTorch nor timm. Its importance
output is bit-identical to the GAICD-only artifact on the deterministic
verification input; only crop scores changed.

Reference hashes:

```text
focalnet-human.onnx  59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439
GAICD best.pt         82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a
CPC best.pt           16cf17814cbeec88180d158fd6ba018355e6ca80aecc7f861d7325e295b9c91b
CPCDataset.tar.gz     dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281
GAIC.zip              b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b
```

The ranking checkpoint identified by these hashes is published at
[huggingface.co/appwrite/focalnet](https://huggingface.co/appwrite/focalnet)
and the
[2026-09-14-rc1 GitHub release](https://github.com/appwrite/focalnet/releases/tag/2026-09-14-rc1)
as `focalnet-human.onnx` and `focalnet-human.pt`. Dataset archives stay off the
Hub and the release. Neither reviewed CPC nor GAICD archive states an explicit
license for its images or annotations; the published weights do not grant rights
to those datasets.
