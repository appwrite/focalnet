# Reference importance-model results

The reference model uses a pretrained RepViT-M0.9 encoder and a 48-channel
feature-pyramid decoder. It was trained for 10 epochs on 500,000 Open Images V7
samples labeled by the U²-Net + YuNet teacher. The fixed, group-disjoint split
contains 490,000 training images and 10,000 validation images. Epoch 7 produced
the lowest validation loss and was selected for export.

| Property | Result |
| --- | ---: |
| Parameters | 4,750,625 |
| FP32 ONNX size | 20,184,361 bytes (19.25 MiB) |
| Best training loss | 0.32967 |
| Best validation loss | 0.34754 |
| Training accelerator | NVIDIA L40S |

## Teacher agreement

The FP32 ONNX checkpoint was evaluated on the complete 10,000-image validation
split. These metrics use the teacher importance map as the reference; they do
not measure human crop preference or segmentation quality.

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

On the same 1,000 validation records used for the earlier 100,000-image
experiment, expanding the training set reduced map MAE by 7.36% and centroid
error by 3.97%. Crop retention was effectively unchanged at 1:1 and increased
by 0.19 and 0.09 percentage points at 16:9 and 4:5.

The experiment demonstrates close imitation of the teacher for the measured
crop ratios. FocalNet predicts a task-specific importance map, so these results
do not establish equivalence to U²-Net segmentation. Human composition quality
is evaluated separately in [the crop-ranking report](results-cpc-gaic-v2.md).

## CPU latency

Sequential batch-one latency was measured on an Apple Silicon arm64 host with
ONNX Runtime 1.29.0. Each result covers 50 iterations after five warmups. The
pipeline measurement includes image read and decode, preprocessing, inference,
focal-point calculation, and square-crop selection. Model startup and service
overhead are excluded.

| CPU threads | Forward median / p95 | Full pipeline median / p95 |
| ---: | ---: | ---: |
| 1 | 33.44 / 33.67 ms | 40.84 / 42.85 ms |
| 4 | 11.35 / 11.39 ms | 17.58 / 19.94 ms |

These measurements characterize one hardware and runtime configuration. Measure
the complete service on its deployment target before setting latency or
throughput expectations.

## Export and quantization

The FP32 artifact passed ONNX validation, embedded/sidecar metadata comparison,
checkpoint-hash verification, and real-image inference. RepViT branch fusion
exceeded the strict pointwise parity tolerance for this checkpoint, so the
verified export retains the unfused graph. The ONNX metadata records the runtime
contract and measured export error.

Post-training INT8 did not pass the crop-quality gate. Static S8/S8 quantization
reduced 16:9 crop retention to 84.90%, below the 85.36% centered baseline. U8/S8
and dynamic INT8 variants also lost too much quality. A reduced-precision release
would require quantization-aware training or another compact backbone.

Reference hashes:

```text
focalnet.onnx  87becceb269a2973c359df789783be49a9f840f47170a015d3776d7c4145a2ce
best.pt        d1942f0652f8ea85f75ffc0cb1bf40d7e70b38e7ad102ab0e6df5c2e07ce52cf
```

The files identified by these hashes are not distributed in this repository.
