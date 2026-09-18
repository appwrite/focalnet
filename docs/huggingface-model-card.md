---
license: mit
library_name: onnxruntime
pipeline_tag: image-to-image
tags:
  - onnx
  - computer-vision
  - image-cropping
  - saliency
base_model: timm/repvit_m0_9
---

# FocalNet

Appwrite FocalNet is a compact vision model for content-aware image cropping. It
predicts a 64×64 importance map, then ranks crop candidates with a small
composition head trained on human preferences. Training code lives in
[appwrite/focalnet](https://github.com/appwrite/focalnet).

The reference RepViT-M0.9 ranker has 4.76 million parameters. The published
`focalnet-human.onnx` artifact is 19.45 MiB in FP32.

## Examples

These photographs come from [Autogravity](https://github.com/appwrite/autogravity).
A centered 1:1 crop of the park scene keeps grass. FocalNet keeps the dog.

![Center 1:1 crop of empty grass versus FocalNet 1:1 crop of the golden retriever](images/comparison-retriever.png)

On a two-person portrait, FocalNet holds the nearer subject instead of splitting
the frame.

![Center 1:1 crop versus FocalNet 1:1 crop of a two-person portrait](images/comparison-faces.png)

## Files

| File | Role | SHA-256 |
| --- | --- | --- |
| `focalnet-human.onnx` | Importance map + crop ranking (format v2) | `59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439` |
| `focalnet.onnx` | Importance map only (format v1) | `87becceb269a2973c359df789783be49a9f840f47170a015d3776d7c4145a2ce` |
| `focalnet-human.pt` | PyTorch ranking checkpoint | `82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a` |
| `focalnet.pt` | PyTorch importance checkpoint | `d1942f0652f8ea85f75ffc0cb1bf40d7e70b38e7ad102ab0e6df5c2e07ce52cf` |

JSON sidecars next to the ONNX files record the preprocessing contract, parameter
counts, and export verification.

Use `focalnet-human.onnx` for inference. The PyTorch files are for continued
training, not for the ONNX Runtime path.

## Usage

Download the ranking model and crop a photo:

```sh
hf download appwrite/focalnet focalnet-human.onnx --local-dir artifacts
uv run focalnet predict-human artifacts/focalnet-human.onnx photo.jpg \
  --aspect-ratio 16:9
```

From Python:

```python
from huggingface_hub import hf_hub_download
from focalnet.human_runtime import HumanCropPredictor

path = hf_hub_download("appwrite/focalnet", "focalnet-human.onnx")
crop = HumanCropPredictor(path).predict("photo.jpg", aspect_ratio=16 / 9)
```

Importance-only inference:

```sh
hf download appwrite/focalnet focalnet.onnx --local-dir artifacts
uv run focalnet predict artifacts/focalnet.onnx photo.jpg --aspect-ratio 16:9
```

`predict-human` generates up to 125 crops across five positions and five zoom
levels, scores a padded maximum of 128 candidates in one ONNX call, and selects
the highest preference score among crops whose retained importance is within
0.05 of the best candidate. The returned human score is a relative logit.
Compare it only among crops for the same image.

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

## Results

The importance model was trained on 500,000 teacher-labeled Open Images V7
samples (RepViT-M0.9 encoder, 48-channel decoder, 10 epochs, best epoch 7).
Teacher agreement on the group-disjoint 10,000-image validation split:

| Metric | Result |
| --- | ---: |
| Map MAE | 0.09975 |
| Normalized centroid error | 0.06430 |
| 1:1 teacher importance retained | 91.09% |
| 16:9 teacher importance retained | 90.86% |
| 4:5 teacher importance retained | 85.56% |

The ranking head (7,069 parameters) was pretrained on CPC and fine-tuned on
GAICD. The importance network stays frozen. On GAICD's 500-image test split:

| Metric | GAICD only | CPC → GAICD |
| --- | ---: | ---: |
| Spearman correlation | 0.75696 | **0.76450** |
| Pearson correlation | 0.78211 | **0.79192** |
| Top-5 accuracy | 42.48% | **43.70%** |
| Top-10 accuracy | 64.45% | **64.54%** |
| Pairwise accuracy | 86.00% | **86.44%** |

The GAICD test split was inspected during several development iterations. Treat
these figures as engineering benchmarks, not an untouched final test. INT8
post-training quantization did not pass the crop-quality gate; FP32 is the
published artifact.

## Training data

- Open Images V7 (teacher-labeled importance maps from U²-Net + YuNet)
- [Comparative Photo Composition (CPC)](https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html)
- [GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch)

The reviewed CPC and GAICD archives do not state explicit image or annotation
licenses. This checkpoint is published under MIT for the model files; that
license does not grant rights to the third-party datasets or teacher weights
used during training.

## License

Model files are available under the [MIT License](https://github.com/appwrite/focalnet/blob/main/LICENSE),
the same license as the training code. Review dataset and teacher-model terms
before redistribution of derived artifacts.
