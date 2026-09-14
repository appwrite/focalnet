#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

IMAGES=downloads/open-images-v7-extension-400k
OLD_IMAGES=downloads/open-images-v7-100k/images.jsonl
MODELS=downloads/teacher-models
RUN=downloads/modal-run-500k

uv sync --frozen

echo "[1/7] Planning 400,000 new Open Images files"
uv run focalnet acquire-open-images \
  --output "$IMAGES" \
  --count 400000 --max-side 640 --workers 128 \
  --storage-limit-gib 40 --min-free-gib 45 \
  --seed 43 --jpeg-quality 88 --mix expanded \
  --exclude-manifest "$OLD_IMAGES" --plan-only

echo "[2/7] Uploading the plan and teacher models to Modal"
uvx modal volume put -f focalnet-data "$IMAGES/candidates-with-metadata.jsonl" \
  /open-images-v7-extension-400k/candidates-with-metadata.jsonl
uvx modal volume put -f focalnet-data "$IMAGES/provenance.json" \
  /open-images-v7-extension-400k/provenance.json
uvx modal volume put -f focalnet-data "$MODELS/u2net.onnx" /teacher-models/u2net.onnx
uvx modal volume put -f focalnet-data "$MODELS/face_detection_yunet_2023mar.onnx" \
  /teacher-models/face_detection_yunet_2023mar.onnx
uvx modal run modal_app.py::verify_teacher_runtime

echo "[3/7] Downloading and normalizing the extension images on Modal"
uvx modal run --detach modal_app.py::download_planned_images

echo "[4/7] Generating U²-Net + YuNet teacher maps on Modal CUDA"
uvx modal run --detach modal_app.py::label_extension

echo "[5/7] Packing resumable training shards"
uvx modal run modal_app.py::pack_expanded_dataset

echo "[6/7] Fine-tuning from the 100k model on an L40S"
uvx modal run --detach modal_app.py::train_expanded

echo "[7/7] Downloading artifacts individually and evaluating the fixed validation set"
mkdir -p "$RUN"
for name in best.pt last.pt history.json focalnet.onnx focalnet.onnx.json; do
  uvx modal volume get --force focalnet-data \
    "runs/repvit-m0-9-500k-v2/$name" "$RUN/$name"
done
rm -f "$RUN/evaluation-fp32.json"
uv run focalnet evaluate "$RUN/focalnet.onnx" \
  --manifest downloads/splits-v1/val.jsonl \
  --ratios 1 1.7777778 0.8 --threads 4 \
  --output "$RUN/evaluation-fp32.json"

echo "500k pipeline complete"
