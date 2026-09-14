#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

RUN=downloads/modal-run-500k

echo "[5/7] Packing resumable training shards in parallel"
uvx modal run modal_app.py::pack_expanded_dataset

echo "[6/7] Fine-tuning from the 100k model on an L40S for 10 epochs"
uvx modal run --detach modal_app.py::train_expanded --epochs 10

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
