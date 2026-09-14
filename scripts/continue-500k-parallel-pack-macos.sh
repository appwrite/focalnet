#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

RUN=downloads/modal-run-500k
LOG_ROOT=downloads/modal-pack-parts
mkdir -p "$LOG_ROOT"

run_part() {
  part=$1
  attempt=1
  while ! uvx modal run modal_app.py::pack_expanded_dataset \
    --part-index "$part" --part-count 4 --pack-workers 8; do
    echo "Pack part $part launch failed on attempt $attempt; retrying in 30s" >&2
    attempt=$((attempt + 1))
    sleep 30
  done
}

echo "[5/7] Packing four disjoint shard groups on Modal CPU containers"
pids=""
part=0
while [ "$part" -lt 4 ]; do
  log="$LOG_ROOT/part-$part.log"
  run_part "$part" >"$log" 2>&1 &
  pids="$pids $!"
  echo "Started pack part $part as local client PID $!"
  part=$((part + 1))
done
for pid in $pids; do
  wait "$pid"
done

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
