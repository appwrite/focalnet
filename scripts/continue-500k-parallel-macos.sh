#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

RUN=downloads/modal-run-500k
LOG_ROOT=downloads/modal-label-shards
mkdir -p "$LOG_ROOT"

echo "[4/7] Labeling four disjoint 100,000-image ranges on Modal L4 GPUs"

run_shard() {
  shard_start=$1
  shard_end=$2
  attempt=1
  while ! uvx modal run --detach modal_app.py::label_extension_shard \
    --start-index "$shard_start" --end-index "$shard_end"; do
    echo "Range $shard_start-$shard_end launch failed on attempt $attempt; retrying in 30s" >&2
    attempt=$((attempt + 1))
    sleep 30
  done
}

pids=""
start=0
while [ "$start" -lt 400000 ]; do
  end=$((start + 100000))
  log="$LOG_ROOT/${start}-${end}.log"
  run_shard "$start" "$end" >"$log" 2>&1 &
  pids="$pids $!"
  echo "Started range $start-$end as local client PID $!"
  start="$end"
done

failed=0
for pid in $pids; do
  if ! wait "$pid"; then
    echo "Label client PID $pid failed; inspect $LOG_ROOT" >&2
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  exit 1
fi

echo "[4/7] Verifying and merging parallel label manifests"
uvx modal run modal_app.py::merge_extension_shards

echo "[5/7] Packing resumable training shards"
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
