#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

DATA_ROOT=${GAIC_DATA_ROOT:-/Volumes/SanDisk/focalnet-downloads/human-crops/gaicd-v2}
ARCHIVE="$DATA_ROOT/GAIC.zip"
SOURCE='https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view?usp=sharing'
EXPECTED_SHA256=b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b
RUN=downloads/modal-run-human-gaic-v1

mkdir -p "$DATA_ROOT" "$RUN"
if [ ! -f "$ARCHIVE" ]; then
  uvx --from gdown gdown "$SOURCE" --output "$ARCHIVE"
fi

ACTUAL_SHA256=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "GAICD archive hash mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi

cat > "$DATA_ROOT/provenance.json" <<EOF
{
  "dataset": "GAICD journal version",
  "source": "$SOURCE",
  "archive_sha256": "$EXPECTED_SHA256",
  "images": {"train": 2636, "val": 200, "test": 500},
  "license_note": "The archive contains no explicit image or annotation license; confirm rights before distribution or commercial use."
}
EOF

uv sync --frozen --extra train
uvx modal volume put -f focalnet-data "$ARCHIVE" /datasets/gaic-v2/GAIC.zip
uvx modal volume put -f focalnet-data "$DATA_ROOT/provenance.json" \
  /datasets/gaic-v2/provenance.json

# Keep this attached so downloads begin only after training, test evaluation,
# export, and Volume commits have completed. Checkpoints make a rerun resumable.
uvx modal run modal_app.py::train_human_crops

for name in \
  best.pt last.pt history.json evaluation-gaicd-test.json \
  focalnet-human.onnx focalnet-human.onnx.json
do
  uvx modal volume get --force focalnet-data \
    "runs/repvit-m0-9-500k-gaic-v1/$name" "$RUN/$name"
done

echo "Human crop-ranking run complete: $RUN"
