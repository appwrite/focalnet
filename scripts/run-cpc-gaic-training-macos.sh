#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

ROOT=${HUMAN_CROP_DATA_ROOT:-/Volumes/SanDisk/focalnet-downloads/human-crops}
CPC_ROOT="$ROOT/cpc"
GAIC_ROOT="$ROOT/gaicd-v2"
CPC_ARCHIVE="$CPC_ROOT/CPCDataset.tar.gz"
GAIC_ARCHIVE="$GAIC_ROOT/GAIC.zip"
CPC_SHA256=dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281
GAIC_SHA256=b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b
CPC_RUN=downloads/modal-run-cpc-pretrain
FINAL_RUN=downloads/modal-run-cpc-gaic-lr3e4

mkdir -p "$CPC_ROOT" "$GAIC_ROOT" "$CPC_RUN" "$FINAL_RUN"
if [ ! -f "$CPC_ARCHIVE" ]; then
  uvx --from gdown gdown --id 1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO --output "$CPC_ARCHIVE"
fi
if [ ! -f "$GAIC_ARCHIVE" ]; then
  uvx --from gdown gdown \
    'https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view?usp=sharing' \
    --output "$GAIC_ARCHIVE"
fi

test "$(shasum -a 256 "$CPC_ARCHIVE" | awk '{print $1}')" = "$CPC_SHA256"
test "$(shasum -a 256 "$GAIC_ARCHIVE" | awk '{print $1}')" = "$GAIC_SHA256"

uv sync --frozen --extra train
uvx modal volume put -f focalnet-data "$CPC_ARCHIVE" /datasets/cpc/CPCDataset.tar.gz
uvx modal volume put -f focalnet-data "$GAIC_ARCHIVE" /datasets/gaic-v2/GAIC.zip

# CPC pretraining and GAICD fine-tuning use only training/validation data. The
# separate evaluation command reads the GAICD test split after model selection.
uvx modal run modal_app.py::pretrain_cpc_ranker
uvx modal run modal_app.py::finetune_cpc_on_gaic_candidate
uvx modal run modal_app.py::evaluate_cpc_gaic_candidate

for name in best.pt last.pt history.json
do
  uvx modal volume get --force focalnet-data \
    "runs/repvit-m0-9-500k-cpc-v1/$name" "$CPC_RUN/$name"
done
for name in \
  best.pt last.pt history.json evaluation.json \
  focalnet-human.onnx focalnet-human.onnx.json
do
  uvx modal volume get --force focalnet-data \
    "runs/repvit-m0-9-500k-cpc-gaic-lr3e4-candidate/$name" "$FINAL_RUN/$name"
done

echo "CPC → GAICD crop-ranking run complete: $FINAL_RUN"
