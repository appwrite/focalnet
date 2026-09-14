#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
export PATH

command -v uv >/dev/null 2>&1 || {
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

for path in \
    downloads/open-images-v7-100k/images \
    downloads/teacher-models/u2net.onnx \
    downloads/teacher-models/face_detection_yunet_2023mar.onnx \
    downloads/teacher-open-images-v7-100k-fp32/provenance.json
do
    if [ ! -e "$path" ]; then
        echo "Missing transferred path: $path" >&2
        exit 1
    fi
done

uv sync --frozen

uv run python - <<'PY'
import onnxruntime as ort

if "CoreMLExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("This ONNX Runtime build does not expose CoreMLExecutionProvider")
print("Core ML provider is available; resuming FP32 teacher labels")
PY

exec uv run focalnet label \
    --images downloads/open-images-v7-100k/images \
    --saliency-model downloads/teacher-models/u2net.onnx \
    --face-model downloads/teacher-models/face_detection_yunet_2023mar.onnx \
    --output downloads/teacher-open-images-v7-100k-fp32 \
    --provider coreml \
    --threads 1 \
    --workers 8 \
    --resume
