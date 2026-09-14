#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
dest="$root/web/public/focalnet-human.onnx"
expected="${FOCALNET_ONNX_SHA256:-59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439}"
model_url="${FOCALNET_ONNX_URL:-https://knowledgeable-sun-ef9f.yeet.page/focalnet-human.onnx}"
volume_path="/runs/repvit-m0-9-500k-cpc-gaic-lr3e4-candidate/focalnet-human.onnx"

mkdir -p "$(dirname "$dest")"

hash_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

install_model() {
  local source="$1"
  cp "$source" "$dest"
  local actual
  actual="$(hash_file "$dest")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Model hash $actual does not match $expected" >&2
    rm -f "$dest"
    return 1
  fi
  echo "Wrote $dest"
}

if ! command -v uvx >/dev/null && [[ -f "$HOME/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

if [[ -f "$dest" ]]; then
  actual="$(hash_file "$dest")"
  if [[ "$actual" == "$expected" ]]; then
    echo "Using existing model at $dest"
    exit 0
  fi
  if [[ "${FOCALNET_ALLOW_DUMMY:-}" == "1" ]]; then
    echo "Using existing dummy model at $dest"
    exit 0
  fi
  echo "Existing model hash $actual does not match $expected; re-downloading" >&2
  rm -f "$dest"
fi

if [[ -n "${FOCALNET_ONNX:-}" ]]; then
  install_model "$FOCALNET_ONNX"
  exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
download="$tmpdir/focalnet-human.onnx"

if command -v curl >/dev/null; then
  if curl -fsSL --retry 3 --retry-delay 2 -o "$download" "$model_url"; then
    install_model "$download"
    exit 0
  fi
elif command -v wget >/dev/null; then
  if wget -q -O "$download" "$model_url"; then
    install_model "$download"
    exit 0
  fi
fi

if command -v uvx >/dev/null && [[ "${FOCALNET_ALLOW_DUMMY:-}" != "1" ]]; then
  if uvx modal volume get focalnet-data "$volume_path" "$tmpdir"; then
    found="$(find "$tmpdir" -name 'focalnet-human.onnx' -type f | head -n 1)"
    if [[ -z "$found" ]]; then
      echo "Modal volume get did not produce $volume_path" >&2
      exit 1
    fi
    install_model "$found"
    exit 0
  fi
fi

if [[ "${FOCALNET_ALLOW_DUMMY:-}" != "1" ]]; then
  echo "Could not fetch the production ONNX from $model_url. Set FOCALNET_ONNX, authenticate Modal, or FOCALNET_ALLOW_DUMMY=1 for UI layout only." >&2
  exit 1
fi

cd "$root"
uv run --with onnx python web/scripts/export-dummy-onnx.py
echo "Wrote dummy ONNX for local UI testing; do not publish this file"
