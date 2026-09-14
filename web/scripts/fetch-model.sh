#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
dest="$root/web/public/focalnet-human.onnx"
expected="${FOCALNET_ONNX_SHA256:-59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439}"
volume_path="/runs/repvit-m0-9-500k-cpc-gaic-lr3e4-candidate/focalnet-human.onnx"

mkdir -p "$(dirname "$dest")"

hash_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
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
  cp "$FOCALNET_ONNX" "$dest"
  actual="$(hash_file "$dest")"
  if [[ "$actual" != "$expected" ]]; then
    echo "FOCALNET_ONNX hash $actual does not match $expected" >&2
    rm -f "$dest"
    exit 1
  fi
  echo "Copied $FOCALNET_ONNX to $dest"
  exit 0
fi

if command -v uvx >/dev/null && [[ "${FOCALNET_ALLOW_DUMMY:-}" != "1" ]]; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  if uvx modal volume get focalnet-data "$volume_path" "$tmpdir"; then
    found="$(find "$tmpdir" -name 'focalnet-human.onnx' -type f | head -n 1)"
    if [[ -z "$found" ]]; then
      echo "Modal volume get did not produce $volume_path" >&2
      exit 1
    fi
    cp "$found" "$dest"
    actual="$(hash_file "$dest")"
    if [[ "$actual" != "$expected" ]]; then
      echo "Downloaded model hash $actual does not match $expected" >&2
      rm -f "$dest"
      exit 1
    fi
    echo "Wrote $dest"
    exit 0
  fi
  trap - EXIT
  rm -rf "$tmpdir"
fi

if [[ "${FOCALNET_ALLOW_DUMMY:-}" != "1" ]]; then
  echo "Could not fetch the production ONNX. Set FOCALNET_ONNX, authenticate Modal, or FOCALNET_ALLOW_DUMMY=1 for UI layout only." >&2
  exit 1
fi

cd "$root"
uv run --with onnx python web/scripts/export-dummy-onnx.py
echo "Wrote dummy ONNX for local UI testing; do not publish this file"
