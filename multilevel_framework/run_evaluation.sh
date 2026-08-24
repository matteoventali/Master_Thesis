#!/usr/bin/env bash
set -Eeuo pipefail

framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
templates_dir="$(cd -- "$framework_dir/../templates" && pwd -P)"
dockerfile="$framework_dir/docker/Dockerfile"
image_name="${IMAGE_NAME:-tesi-multilevel}"
gpu_id="${GPU_ID:-}"
if [[ -n "$gpu_id" && ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  printf 'Errore: GPU_ID deve essere un intero non negativo.\n' >&2
  exit 2
fi
if [[ -n "$gpu_id" ]]; then
  gpu_args=(--gpus "device=$gpu_id")
else
  gpu_args=(--gpus all)
fi

if (( $# == 0 )); then
  printf 'Usage: %s <policy> [<policy> ...] [evaluate.py options]\n' "$0" >&2
  exit 2
fi

docker build -f "$dockerfile" -t "$image_name" "$framework_dir"
docker run --rm \
  "${gpu_args[@]}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$framework_dir,target=/workspace" \
  --mount "type=bind,src=$templates_dir,target=/templates,readonly" \
  --workdir /workspace \
  --entrypoint python \
  "$image_name" /workspace/src/evaluate.py "$@"
