#!/usr/bin/env bash
set -Eeuo pipefail

framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dockerfile="$framework_dir/docker/Dockerfile"
image_name="${IMAGE_NAME:-tesi-multilevel-dual-learner}"

if (( $# == 0 )); then
  printf 'Usage: %s <policy> [<policy> ...] [evaluate.py options]\n' "$0" >&2
  exit 2
fi

docker build -f "$dockerfile" -t "$image_name" "$framework_dir"
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$framework_dir,target=/workspace" \
  --workdir /workspace \
  --entrypoint python \
  "$image_name" /workspace/src/evaluate.py "$@"
