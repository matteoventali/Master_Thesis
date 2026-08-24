#!/usr/bin/env bash
set -Eeuo pipefail

framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dockerfile="$framework_dir/docker/Dockerfile"
image_name="${IMAGE_NAME:-tesi-multilevel}"
container_name="${CONTAINER_NAME:-traj2-value-iteration-benchmark}"
benchmark_args=("$@")

docker build -f "$dockerfile" -t "$image_name" "$framework_dir"

docker run -d --rm \
  --name "$container_name" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$framework_dir,target=/workspace" \
  --workdir /workspace \
  --entrypoint python \
  "$image_name" \
  /workspace/benchmark_value_iteration.py \
  --initial-size 3 \
  --timeout 3600 \
  --theta 0.001 \
  --csv /workspace/value_iteration_traj2.csv \
  "${benchmark_args[@]}"

printf '\nBenchmark avviato: %s\n' "$container_name"
printf 'Log:       docker logs -f %s\n' "$container_name"
printf 'Stato:     docker ps -a --filter name=%s\n' "$container_name"
printf 'Stop:      docker stop %s\n' "$container_name"
printf 'CSV:       %s/value_iteration_traj2.csv\n' "$framework_dir"
