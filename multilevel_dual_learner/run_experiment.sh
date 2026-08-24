#!/usr/bin/env bash
set -Eeuo pipefail

framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
templates_dir="$(cd -- "$framework_dir/../templates" && pwd -P)"
dockerfile="$framework_dir/docker/Dockerfile"
image_name="${IMAGE_NAME:-tesi-multilevel-dual-learner}"
trainer_args=("$@")
experiment_name=""
for ((index = 0; index < ${#trainer_args[@]}; index++)); do
  case "${trainer_args[index]}" in
    --experiment-name) experiment_name="${trainer_args[index + 1]:-}" ;;
    --experiment-name=*) experiment_name="${trainer_args[index]#*=}" ;;
  esac
done
if [[ ! "$experiment_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$ ]]; then
  printf 'Errore: specifica --experiment-name con un nome sicuro (lettere, numeri, punto, trattino o underscore).\n' >&2
  exit 2
fi
container_name="${CONTAINER_NAME:-$experiment_name}"

docker build -f "$dockerfile" -t "$image_name" "$framework_dir"
docker run --rm --gpus all --entrypoint python "$image_name" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA non disponibile nel container"; print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")'
docker run -d \
  --name "$container_name" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$framework_dir,target=/workspace" \
  --mount "type=bind,src=$templates_dir,target=/templates,readonly" \
  "$image_name" "${trainer_args[@]}"

printf '\nEsperimento avviato: %s\n' "$container_name"
printf 'Log:       docker logs -f %s\n' "$container_name"
printf 'Stato:     docker ps -a --filter name=%s\n' "$container_name"
printf 'Stop:      docker stop %s\n' "$container_name"
printf 'Risultati: %s/results/%s\n' "$framework_dir" "$experiment_name"
