#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "[ERROR] No ejecutes este script con 'source' o '.'."
    echo "[ERROR] Usalo como proceso independiente: bash scripts/entrenar_marcha_ppo_vectorizado_real.sh"
    return 1
fi

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_BASE="$RAIZ_REPO/scripts/entrenar_marcha_ppo_vectorizado.sh"

echo "[INFO] Lanzando corrida real PPO vectorizada."
echo "[INFO] Preset: num_envs=4096, iteraciones=10001, pasos_rollout=64"

exec "$SCRIPT_BASE" \
    --num-envs 4096 \
    --iteraciones 10001 \
    --pasos-rollout 64 \
    "$@"
