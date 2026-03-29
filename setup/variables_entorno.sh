#!/usr/bin/env bash

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RAIZ_REPO
export DIRECTORIO_ISAACLAB="$RAIZ_REPO/isaac_lab/IsaacLab"
export DIRECTORIO_CODIGO="$RAIZ_REPO/codigo"
export DIRECTORIO_ASSETS="$RAIZ_REPO/assets"
export DIRECTORIO_SALIDAS="$RAIZ_REPO/salidas"
