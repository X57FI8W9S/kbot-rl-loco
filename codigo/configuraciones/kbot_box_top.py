"""
kbot_box_top.py

Bloque base para el futuro entorno de caminata.

Objetivo:
- preservar la logica que YA existe en el repo para el box top:
  - mismo USD por defecto
  - misma pose inicial del root
  - misma rigidez y amortiguamiento implicitos
  - misma pose objetivo basada en default_joint_pos + offsets
- separar claramente lo que es:
  1) configuracion del robot
  2) pose nominal / pose objetivo
  3) comentario sobre el USD con "cuartito" o entorno embebido

NOTA IMPORTANTE:
Este archivo NO elimina nada del USD.
Si el USD trae geometria de entorno visual, eso sigue estando en el archivo USD.
La idea aqui es NO perder la configuracion actual mientras preparamos un entorno de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable


# -----------------------------------------------------------------------------
# Configuracion base heredada del spawn actual del repo
# -----------------------------------------------------------------------------

def resolver_ruta_usd_box_top() -> str:
    """Resuelve la ruta del USD del robot sin depender de un único path duro."""

    raiz_repo = Path(__file__).resolve().parents[2]
    candidatos = [
        raiz_repo / "assets" / "robot" / "usd" / "kbot_box_top3.usd",
        Path("/media/rnyx/Tapioka/TPs/isaaclab/IsaacLab/source/isaaclab_assets/robots/kbot_box_top/kbot_box_top3.usd"),
        Path("/media/rnyx/Tapioka/TPs/isaaclab (Copy)/IsaacLab/source/isaaclab_assets/robots/kbot_box_top/kbot_box_top3.usd"),
    ]

    for candidato in candidatos:
        if candidato.is_file():
            return str(candidato)

    raise FileNotFoundError(
        "No se encontro kbot_box_top3.usd. "
        "Copia el USD a assets/robot/usd/ o ajusta la ruta en resolver_ruta_usd_box_top()."
    )


RUTA_USD_BOX_TOP_POR_DEFECTO = resolver_ruta_usd_box_top()


@dataclass
class ConfiguracionActuadoresKBot:
    """Mantiene los valores del spawn actual del repo.

    En el archivo actual `kbot_spawn_box_top.py` del repo, la articulacion se crea con:
    - effort_limit_sim = 200.0
    - stiffness = 600.0
    - damping = 20.0
    """

    limite_esfuerzo_sim: float = 200.0
    rigidez: float = 600.0
    amortiguamiento: float = 20.0
    expresiones_nombres_articulaciones: Iterable[str] = field(default_factory=lambda: [".*"])


@dataclass
class ConfiguracionKBotBoxTop:
    """Configuracion minima que preserva el comportamiento actual de spawn."""

    ruta_usd: str = RUTA_USD_BOX_TOP_POR_DEFECTO
    posicion_inicial_root: tuple[float, float, float] = (0.0, 0.0, 0.90)
    activar_sensores_contacto: bool = True
    actuadores: ConfiguracionActuadoresKBot = field(default_factory=ConfiguracionActuadoresKBot)

    # Si este flag es True, asumimos que el USD trae geometria de entorno visual
    # (por ejemplo, un cuarto o una escena decorativa) mezclada con el robot.
    # Esto NO rompe necesariamente un spawn de prueba, pero NO es lo ideal para
    # entrenamiento masivo / replicado.
    usd_tiene_entorno_embebido: bool = True


# -----------------------------------------------------------------------------
# Pose nominal / pose objetivo preservando la logica del repo
# -----------------------------------------------------------------------------

# Estos offsets salen directamente del archivo actual `kbot_spawn_box_top.py`.
OFFSETS_POSE_BOX_TOP: Dict[str, float] = {
    "right_hip_pitch_04": -0.30,
    "left_hip_pitch_04": 0.30,
    "right_knee_04": -0.50,
    "left_knee_04": 0.50,
    "right_ankle_02": 0.20,
    "left_ankle_02": -0.20,
}


def crear_pose_objetivo_desde_pose_por_defecto(robot, offsets_articulaciones: Dict[str, float] | None = None):
    """Crea la pose objetivo como:

    pose_objetivo = default_joint_pos + offsets

    Esto replica exactamente la idea del spawn actual del repo.

    Parametros
    ----------
    robot:
        Articulation de IsaacLab ya creada.
    offsets_articulaciones:
        Diccionario opcional con offsets por nombre de articulacion.

    Retorna
    -------
    tensor
        Tensor con la pose objetivo por entorno.
    """

    if offsets_articulaciones is None:
        offsets_articulaciones = OFFSETS_POSE_BOX_TOP

    pose_objetivo = robot.data.default_joint_pos.clone()

    for indice_articulacion, nombre_articulacion in enumerate(robot.data.joint_names):
        if nombre_articulacion in offsets_articulaciones:
            pose_objetivo[:, indice_articulacion] += offsets_articulaciones[nombre_articulacion]

    return pose_objetivo


# -----------------------------------------------------------------------------
# Recomendaciones practicas para el siguiente paso
# -----------------------------------------------------------------------------

RECOMENDACIONES = {
    "spawn_y_prueba_visual": (
        "Se puede seguir usando el USD actual para spawnear y verificar la pose, "
        "las inercias y las articulaciones."
    ),
    "entrenamiento": (
        "Para entrenamiento conviene mucho mas un USD solo-del-robot, sin cuartito, "
        "sin geometria decorativa y sin extras no esenciales."
    ),
    "pose_y_pd": (
        "La pose base y el PD del spawn actual no se perdieron: deben preservarse como "
        "baseline del entorno de caminata."
    ),
}
