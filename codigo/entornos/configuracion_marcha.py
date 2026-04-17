from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RUTA_USD_PISO_REMOTA = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd"
)


def resolver_ruta_usd_piso() -> str:
    raiz_repo = Path(__file__).resolve().parents[2]
    ruta_local = raiz_repo / "assets" / "environment" / "usd" / "default_environment.usd"
    if ruta_local.is_file():
        return str(ruta_local)
    return RUTA_USD_PISO_REMOTA


NOMBRES_ARTICULACIONES_CONTROLADAS = [
    "left_hip_pitch_04", "left_hip_roll_03", "left_hip_yaw_03", "left_knee_04", "left_ankle_02",
    "right_hip_pitch_04", "right_hip_roll_03", "right_hip_yaw_03", "right_knee_04", "right_ankle_02",
]


@dataclass
class ConfiguracionEntornoMarchaRslRl:
    dispositivo: str = "cuda:0"
    num_entornos: int = 64
    espaciado_entornos: float = 2.5
    usar_clone_en_fabric: bool = False
    dt_simulacion: float = 1.0 / 120.0
    decimacion: int = 2
    pasos_maximos_episodio: int = 1200
    is_finite_horizon: bool = True

    escala_velocidad_angular_base: float = 0.25
    escala_velocidad_articulaciones: float = 0.05
    escala_accion: float = 0.20

    velocidad_objetivo_min: float = 0.20
    velocidad_objetivo_max: float = 0.20
    intervalo_reinicio_comando: int = 240

    altura_minima_base: float = 0.45
    coseno_minimo_vertical: float = 0.70

    peso_vx: float = 2.0
    peso_vy: float = 10.00
    peso_y: float = 0.05
    peso_yaw: float = 10.00
    peso_verticalidad: float = 1.00
    peso_supervivencia: float = 0.5
    peso_progreso: float = 0.50
    peso_suavidad_accion: float = 0.01
    peso_torque: float = 0.000005
    peso_pose_nominal: float = 0.000
    peso_fin_supervivencia: float = 25.0
    peso_fin_distancia_objetivo: float = 50.0
    peso_fin_error_vx: float = 10.0
    peso_fin_caida: float = 25.0

    sigma_vx: float = 0.005
    sigma_vy: float = 0.003
    sigma_yaw: float = 0.02
    intensidad_luz: float = 3000.0
    ruta_usd_piso: str = resolver_ruta_usd_piso()
