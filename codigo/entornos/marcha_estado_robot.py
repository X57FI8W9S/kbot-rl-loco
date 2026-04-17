from __future__ import annotations

import torch

from configuraciones.kbot_box_top import (
    OFFSETS_POSE_BOX_TOP,
    crear_pose_objetivo_desde_pose_por_defecto,
)
from entornos.configuracion_marcha import NOMBRES_ARTICULACIONES_CONTROLADAS

Tensor = torch.Tensor


def mapear_articulaciones_controladas(entorno) -> None:
    entorno.nombres_articulaciones = list(entorno.robot.data.joint_names)
    entorno.indices_controlados = []
    for nombre in NOMBRES_ARTICULACIONES_CONTROLADAS:
        if nombre not in entorno.nombres_articulaciones:
            raise ValueError(f"No encontre la articulacion controlada: {nombre}")
        entorno.indices_controlados.append(entorno.nombres_articulaciones.index(nombre))

    entorno.indices_controlados = torch.tensor(
        entorno.indices_controlados, dtype=torch.long, device=entorno.device
    )


def preparar_pose_nominal(entorno) -> None:
    pose_objetivo = crear_pose_objetivo_desde_pose_por_defecto(
        entorno.robot, offsets_articulaciones=OFFSETS_POSE_BOX_TOP
    )
    entorno.pose_nominal_completa = pose_objetivo.clone()
    entorno.pose_nominal_controlada = pose_objetivo[:, entorno.indices_controlados].clone()


def samplear_comando(entorno, env_ids: Tensor | None = None) -> None:
    if env_ids is None:
        env_ids = entorno.env_ids_todos
    entorno.velocidad_objetivo_x[env_ids] = torch.empty(
        len(env_ids), device=entorno.device
    ).uniform_(entorno.cfg.velocidad_objetivo_min, entorno.cfg.velocidad_objetivo_max)


def accion_reducida_a_objetivo_completo(entorno, accion_reducida: Tensor) -> Tensor:
    objetivo = entorno.pose_nominal_completa.clone()
    objetivo[:, entorno.indices_controlados] = (
        entorno.pose_nominal_controlada + entorno.cfg.escala_accion * accion_reducida
    )
    return objetivo


def obtener_torque_controlado(entorno) -> Tensor:
    for nombre in ["computed_torque", "applied_torque", "joint_torque", "joint_effort"]:
        if hasattr(entorno.robot.data, nombre):
            valor = getattr(entorno.robot.data, nombre)
            if isinstance(valor, torch.Tensor):
                return valor[:, entorno.indices_controlados]
    return torch.zeros((entorno.num_envs, entorno.num_actions), device=entorno.device)
