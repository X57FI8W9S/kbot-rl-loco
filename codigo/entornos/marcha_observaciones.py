from __future__ import annotations

import torch

Tensor = torch.Tensor


def construir_observacion(entorno) -> Tensor:
    gravedad = entorno.robot.data.projected_gravity_b
    vel_ang_base = entorno.robot.data.root_ang_vel_b * entorno.cfg.escala_velocidad_angular_base
    q = entorno.robot.data.joint_pos[:, entorno.indices_controlados]
    qd = (
        entorno.robot.data.joint_vel[:, entorno.indices_controlados]
        * entorno.cfg.escala_velocidad_articulaciones
    )
    q_err = q - entorno.pose_nominal_controlada
    cmd = entorno.velocidad_objetivo_x.view(-1, 1)

    return torch.cat([gravedad, vel_ang_base, cmd, q_err, qd, entorno.previous_actions], dim=1)
