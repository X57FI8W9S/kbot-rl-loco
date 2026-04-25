from __future__ import annotations

import torch

from entornos.marcha_estado_robot import obtener_torque_controlado

Tensor = torch.Tensor


def calcular_recompensa_paso(entorno, accion_actual: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    pos_x = entorno.robot.data.root_pos_w[:, 0] - entorno.scene.env_origins[:, 0]
    pos_y = entorno.robot.data.root_pos_w[:, 1] - entorno.scene.env_origins[:, 1]
    vel_x = entorno.robot.data.root_lin_vel_w[:, 0]
    vel_y = entorno.robot.data.root_lin_vel_w[:, 1]
    vel_wz = entorno.robot.data.root_ang_vel_b[:, 2]
    verticalidad_base = torch.abs(entorno.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    q = entorno.robot.data.joint_pos[:, entorno.indices_controlados]
    q_err = q - entorno.pose_nominal_controlada
    torque = obtener_torque_controlado(entorno)
    dt_entorno = entorno.dt * entorno.cfg.decimacion
    tiempo_episodio = entorno.episode_length_buf.float() * dt_entorno
    avance_objetivo = entorno.velocidad_objetivo_x * tiempo_episodio
    avance = torch.clamp(pos_x / avance_objetivo.clamp_min(1.0e-6), min=0.0, max=1.0)
    velocidad_activacion_inicio = 0.25 * entorno.velocidad_objetivo_x
    velocidad_activacion_completa = 0.75 * entorno.velocidad_objetivo_x
    rango_activacion_velocidad = (
        velocidad_activacion_completa - velocidad_activacion_inicio
    ).clamp_min(1.0e-6)
    factor_marcha = torch.clamp(
        (vel_x - velocidad_activacion_inicio) / rango_activacion_velocidad,
        min=0.0,
        max=1.0,
    )

    recompensa_vx = torch.exp(-((vel_x - entorno.velocidad_objetivo_x) ** 2) / entorno.cfg.sigma_vx)
    recompensa_avance = avance
    recompensa_vy = torch.exp(-(vel_y ** 2) / entorno.cfg.sigma_vy)
    recompensa_yaw = torch.exp(-(vel_wz ** 2) / entorno.cfg.sigma_yaw)
    recompensa_vertical = verticalidad_base
    recompensa_supervivencia = torch.ones_like(recompensa_vx)
    costo_y_base = pos_y ** 2
    costo_error_vx = (vel_x - entorno.velocidad_objetivo_x) ** 2

    costo_suavidad_base = torch.sum((accion_actual - entorno.previous_actions) ** 2, dim=1)
    costo_torque_base = torch.sum(torque ** 2, dim=1)
    costo_pose_base = torch.sum(q_err ** 2, dim=1)

    aporte_vx = entorno.cfg.peso_vx * recompensa_vx
    aporte_x = entorno.cfg.peso_avance * recompensa_avance
    aporte_vy = factor_marcha * entorno.cfg.peso_vy * recompensa_vy
    aporte_yaw = factor_marcha * entorno.cfg.peso_yaw * recompensa_yaw
    aporte_vert = factor_marcha * entorno.cfg.peso_verticalidad * recompensa_vertical
    aporte_superv = (
        factor_marcha * entorno.cfg.peso_supervivencia * recompensa_supervivencia
    )
    costo_y = entorno.cfg.peso_y * costo_y_base
    costo_suavidad = entorno.cfg.peso_suavidad_accion * costo_suavidad_base
    costo_torque = entorno.cfg.peso_torque * costo_torque_base
    costo_pose = entorno.cfg.peso_pose_nominal * costo_pose_base

    recompensa = (
        aporte_vx
        + aporte_x
        + aporte_vy
        + aporte_yaw
        + aporte_vert
        + aporte_superv
        - costo_y
        - costo_suavidad
        - costo_torque
        - costo_pose
    )

    componentes = {
        "aporte_vx": aporte_vx.detach(),
        "aporte_x": aporte_x.detach(),
        "aporte_vy": aporte_vy.detach(),
        "aporte_yaw": aporte_yaw.detach(),
        "aporte_vert": aporte_vert.detach(),
        "aporte_superv": aporte_superv.detach(),
        "costo_y": costo_y.detach(),
        "costo_suavidad": costo_suavidad.detach(),
        "costo_torque": costo_torque.detach(),
        "costo_pose": costo_pose.detach(),
        "recompensa_vx": recompensa_vx.detach(),
        "recompensa_avance": recompensa_avance.detach(),
        "recompensa_vy": recompensa_vy.detach(),
        "recompensa_yaw": recompensa_yaw.detach(),
        "recompensa_vertical": recompensa_vertical.detach(),
        "recompensa_supervivencia": recompensa_supervivencia.detach(),
        "factor_marcha": factor_marcha.detach(),
        "pos_x": pos_x.detach(),
        "avance": avance.detach(),
        "avance_objetivo": avance_objetivo.detach(),
        "pos_y": pos_y.detach(),
        "vel_x": vel_x.detach(),
        "vel_y": vel_y.detach(),
        "vel_wz": vel_wz.detach(),
        "costo_y_base": costo_y_base.detach(),
        "costo_error_vx": costo_error_vx.detach(),
        "costo_suavidad_base": costo_suavidad_base.detach(),
        "costo_torque_base": costo_torque_base.detach(),
        "costo_pose_base": costo_pose_base.detach(),
    }
    return recompensa, componentes


def calcular_recompensa_terminal(
    entorno, terminated: Tensor, truncated: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    bonus_final_superv = torch.zeros((entorno.num_envs,), device=entorno.device)
    bonus_final_x = torch.zeros((entorno.num_envs,), device=entorno.device)
    malus_final_vx = torch.zeros((entorno.num_envs,), device=entorno.device)
    malus_final_caida = torch.zeros((entorno.num_envs,), device=entorno.device)

    dt_entorno = entorno.dt * entorno.cfg.decimacion
    duracion_maxima_episodio = entorno.cfg.pasos_maximos_episodio * dt_entorno

    env_ids_caida = torch.nonzero(terminated, as_tuple=False).squeeze(-1)
    if env_ids_caida.numel() > 0:
        pos_x_caida = (
            entorno.robot.data.root_pos_w[env_ids_caida, 0] - entorno.scene.env_origins[env_ids_caida, 0]
        )
        distancia_objetivo_caida = entorno.velocidad_objetivo_x[env_ids_caida] * duracion_maxima_episodio
        distancia_objetivo_abs_caida = distancia_objetivo_caida.abs().clamp_min(1.0e-6)
        avance_hacia_objetivo = torch.clamp(
            pos_x_caida / distancia_objetivo_abs_caida, min=0.0, max=1.0
        )
        malus_final_caida[env_ids_caida] = (
            entorno.cfg.peso_fin_caida * (1.0 - avance_hacia_objetivo) ** 2
        )

    env_ids_fin = torch.nonzero(truncated & ~terminated, as_tuple=False).squeeze(-1)
    if env_ids_fin.numel() == 0:
        recompensa_fin = -malus_final_caida
        return recompensa_fin, {
            "bonus_final_superv": bonus_final_superv,
            "bonus_final_x": bonus_final_x,
            "malus_final_vx": malus_final_vx,
            "malus_final_caida": malus_final_caida,
        }

    pos_x = entorno.robot.data.root_pos_w[env_ids_fin, 0] - entorno.scene.env_origins[env_ids_fin, 0]
    distancia_objetivo = entorno.velocidad_objetivo_x[env_ids_fin] * duracion_maxima_episodio
    distancia_objetivo_abs = distancia_objetivo.abs().clamp_min(1.0e-6)
    error_distancia_relativo = torch.abs(pos_x - distancia_objetivo) / distancia_objetivo_abs

    lengths = entorno.episode_length_buf[env_ids_fin].float().clamp_min(1.0)
    error_vx_medio = entorno._suma_costo_error_vx_episodio[env_ids_fin] / lengths

    bonus_final_superv[env_ids_fin] = entorno.cfg.peso_fin_supervivencia
    bonus_final_x[env_ids_fin] = (
        entorno.cfg.peso_fin_distancia_objetivo
        * torch.clamp(1.0 - error_distancia_relativo, min=0.0, max=1.0)
    )
    malus_final_vx[env_ids_fin] = entorno.cfg.peso_fin_error_vx * error_vx_medio

    recompensa_fin = (
        bonus_final_superv
        + bonus_final_x
        - malus_final_vx
        - malus_final_caida
    )
    return recompensa_fin, {
        "bonus_final_superv": bonus_final_superv,
        "bonus_final_x": bonus_final_x,
        "malus_final_vx": malus_final_vx,
        "malus_final_caida": malus_final_caida,
    }
