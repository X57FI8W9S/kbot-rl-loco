from __future__ import annotations

import torch

from entornos.marcha_estado_robot import obtener_torque_controlado

Tensor = torch.Tensor


def calcular_recompensa_paso(entorno, accion_actual: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    x_env = entorno.robot.data.root_pos_w[:, 0] - entorno.scene.env_origins[:, 0]
    y_env = entorno.robot.data.root_pos_w[:, 1] - entorno.scene.env_origins[:, 1]
    vx_real = entorno.robot.data.root_lin_vel_w[:, 0]
    vy_real = entorno.robot.data.root_lin_vel_w[:, 1]
    wz_real = entorno.robot.data.root_ang_vel_b[:, 2]
    gravedad_z = torch.abs(entorno.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    q = entorno.robot.data.joint_pos[:, entorno.indices_controlados]
    q_err = q - entorno.pose_nominal_controlada
    torque = obtener_torque_controlado(entorno)
    dt_entorno = entorno.dt * entorno.cfg.decimacion
    tiempo_episodio = entorno.episode_length_buf.float() * dt_entorno
    progreso_objetivo = entorno.velocidad_objetivo_x * tiempo_episodio
    progreso = torch.clamp(x_env / progreso_objetivo.clamp_min(1.0e-6), min=0.0, max=1.0)
    velocidad_activacion_inicio = 0.25 * entorno.velocidad_objetivo_x
    velocidad_activacion_completa = 0.75 * entorno.velocidad_objetivo_x
    rango_activacion_velocidad = (
        velocidad_activacion_completa - velocidad_activacion_inicio
    ).clamp_min(1.0e-6)
    activacion_marcha = torch.clamp(
        (vx_real - velocidad_activacion_inicio) / rango_activacion_velocidad,
        min=0.0,
        max=1.0,
    )

    recompensa_vx = torch.exp(-((vx_real - entorno.velocidad_objetivo_x) ** 2) / entorno.cfg.sigma_vx)
    recompensa_progreso = progreso
    recompensa_vy = torch.exp(-(vy_real ** 2) / entorno.cfg.sigma_vy)
    recompensa_yaw = torch.exp(-(wz_real ** 2) / entorno.cfg.sigma_yaw)
    recompensa_vertical = gravedad_z
    recompensa_supervivencia = torch.ones_like(recompensa_vx)
    penalizacion_y = y_env ** 2
    penalizacion_error_vx = (vx_real - entorno.velocidad_objetivo_x) ** 2

    penalizacion_suavidad = torch.sum((accion_actual - entorno.previous_actions) ** 2, dim=1)
    penalizacion_torque = torch.sum(torque ** 2, dim=1)
    penalizacion_pose = torch.sum(q_err ** 2, dim=1)

    recompensa = (
        entorno.cfg.peso_vx * recompensa_vx
        + entorno.cfg.peso_progreso * recompensa_progreso
        + activacion_marcha * (
            entorno.cfg.peso_vy * recompensa_vy
            + entorno.cfg.peso_yaw * recompensa_yaw
            + entorno.cfg.peso_verticalidad * recompensa_vertical
            + entorno.cfg.peso_supervivencia * recompensa_supervivencia
        )
        - entorno.cfg.peso_y * penalizacion_y
        - entorno.cfg.peso_suavidad_accion * penalizacion_suavidad
        - entorno.cfg.peso_torque * penalizacion_torque
        - entorno.cfg.peso_pose_nominal * penalizacion_pose
    )

    componentes = {
        "recompensa_vx": recompensa_vx.detach(),
        "recompensa_progreso": recompensa_progreso.detach(),
        "recompensa_vy": recompensa_vy.detach(),
        "recompensa_yaw": recompensa_yaw.detach(),
        "recompensa_vertical": recompensa_vertical.detach(),
        "recompensa_supervivencia": recompensa_supervivencia.detach(),
        "activacion_marcha": activacion_marcha.detach(),
        "x_env": x_env.detach(),
        "progreso_x": progreso.detach(),
        "progreso_objetivo_x": progreso_objetivo.detach(),
        "y_env": y_env.detach(),
        "velocidad_mundo_x": vx_real.detach(),
        "velocidad_mundo_y": vy_real.detach(),
        "velocidad_yaw": wz_real.detach(),
        "penalizacion_y": penalizacion_y.detach(),
        "penalizacion_error_vx": penalizacion_error_vx.detach(),
        "penalizacion_suavidad": penalizacion_suavidad.detach(),
        "penalizacion_torque": penalizacion_torque.detach(),
        "penalizacion_pose": penalizacion_pose.detach(),
    }
    return recompensa, componentes


def calcular_recompensa_terminal(
    entorno, terminated: Tensor, truncated: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    recompensa_fin_supervivencia = torch.zeros((entorno.num_envs,), device=entorno.device)
    recompensa_fin_distancia_objetivo = torch.zeros((entorno.num_envs,), device=entorno.device)
    penalizacion_fin_error_vx = torch.zeros((entorno.num_envs,), device=entorno.device)
    penalizacion_fin_caida = torch.zeros((entorno.num_envs,), device=entorno.device)

    dt_entorno = entorno.dt * entorno.cfg.decimacion
    duracion_maxima_episodio = entorno.cfg.pasos_maximos_episodio * dt_entorno

    env_ids_caida = torch.nonzero(terminated, as_tuple=False).squeeze(-1)
    if env_ids_caida.numel() > 0:
        x_env_caida = (
            entorno.robot.data.root_pos_w[env_ids_caida, 0] - entorno.scene.env_origins[env_ids_caida, 0]
        )
        distancia_objetivo_caida = entorno.velocidad_objetivo_x[env_ids_caida] * duracion_maxima_episodio
        distancia_objetivo_abs_caida = distancia_objetivo_caida.abs().clamp_min(1.0e-6)
        progreso_hacia_objetivo = torch.clamp(
            x_env_caida / distancia_objetivo_abs_caida, min=0.0, max=1.0
        )
        penalizacion_fin_caida[env_ids_caida] = (
            entorno.cfg.peso_fin_caida * (1.0 - progreso_hacia_objetivo) ** 2
        )

    env_ids_fin = torch.nonzero(truncated & ~terminated, as_tuple=False).squeeze(-1)
    if env_ids_fin.numel() == 0:
        recompensa_fin = -penalizacion_fin_caida
        return recompensa_fin, {
            "recompensa_fin_supervivencia": recompensa_fin_supervivencia,
            "recompensa_fin_distancia_objetivo": recompensa_fin_distancia_objetivo,
            "penalizacion_fin_error_vx": penalizacion_fin_error_vx,
            "penalizacion_fin_caida": penalizacion_fin_caida,
        }

    x_env = entorno.robot.data.root_pos_w[env_ids_fin, 0] - entorno.scene.env_origins[env_ids_fin, 0]
    distancia_objetivo = entorno.velocidad_objetivo_x[env_ids_fin] * duracion_maxima_episodio
    distancia_objetivo_abs = distancia_objetivo.abs().clamp_min(1.0e-6)
    error_distancia_relativo = torch.abs(x_env - distancia_objetivo) / distancia_objetivo_abs

    lengths = entorno.episode_length_buf[env_ids_fin].float().clamp_min(1.0)
    error_vx_medio = entorno._episode_penalty_error_vx_sum[env_ids_fin] / lengths

    recompensa_fin_supervivencia[env_ids_fin] = entorno.cfg.peso_fin_supervivencia
    recompensa_fin_distancia_objetivo[env_ids_fin] = (
        entorno.cfg.peso_fin_distancia_objetivo
        * torch.clamp(1.0 - error_distancia_relativo, min=0.0, max=1.0)
    )
    penalizacion_fin_error_vx[env_ids_fin] = entorno.cfg.peso_fin_error_vx * error_vx_medio

    recompensa_fin = (
        recompensa_fin_supervivencia
        + recompensa_fin_distancia_objetivo
        - penalizacion_fin_error_vx
        - penalizacion_fin_caida
    )
    return recompensa_fin, {
        "recompensa_fin_supervivencia": recompensa_fin_supervivencia,
        "recompensa_fin_distancia_objetivo": recompensa_fin_distancia_objetivo,
        "penalizacion_fin_error_vx": penalizacion_fin_error_vx,
        "penalizacion_fin_caida": penalizacion_fin_caida,
    }
