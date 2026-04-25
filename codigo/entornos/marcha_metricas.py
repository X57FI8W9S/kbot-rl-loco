from __future__ import annotations

import torch

Tensor = torch.Tensor


def actualizar_sumas_episodio(entorno, reward: Tensor, reward_info: dict[str, Tensor]) -> None:
    entorno._suma_recompensa_episodio += reward
    entorno._suma_aporte_vx_episodio += reward_info["aporte_vx"]
    entorno._suma_aporte_x_episodio += reward_info["aporte_x"]
    entorno._suma_aporte_vy_episodio += reward_info["aporte_vy"]
    entorno._suma_aporte_yaw_episodio += reward_info["aporte_yaw"]
    entorno._suma_aporte_vert_episodio += reward_info["aporte_vert"]
    entorno._suma_aporte_superv_episodio += reward_info["aporte_superv"]
    entorno._suma_costo_y_episodio += reward_info["costo_y"]
    entorno._suma_costo_suavidad_episodio += reward_info["costo_suavidad"]
    entorno._suma_costo_torque_episodio += reward_info["costo_torque"]
    entorno._suma_costo_pose_episodio += reward_info["costo_pose"]
    entorno._suma_recompensa_vx_episodio += reward_info["recompensa_vx"]
    entorno._suma_recompensa_avance_episodio += reward_info["recompensa_avance"]
    entorno._suma_recompensa_vy_episodio += reward_info["recompensa_vy"]
    entorno._suma_recompensa_yaw_episodio += reward_info["recompensa_yaw"]
    entorno._suma_recompensa_vertical_episodio += reward_info["recompensa_vertical"]
    entorno._suma_recompensa_supervivencia_episodio += reward_info["recompensa_supervivencia"]
    entorno._suma_factor_marcha_episodio += reward_info["factor_marcha"]
    entorno._suma_costo_y_base_episodio += reward_info["costo_y_base"]
    entorno._suma_costo_suavidad_base_episodio += reward_info["costo_suavidad_base"]
    entorno._suma_costo_torque_base_episodio += reward_info["costo_torque_base"]
    entorno._suma_costo_pose_base_episodio += reward_info["costo_pose_base"]
    entorno._suma_costo_error_vx_episodio += reward_info["costo_error_vx"]


def construir_extras_episodio(entorno, done_env_ids: Tensor) -> dict[str, Tensor]:
    lengths = entorno.episode_length_buf[done_env_ids].float().clamp_min(1.0)
    return {
        "recompensa": entorno._suma_recompensa_episodio[done_env_ids].mean(),
        "longitud": lengths.mean(),
        "aporte_vx": (entorno._suma_aporte_vx_episodio[done_env_ids] / lengths).mean(),
        "aporte_x": (entorno._suma_aporte_x_episodio[done_env_ids] / lengths).mean(),
        "aporte_vy": (entorno._suma_aporte_vy_episodio[done_env_ids] / lengths).mean(),
        "aporte_yaw": (entorno._suma_aporte_yaw_episodio[done_env_ids] / lengths).mean(),
        "aporte_vert": (entorno._suma_aporte_vert_episodio[done_env_ids] / lengths).mean(),
        "aporte_superv": (entorno._suma_aporte_superv_episodio[done_env_ids] / lengths).mean(),
        "costo_y": (entorno._suma_costo_y_episodio[done_env_ids] / lengths).mean(),
        "costo_suavidad": (entorno._suma_costo_suavidad_episodio[done_env_ids] / lengths).mean(),
        "costo_torque": (entorno._suma_costo_torque_episodio[done_env_ids] / lengths).mean(),
        "costo_pose": (entorno._suma_costo_pose_episodio[done_env_ids] / lengths).mean(),
        "recompensa_vx": (entorno._suma_recompensa_vx_episodio[done_env_ids] / lengths).mean(),
        "recompensa_avance": (entorno._suma_recompensa_avance_episodio[done_env_ids] / lengths).mean(),
        "recompensa_vy": (entorno._suma_recompensa_vy_episodio[done_env_ids] / lengths).mean(),
        "recompensa_yaw": (entorno._suma_recompensa_yaw_episodio[done_env_ids] / lengths).mean(),
        "recompensa_vertical": (entorno._suma_recompensa_vertical_episodio[done_env_ids] / lengths).mean(),
        "recompensa_supervivencia": (
            entorno._suma_recompensa_supervivencia_episodio[done_env_ids] / lengths
        ).mean(),
        "factor_marcha": (entorno._suma_factor_marcha_episodio[done_env_ids] / lengths).mean(),
        "costo_y_base": (entorno._suma_costo_y_base_episodio[done_env_ids] / lengths).mean(),
        "costo_error_vx": (entorno._suma_costo_error_vx_episodio[done_env_ids] / lengths).mean(),
        "costo_suavidad_base": (
            entorno._suma_costo_suavidad_base_episodio[done_env_ids] / lengths
        ).mean(),
        "costo_torque_base": (entorno._suma_costo_torque_base_episodio[done_env_ids] / lengths).mean(),
        "costo_pose_base": (entorno._suma_costo_pose_base_episodio[done_env_ids] / lengths).mean(),
    }


def limpiar_sumas_episodio(entorno, env_ids: Tensor) -> None:
    entorno._suma_recompensa_episodio[env_ids] = 0.0
    entorno._suma_aporte_vx_episodio[env_ids] = 0.0
    entorno._suma_aporte_x_episodio[env_ids] = 0.0
    entorno._suma_aporte_vy_episodio[env_ids] = 0.0
    entorno._suma_aporte_yaw_episodio[env_ids] = 0.0
    entorno._suma_aporte_vert_episodio[env_ids] = 0.0
    entorno._suma_aporte_superv_episodio[env_ids] = 0.0
    entorno._suma_costo_y_episodio[env_ids] = 0.0
    entorno._suma_costo_suavidad_episodio[env_ids] = 0.0
    entorno._suma_costo_torque_episodio[env_ids] = 0.0
    entorno._suma_costo_pose_episodio[env_ids] = 0.0
    entorno._suma_recompensa_vx_episodio[env_ids] = 0.0
    entorno._suma_recompensa_avance_episodio[env_ids] = 0.0
    entorno._suma_recompensa_vy_episodio[env_ids] = 0.0
    entorno._suma_recompensa_yaw_episodio[env_ids] = 0.0
    entorno._suma_recompensa_vertical_episodio[env_ids] = 0.0
    entorno._suma_recompensa_supervivencia_episodio[env_ids] = 0.0
    entorno._suma_factor_marcha_episodio[env_ids] = 0.0
    entorno._suma_costo_y_base_episodio[env_ids] = 0.0
    entorno._suma_costo_suavidad_base_episodio[env_ids] = 0.0
    entorno._suma_costo_torque_base_episodio[env_ids] = 0.0
    entorno._suma_costo_pose_base_episodio[env_ids] = 0.0
    entorno._suma_costo_error_vx_episodio[env_ids] = 0.0
