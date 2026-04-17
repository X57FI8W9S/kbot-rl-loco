from __future__ import annotations

import torch

Tensor = torch.Tensor


def actualizar_sumas_episodio(entorno, reward: Tensor, reward_info: dict[str, Tensor]) -> None:
    entorno._episode_reward_sum += reward
    entorno._episode_reward_vx_sum += reward_info["recompensa_vx"]
    entorno._episode_reward_progreso_sum += reward_info["recompensa_progreso"]
    entorno._episode_reward_vy_sum += reward_info["recompensa_vy"]
    entorno._episode_reward_yaw_sum += reward_info["recompensa_yaw"]
    entorno._episode_reward_vertical_sum += reward_info["recompensa_vertical"]
    entorno._episode_reward_supervivencia_sum += reward_info["recompensa_supervivencia"]
    entorno._episode_penalty_smoothness_sum += reward_info["penalizacion_suavidad"]
    entorno._episode_penalty_torque_sum += reward_info["penalizacion_torque"]
    entorno._episode_penalty_pose_sum += reward_info["penalizacion_pose"]
    entorno._episode_penalty_error_vx_sum += reward_info["penalizacion_error_vx"]


def construir_extras_episodio(entorno, done_env_ids: Tensor) -> dict[str, Tensor]:
    lengths = entorno.episode_length_buf[done_env_ids].float().clamp_min(1.0)
    return {
        "reward": entorno._episode_reward_sum[done_env_ids].mean(),
        "length": lengths.mean(),
        "reward_vx": (entorno._episode_reward_vx_sum[done_env_ids] / lengths).mean(),
        "reward_progreso": (entorno._episode_reward_progreso_sum[done_env_ids] / lengths).mean(),
        "reward_vy": (entorno._episode_reward_vy_sum[done_env_ids] / lengths).mean(),
        "reward_yaw": (entorno._episode_reward_yaw_sum[done_env_ids] / lengths).mean(),
        "reward_vertical": (entorno._episode_reward_vertical_sum[done_env_ids] / lengths).mean(),
        "reward_supervivencia": (
            entorno._episode_reward_supervivencia_sum[done_env_ids] / lengths
        ).mean(),
        "penalty_error_vx": (entorno._episode_penalty_error_vx_sum[done_env_ids] / lengths).mean(),
        "penalty_smoothness": (
            entorno._episode_penalty_smoothness_sum[done_env_ids] / lengths
        ).mean(),
        "penalty_torque": (entorno._episode_penalty_torque_sum[done_env_ids] / lengths).mean(),
        "penalty_pose": (entorno._episode_penalty_pose_sum[done_env_ids] / lengths).mean(),
    }


def limpiar_sumas_episodio(entorno, env_ids: Tensor) -> None:
    entorno._episode_reward_sum[env_ids] = 0.0
    entorno._episode_reward_vx_sum[env_ids] = 0.0
    entorno._episode_reward_progreso_sum[env_ids] = 0.0
    entorno._episode_reward_vy_sum[env_ids] = 0.0
    entorno._episode_reward_yaw_sum[env_ids] = 0.0
    entorno._episode_reward_vertical_sum[env_ids] = 0.0
    entorno._episode_reward_supervivencia_sum[env_ids] = 0.0
    entorno._episode_penalty_smoothness_sum[env_ids] = 0.0
    entorno._episode_penalty_torque_sum[env_ids] = 0.0
    entorno._episode_penalty_pose_sum[env_ids] = 0.0
    entorno._episode_penalty_error_vx_sum[env_ids] = 0.0
