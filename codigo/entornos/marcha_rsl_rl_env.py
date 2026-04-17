#!/usr/bin/env python3
from __future__ import annotations

import inspect
from typing import Any

import torch
import gymnasium as gym
from gymnasium.vector.utils import batch_space

from isaaclab.envs import DirectRLEnv

from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop
from entornos.configuracion_marcha import (
    ConfiguracionEntornoMarchaRslRl,
    NOMBRES_ARTICULACIONES_CONTROLADAS,
)
from entornos.marcha_metricas import (
    actualizar_sumas_episodio,
    construir_extras_episodio,
    limpiar_sumas_episodio,
)
from entornos.marcha_observaciones import construir_observacion
from entornos.marcha_recompensas import (
    calcular_recompensa_paso,
    calcular_recompensa_terminal,
)
from entornos.marcha_estado_robot import (
    accion_reducida_a_objetivo_completo,
    mapear_articulaciones_controladas,
    obtener_torque_controlado,
    preparar_pose_nominal,
    samplear_comando,
)
from entornos.marcha_escena import crear_escena, crear_simulacion
from entornos.marcha_terminaciones import calcular_terminaciones

Tensor = torch.Tensor


class EntornoMarchaRslRl(DirectRLEnv):
    """Entorno vectorizado limpio con API compatible con rsl_rl."""

    def __init__(
        self,
        configuracion_entorno: ConfiguracionEntornoMarchaRslRl,
        configuracion_robot: ConfiguracionKBotBoxTop,
        headless: bool = True,
    ) -> None:
        self.cfg = configuracion_entorno
        self.cfg_robot = configuracion_robot
        self.headless = headless

        self._device = torch.device(self.cfg.dispositivo)
        self.dispositivo = self._device
        self._num_envs = self.cfg.num_entornos
        self._max_episode_length = self.cfg.pasos_maximos_episodio
        self.render_mode = None
        self.metadata = {}

        self.env_ids_todos = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        self._create_simulation()
        self._create_scene()
        self.simulation.reset()
        self.scene.update(self.dt)

        self._map_controlled_joints()
        self._prepare_nominal_pose()

        self.num_actions = len(NOMBRES_ARTICULACIONES_CONTROLADAS)
        self.num_obs = 3 + 3 + 1 + self.num_actions + self.num_actions + self.num_actions
        self.num_privileged_obs = 0

        # Standard Gym Space definitions
        obs_bound = torch.inf * torch.ones(self.num_obs, dtype=torch.float32)
        act_bound = torch.ones(self.num_actions, dtype=torch.float32)

        self.single_observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Box(low=-obs_bound.cpu().numpy(), high=obs_bound.cpu().numpy())
        })
        self.single_action_space = gym.spaces.Box(low=-act_bound.cpu().numpy(), high=act_bound.cpu().numpy())
        
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # RL Buffers
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.velocidad_objetivo_x = torch.zeros((self.num_envs,), device=self.device)
        self.x_anterior = torch.zeros((self.num_envs,), device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self._current_obs = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self._last_extras: dict[str, Any] = {}
        
        self._episode_reward_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_vx_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_progreso_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_vy_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_yaw_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_vertical_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_supervivencia_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_smoothness_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_torque_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_pose_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_error_vx_sum = torch.zeros((self.num_envs,), device=self.device)

        self.reset()

    @property
    def obs_dim(self) -> int: return self.num_obs

    @property
    def act_dim(self) -> int: return self.num_actions

    @property
    def device(self) -> torch.device: return self._device

    @property
    def num_envs(self) -> int: return self._num_envs

    @property
    def max_episode_length(self) -> int: return self._max_episode_length

    @property
    def unwrapped(self): return self

    @property
    def action_manager(self) -> None:
        raise AttributeError("Entorno directo sin action_manager")

    @property
    def observation_manager(self) -> None:
        raise AttributeError("Entorno directo sin observation_manager")

    def _create_simulation(self) -> None:
        crear_simulacion(self)

    def _create_scene(self) -> None:
        crear_escena(self)

    def _map_controlled_joints(self) -> None:
        mapear_articulaciones_controladas(self)

    def _prepare_nominal_pose(self) -> None:
        preparar_pose_nominal(self)

    def _sample_command(self, env_ids: Tensor | None = None) -> None:
        samplear_comando(self, env_ids)

    def _reduced_action_to_full_target(self, reduced_action: Tensor) -> Tensor:
        return accion_reducida_a_objetivo_completo(self, reduced_action)

    def _get_controlled_torque(self) -> Tensor:
        return obtener_torque_controlado(self)

    def _build_observation(self) -> Tensor:
        return construir_observacion(self)

    def _compute_reward(self, current_action: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return calcular_recompensa_paso(self, current_action)

    def _compute_dones(self) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return calcular_terminaciones(self)

    def get_observations(self) -> Tensor:
        self._current_obs = self._build_observation()
        return self._current_obs

    def _get_observations(self) -> dict[str, Tensor]:
        return {"policy": self.get_observations()}

    def get_privileged_observations(self) -> None:
        return None

    def _update_episode_sums(self, reward: Tensor, reward_info: dict[str, Tensor]) -> None:
        actualizar_sumas_episodio(self, reward, reward_info)

    def _build_episode_extras(self, done_env_ids: Tensor) -> dict[str, Tensor]:
        return construir_extras_episodio(self, done_env_ids)

    def _clear_episode_sums(self, env_ids: Tensor) -> None:
        limpiar_sumas_episodio(self, env_ids)

    def _compute_terminal_rewards(
        self, terminated: Tensor, truncated: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return calcular_recompensa_terminal(self, terminated, truncated)

    def _reset_idx(self, env_ids: Tensor) -> None:
        if env_ids.numel() == 0:
            return

        self.scene.reset(env_ids)

        pose_root = self.robot.data.default_root_state[env_ids, :7].clone()
        pose_root[:, :3] += self.scene.env_origins[env_ids]

        vel_root = self.robot.data.default_root_state[env_ids, 7:].clone()
        q = self.pose_nominal_completa[env_ids].clone()
        qd = torch.zeros_like(q)

        self.robot.write_root_pose_to_sim(pose_root, env_ids)
        self.robot.write_root_velocity_to_sim(vel_root, env_ids)
        self.robot.write_joint_state_to_sim(q, qd, None, env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.x_anterior[env_ids] = pose_root[:, 0] - self.scene.env_origins[env_ids, 0]
        self.episode_length_buf[env_ids] = 0
        self.reset_terminated[env_ids] = False
        self.reset_time_outs[env_ids] = False
        self._clear_episode_sums(env_ids)
        self._sample_command(env_ids)

        self.robot.set_joint_position_target(self.pose_nominal_completa)
        self.scene.write_data_to_sim()
        self.simulation.step()
        self.scene.update(self.dt)

    def reset(
        self,
        env_ids: Tensor | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        if env_ids is None:
            env_ids = self.env_ids_todos
        self._reset_idx(env_ids)
        self._last_extras = {}
        return {"policy": self.get_observations()}, self._last_extras

    def step(self, action: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        action = action.to(self.device).view(self.num_envs, self.num_actions).clamp(-1.0, 1.0)
        self.actions.copy_(action)

        target_q = self._reduced_action_to_full_target(action)
        for _ in range(self.cfg.decimacion):
            self.robot.set_joint_position_target(target_q)
            self.scene.write_data_to_sim()
            if self._sim_step_supports_render:
                self.simulation.step(render=not self.headless)
            else:
                self.simulation.step()
            self.scene.update(self.dt)

        self.episode_length_buf += 1

        if self.cfg.intervalo_reinicio_comando > 0:
            cambiar_cmd = (self.episode_length_buf % self.cfg.intervalo_reinicio_comando) == 0
            if torch.any(cambiar_cmd):
                self._sample_command(torch.nonzero(cambiar_cmd, as_tuple=False).squeeze(-1))

        reward, reward_info = self._compute_reward(action)
        self.x_anterior.copy_(self.robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0])
        terminated, truncated, termination_info = self._compute_dones()
        dones = terminated | truncated
        recompensa_fin, reward_info_fin = self._compute_terminal_rewards(terminated, truncated)
        reward = reward + recompensa_fin

        self._update_episode_sums(reward, reward_info)
        self.previous_actions.copy_(action)
        self.reset_terminated.copy_(terminated)
        self.reset_time_outs.copy_(truncated)

        extras: dict[str, Any] = {
            **reward_info,
            **reward_info_fin,
            **termination_info,
            "time_outs": truncated.clone(),
            "velocidad_objetivo_x": self.velocidad_objetivo_x.clone(),
            "velocidad_real_x": self.robot.data.root_lin_vel_w[:, 0].clone(),
            "velocidad_real_y": self.robot.data.root_lin_vel_w[:, 1].clone(),
            "x_env": (self.robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]).clone(),
            "yaw_rate_real": self.robot.data.root_ang_vel_b[:, 2].clone(),
            "altura_base": self.robot.data.root_pos_w[:, 2].clone(),
        }

        if torch.any(dones):
            env_ids_reset = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            extras["episode"] = {
                **self._build_episode_extras(env_ids_reset),
                "reward_fin_supervivencia": reward_info_fin["recompensa_fin_supervivencia"][env_ids_reset].mean(),
                "reward_fin_distancia_objetivo": reward_info_fin["recompensa_fin_distancia_objetivo"][env_ids_reset].mean(),
                "penalty_fin_error_vx": reward_info_fin["penalizacion_fin_error_vx"][env_ids_reset].mean(),
                "penalty_fin_caida": reward_info_fin["penalizacion_fin_caida"][env_ids_reset].mean(),
            }
            self._reset_idx(env_ids_reset)

        self._last_extras = extras
        return {"policy": self.get_observations()}, reward, terminated, truncated, extras

    def close(self) -> None:
        pass

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            return []
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return [seed]

    @property
    def _sim_step_supports_render(self) -> bool:
        try:
            return "render" in inspect.signature(self.simulation.step).parameters
        except Exception:
            return False
