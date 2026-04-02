#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector.utils import batch_space

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.envs import DirectRLEnv

from configuraciones.kbot_box_top import (
    OFFSETS_POSE_BOX_TOP,
    ConfiguracionKBotBoxTop,
    crear_pose_objetivo_desde_pose_por_defecto,
)

Tensor = torch.Tensor

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

    velocidad_objetivo_min: float = 0.10
    velocidad_objetivo_max: float = 1.20
    intervalo_reinicio_comando: int = 240

    altura_minima_base: float = 0.45
    coseno_minimo_vertical: float = 0.55

    peso_vx: float = 2.0
    peso_verticalidad: float = 0.5
    peso_supervivencia: float = 0.2
    peso_suavidad_accion: float = 0.01
    peso_torque: float = 0.00002
    peso_pose_nominal: float = 0.05

    sigma_vx: float = 0.25
    intensidad_luz: float = 3000.0


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
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self._current_obs = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self._last_extras: dict[str, Any] = {}
        
        self._episode_reward_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_vx_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_reward_vertical_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_smoothness_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_torque_sum = torch.zeros((self.num_envs,), device=self.device)
        self._episode_penalty_pose_sum = torch.zeros((self.num_envs,), device=self.device)

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
        cfg_sim = sim_utils.SimulationCfg(dt=self.cfg.dt_simulacion, device=self.cfg.dispositivo)
        self.simulation = SimulationContext(cfg_sim)
        self.dt = cfg_sim.dt
        self.simulation.set_camera_view([6, 6, 4], [0, 0, 0])

    def _create_scene(self) -> None:
        floor_cfg = sim_utils.GroundPlaneCfg()
        floor_cfg.func("/World/piso", floor_cfg)

        light_cfg = sim_utils.DistantLightCfg(intensity=self.cfg.intensidad_luz)
        light_cfg.func("/World/luz", light_cfg)

        self.scene = InteractiveScene(
            InteractiveSceneCfg(
                num_envs=self.cfg.num_entornos,
                env_spacing=self.cfg.espaciado_entornos,
                replicate_physics=True,
                clone_in_fabric=self.cfg.usar_clone_en_fabric,
            )
        )

        articulation_cfg = ArticulationCfg(
            prim_path=f"{self.scene.env_regex_ns}/KBotBoxTop",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg_robot.ruta_usd,
                activate_contact_sensors=self.cfg_robot.activar_sensores_contacto,
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=self.cfg_robot.posicion_inicial_root),
            actuators={
                "todos": ImplicitActuatorCfg(
                    joint_names_expr=list(self.cfg_robot.actuadores.expresiones_nombres_articulaciones),
                    effort_limit_sim=self.cfg_robot.actuadores.limite_esfuerzo_sim,
                    stiffness=self.cfg_robot.actuadores.rigidez,
                    damping=self.cfg_robot.actuadores.amortiguamiento,
                )
            },
        )

        self.robot = Articulation(articulation_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)

        if self.device.type == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/piso"])

    def _map_controlled_joints(self) -> None:
        self.nombres_articulaciones = list(self.robot.data.joint_names)
        self.indices_controlados = []
        for nombre in NOMBRES_ARTICULACIONES_CONTROLADAS:
            if nombre not in self.nombres_articulaciones:
                raise ValueError(f"No encontre la articulacion controlada: {nombre}")
            self.indices_controlados.append(self.nombres_articulaciones.index(nombre))

        self.indices_controlados = torch.tensor(self.indices_controlados, dtype=torch.long, device=self.device)

    def _prepare_nominal_pose(self) -> None:
        pose_objetivo = crear_pose_objetivo_desde_pose_por_defecto(
            self.robot, offsets_articulaciones=OFFSETS_POSE_BOX_TOP
        )
        self.pose_nominal_completa = pose_objetivo.clone()
        self.pose_nominal_controlada = pose_objetivo[:, self.indices_controlados].clone()

    def _sample_command(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.env_ids_todos
        self.velocidad_objetivo_x[env_ids] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(self.cfg.velocidad_objetivo_min, self.cfg.velocidad_objetivo_max)

    def _reduced_action_to_full_target(self, reduced_action: Tensor) -> Tensor:
        objetivo = self.pose_nominal_completa.clone()
        objetivo[:, self.indices_controlados] = (
            self.pose_nominal_controlada + self.cfg.escala_accion * reduced_action
        )
        return objetivo

    def _get_controlled_torque(self) -> Tensor:
        for nombre in ["computed_torque", "applied_torque", "joint_torque", "joint_effort"]:
            if hasattr(self.robot.data, nombre):
                valor = getattr(self.robot.data, nombre)
                if isinstance(valor, torch.Tensor):
                    return valor[:, self.indices_controlados]
        return torch.zeros((self.num_envs, self.num_actions), device=self.device)

    def _build_observation(self) -> Tensor:
        gravedad = self.robot.data.projected_gravity_b
        vel_ang_base = self.robot.data.root_ang_vel_b * self.cfg.escala_velocidad_angular_base
        q = self.robot.data.joint_pos[:, self.indices_controlados]
        qd = self.robot.data.joint_vel[:, self.indices_controlados] * self.cfg.escala_velocidad_articulaciones
        q_err = q - self.pose_nominal_controlada
        cmd = self.velocidad_objetivo_x.view(-1, 1)

        return torch.cat([gravedad, vel_ang_base, cmd, q_err, qd, self.previous_actions], dim=1)

    def _compute_reward(self, current_action: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        vx_real = self.robot.data.root_lin_vel_b[:, 0]
        gravedad_z = torch.abs(self.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        q = self.robot.data.joint_pos[:, self.indices_controlados]
        q_err = q - self.pose_nominal_controlada
        torque = self._get_controlled_torque()

        recompensa_vx = torch.exp(-((vx_real - self.velocidad_objetivo_x) ** 2) / self.cfg.sigma_vx)
        recompensa_vertical = gravedad_z
        recompensa_supervivencia = torch.ones_like(recompensa_vx)

        penalizacion_suavidad = torch.sum((current_action - self.previous_actions) ** 2, dim=1)
        penalizacion_torque = torch.sum(torque ** 2, dim=1)
        penalizacion_pose = torch.sum(q_err ** 2, dim=1)

        recompensa = (
            self.cfg.peso_vx * recompensa_vx
            + self.cfg.peso_verticalidad * recompensa_vertical
            + self.cfg.peso_supervivencia * recompensa_supervivencia
            - self.cfg.peso_suavidad_accion * penalizacion_suavidad
            - self.cfg.peso_torque * penalizacion_torque
            - self.cfg.peso_pose_nominal * penalizacion_pose
        )

        componentes = {
            "recompensa_vx": recompensa_vx.detach(),
            "recompensa_vertical": recompensa_vertical.detach(),
            "penalizacion_suavidad": penalizacion_suavidad.detach(),
            "penalizacion_torque": penalizacion_torque.detach(),
            "penalizacion_pose": penalizacion_pose.detach(),
        }
        return recompensa, componentes

    def _compute_dones(self) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        altura = self.robot.data.root_pos_w[:, 2]
        coseno_vertical = torch.abs(self.robot.data.projected_gravity_b[:, 2])

        caida_por_altura = altura < self.cfg.altura_minima_base
        caida_por_inclinacion = coseno_vertical < self.cfg.coseno_minimo_vertical
        truncado_por_tiempo = self.episode_length_buf >= self.cfg.pasos_maximos_episodio

        terminated = caida_por_altura | caida_por_inclinacion
        truncated = truncado_por_tiempo
        info = {
            "caida_por_altura": caida_por_altura,
            "caida_por_inclinacion": caida_por_inclinacion,
            "truncado_por_tiempo": truncado_por_tiempo,
        }
        return terminated, truncated, info

    def get_observations(self) -> Tensor:
        self._current_obs = self._build_observation()
        return self._current_obs

    def _get_observations(self) -> dict[str, Tensor]:
        return {"policy": self.get_observations()}

    def get_privileged_observations(self) -> None:
        return None

    def _update_episode_sums(self, reward: Tensor, reward_info: dict[str, Tensor]) -> None:
        self._episode_reward_sum += reward
        self._episode_reward_vx_sum += reward_info["recompensa_vx"]
        self._episode_reward_vertical_sum += reward_info["recompensa_vertical"]
        self._episode_penalty_smoothness_sum += reward_info["penalizacion_suavidad"]
        self._episode_penalty_torque_sum += reward_info["penalizacion_torque"]
        self._episode_penalty_pose_sum += reward_info["penalizacion_pose"]

    def _build_episode_extras(self, done_env_ids: Tensor) -> dict[str, Tensor]:
        lengths = self.episode_length_buf[done_env_ids].float().clamp_min(1.0)
        return {
            "reward": self._episode_reward_sum[done_env_ids].mean(),
            "length": lengths.mean(),
            "reward_vx": (self._episode_reward_vx_sum[done_env_ids] / lengths).mean(),
            "reward_vertical": (self._episode_reward_vertical_sum[done_env_ids] / lengths).mean(),
            "penalty_smoothness": (self._episode_penalty_smoothness_sum[done_env_ids] / lengths).mean(),
            "penalty_torque": (self._episode_penalty_torque_sum[done_env_ids] / lengths).mean(),
            "penalty_pose": (self._episode_penalty_pose_sum[done_env_ids] / lengths).mean(),
        }

    def _clear_episode_sums(self, env_ids: Tensor) -> None:
        self._episode_reward_sum[env_ids] = 0.0
        self._episode_reward_vx_sum[env_ids] = 0.0
        self._episode_reward_vertical_sum[env_ids] = 0.0
        self._episode_penalty_smoothness_sum[env_ids] = 0.0
        self._episode_penalty_torque_sum[env_ids] = 0.0
        self._episode_penalty_pose_sum[env_ids] = 0.0

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
        terminated, truncated, termination_info = self._compute_dones()
        dones = terminated | truncated

        self._update_episode_sums(reward, reward_info)
        self.previous_actions.copy_(action)
        self.reset_terminated.copy_(terminated)
        self.reset_time_outs.copy_(truncated)

        extras: dict[str, Any] = {
            **reward_info,
            **termination_info,
            "time_outs": truncated.clone(),
            "velocidad_objetivo_x": self.velocidad_objetivo_x.clone(),
            "velocidad_real_x": self.robot.data.root_lin_vel_b[:, 0].clone(),
            "altura_base": self.robot.data.root_pos_w[:, 2].clone(),
        }

        if torch.any(dones):
            env_ids_reset = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            extras["episode"] = self._build_episode_extras(env_ids_reset)
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
