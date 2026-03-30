#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict

import torch

import isaaclab.sim as utilidades_sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from configuraciones.kbot_box_top import (
    ConfiguracionKBotBoxTop,
    OFFSETS_POSE_BOX_TOP,
    crear_pose_objetivo_desde_pose_por_defecto,
)

Tensor = torch.Tensor


NOMBRES_ARTICULACIONES_CONTROLADAS = [
    "left_hip_pitch_04",
    "left_hip_roll_03",
    "left_hip_yaw_03",
    "left_knee_04",
    "left_ankle_02",
    "right_hip_pitch_04",
    "right_hip_roll_03",
    "right_hip_yaw_03",
    "right_knee_04",
    "right_ankle_02",
]


@dataclass
class ConfiguracionEntornoMarchaPPOVec:
    dispositivo: str = "cuda:0"
    num_entornos: int = 64
    espaciado_entornos: float = 2.5
    dt_simulacion: float = 1.0 / 120.0
    decimacion: int = 2
    pasos_maximos_episodio: int = 1200

    escala_velocidad_angular_base: float = 0.25
    escala_velocidad_articulaciones: float = 0.05
    escala_accion: float = 0.20

    velocidad_objetivo_min: float = 0.10
    velocidad_objetivo_max: float = 0.60
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


class EntornoMarchaPPOVec:
    """Entorno vectorizado para entrenar PPO con multiples instancias en paralelo."""

    def __init__(
        self,
        configuracion_entorno: ConfiguracionEntornoMarchaPPOVec,
        configuracion_robot: ConfiguracionKBotBoxTop,
        headless: bool = True,
    ) -> None:
        self.cfg = configuracion_entorno
        self.cfg_robot = configuracion_robot
        self.headless = headless
        self.dispositivo = torch.device(self.cfg.dispositivo)
        self.num_entornos = self.cfg.num_entornos
        self.env_ids_todos = torch.arange(self.num_entornos, dtype=torch.long, device=self.dispositivo)

        self._crear_simulacion()
        self._crear_escena()
        self.simulacion.reset()
        self.scene.update(self.dt)

        self._mapear_articulaciones_controladas()
        self._preparar_pose_nominal()

        self.accion_anterior = torch.zeros((self.num_entornos, self.num_acciones), device=self.dispositivo)
        self.velocidad_objetivo_x = torch.zeros((self.num_entornos,), device=self.dispositivo)
        self.contador_pasos = torch.zeros((self.num_entornos,), dtype=torch.long, device=self.dispositivo)

        self.obs_dim = 3 + 3 + 1 + self.num_acciones + self.num_acciones + self.num_acciones
        self.act_dim = self.num_acciones

        self.reiniciar()

    def _crear_simulacion(self) -> None:
        cfg_sim = utilidades_sim.SimulationCfg(
            dt=self.cfg.dt_simulacion,
            device=self.cfg.dispositivo,
        )
        self.simulacion = SimulationContext(cfg_sim)
        self.dt = cfg_sim.dt
        self.simulacion.set_camera_view([6, 6, 4], [0, 0, 0])

    def _crear_escena(self) -> None:
        cfg_piso = utilidades_sim.GroundPlaneCfg()
        cfg_piso.func("/World/piso", cfg_piso)

        cfg_luz = utilidades_sim.DistantLightCfg(intensity=self.cfg.intensidad_luz)
        cfg_luz.func("/World/luz", cfg_luz)

        self.scene = InteractiveScene(
            InteractiveSceneCfg(
                num_envs=self.cfg.num_entornos,
                env_spacing=self.cfg.espaciado_entornos,
                replicate_physics=True,
                clone_in_fabric=True,
            )
        )

        cfg_articulacion = ArticulationCfg(
            prim_path=f"{self.scene.env_regex_ns}/KBotBoxTop",
            spawn=utilidades_sim.UsdFileCfg(
                usd_path=self.cfg_robot.ruta_usd,
                activate_contact_sensors=self.cfg_robot.activar_sensores_contacto,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.cfg_robot.posicion_inicial_root
            ),
            actuators={
                "todos": ImplicitActuatorCfg(
                    joint_names_expr=list(self.cfg_robot.actuadores.expresiones_nombres_articulaciones),
                    effort_limit_sim=self.cfg_robot.actuadores.limite_esfuerzo_sim,
                    stiffness=self.cfg_robot.actuadores.rigidez,
                    damping=self.cfg_robot.actuadores.amortiguamiento,
                )
            },
        )

        self.robot = Articulation(cfg_articulacion)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)

        if self.dispositivo.type == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/piso"])

    def _mapear_articulaciones_controladas(self) -> None:
        self.nombres_articulaciones = list(self.robot.data.joint_names)
        self.indices_controlados = []
        for nombre in NOMBRES_ARTICULACIONES_CONTROLADAS:
            if nombre not in self.nombres_articulaciones:
                raise ValueError(f"No encontre la articulacion controlada: {nombre}")
            self.indices_controlados.append(self.nombres_articulaciones.index(nombre))

        self.indices_controlados = torch.tensor(
            self.indices_controlados, dtype=torch.long, device=self.dispositivo
        )
        self.num_acciones = len(NOMBRES_ARTICULACIONES_CONTROLADAS)

    def _preparar_pose_nominal(self) -> None:
        pose_objetivo = crear_pose_objetivo_desde_pose_por_defecto(
            self.robot, offsets_articulaciones=OFFSETS_POSE_BOX_TOP
        )
        self.pose_nominal_completa = pose_objetivo.clone()
        self.pose_nominal_controlada = pose_objetivo[:, self.indices_controlados].clone()

    def _muestrear_comando(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.env_ids_todos
        self.velocidad_objetivo_x[env_ids] = torch.empty(
            len(env_ids), device=self.dispositivo
        ).uniform_(self.cfg.velocidad_objetivo_min, self.cfg.velocidad_objetivo_max)

    def _accion_reducida_a_objetivo_completo(self, accion_reducida: Tensor) -> Tensor:
        objetivo = self.pose_nominal_completa.clone()
        objetivo[:, self.indices_controlados] = (
            self.pose_nominal_controlada + self.cfg.escala_accion * accion_reducida
        )
        return objetivo

    def _obtener_torque_controlado(self) -> Tensor:
        for nombre in ["computed_torque", "applied_torque", "joint_torque", "joint_effort"]:
            if hasattr(self.robot.data, nombre):
                valor = getattr(self.robot.data, nombre)
                if isinstance(valor, torch.Tensor):
                    return valor[:, self.indices_controlados]
        return torch.zeros((self.num_entornos, self.num_acciones), device=self.dispositivo)

    def _construir_observacion(self) -> Tensor:
        gravedad = self.robot.data.projected_gravity_b
        vel_ang_base = self.robot.data.root_ang_vel_b * self.cfg.escala_velocidad_angular_base
        q = self.robot.data.joint_pos[:, self.indices_controlados]
        qd = self.robot.data.joint_vel[:, self.indices_controlados] * self.cfg.escala_velocidad_articulaciones
        q_err = q - self.pose_nominal_controlada
        cmd = self.velocidad_objetivo_x.view(-1, 1)

        observacion = torch.cat(
            [gravedad, vel_ang_base, cmd, q_err, qd, self.accion_anterior], dim=1
        )
        return observacion

    def _calcular_recompensa(self, accion_actual: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        vx_real = self.robot.data.root_lin_vel_b[:, 0]
        gravedad_z = torch.abs(self.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        q = self.robot.data.joint_pos[:, self.indices_controlados]
        q_err = q - self.pose_nominal_controlada
        torque = self._obtener_torque_controlado()

        recompensa_vx = torch.exp(-((vx_real - self.velocidad_objetivo_x) ** 2) / self.cfg.sigma_vx)
        recompensa_vertical = gravedad_z
        recompensa_supervivencia = torch.ones_like(recompensa_vx)

        penalizacion_suavidad = torch.sum((accion_actual - self.accion_anterior) ** 2, dim=1)
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

    def _calcular_terminacion(self) -> tuple[Tensor, Dict[str, Tensor]]:
        altura = self.robot.data.root_pos_w[:, 2]
        coseno_vertical = torch.abs(self.robot.data.projected_gravity_b[:, 2])

        caida_por_altura = altura < self.cfg.altura_minima_base
        caida_por_inclinacion = coseno_vertical < self.cfg.coseno_minimo_vertical
        truncado_por_tiempo = self.contador_pasos >= self.cfg.pasos_maximos_episodio

        terminado = caida_por_altura | caida_por_inclinacion | truncado_por_tiempo
        info = {
            "caida_por_altura": caida_por_altura,
            "caida_por_inclinacion": caida_por_inclinacion,
            "truncado_por_tiempo": truncado_por_tiempo,
        }
        return terminado, info

    def reiniciar(self, env_ids: Tensor | None = None) -> Tensor:
        if env_ids is None:
            env_ids = self.env_ids_todos

        self.scene.reset(env_ids)

        pose_root = self.robot.data.default_root_state[env_ids, :7].clone()
        pose_root[:, :3] += self.scene.env_origins[env_ids]

        vel_root = self.robot.data.default_root_state[env_ids, 7:].clone()
        q = self.pose_nominal_completa[env_ids].clone()
        qd = torch.zeros_like(q)

        self.robot.write_root_pose_to_sim(pose_root, env_ids)
        self.robot.write_root_velocity_to_sim(vel_root, env_ids)
        self.robot.write_joint_state_to_sim(q, qd, None, env_ids)

        self.accion_anterior[env_ids] = 0.0
        self.contador_pasos[env_ids] = 0
        self._muestrear_comando(env_ids)

        self.robot.set_joint_position_target(self.pose_nominal_completa)
        self.scene.write_data_to_sim()
        self.simulacion.step()
        self.scene.update(self.dt)

        return self._construir_observacion()

    def paso(self, accion: Tensor):
        accion = accion.to(self.dispositivo).view(self.num_entornos, self.num_acciones).clamp(-1.0, 1.0)

        objetivo_q = self._accion_reducida_a_objetivo_completo(accion)
        for _ in range(self.cfg.decimacion):
            self.robot.set_joint_position_target(objetivo_q)
            self.scene.write_data_to_sim()
            if self._sim_step_tiene_render:
                self.simulacion.step(render=not self.headless)
            else:
                self.simulacion.step()
            self.scene.update(self.dt)

        self.contador_pasos += 1

        if self.cfg.intervalo_reinicio_comando > 0:
            cambiar_cmd = (self.contador_pasos % self.cfg.intervalo_reinicio_comando) == 0
            if torch.any(cambiar_cmd):
                self._muestrear_comando(torch.nonzero(cambiar_cmd, as_tuple=False).squeeze(-1))

        recompensa, info_recompensa = self._calcular_recompensa(accion)
        terminado, info_terminacion = self._calcular_terminacion()

        self.accion_anterior.copy_(accion)

        if torch.any(terminado):
            env_ids_reset = torch.nonzero(terminado, as_tuple=False).squeeze(-1)
            self.reiniciar(env_ids_reset)

        observacion = self._construir_observacion()
        info = {
            **info_recompensa,
            **info_terminacion,
            "velocidad_objetivo_x": self.velocidad_objetivo_x.clone(),
            "velocidad_real_x": self.robot.data.root_lin_vel_b[:, 0].clone(),
            "altura_base": self.robot.data.root_pos_w[:, 2].clone(),
        }
        return observacion, recompensa, terminado, info

    @property
    def _sim_step_tiene_render(self) -> bool:
        try:
            return "render" in inspect.signature(self.simulacion.step).parameters
        except Exception:
            return False
