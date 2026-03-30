#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict

import torch

import isaaclab.sim as utilidades_sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
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
class ConfiguracionEntornoMarchaPPO:
    dispositivo: str = "cuda:0"
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


class EntornoMarchaPPO:
    """
    Entorno mínimo para entrenar PPO.
    Una sola instancia del robot, variables y comentarios en español.
    """

    def __init__(
        self,
        configuracion_entorno: ConfiguracionEntornoMarchaPPO,
        configuracion_robot: ConfiguracionKBotBoxTop,
        headless: bool = True,
    ) -> None:
        self.cfg = configuracion_entorno
        self.cfg_robot = configuracion_robot
        self.headless = headless
        self.dispositivo = torch.device(self.cfg.dispositivo)

        self._crear_simulacion()
        self._crear_robot()
        self.simulacion.reset()
        self.robot.update(self.dt)

        self._mapear_articulaciones_controladas()
        self._preparar_pose_nominal()

        self.accion_anterior = torch.zeros((1, self.num_acciones), device=self.dispositivo)
        self.velocidad_objetivo_x = torch.zeros((1,), device=self.dispositivo)
        self.contador_pasos = 0

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
        self.simulacion.set_camera_view([3, 3, 2], [0, 0, 1])

        cfg_piso = utilidades_sim.GroundPlaneCfg()
        cfg_piso.func("/World/piso", cfg_piso)

        cfg_luz = utilidades_sim.DistantLightCfg(intensity=self.cfg.intensidad_luz)
        cfg_luz.func("/World/luz", cfg_luz)

    def _crear_robot(self) -> None:
        cfg_articulacion = ArticulationCfg(
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
        self.robot = Articulation(cfg_articulacion.replace(prim_path="/World/KBotBoxTop"))

    def _mapear_articulaciones_controladas(self) -> None:
        self.nombres_articulaciones = list(self.robot.data.joint_names)
        self.indices_controlados = []
        for nombre in NOMBRES_ARTICULACIONES_CONTROLADAS:
            if nombre not in self.nombres_articulaciones:
                raise ValueError(f"No encontré la articulación controlada: {nombre}")
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

    def _muestrear_comando(self) -> None:
        self.velocidad_objetivo_x[:] = torch.empty(
            1, device=self.dispositivo
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
        return torch.zeros((1, self.num_acciones), device=self.dispositivo)

    def _construir_observacion(self) -> Tensor:
        gravedad = self.robot.data.projected_gravity_b
        vel_ang_base = self.robot.data.root_ang_vel_b * self.cfg.escala_velocidad_angular_base
        q = self.robot.data.joint_pos[:, self.indices_controlados]
        qd = self.robot.data.joint_vel[:, self.indices_controlados] * self.cfg.escala_velocidad_articulaciones
        q_err = q - self.pose_nominal_controlada
        cmd = self.velocidad_objetivo_x.view(1, 1)

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
        truncado_por_tiempo = torch.tensor(
            [self.contador_pasos >= self.cfg.pasos_maximos_episodio],
            dtype=torch.bool,
            device=self.dispositivo,
        )

        terminado = caida_por_altura | caida_por_inclinacion | truncado_por_tiempo
        info = {
            "caida_por_altura": caida_por_altura,
            "caida_por_inclinacion": caida_por_inclinacion,
            "truncado_por_tiempo": truncado_por_tiempo,
        }
        return terminado, info

    def reiniciar(self) -> Tensor:
        pose_root = torch.zeros((1, 7), dtype=torch.float32, device=self.dispositivo)
        pose_root[:, 2] = self.cfg_robot.posicion_inicial_root[2]
        pose_root[:, 3] = 1.0

        vel_root = torch.zeros((1, 6), dtype=torch.float32, device=self.dispositivo)
        q = self.pose_nominal_completa.clone()
        qd = torch.zeros_like(q)

        self.robot.write_root_pose_to_sim(pose_root)
        self.robot.write_root_velocity_to_sim(vel_root)
        self.robot.write_joint_state_to_sim(q, qd)

        self.robot.reset()
        self.robot.set_joint_position_target(q)
        self.robot.write_data_to_sim()
        self.simulacion.step()
        self.robot.update(self.dt)

        self.accion_anterior.zero_()
        self.contador_pasos = 0
        self._muestrear_comando()

        return self._construir_observacion()

    def paso(self, accion: Tensor):
        accion = accion.to(self.dispositivo).view(1, self.num_acciones).clamp(-1.0, 1.0)

        objetivo_q = self._accion_reducida_a_objetivo_completo(accion)
        for _ in range(self.cfg.decimacion):
            self.robot.set_joint_position_target(objetivo_q)
            self.robot.write_data_to_sim()
            if self._sim_step_tiene_render:
                self.simulacion.step(render=not self.headless)
            else:
                self.simulacion.step()
            self.robot.update(self.dt)

        self.contador_pasos += 1

        if self.cfg.intervalo_reinicio_comando > 0:
            if self.contador_pasos % self.cfg.intervalo_reinicio_comando == 0:
                self._muestrear_comando()

        observacion = self._construir_observacion()
        recompensa, info_recompensa = self._calcular_recompensa(accion)
        terminado, info_terminacion = self._calcular_terminacion()

        self.accion_anterior.copy_(accion)

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