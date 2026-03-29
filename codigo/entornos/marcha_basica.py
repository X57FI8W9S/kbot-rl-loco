#!/usr/bin/env python3
"""
marcha_basica.py

Entorno mínimo de caminata propioceptiva para KBot box-top en Isaac Lab.

Objetivos de este archivo:
- preservar el spawn actual del repo:
  - mismo USD del robot
  - misma altura inicial del root
  - mismos valores de rigidez / amortiguamiento implicitos
  - misma pose nominal basada en default_joint_pos + offsets
- conectar los componentes base ya definidos en:
  - componentes/marcha_basica.py
  - configuraciones/kbot_box_top.py
- ofrecer un entorno pequeño y legible, con nombres y comentarios en español
- incluir un bucle de prueba al final para verificar que la tubería de:
  observación -> acción -> recompensa -> terminación
  está conectada

Notas importantes:
- Este archivo NO registra todavía una tarea formal de Isaac Lab para PPO.
- Está pensado como paso intermedio útil antes de construir la versión completa de entrenamiento.
- El robot se trata aquí como articulación única sobre plano, con una consigna simple de velocidad
  hacia adelante: velocidad_objetivo_x.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Argumentos de línea de comandos
# -----------------------------------------------------------------------------

analizador_argumentos = argparse.ArgumentParser()
analizador_argumentos.add_argument("--headless", action="store_true")
analizador_argumentos.add_argument("--num-entornos", type=int, default=1)
analizador_argumentos.add_argument("--num-pasos", type=int, default=4000)
analizador_argumentos.add_argument("--intervalo-reinicio-comando", type=int, default=300)
analizador_argumentos.add_argument("--pasos-espera-inicial", type=int, default=240)
analizador_argumentos.add_argument("--mantener-abierto", action="store_true")
argumentos = analizador_argumentos.parse_args()

print(f"[INFO] Iniciando script marcha_basica.py con argumentos: {argumentos}", flush=True)

lanzador_app = AppLauncher(headless=argumentos.headless)
aplicacion_simulacion = lanzador_app.app


try:
    import torch
    import isaaclab.sim as utilidades_sim
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext

    from componentes.marcha_basica import (
        ConfiguracionCaminarBasico,
        EstadoRobot,
        aplicar_penalizacion_por_caida,
        calcular_recompensa,
        calcular_terminacion,
        construir_observacion,
        crear_objetivo_posicion_articulaciones,
        muestrear_velocidad_objetivo_x,
    )
    from configuraciones.kbot_box_top import (
        ConfiguracionKBotBoxTop,
        OFFSETS_POSE_BOX_TOP,
        crear_pose_objetivo_desde_pose_por_defecto,
    )


    # -------------------------------------------------------------------------
    # Configuración auxiliar del entorno
    # -------------------------------------------------------------------------

    @dataclass
    class ConfiguracionEntornoCaminarBasico:
        """Parámetros prácticos del entorno mínimo."""

        num_entornos: int = 1
        dt_simulacion: float = 1.0 / 120.0
        dispositivo: str = "cuda:0"
        intensidad_luz: float = 3000.0
        intervalo_reinicio_comando: int = 300
        escala_accion_politica: float = 1.0
        amplitud_marcha_prueba: float = 0.35
        frecuencia_marcha_prueba_hz: float = 1.4
        usar_politica_prueba: bool = True


    # -------------------------------------------------------------------------
    # Entorno
    # -------------------------------------------------------------------------

    class EntornoKBotWalkBasico:
        """
        Entorno mínimo para pruebas de caminata ciega.

        Este objeto crea:
        - simulación
        - plano y luz
        - articulación del robot
        - estado necesario para observación / recompensa / terminación

        También ofrece:
        - reinicio de comandos
        - paso de simulación
        - política abierta simple de prueba
        """

        def __init__(
            self,
            configuracion_entorno: ConfiguracionEntornoCaminarBasico,
            configuracion_robot: ConfiguracionKBotBoxTop,
        ) -> None:
            self.configuracion_entorno = configuracion_entorno
            self.configuracion_robot = configuracion_robot
            self.dispositivo = torch.device(configuracion_entorno.dispositivo)

            print("[DEBUG] Entorno: creando simulacion...", flush=True)
            self._crear_simulacion()
            print("[DEBUG] Entorno: creando robot...", flush=True)
            self._crear_robot()
            print("[DEBUG] Entorno: reseteando simulacion...", flush=True)
            self.simulacion.reset()
            print("[DEBUG] Entorno: actualizando robot tras reset...", flush=True)
            self.robot.update(self.dt)
            print("[DEBUG] Entorno: preparando metadatos...", flush=True)
            self._preparar_metadatos_articulaciones()
            print("[DEBUG] Entorno: creando configuracion de caminata...", flush=True)
            self._crear_configuracion_caminata()
            print("[DEBUG] Entorno: reiniciando buffers internos...", flush=True)
            self._reiniciar_buffers_internos()
            print("[DEBUG] Entorno: reiniciando entornos...", flush=True)
            self._reiniciar_entornos()
            print("[DEBUG] Entorno: manteniendo pose inicial...", flush=True)
            self._mantener_pose_actual_durante_pasos( max(1, int(argumentos.pasos_espera_inicial)) )
            print("[DEBUG] Entorno: inicializacion completa.", flush=True)

        # ------------------------------------------------------------------
        # Construcción de escena
        # ------------------------------------------------------------------

        def _crear_simulacion(self) -> None:
            configuracion_simulacion = utilidades_sim.SimulationCfg(
                dt=self.configuracion_entorno.dt_simulacion,
                device=self.configuracion_entorno.dispositivo,
            )
            self.simulacion = SimulationContext(configuracion_simulacion)
            self.dt = configuracion_simulacion.dt
            self.simulacion.set_camera_view([3, 3, 2], [0, 0, 1])

            configuracion_piso = utilidades_sim.GroundPlaneCfg()
            configuracion_piso.func("/World/piso", configuracion_piso)

            configuracion_luz = utilidades_sim.DistantLightCfg(
                intensity=self.configuracion_entorno.intensidad_luz
            )
            configuracion_luz.func("/World/luz", configuracion_luz)

        def _crear_robot(self) -> None:
            configuracion_articulacion = ArticulationCfg(
                spawn=utilidades_sim.UsdFileCfg(
                    usd_path=self.configuracion_robot.ruta_usd,
                    activate_contact_sensors=self.configuracion_robot.activar_sensores_contacto,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=self.configuracion_robot.posicion_inicial_root
                ),
                actuators={
                    "todos": ImplicitActuatorCfg(
                        joint_names_expr=list(
                            self.configuracion_robot.actuadores.expresiones_nombres_articulaciones
                        ),
                        effort_limit_sim=self.configuracion_robot.actuadores.limite_esfuerzo_sim,
                        stiffness=self.configuracion_robot.actuadores.rigidez,
                        damping=self.configuracion_robot.actuadores.amortiguamiento,
                    )
                },
            )

            self.robot = Articulation(configuracion_articulacion.replace(prim_path="/World/KBotBoxTop"))

        def _preparar_metadatos_articulaciones(self) -> None:
            self.nombres_articulaciones = list(self.robot.data.joint_names)
            self.nombres_cuerpos = list(self.robot.data.body_names)
            self.num_articulaciones = len(self.nombres_articulaciones)
            self.num_cuerpos = len(self.nombres_cuerpos)
            self.num_entornos = self.configuracion_entorno.num_entornos

            self.indices_articulaciones_izquierda = [
                indice
                for indice, nombre in enumerate(self.nombres_articulaciones)
                if "left" in nombre.lower()
            ]
            self.indices_articulaciones_derecha = [
                indice
                for indice, nombre in enumerate(self.nombres_articulaciones)
                if "right" in nombre.lower()
            ]

            self.indices_marcha_prueba: Dict[str, int] = {}
            for nombre_objetivo in [
                "left_hip_pitch_04",
                "right_hip_pitch_04",
                "left_knee_04",
                "right_knee_04",
                "left_ankle_02",
                "right_ankle_02",
            ]:
                if nombre_objetivo in self.nombres_articulaciones:
                    self.indices_marcha_prueba[nombre_objetivo] = self.nombres_articulaciones.index(nombre_objetivo)

            self.indices_cuerpos_pies = [
                indice
                for indice, nombre in enumerate(self.nombres_cuerpos)
                if any(clave in nombre.lower() for clave in ["foot", "pie", "ankle"])
            ]

            self.indices_cuerpos_no_pies = [
                indice for indice in range(self.num_cuerpos) if indice not in self.indices_cuerpos_pies
            ]

        def _crear_configuracion_caminata(self) -> None:
            pose_nominal = crear_pose_objetivo_desde_pose_por_defecto(
                self.robot,
                offsets_articulaciones=OFFSETS_POSE_BOX_TOP,
            )[0].detach().clone()

            self.pose_nominal_articulaciones = pose_nominal
            self.configuracion_caminata = ConfiguracionCaminarBasico(
                num_articulaciones_controladas=self.num_articulaciones,
                pose_nominal_articulaciones=self.pose_nominal_articulaciones,
                escala_accion=0.20,
                velocidad_objetivo_min=0.10,
                velocidad_objetivo_max=0.60,
                altura_minima_base=0.45,
                coseno_minimo_vertical=0.55,
            )
            self.configuracion_caminata.validar()

        def _reiniciar_buffers_internos(self) -> None:
            self.accion_anterior = torch.zeros(
                (self.num_entornos, self.num_articulaciones),
                dtype=torch.float32,
                device=self.dispositivo,
            )
            self.velocidad_objetivo_x = torch.zeros(
                self.num_entornos,
                dtype=torch.float32,
                device=self.dispositivo,
            )
            self.contador_pasos = torch.zeros(
                self.num_entornos,
                dtype=torch.long,
                device=self.dispositivo,
            )
            self.tiempo_acumulado = 0.0

        # ------------------------------------------------------------------
        # Lectura de estado
        # ------------------------------------------------------------------

        def _obtener_torques_articulaciones(self) -> torch.Tensor:
            for nombre_atributo in [
                "computed_torque",
                "applied_torque",
                "joint_torque",
                "joint_effort",
            ]:
                if hasattr(self.robot.data, nombre_atributo):
                    valor = getattr(self.robot.data, nombre_atributo)
                    if isinstance(valor, torch.Tensor):
                        return valor
            return torch.zeros(
                (self.num_entornos, self.num_articulaciones),
                dtype=torch.float32,
                device=self.dispositivo,
            )

        def _obtener_fuerzas_contacto_cuerpo(self) -> torch.Tensor | None:
            for nombre_atributo in [
                "net_contact_forces_w",
                "body_net_contact_forces_w",
            ]:
                if hasattr(self.robot.data, nombre_atributo):
                    valor = getattr(self.robot.data, nombre_atributo)
                    if isinstance(valor, torch.Tensor):
                        if valor.ndim == 3:
                            return valor[:, self.indices_cuerpos_no_pies, :] if self.indices_cuerpos_no_pies else None
            return None

        def _obtener_velocidades_pies_xy(self) -> torch.Tensor | None:
            for nombre_atributo in [
                "body_lin_vel_w",
                "body_lin_vel_b",
            ]:
                if hasattr(self.robot.data, nombre_atributo):
                    valor = getattr(self.robot.data, nombre_atributo)
                    if isinstance(valor, torch.Tensor) and valor.ndim == 3 and self.indices_cuerpos_pies:
                        return valor[:, self.indices_cuerpos_pies, :2]
            return None

        def construir_estado_robot(self) -> EstadoRobot:
            altura_base = self.robot.data.root_pos_w[:, 2]

            estado = EstadoRobot(
                gravedad_proyectada=self.robot.data.projected_gravity_b,
                velocidad_angular_base=self.robot.data.root_ang_vel_b,
                posicion_articulaciones=self.robot.data.joint_pos,
                velocidad_articulaciones=self.robot.data.joint_vel,
                accion_anterior=self.accion_anterior,
                altura_base=altura_base,
                velocidad_lineal_base_cuerpo=self.robot.data.root_lin_vel_b,
                torques_articulaciones=self._obtener_torques_articulaciones(),
                fuerzas_contacto_cuerpo=self._obtener_fuerzas_contacto_cuerpo(),
                velocidades_pies_xy=self._obtener_velocidades_pies_xy(),
            )
            return estado

        # ------------------------------------------------------------------
        # Reinicios
        # ------------------------------------------------------------------

        def reiniciar_comandos(self, indices_entornos: torch.Tensor | None = None) -> None:
            if indices_entornos is None:
                self.velocidad_objetivo_x = muestrear_velocidad_objetivo_x(
                    self.num_entornos,
                    self.configuracion_caminata,
                    self.dispositivo,
                )
                return

            nuevas_velocidades = muestrear_velocidad_objetivo_x(
                int(indices_entornos.numel()),
                self.configuracion_caminata,
                self.dispositivo,
            )
            self.velocidad_objetivo_x[indices_entornos] = nuevas_velocidades

        def _estado_root_por_defecto(self) -> tuple[torch.Tensor, torch.Tensor]:
            pose_root = torch.zeros((self.num_entornos, 7), dtype=torch.float32, device=self.dispositivo)
            pose_root[:, 0] = self.configuracion_robot.posicion_inicial_root[0]
            pose_root[:, 1] = self.configuracion_robot.posicion_inicial_root[1]
            pose_root[:, 2] = self.configuracion_robot.posicion_inicial_root[2]
            pose_root[:, 3] = 1.0

            velocidad_root = torch.zeros((self.num_entornos, 6), dtype=torch.float32, device=self.dispositivo)

            if hasattr(self.robot.data, "default_root_state"):
                estado_root = self.robot.data.default_root_state.clone()
                if estado_root.shape[1] >= 13:
                    pose_root = estado_root[:, :7]
                    velocidad_root = estado_root[:, 7:13]
            return pose_root, velocidad_root

        def _reiniciar_entornos(self, indices_entornos: torch.Tensor | None = None) -> None:
            if indices_entornos is None:
                indices_entornos = torch.arange(self.num_entornos, device=self.dispositivo, dtype=torch.long)

            pose_root, velocidad_root = self._estado_root_por_defecto()
            pose_root = pose_root[indices_entornos]
            velocidad_root = velocidad_root[indices_entornos]

            posicion_articulaciones = self.pose_nominal_articulaciones.unsqueeze(0).repeat(indices_entornos.numel(), 1)
            velocidad_articulaciones = torch.zeros_like(posicion_articulaciones)

            if hasattr(self.robot, "write_root_pose_to_sim"):
                self.robot.write_root_pose_to_sim(pose_root, env_ids=indices_entornos)
            if hasattr(self.robot, "write_root_velocity_to_sim"):
                self.robot.write_root_velocity_to_sim(velocidad_root, env_ids=indices_entornos)
            if hasattr(self.robot, "write_joint_state_to_sim"):
                self.robot.write_joint_state_to_sim(
                    posicion_articulaciones,
                    velocidad_articulaciones,
                    env_ids=indices_entornos,
                )

            self.accion_anterior[indices_entornos] = 0.0
            self.contador_pasos[indices_entornos] = 0
            self.reiniciar_comandos(indices_entornos)

            self.robot.reset(env_ids=indices_entornos)
            self.robot.set_joint_position_target(
                posicion_articulaciones,
                env_ids=indices_entornos,
            )
            self.robot.write_data_to_sim()

        def _mantener_pose_actual_durante_pasos(self, num_pasos_espera: int) -> None:
            """Mantiene la pose nominal unos pasos para que en GUI se vea claro el spawn inicial."""

            objetivo = self.pose_nominal_articulaciones.unsqueeze(0).repeat(self.num_entornos, 1)
            for _ in range(num_pasos_espera):
                self.robot.set_joint_position_target(objetivo)
                self.robot.write_data_to_sim()

                if self._sim_step_tiene_render:
                    self.simulacion.step(render=not argumentos.headless)
                else:
                    self.simulacion.step()
                    if not argumentos.headless:
                        aplicacion_simulacion.update()

                self.robot.update(self.dt)
                self.tiempo_acumulado += self.dt

        # ------------------------------------------------------------------
        # Política de prueba
        # ------------------------------------------------------------------

        def politica_abierta_de_prueba(self) -> torch.Tensor:
            """
            Produce una acción oscilatoria pequeña para verificar la tubería.

            No pretende ser una política buena de caminata.
            Solo busca excitar ligeramente caderas, rodillas y tobillos.
            """

            accion = torch.zeros(
                (self.num_entornos, self.num_articulaciones),
                dtype=torch.float32,
                device=self.dispositivo,
            )

            fase = 2.0 * math.pi * self.configuracion_entorno.frecuencia_marcha_prueba_hz * self.tiempo_acumulado
            seno = math.sin(fase)
            seno_opuesto = math.sin(fase + math.pi)
            amplitud = self.configuracion_entorno.amplitud_marcha_prueba

            if "left_hip_pitch_04" in self.indices_marcha_prueba:
                accion[:, self.indices_marcha_prueba["left_hip_pitch_04"]] = amplitud * seno
            if "right_hip_pitch_04" in self.indices_marcha_prueba:
                accion[:, self.indices_marcha_prueba["right_hip_pitch_04"]] = amplitud * seno_opuesto
            if "left_knee_04" in self.indices_marcha_prueba:
                accion[:, self.indices_marcha_prueba["left_knee_04"]] = -0.8 * amplitud * seno
            if "right_knee_04" in self.indices_marcha_prueba:
                accion[:, self.indices_marcha_prueba["right_knee_04"]] = -0.8 * amplitud * seno_opuesto
            if "left_ankle_02" in self.indices_marcha_prueba:
                accion[:, self.indices_marcha_prueba["left_ankle_02"]] = -0.4 * amplitud * seno
            if "right_ankle_02" in self.indices_marcha_prueba:
                accion[:, self.indices_marcha_prueba["right_ankle_02"]] = -0.4 * amplitud * seno_opuesto

            return accion.clamp(-1.0, 1.0)

        # ------------------------------------------------------------------
        # Paso principal
        # ------------------------------------------------------------------

        def paso(self, accion_politica: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            accion_politica = accion_politica.to(device=self.dispositivo, dtype=torch.float32)
            accion_politica = accion_politica * self.configuracion_entorno.escala_accion_politica

            objetivo_posicion_articulaciones = crear_objetivo_posicion_articulaciones(
                accion_politica,
                self.configuracion_caminata,
            )

            self.robot.set_joint_position_target(objetivo_posicion_articulaciones)
            self.robot.write_data_to_sim()

            if self._sim_step_tiene_render:
                self.simulacion.step(render=not argumentos.headless)
            else:
                self.simulacion.step()
                if not argumentos.headless:
                    aplicacion_simulacion.update()

            self.robot.update(self.dt)
            self.tiempo_acumulado += self.dt
            self.contador_pasos += 1

            estado = self.construir_estado_robot()
            observacion = construir_observacion(
                estado,
                self.velocidad_objetivo_x,
                self.configuracion_caminata,
            )
            recompensa, componentes_recompensa = calcular_recompensa(
                estado,
                accion_politica,
                self.velocidad_objetivo_x,
                self.configuracion_caminata,
            )
            termino, detalles_terminacion = calcular_terminacion(
                estado,
                self.configuracion_caminata,
            )
            recompensa_final = aplicar_penalizacion_por_caida(
                recompensa,
                termino,
                self.configuracion_caminata,
            )

            info = {
                **componentes_recompensa,
                **detalles_terminacion,
                "velocidad_objetivo_x": self.velocidad_objetivo_x.clone(),
                "altura_base": estado.altura_base.clone(),
                "velocidad_lineal_base_x": estado.velocidad_lineal_base_cuerpo[:, 0].clone(),
            }

            if torch.any(termino):
                indices_terminados = torch.nonzero(termino, as_tuple=False).squeeze(-1)
                self._reiniciar_entornos(indices_terminados)

            if self.configuracion_entorno.intervalo_reinicio_comando > 0:
                hay_reinicio = (self.contador_pasos % self.configuracion_entorno.intervalo_reinicio_comando) == 0
                if torch.any(hay_reinicio):
                    indices_reinicio = torch.nonzero(hay_reinicio, as_tuple=False).squeeze(-1)
                    self.reiniciar_comandos(indices_reinicio)

            self.accion_anterior = accion_politica.detach().clone()
            return observacion, recompensa_final, termino, info

        # ------------------------------------------------------------------
        # Utilidad
        # ------------------------------------------------------------------

        @property
        def _sim_step_tiene_render(self) -> bool:
            try:
                return "render" in inspect.signature(self.simulacion.step).parameters
            except (TypeError, ValueError):
                return False


    # -------------------------------------------------------------------------
    # Programa principal de prueba
    # -------------------------------------------------------------------------

    configuracion_entorno = ConfiguracionEntornoCaminarBasico(
        num_entornos=argumentos.num_entornos,
        intervalo_reinicio_comando=argumentos.intervalo_reinicio_comando,
    )
    configuracion_robot = ConfiguracionKBotBoxTop(
        usd_tiene_entorno_embebido=False,
    )

    print("[DEBUG] Programa: construyendo entorno...", flush=True)
    entorno = EntornoKBotWalkBasico(configuracion_entorno, configuracion_robot)
    print("[DEBUG] Programa: entorno construido.", flush=True)

    print("Cuerpos:", entorno.nombres_cuerpos, flush=True)
    print("Articulaciones:", entorno.nombres_articulaciones, flush=True)
    print("Indices pies:", entorno.indices_cuerpos_pies, flush=True)
    print("Pose nominal preservada desde default_joint_pos + offsets.", flush=True)
    print(
        "PD implicito preservado:",
        configuracion_robot.actuadores.rigidez,
        configuracion_robot.actuadores.amortiguamiento,
        flush=True,
    )
    print("USD:", configuracion_robot.ruta_usd, flush=True)
    print("Espera inicial (pasos):", argumentos.pasos_espera_inicial, flush=True)

    if argumentos.num_pasos > 0:
        print(
            f"[INFO] Demo finita: se ejecutaran {argumentos.num_pasos} pasos y luego terminara.",
            flush=True,
        )
        iterador_pasos: Iterable[int] = range(argumentos.num_pasos)
    else:
        print("[INFO] Demo continua: num-pasos <= 0, correra hasta que cierres la ventana.", flush=True)

        def _iterador_continuo() -> Iterable[int]:
            paso = 0
            while True:
                yield paso
                paso += 1

        iterador_pasos = _iterador_continuo()

    for paso in iterador_pasos:
        if configuracion_entorno.usar_politica_prueba:
            accion = entorno.politica_abierta_de_prueba()
        else:
            accion = torch.zeros(
                (entorno.num_entornos, entorno.num_articulaciones),
                dtype=torch.float32,
                device=entorno.dispositivo,
            )

        observacion, recompensa, termino, info = entorno.paso(accion)

        if (paso % 100 == 0) or (paso == argumentos.num_pasos - 1):
            print(
                f"paso={paso:04d} "
                f"obs={tuple(observacion.shape)} "
                f"recompensa_media={float(recompensa.mean()): .4f} "
                f"vx_obj_media={float(info['velocidad_objetivo_x'].mean()): .3f} "
                f"vx_real_media={float(info['velocidad_lineal_base_x'].mean()): .3f} "
                f"altura_media={float(info['altura_base'].mean()): .3f} "
                f"terminos={int(termino.sum())}",
                flush=True,
            )


    if (not argumentos.headless) and argumentos.mantener_abierto:
        print("Modo GUI: manteniendo la ventana abierta. Cierra Isaac Lab manualmente cuando termines.", flush=True)
        while aplicacion_simulacion.is_running():
            if entorno._sim_step_tiene_render:
                entorno.simulacion.step(render=True)
            else:
                aplicacion_simulacion.update()

finally:
    try:
        aplicacion_simulacion.close()
    except Exception:
        pass
