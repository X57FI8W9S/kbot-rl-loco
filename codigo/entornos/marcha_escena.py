from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext


def crear_simulacion(entorno) -> None:
    cfg_sim = sim_utils.SimulationCfg(dt=entorno.cfg.dt_simulacion, device=entorno.cfg.dispositivo)
    entorno.simulation = SimulationContext(cfg_sim)
    entorno.dt = cfg_sim.dt
    entorno.simulation.set_camera_view([6, 6, 4], [0, 0, 0])


def crear_escena(entorno) -> None:
    floor_cfg = sim_utils.GroundPlaneCfg(usd_path=entorno.cfg.ruta_usd_piso)
    floor_cfg.func("/World/piso", floor_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=entorno.cfg.intensidad_luz)
    light_cfg.func("/World/luz", light_cfg)

    entorno.scene = InteractiveScene(
        InteractiveSceneCfg(
            num_envs=entorno.cfg.num_entornos,
            env_spacing=entorno.cfg.espaciado_entornos,
            replicate_physics=True,
            clone_in_fabric=entorno.cfg.usar_clone_en_fabric,
        )
    )

    articulation_cfg = ArticulationCfg(
        prim_path=f"{entorno.scene.env_regex_ns}/KBotBoxTop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=entorno.cfg_robot.ruta_usd,
            activate_contact_sensors=entorno.cfg_robot.activar_sensores_contacto,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=entorno.cfg_robot.posicion_inicial_root),
        actuators={
            "todos": ImplicitActuatorCfg(
                joint_names_expr=list(entorno.cfg_robot.actuadores.expresiones_nombres_articulaciones),
                effort_limit_sim=entorno.cfg_robot.actuadores.limite_esfuerzo_sim,
                stiffness=entorno.cfg_robot.actuadores.rigidez,
                damping=entorno.cfg_robot.actuadores.amortiguamiento,
            )
        },
    )

    entorno.robot = Articulation(articulation_cfg)
    entorno.scene.articulations["robot"] = entorno.robot
    entorno.scene.clone_environments(copy_from_source=False)

    if entorno.device.type == "cpu":
        entorno.scene.filter_collisions(global_prim_paths=["/World/piso"])
