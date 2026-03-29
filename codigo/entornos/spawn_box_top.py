#!/usr/bin/env python3
"""
spawn_box_top.py

Spawn de prueba para KBot box-top.
"""

import argparse
import inspect

from isaaclab.app import AppLauncher

from configuraciones.kbot_box_top import OFFSETS_POSE_BOX_TOP, ConfiguracionKBotBoxTop

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext

    configuracion_robot = ConfiguracionKBotBoxTop()

    # --- sim ---
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    dt = sim_cfg.dt
    sim.set_camera_view([3, 3, 2], [0, 0, 1])

    # --- ground + light ---
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=3000)
    light_cfg.func("/World/light", light_cfg)

    # --- robot cfg ---
    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=configuracion_robot.ruta_usd,
            activate_contact_sensors=configuracion_robot.activar_sensores_contacto,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=configuracion_robot.posicion_inicial_root),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=list(configuracion_robot.actuadores.expresiones_nombres_articulaciones),
                effort_limit_sim=configuracion_robot.actuadores.limite_esfuerzo_sim,
                stiffness=configuracion_robot.actuadores.rigidez,
                damping=configuracion_robot.actuadores.amortiguamiento,
            )
        },
    )

    robot = Articulation(cfg.replace(prim_path="/World/KBotBoxTop"))

    # Reset sim AFTER creating the robot/scene
    sim.reset()

    print("Bodies:", robot.data.body_names)
    print("Joints:", robot.data.joint_names)

    # --- targets ---
    qpos_target = robot.data.default_joint_pos.clone()

    for j, name in enumerate(robot.data.joint_names):
        if name in OFFSETS_POSE_BOX_TOP:
            qpos_target[:, j] += OFFSETS_POSE_BOX_TOP[name]

    # detect whether sim.step(render=...) exists (depends on IsaacLab build)
    try:
        STEP_HAS_RENDER = "render" in inspect.signature(sim.step).parameters
    except (TypeError, ValueError):
        STEP_HAS_RENDER = False

    # --- loop ---
    for i in range(2000):
        robot.set_joint_position_target(qpos_target)
        robot.write_data_to_sim()

        if STEP_HAS_RENDER:
            sim.step(render=not args.headless)
        else:
            sim.step()
            # Yield to the UI so it doesn't *look* frozen in non-headless mode
            if not args.headless:
                simulation_app.update()

        robot.update(dt)

        if (i % 100 == 0) or (i == 1999):
            print("step", i)

finally:
    try:
        simulation_app.close()
    except Exception:
        pass
