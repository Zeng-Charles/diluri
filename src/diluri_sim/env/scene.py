from __future__ import annotations

import numpy as np

from diluri_sim.config import DualFr3Config, validate_assets


def build_scene(config: DualFr3Config, *, show_viewer: bool | None = None):
    """Build a Genesis scene from config and return (scene, robots dict)."""
    import genesis as gs

    validate_assets(config)
    _init_genesis(gs, config.scene.backend)

    viewer_enabled = config.scene.viewer if show_viewer is None else show_viewer
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.scene.dt, substeps=config.scene.substeps),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=config.scene.camera.pos,
            camera_lookat=config.scene.camera.lookat,
            camera_fov=config.scene.camera.fov,
            enable_gui=True,
        ),
        show_viewer=viewer_enabled,
    )

    scene.add_entity(gs.morphs.Plane())

    robots = {}
    for robot in config.robots:
        robots[robot.name] = scene.add_entity(
            gs.morphs.URDF(
                file=str(robot.asset),
                pos=robot.base_pos,
                euler=robot.base_euler,
                fixed=robot.fixed_base,
                requires_jac_and_IK=True,
            )
        )

    scene.build()

    # Set home positions and flush the viewer state before user code runs.
    # Genesis viewer renders on scene.step(), so without this step the viewer
    # shows q=0 (URDF default) until the first user step — causing a visible jump.
    for robot_cfg in config.robots:
        idx = list(range(len(robot_cfg.home_q)))
        robots[robot_cfg.name].set_dofs_position(
            np.array(robot_cfg.home_q, dtype=float), dofs_idx_local=idx
        )
        robots[robot_cfg.name].set_dofs_velocity(
            np.zeros(len(idx), dtype=float), dofs_idx_local=idx
        )
    scene.step()   # flush: viewer now shows home_q as the first visible frame

    if viewer_enabled:
        from genesis.vis.viewer_plugins import MouseInteractionPlugin
        scene.viewer.add_plugin(MouseInteractionPlugin())

    return scene, robots


def _init_genesis(gs, backend: str) -> None:
    b = backend.lower()
    if b == "gpu":
        gs.init(backend=gs.gpu)
    elif b == "cpu":
        gs.init(backend=gs.cpu)
    else:
        raise ValueError(f"Unsupported Genesis backend: {backend!r}. Use 'cpu' or 'gpu'.")
