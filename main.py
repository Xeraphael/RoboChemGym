import os
import argparse
from pathlib import Path

from utils.isaacsim_runtime import prepare_isaacsim_argv


def _resolve_config_dir(config_dir, *, repository_root=None):
    repository_root = (
        Path(__file__).resolve().parent
        if repository_root is None
        else Path(repository_root)
    )
    if config_dir is None:
        return (repository_root / "config").resolve()
    config_path = Path(config_dir)
    if config_path.is_absolute():
        return config_path
    return (Path.cwd() / config_path).resolve()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    parser.add_argument('--config-name', type=str, default='Level2_Protocol1',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', type=str, default=None,
                       help='Configuration file directory')
    return parser.parse_known_args()

# Get command line arguments
args, kit_args = parse_args()
prepare_isaacsim_argv(kit_args)

from isaacsim import SimulationApp

# Set up simulation app based on arguments
simulation_config = {"headless": args.headless}
simulation_app = SimulationApp(simulation_config)

import hydra
from omegaconf import OmegaConf
import cv2
import numpy as np

import omni
import omni.physx
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller
from data_collectors.attempt_video_recorder import (
    AttemptVideoConfig,
    AttemptVideoRecorder,
)

def main():
    config_dir = _resolve_config_dir(args.config_dir)
    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        job_name=args.config_name,
        version_base=None,
    ):
        cfg = hydra.compose(config_name=args.config_name)
    if cfg.controller_type in {"plan_executor", "policy"}:
        omni.timeline.get_timeline_interface().set_end_time(3600.0)
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")

    # Set backend based on command line arguments
    if args.backend == 'gpu':
        world = World(stage_units_in_meters=1, device="cpu")
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
    else:
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    
    video_mapping = None
    if "collector" in cfg and "video" in cfg.collector:
        video_mapping = OmegaConf.to_container(cfg.collector.video, resolve=True)
    attempt_video_config = AttemptVideoConfig.from_mapping(video_mapping)
    attempt_video_recorder = None
    if attempt_video_config.enabled and not args.no_video:
        camera_configs = OmegaConf.to_container(cfg.cameras, resolve=True)
        attempt_video_recorder = AttemptVideoRecorder(
            Path(cfg.multi_run.run_dir) / "videos",
            camera_configs,
            attempt_video_config,
        )

    show_video = not args.no_video and not args.headless
    save_video = show_video and attempt_video_recorder is None

    # 机器人位姿：位置必填，方向可选（如 cfg.robot.orientation 存在则使用）
    robot_kwargs = {
        "position": np.array(cfg.robot.position)
    }
    if "orientation" in cfg.robot:
        robot_kwargs["orientation"] = np.array(cfg.robot.orientation)

    robot = create_robot(
        cfg.robot.type,
        **robot_kwargs
    )
    
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )
    if cfg.mode == "collect":
        task.episode_index = task_controller.episode_num() - 1

    video_writer = None
    try:
        task.reset()
        start_collection_episode = getattr(
            task_controller, "start_collection_episode", None
        )
        if callable(start_collection_episode):
            start_collection_episode(task.episode_metadata())
        start_evaluation_episode = getattr(
            task_controller, "start_evaluation_episode", None
        )
        if callable(start_evaluation_episode):
            start_evaluation_episode(task.episode_metadata())
        if attempt_video_recorder is not None:
            attempt_video_recorder.start_attempt()

        while simulation_app.is_running():
            world.step(render=True)

            if world.is_stopped():
                abort = getattr(task_controller, "abort", None)
                if callable(abort):
                    abort(
                        "PHYSICS_STOPPED",
                        "Isaac physics stopped before the task completed",
                    )
                if attempt_video_recorder is not None:
                    attempt_video_recorder.finish(success=False)
                task.on_task_complete(False)
                break

            if world.is_playing():
                if task_controller.need_reset() or task.need_reset():
                    if not task_controller.need_reset():
                        abort = getattr(task_controller, "abort", None)
                        if callable(abort):
                            abort(
                                "TASK_RESET",
                                "task requested reset before controller completion",
                            )
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    if attempt_video_recorder is not None:
                        attempt_video_recorder.abort()

                    if task_controller.episode_num() >= cfg.max_episodes:
                        break

                    task_controller.reset()
                    if task_controller.episode_num() >= cfg.max_episodes:
                        break
                    task.reset()
                    if callable(start_collection_episode):
                        start_collection_episode(task.episode_metadata())
                    if callable(start_evaluation_episode):
                        start_evaluation_episode(task.episode_metadata())
                    if attempt_video_recorder is not None:
                        attempt_video_recorder.start_attempt()

                    continue

                state = task.step()
                if state is None:
                    continue

                if attempt_video_recorder is not None:
                    attempt_video_recorder.capture(state.get("camera_data", {}))

                action, done, is_success = task_controller.step(state)
                if action is not None:
                    record_applied_action = getattr(
                        task_controller, "record_applied_action", None
                    )
                    if callable(record_applied_action):
                        record_applied_action(state, action)
                    robot.get_articulation_controller().apply_action(action)
                if done:
                    finalize_collection_episode = getattr(
                        task_controller, "finalize_collection_episode", None
                    )
                    if callable(finalize_collection_episode):
                        finalize_collection_episode()
                    if attempt_video_recorder is not None:
                        attempt_video_recorder.finish(success=is_success)
                    task.on_task_complete(is_success)
                    continue

                if save_video or show_video:
                    camera_images = []
                    for _, image_data in state['camera_display'].items():
                        display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                        camera_images.append(display_img)

                    if camera_images:
                        combined_img = np.hstack(camera_images)
                        total_width = 0
                        for idx, img in enumerate(camera_images):
                            label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                            cv2.putText(combined_img, label, (total_width + 2, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                            total_width += img.shape[1]
                        if show_video:
                            cv2.imshow('Camera Views', combined_img)
                            cv2.waitKey(1)
                        if save_video:
                            output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                            if video_writer is None:
                                height, width = combined_img.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                            video_writer.write(combined_img)
    finally:
        if video_writer is not None:
            video_writer.release()
        try:
            if attempt_video_recorder is not None:
                attempt_video_recorder.close()
        finally:
            try:
                task_controller.close()
            finally:
                simulation_app.close()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
