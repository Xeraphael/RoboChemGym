"""LabUtopia extensions for Isaac Sim's installed Franka robot class."""

from typing import List, Optional

import numpy as np
from omni.isaac.franka import Franka as IsaacFranka
from omni.isaac.sensor import Camera, ContactSensor

from utils.object_utils import ObjectUtils


class Franka(IsaacFranka):
    """Add LabUtopia contact sensors and a wrist camera to Isaac's Franka."""

    def __init__(
        self,
        prim_path: str = "/World/Franka",
        name: str = "franka",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            prim_path=prim_path,
            name=name,
            usd_path=usd_path,
            position=position,
            orientation=orientation,
            end_effector_prim_name=end_effector_prim_name,
            gripper_dof_names=gripper_dof_names,
            gripper_open_position=gripper_open_position,
            gripper_closed_position=gripper_closed_position,
            deltas=deltas,
        )
        self.prim_path_str = prim_path
        self.gripper.set_action_deltas(np.array([0.03, 0.03]))

        self.left_contact_sensor = ContactSensor(
            prim_path=f"{prim_path}/panda_leftfinger/contact_sensor",
            name="left_finger_contact_sensor",
            min_threshold=0,
            max_threshold=10_000_000,
            radius=0.1,
        )
        self.right_contact_sensor = ContactSensor(
            prim_path=f"{prim_path}/panda_rightfinger/contact_sensor",
            name="right_finger_contact_sensor",
            min_threshold=0,
            max_threshold=10_000_000,
            radius=0.1,
        )
        self.camera = Camera(
            prim_path=f"{prim_path}/panda_hand/arm_camera",
            translation=np.array([-0.2, 0.0, -0.02]),
            frequency=60,
            resolution=(256, 256),
            orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]),
        )
        self.camera.set_local_pose(
            orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]),
            camera_axes="usd",
        )
        self.camera.set_clipping_range(near_distance=0.05)
        self.camera.set_focal_length(1.0)

    def get_contact_sensor(self):
        return self.left_contact_sensor, self.right_contact_sensor

    def get_gripper_position(self) -> np.ndarray:
        return ObjectUtils.get_instance().get_object_xform_position(
            object_path=f"{self.prim_path_str}/panda_hand/tool_center"
        )
