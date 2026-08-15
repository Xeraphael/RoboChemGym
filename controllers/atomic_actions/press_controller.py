from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PressController(BaseController):
    """
    A pressing state machine controller.
    
    This controller handles the process of pressing a button, including the following phases:
    - Phase 0: Move end effector above the target object (12cm above button), 7cm behind button along X-axis.
    - Phase 1: Lower end effector to 5cm height, keeping 7cm behind button along X-axis.
    - Phase 2: Close the gripper.
    - Phase 3: Press forward along X-axis (7cm forward to button position).
    - Phase 4: Complete the sequence.
    
    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that returns ArticulationAction.
        gripper (Gripper): Controller for opening/closing the gripper.
        initial_offset (float, optional): Initial offset distance (along X-axis), defaults to 0.07 meters (7cm).
        events_dt (list of float, optional): Duration for each phase, defaults to [0.005, 0.005, 0.1, 0.01, 0.01].
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        end_effector_initial_height: typing.Optional[float] = None,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
    ) -> None:
        # Initialize parent BaseController
        BaseController.__init__(self, name=name)
        
        self._event = 0  # Current phase number
        self._t = 0  # Current phase time counter
        self._initial_offset = (initial_offset if initial_offset is not None else 0.07) / get_stage_units()
        # Initial offset distance, default 0.07 meters (7cm) - distance behind button along X-axis (adjusted by stage units)
        self._pre_offset_z = 0.12 / get_stage_units()  # Height offset above object for Phase 0, default 0.12 meters (12cm)
        self._lower_offset_z = 0.05 / get_stage_units()  # Lower height offset for Phase 1, default 0.05 meters (5cm)
        
        if events_dt is None:
            self._events_dt = [0.005, 0.005, 0.1, 0.01, 0.01]  # Default phase durations for 5 phases
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 5:
                raise Exception("events_dt length must be exactly 5")
        
        self._cspace_controller = cspace_controller  # Store Cartesian space controller
        self._start = True
        self._position_threshold = position_threshold / get_stage_units()  # Position threshold for phase transitions
        self._target_positions = {}  # Store target positions for each phase

    def get_current_event(self) -> int:
        """
        Get the current phase/event of the state machine.

        Returns:
            int: Current phase/event number.
        """
        return self._event
    
    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        press_distance: typing.Optional[float] = None
    ) -> ArticulationAction:
        """
        Execute one step of the pressing action.
        
        Args:
            target_position (np.ndarray): Target pressing position.
            current_joint_positions (np.ndarray): Current robot joint positions.
            gripper_control: Gripper controller.
            gripper_position (np.ndarray): Current gripper position for position-based phase transitions.
            end_effector_orientation (np.ndarray, optional): End effector orientation.
            press_distance (float): Distance to press forward, default 0.07 meters (7cm). Note: This should match initial_offset for consistent behavior.
        
        Returns:
            ArticulationAction: Robot control action.
        """
        
        if self._start:
            # Initial state: Open the gripper
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = 0.04 / get_stage_units()  # Open the gripper
            target_joint_positions[8] = 0.04 / get_stage_units()  # Open the gripper
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if self.is_done():
            # Pause or done state: Maintain current joint positions
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # Execute the current phase action
        if self._event == 0:
            # Phase 0: Move end effector above the target object (12cm above), 7cm behind button along X-axis
            above_position = target_position.copy()
            above_position[0] -= self._initial_offset  # 7cm behind button along X-axis
            above_position[2] += self._pre_offset_z  # 12cm above button
            self._target_positions[0] = above_position.copy()
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=above_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # Check if reached target position (position-based phase transition)
            xy_distance = np.linalg.norm(gripper_position[:2] - above_position[:2])
            z_distance = abs(gripper_position[2] - above_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                    
        elif self._event == 1:
            # Phase 1: Lower end effector to 5cm height, keeping 7cm behind button along X-axis
            approach_position = target_position.copy()
            approach_position[0] -= self._initial_offset  # Keep 7cm behind button along X-axis
            approach_position[2] += self._lower_offset_z  # Lower to 5cm above button
            self._target_positions[1] = approach_position.copy()
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=approach_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # Check if reached target position (position-based phase transition)
            xy_distance = np.linalg.norm(gripper_position[:2] - approach_position[:2])
            z_distance = abs(gripper_position[2] - approach_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
        elif self._event == 2:
            # Phase 2: Close the gripper (time-based, as it's an action not movement)
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.0015 / get_stage_units()  # Default gripper close distance (adjustable)
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            
            # Time-based transition for gripper closing
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
                
        elif self._event == 3:
            # Phase 3: Press forward along X-axis (7cm forward from Phase 1 position to button position)
            press_target = target_position.copy()
            # Keep Y and Z the same as Phase 1 position to ensure pure X-axis movement
            if 1 in self._target_positions:
                # Use Phase 1's Y and Z coordinates, only change X to button position
                press_target[1] = self._target_positions[1][1]  # Keep Y from Phase 1
                press_target[2] = self._target_positions[1][2]  # Keep Z from Phase 1
            # X coordinate is target_position[0] (button position), which is 7cm forward from Phase 1
            self._target_positions[3] = press_target.copy()
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=press_target,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # Check if reached target position (position-based phase transition)
            # Only check X-axis distance for pure X-axis movement
            x_distance = abs(gripper_position[0] - press_target[0])
            if x_distance < self._position_threshold:
                self._event += 1
                self._t = 0
        
        elif self._event == 4:
            # Phase 4: Complete the sequence
            # Mark as done by incrementing event counter
            self._event += 1
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        return target_joint_positions

    
    def reset(
        self,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None
    ) -> None:
        """
        Reset the state machine to initial state.
        
        Args:
            initial_offset (float, optional): New initial offset distance.
            events_dt (list of float, optional): New list of phase durations.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if initial_offset is not None:
            self._initial_offset = initial_offset / get_stage_units()
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 5:
                raise Exception("events_dt length must be exactly 5")
        self._start = True
        self._target_positions = {}  # Reset target positions
    
    def is_done(self) -> bool:
        # Check if the state machine is done
        return self._event >= len(self._events_dt)
