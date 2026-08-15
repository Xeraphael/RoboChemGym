from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PressZController(BaseController):
    """
    A vertical pressing state machine controller.
    
    This controller handles the process of pressing a button vertically, including the following phases:
    - Phase 0: Move end effector above the target object (5cm above button).
    - Phase 1: (Reserved for future use)
    - Phase 2: Close the gripper.
    - Phase 3: Press downward along Z-axis (5cm down to button position).
    - Phase 4: Complete the sequence.
    
    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that returns ArticulationAction.
        gripper (Gripper, optional): Controller for opening/closing the gripper.
        press_distance (float, optional): Distance to press downward, defaults to 0.05 meters (5cm).
        events_dt (list of float, optional): Duration for each phase, defaults to [0.005, 0.005, 0.1, 0.01, 0.01].
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        press_distance: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
    ) -> None:
        # Initialize parent BaseController
        BaseController.__init__(self, name=name)
        
        self._event = 0  # Current phase number
        self._t = 0  # Current phase time counter
        self._press_distance = (press_distance if press_distance is not None else 0.05) / get_stage_units()
        # Press distance, default 0.05 meters (5cm) - distance to press downward along Z-axis (adjusted by stage units)
        self._above_offset_z = 0.05 / get_stage_units()  # Height offset above object for Phase 0, default 0.05 meters (5cm)
        
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
    ) -> ArticulationAction:
        """
        Execute one step of the vertical pressing action.
        
        Args:
            target_position (np.ndarray): Target pressing position.
            current_joint_positions (np.ndarray): Current robot joint positions.
            gripper_control: Gripper controller.
            gripper_position (np.ndarray): Current gripper position for position-based phase transitions.
            end_effector_orientation (np.ndarray, optional): End effector orientation.
        
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
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2 - np.radians(10), 0]))
        
        # Execute the current phase action
        if self._event == 0:
            # Phase 0: Move end effector above the target object (5cm above button)
            above_position = target_position.copy()
            above_position[2] += self._press_distance  # Configured height above button
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
            # Phase 1: Reserved (can be used for additional positioning if needed)
            # Skip to next phase immediately
            self._event += 1
            self._t = 0
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
                
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
            # Phase 3: Press downward along Z-axis (5cm down from Phase 0 position to button position)
            press_target = target_position.copy()
            # Keep X and Y the same as Phase 0 position to ensure pure Z-axis movement
            if 0 in self._target_positions:
                # Use Phase 0's X and Y coordinates, only change Z to button position
                press_target[0] = self._target_positions[0][0]  # Keep X from Phase 0
                press_target[1] = self._target_positions[0][1]  # Keep Y from Phase 0
            # Z coordinate is target_position[2] (button position), which is 5cm down from Phase 0
            self._target_positions[3] = press_target.copy()
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=press_target,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # Check if reached target position (position-based phase transition)
            # Only check Z-axis distance for pure Z-axis movement
            z_distance = abs(gripper_position[2] - press_target[2])
            if z_distance < self._position_threshold:
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
        press_distance: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None
    ) -> None:
        """
        Reset the state machine to initial state.
        
        Args:
            press_distance (float, optional): New press distance.
            events_dt (list of float, optional): New list of phase durations.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if press_distance is not None:
            self._press_distance = press_distance / get_stage_units()
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
