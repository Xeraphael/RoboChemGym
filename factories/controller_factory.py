from typing import Dict, Type
from controllers.base_controller import BaseController
from controllers.open_controller import OpenTaskController
from controllers.pickpour_controller import PickPourTaskController
from controllers.placepress_controller import PlacePressTaskController
from controllers.pick_controller import PickTaskController
from controllers.pour_controller import PourTaskController
from controllers.place_controller import PlaceTaskController
from controllers.press_controller import PressTaskController
from controllers.shake_controller import ShakeTaskController
from controllers.stir_controller import StirTaskController
from controllers.stirglassrod_controller import StirGlassrodTaskController
from controllers.pickplace_controller import PickPlaceTaskController
from controllers.shakebeaker_controller import ShakeBeakerTaskController
from controllers.cleanbeaker_controller import CleanBeakerTaskController
from controllers.cleanbeaker7policy_controller import CleanBeaker7PolicyTaskController
from controllers.device_operate_controller import DeviceOperateController
from controllers.opentransportpour_controller import OpenTransportPourController
from controllers.LiquidMixing_controller import LiquidMixingController
from controllers.close_controller import CloseTaskController
from controllers.openclose_controller import OpenCloseTaskController
from controllers.grasp_controller import GraspObjectTaskController
from controllers.door_pick_pour_controller import DoorPickPourTaskController
from controllers.benzoic_acid_synthesis_controller import BenzoicAcidSynthesisController
from controllers.synthesize_controller import SynthesizeController
from controllers.benzoic_acid_dissolution_controller import BenzoicAcidDissolutionController
from controllers.beaker_pick_controller import BeakerPickTaskController
from controllers.group_beaker_scale_controller import GroupBeakerScaleController
from controllers.beaker_flask_experiment_controller import BeakerFlaskExperimentController
from controllers.plan_executor import PlanExecutorController
from controllers.policy_controller import PolicyController
_controller_registry: Dict[str, Type[BaseController]] = {}

def register_controller(name: str, controller_class: Type[BaseController]):
    _controller_registry[name] = controller_class

def create_controller(controller_name: str, *args, **kwargs) -> BaseController:
    """
    创建controller实例
    
    如果设置了 AGENT_MONITOR_MODE 环境变量，自动包装监控器
    """
    import os
    
    if controller_name not in _controller_registry:
        raise ValueError(f": {controller_name}")
    
    # 创建原始 controller
    controller = _controller_registry[controller_name](*args, **kwargs)
    
    # 检查是否启用监控模式
    if os.getenv("AGENT_MONITOR_MODE") == "true":
        log_file = os.getenv("AGENT_LOG_FILE")
        if log_file:
            try:
                import sys
                from pathlib import Path
                
                # 添加 agent/action 目录到路径
                agent_action_path = Path(__file__).parent.parent / "agent" / "action"
                if str(agent_action_path) not in sys.path:
                    sys.path.insert(0, str(agent_action_path))
                
                from monitoring.execution_monitor import ExecutionMonitor
                
                print(f"[ControllerFactory] Wrapping controller with ExecutionMonitor")
                print(f"[ControllerFactory] Log file: {log_file}")
                
                controller = ExecutionMonitor(
                    controller=controller,
                    log_file=log_file,
                    frame_interval=10,
                    enable_verification=True,
                    strict_mode=True
                )
            except Exception as e:
                print(f"[ControllerFactory] Warning: Failed to wrap monitor: {e}")
    
    return controller

register_controller("pickpour", PickPourTaskController)
register_controller("open", OpenTaskController)
register_controller("close", CloseTaskController)
register_controller("openclose", OpenCloseTaskController)
register_controller("pick", PickTaskController)
register_controller("pour", PourTaskController)
register_controller("place", PlaceTaskController)
register_controller("pickplace", PickPlaceTaskController)
register_controller("placepress", PlacePressTaskController)
register_controller("press", PressTaskController)
register_controller("shake", ShakeTaskController)
register_controller("stir", StirTaskController)
register_controller("stirglassrod", StirGlassrodTaskController)
register_controller("shakebeaker", ShakeBeakerTaskController)
register_controller("cleanbeaker", CleanBeakerTaskController)
register_controller("cleanbeaker7policy", CleanBeaker7PolicyTaskController)
register_controller("device_operate", DeviceOperateController)
register_controller("OpenTransportPour", OpenTransportPourController)
register_controller("LiquidMixing", LiquidMixingController)
register_controller("grasp", GraspObjectTaskController)
register_controller("door_pick_pour", DoorPickPourTaskController)
register_controller("benzoic_acid_synthesis_experiment", BenzoicAcidSynthesisController)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
register_controller("synthesize_experiment", SynthesizeController)
register_controller("benzoic_acid_dissolution_experiment", BenzoicAcidDissolutionController)
register_controller("beaker_pick", BeakerPickTaskController)
register_controller("group_beaker_scale_experiment", GroupBeakerScaleController)
register_controller("beaker_flask_experiment", BeakerFlaskExperimentController)
register_controller("plan_executor", PlanExecutorController)
register_controller("policy", PolicyController)
