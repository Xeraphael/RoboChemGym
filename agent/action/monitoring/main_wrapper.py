#!/usr/bin/env python3
"""
Main Wrapper - 监控模式启动包装器

这个脚本包装 main.py，在不修改任何现有代码的情况下启用监控功能。
它会动态注入监控器到 controller 的创建过程中。

使用方式：
    python agent/action/monitoring/main_wrapper.py --config-name level1_pick
    
环境变量：
    AGENT_MONITOR_MODE=true  # 启用监控
    AGENT_LOG_FILE=path.jsonl  # 日志文件路径
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    print("[MainWrapper] Starting main.py with monitoring support...")
    
    # 重要：必须先导入并执行 main，让 SimulationApp 先创建
    # 然后才能包装 controller_factory（因为它会导入 omni 模块）
    
    # 导入 main 模块
    import main as original_main
    
    # 保存原始的 main 函数
    original_main_func = original_main.main
    
    def wrapped_main():
        """包装后的 main 函数"""
        # 先执行 main 的前半部分（创建 SimulationApp）
        # 通过 monkey patch create_controller 来注入监控
        
        # 等待 SimulationApp 创建后，再包装 controller factory
        import sys
        original_argv = sys.argv.copy()
        
        try:
            # 启动原始 main，但在 create_controller 被调用前拦截
            from factories import controller_factory
            
            # 检查是否需要监控
            if os.getenv("AGENT_MONITOR_MODE") == "true":
                original_create_controller = controller_factory.create_controller
                
                def monitored_create_controller(controller_name: str, *args, **kwargs):
                    """包装后的 create_controller"""
                    controller = original_create_controller(controller_name, *args, **kwargs)
                    
                    log_file = os.getenv("AGENT_LOG_FILE")
                    if log_file:
                        # 延迟导入 ExecutionMonitor（此时 SimulationApp 已创建）
                        sys.path.insert(0, str(Path(__file__).parent.parent))
                        from monitoring.execution_monitor import ExecutionMonitor
                        
                        print(f"[MainWrapper] Wrapping controller with ExecutionMonitor")
                        print(f"[MainWrapper] Log file: {log_file}")
                        
                        controller = ExecutionMonitor(
                            controller=controller,
                            log_file=log_file,
                            frame_interval=10,
                            enable_verification=True,
                            strict_mode=True
                        )
                    
                    return controller
                
                controller_factory.create_controller = monitored_create_controller
                print("[MainWrapper] Controller factory wrapped")
            
            # 执行原始 main
            original_main_func()
            
        finally:
            sys.argv = original_argv
    
    # 替换 main 函数
    original_main.main = wrapped_main
    
    # 运行包装后的 main
    original_main.main()


if __name__ == "__main__":
    main()

