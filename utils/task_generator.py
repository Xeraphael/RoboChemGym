"""
基于图的任务生成器
根据场景中的物体和可用的原子动作，自动生成符合逻辑的复合任务
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import argparse
import re


class ObjectType(Enum):
    """物体类型枚举"""
    CONTAINER = "container"  # 容器：beaker, bottle, cylinder等
    FURNITURE = "furniture"  # 家具：Cabinet, DryingBox, MuffleFurnace
    PLATFORM = "platform"  # 平台：table, target_plat等
    DOOR = "door"  # 门
    DRAWER = "drawer"  # 抽屉
    BUTTON = "button"  # 按钮
    HANDLE = "handle"  # 把手
    JOINT = "joint"  # 关节
    OTHER = "other"  # 其他


class AtomicAction(Enum):
    """原子动作枚举"""
    PICK = "pick"
    PLACE = "place"
    POUR = "pour"
    OPEN = "open"
    CLOSE = "close"
    PRESS = "press"
    PRESSZ = "pressZ"
    SHAKE = "shake"
    STIR = "stir"
    MOVE = "move"


@dataclass
class ObjectInfo:
    """物体信息"""
    path: str
    name: str
    obj_type: str  # USD类型
    position: Tuple[float, float, float]
    obj_type_category: ObjectType
    parent_path: Optional[str] = None
    
    @property
    def short_name(self) -> str:
        """获取简短的物体名称"""
        return self.name.split('/')[-1]
    
    def is_container(self) -> bool:
        """判断是否是容器"""
        name_lower = self.short_name.lower()
        return any(keyword in name_lower for keyword in 
                  ['beaker', 'bottle', 'cylinder', 'flask'])
    
    def is_furniture(self) -> bool:
        """判断是否是家具"""
        name_lower = self.short_name.lower()
        return any(keyword in name_lower for keyword in 
                  ['cabinet', 'dryingbox', 'mufflefurnace', 'box'])
    
    def is_platform(self) -> bool:
        """判断是否是平台"""
        name_lower = self.short_name.lower()
        return any(keyword in name_lower for keyword in 
                  ['table', 'plat', 'platform', 'surface'])
    
    def is_door(self) -> bool:
        """判断是否是门"""
        return 'door' in self.short_name.lower()
    
    def is_drawer(self) -> bool:
        """判断是否是抽屉"""
        return 'drawer' in self.short_name.lower()
    
    def is_button(self) -> bool:
        """判断是否是按钮"""
        return 'button' in self.short_name.lower()
    
    def is_handle(self) -> bool:
        """判断是否是把手"""
        return 'handle' in self.short_name.lower()


class ActionDependencyGraph:
    """动作依赖图"""
    
    def __init__(self):
        # 定义动作的前置条件（prerequisites）
        self.prerequisites: Dict[AtomicAction, Set[AtomicAction]] = {
            AtomicAction.PLACE: {AtomicAction.PICK},  # 放置前需要先抓取
            AtomicAction.POUR: {AtomicAction.PICK},  # 倾倒前需要先抓取
            AtomicAction.SHAKE: {AtomicAction.PICK},  # 摇晃前需要先抓取
            AtomicAction.STIR: {AtomicAction.PICK},  # 搅拌前需要先抓取
            AtomicAction.CLOSE: {AtomicAction.OPEN},  # 关闭前通常需要先打开（可选）
        }
        
        # 定义动作的适用物体类型
        self.action_object_types: Dict[AtomicAction, Set[ObjectType]] = {
            AtomicAction.PICK: {ObjectType.CONTAINER},
            AtomicAction.PLACE: {ObjectType.PLATFORM, ObjectType.FURNITURE},
            AtomicAction.POUR: {ObjectType.CONTAINER},  # 目标物体必须是容器
            AtomicAction.OPEN: {ObjectType.DOOR, ObjectType.DRAWER},
            AtomicAction.CLOSE: {ObjectType.DOOR, ObjectType.DRAWER},
            AtomicAction.PRESS: {ObjectType.BUTTON},
            AtomicAction.PRESSZ: {ObjectType.BUTTON},
            AtomicAction.SHAKE: {ObjectType.CONTAINER},
            AtomicAction.STIR: {ObjectType.CONTAINER},
            AtomicAction.MOVE: {ObjectType.CONTAINER},
        }
        
        # 定义动作序列模板（常见的有意义的组合）
        self.action_templates: Dict[str, List[AtomicAction]] = {
            "pick_place": [AtomicAction.PICK, AtomicAction.PLACE],
            "pick_pour": [AtomicAction.PICK, AtomicAction.POUR],
            "open_pick_place": [AtomicAction.OPEN, AtomicAction.PICK, AtomicAction.PLACE],
            "pick_place_press": [AtomicAction.PICK, AtomicAction.PLACE, AtomicAction.PRESS],
            "open_pick_place_close": [AtomicAction.OPEN, AtomicAction.PICK, AtomicAction.PLACE, AtomicAction.CLOSE],
            "pick_pour_place": [AtomicAction.PICK, AtomicAction.POUR, AtomicAction.PLACE],
            "open_pick_pour_place": [AtomicAction.OPEN, AtomicAction.PICK, AtomicAction.POUR, AtomicAction.PLACE],
        }
    
    def can_execute_after(self, action: AtomicAction, previous_actions: List[AtomicAction]) -> bool:
        """判断动作是否可以在给定前置动作后执行"""
        if action not in self.prerequisites:
            return True
        
        required = self.prerequisites[action]
        previous_set = set(previous_actions)
        
        # 检查是否满足所有前置条件
        return required.issubset(previous_set)
    
    def is_action_applicable_to_object(self, action: AtomicAction, obj: ObjectInfo) -> bool:
        """判断动作是否适用于物体"""
        if action not in self.action_object_types:
            return False
        
        allowed_types = self.action_object_types[action]
        return obj.obj_type_category in allowed_types


class SceneParser:
    """场景解析器"""
    
    def __init__(self, scene_file_path: str):
        self.scene_file_path = scene_file_path
        self.objects: List[ObjectInfo] = []
        self._parse_scene()
    
    def _parse_scene(self):
        """解析场景文件"""
        with open(self.scene_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过头部信息，找到数据开始位置
        data_start = False
        for i, line in enumerate(lines):
            if '路径' in line and '类型' in line and '位置' in line:
                data_start = True
                continue
            
            if not data_start or not line.strip() or line.startswith('-'):
                continue
            
            # 解析每一行：路径 | 类型 | 位置
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            # 重新组合路径（可能包含空格）
            path_parts = []
            type_part = None
            pos_part = None
            
            # 找到类型列（包含特殊字符如Xform, Mesh等）
            for j, part in enumerate(parts):
                if part in ['Xform', 'Mesh', 'PhysicsFixedJoint', 'PhysicsRevoluteJoint', 
                           'PhysicsPrismaticJoint', 'Plane', 'CylinderLight', 'Material',
                           'Shader', 'Scope', 'GeomSubset']:
                    type_part = part
                    path_parts = parts[:j]
                    pos_parts = parts[j+1:]
                    break
            
            if not type_part:
                continue
            
            # 解析路径
            path = ' '.join(path_parts).strip()
            if not path.startswith('/World'):
                continue
            
            # 解析位置 [x, y, z] 或 N/A
            position = None
            if pos_parts and pos_parts[0] != 'N/A':
                pos_str = ' '.join(pos_parts).strip('[]')
                try:
                    coords = [float(x.strip(',')) for x in pos_str.split(',')]
                    if len(coords) >= 3:
                        position = tuple(coords[:3])
                except:
                    pass
            
            # 创建物体信息
            name = path.split('/')[-1]
            obj_type_category = self._classify_object_type(path, name, type_part)
            
            # 获取父路径
            parent_path = None
            if '/' in path and path != '/World':
                parent_path = '/'.join(path.split('/')[:-1])
            
            obj = ObjectInfo(
                path=path,
                name=name,
                obj_type=type_part,
                position=position if position else (0.0, 0.0, 0.0),
                obj_type_category=obj_type_category,
                parent_path=parent_path
            )
            
            self.objects.append(obj)
    
    def _classify_object_type(self, path: str, name: str, usd_type: str) -> ObjectType:
        """分类物体类型"""
        name_lower = name.lower()
        path_lower = path.lower()
        
        # 检查是否是关节或物理对象（跳过）
        if usd_type in ['PhysicsFixedJoint', 'PhysicsRevoluteJoint', 'PhysicsPrismaticJoint']:
            return ObjectType.JOINT
        
        # 检查是否是门
        if 'door' in name_lower or 'door' in path_lower:
            return ObjectType.DOOR
        
        # 检查是否是抽屉
        if 'drawer' in name_lower or 'drawer' in path_lower:
            return ObjectType.DRAWER
        
        # 检查是否是按钮
        if 'button' in name_lower:
            return ObjectType.BUTTON
        
        # 检查是否是把手
        if 'handle' in name_lower or 'nob' in name_lower:
            return ObjectType.HANDLE
        
        # 检查是否是容器
        if any(keyword in name_lower for keyword in ['beaker', 'bottle', 'cylinder', 'flask']):
            return ObjectType.CONTAINER
        
        # 检查是否是家具
        if any(keyword in name_lower for keyword in ['cabinet', 'dryingbox', 'mufflefurnace', 'box']):
            return ObjectType.FURNITURE
        
        # 检查是否是平台
        if any(keyword in name_lower for keyword in ['table', 'plat', 'platform', 'surface']):
            return ObjectType.PLATFORM
        
        return ObjectType.OTHER
    
    def get_containers(self) -> List[ObjectInfo]:
        """获取所有容器"""
        return [obj for obj in self.objects if obj.obj_type_category == ObjectType.CONTAINER]
    
    def get_furniture(self) -> List[ObjectInfo]:
        """获取所有家具"""
        return [obj for obj in self.objects if obj.obj_type_category == ObjectType.FURNITURE]
    
    def get_platforms(self) -> List[ObjectInfo]:
        """获取所有平台"""
        return [obj for obj in self.objects if obj.obj_type_category == ObjectType.PLATFORM]
    
    def get_doors(self) -> List[ObjectInfo]:
        """获取所有门"""
        return [obj for obj in self.objects if obj.obj_type_category == ObjectType.DOOR]
    
    def get_drawers(self) -> List[ObjectInfo]:
        """获取所有抽屉"""
        return [obj for obj in self.objects if obj.obj_type_category == ObjectType.DRAWER]
    
    def get_buttons(self) -> List[ObjectInfo]:
        """获取所有按钮"""
        return [obj for obj in self.objects if obj.obj_type_category == ObjectType.BUTTON]


class TaskGenerator:
    """任务生成器"""
    
    def __init__(self, scene_parser: SceneParser, dependency_graph: ActionDependencyGraph):
        self.scene_parser = scene_parser
        self.dependency_graph = dependency_graph
    
    def generate_tasks(self, max_tasks: int = 20) -> List[Dict]:
        """生成任务列表"""
        tasks = []
        
        containers = self.scene_parser.get_containers()
        platforms = self.scene_parser.get_platforms()
        furniture = self.scene_parser.get_furniture()
        doors = self.scene_parser.get_doors()
        drawers = self.scene_parser.get_drawers()
        buttons = self.scene_parser.get_buttons()
        
        # 1. 简单的pick_place任务
        for container in containers[:3]:  # 限制数量
            for platform in platforms[:2]:
                task = self._create_pick_place_task(container, platform)
                if task:
                    tasks.append(task)
        
        # 2. pick_pour任务
        for container1 in containers[:2]:
            for container2 in containers[:2]:
                if container1.path != container2.path:
                    task = self._create_pick_pour_task(container1, container2)
                    if task:
                        tasks.append(task)
        
        # 3. open_pick_place任务（从家具中取出容器）
        for furniture_obj in furniture[:2]:
            for container in containers[:2]:
                task = self._create_open_pick_place_task(furniture_obj, container, platforms[0] if platforms else None)
                if task:
                    tasks.append(task)
        
        # 4. 完整的设备操作任务（open -> pick -> place -> press -> close）
        for furniture_obj in furniture[:1]:
            for container in containers[:1]:
                button = buttons[0] if buttons else None
                task = self._create_device_operation_task(furniture_obj, container, button, platforms[0] if platforms else None)
                if task:
                    tasks.append(task)
        
        return tasks[:max_tasks]
    
    def _create_pick_place_task(self, container: ObjectInfo, platform: ObjectInfo) -> Optional[Dict]:
        """创建pick_place任务"""
        return {
            "name": f"pick_place_{container.short_name}_{platform.short_name}",
            "description": f"抓取{container.short_name}并放置到{platform.short_name}",
            "actions": [
                {"type": "pick", "object": container.path, "object_name": container.short_name},
                {"type": "place", "object": platform.path, "object_name": platform.short_name}
            ],
            "objects": {
                "container": container.path,
                "platform": platform.path
            },
            "task_type": "pick_place"
        }
    
    def _create_pick_pour_task(self, source_container: ObjectInfo, target_container: ObjectInfo) -> Optional[Dict]:
        """创建pick_pour任务"""
        return {
            "name": f"pick_pour_{source_container.short_name}_to_{target_container.short_name}",
            "description": f"抓取{source_container.short_name}并倒入{target_container.short_name}",
            "actions": [
                {"type": "pick", "object": source_container.path, "object_name": source_container.short_name},
                {"type": "pour", "object": target_container.path, "object_name": target_container.short_name},
                {"type": "place", "object": source_container.path, "object_name": source_container.short_name}  # 放回原处
            ],
            "objects": {
                "source_container": source_container.path,
                "target_container": target_container.path
            },
            "task_type": "pick_pour"
        }
    
    def _create_open_pick_place_task(self, furniture: ObjectInfo, container: ObjectInfo, platform: Optional[ObjectInfo]) -> Optional[Dict]:
        """创建open_pick_place任务"""
        if not platform:
            return None
        
        # 找到家具的门或抽屉
        doors = [obj for obj in self.scene_parser.objects 
                if furniture.path in obj.path and (obj.is_door() or obj.is_drawer())]
        
        if not doors:
            return None
        
        door = doors[0]
        furniture_type = "door" if door.is_door() else "drawer"
        
        return {
            "name": f"open_pick_place_{furniture.short_name}_{container.short_name}",
            "description": f"打开{furniture.short_name}，取出{container.short_name}并放置到{platform.short_name}",
            "actions": [
                {"type": "open", "object": door.path, "object_name": door.short_name, "furniture_type": furniture_type},
                {"type": "pick", "object": container.path, "object_name": container.short_name},
                {"type": "place", "object": platform.path, "object_name": platform.short_name}
            ],
            "objects": {
                "furniture": furniture.path,
                "door": door.path,
                "container": container.path,
                "platform": platform.path
            },
            "task_type": "open_pick_place"
        }
    
    def _create_device_operation_task(self, furniture: ObjectInfo, container: ObjectInfo, 
                                      button: Optional[ObjectInfo], platform: Optional[ObjectInfo]) -> Optional[Dict]:
        """创建设备操作任务"""
        if not platform:
            return None
        
        # 找到家具的门和按钮
        doors = [obj for obj in self.scene_parser.objects 
                if furniture.path in obj.path and (obj.is_door() or obj.is_drawer())]
        buttons = [obj for obj in self.scene_parser.objects 
                  if furniture.path in obj.path and obj.is_button()]
        
        if not doors:
            return None
        
        door = doors[0]
        furniture_type = "door" if door.is_door() else "drawer"
        button_obj = button or (buttons[0] if buttons else None)
        
        actions = [
            {"type": "open", "object": door.path, "object_name": door.short_name, "furniture_type": furniture_type},
            {"type": "pick", "object": container.path, "object_name": container.short_name},
            {"type": "place", "object": furniture.path, "object_name": furniture.short_name}  # 放置到设备内
        ]
        
        if button_obj:
            actions.append({"type": "press", "object": button_obj.path, "object_name": button_obj.short_name})
        
        actions.append({"type": "close", "object": door.path, "object_name": door.short_name, "furniture_type": furniture_type})
        
        return {
            "name": f"device_operation_{furniture.short_name}_{container.short_name}",
            "description": f"操作{furniture.short_name}：打开、放入{container.short_name}、{'按压按钮、' if button_obj else ''}关闭",
            "actions": actions,
            "objects": {
                "furniture": furniture.path,
                "door": door.path,
                "container": container.path,
                "platform": platform.path,
                "button": button_obj.path if button_obj else None
            },
            "task_type": "device_operation"
        }
    
    def export_to_yaml(self, tasks: List[Dict], output_file: str):
        """导出任务为YAML格式（简化版）"""
        import yaml
        
        yaml_tasks = []
        for task in tasks:
            yaml_task = {
                "name": task["name"],
                "description": task["description"],
                "task_type": task["task_type"],
                "actions": task["actions"],
                "objects": task["objects"]
            }
            yaml_tasks.append(yaml_task)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_tasks, f, allow_unicode=True, default_flow_style=False)
    
    def export_to_markdown(self, tasks: List[Dict], output_file: str):
        """导出任务为Markdown格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 自动生成的任务列表\n\n")
            f.write(f"共生成 {len(tasks)} 个任务\n\n")
            
            for i, task in enumerate(tasks, 1):
                f.write(f"## 任务 {i}: {task['name']}\n\n")
                f.write(f"**描述**: {task['description']}\n\n")
                f.write(f"**任务类型**: {task['task_type']}\n\n")
                f.write("**动作序列**:\n\n")
                for j, action in enumerate(task['actions'], 1):
                    action_str = action['type']
                    if 'furniture_type' in action:
                        action_str += f" (furniture_type: {action['furniture_type']})"
                    f.write(f"{j}. {action_str}: {action['object_name']} ({action['object']})\n")
                f.write("\n**涉及物体**:\n\n")
                for obj_type, obj_path in task['objects'].items():
                    if obj_path:
                        f.write(f"- {obj_type}: {obj_path}\n")
                f.write("\n---\n\n")


def main():
    """Generate task suggestions from an extracted scene listing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated_tasks.md"))
    args = parser.parse_args()
    scene_file = str(args.scene_file)
    
    print("解析场景文件...")
    scene_parser = SceneParser(scene_file)
    print(f"解析到 {len(scene_parser.objects)} 个物体")
    print(f"- 容器: {len(scene_parser.get_containers())}")
    print(f"- 家具: {len(scene_parser.get_furniture())}")
    print(f"- 平台: {len(scene_parser.get_platforms())}")
    print(f"- 门: {len(scene_parser.get_doors())}")
    print(f"- 抽屉: {len(scene_parser.get_drawers())}")
    print(f"- 按钮: {len(scene_parser.get_buttons())}")
    
    print("\n创建动作依赖图...")
    dependency_graph = ActionDependencyGraph()
    
    print("\n生成任务...")
    task_generator = TaskGenerator(scene_parser, dependency_graph)
    tasks = task_generator.generate_tasks(max_tasks=15)
    
    print(f"\n生成了 {len(tasks)} 个任务")
    
    # 导出为Markdown
    output_md = str(args.output)
    task_generator.export_to_markdown(tasks, output_md)
    print(f"\n任务已导出到: {output_md}")
    
    # 显示前几个任务
    print("\n前3个任务预览:")
    for i, task in enumerate(tasks[:3], 1):
        print(f"\n{i}. {task['name']}")
        print(f"   描述: {task['description']}")
        print(f"   动作序列: {' -> '.join([a['type'] for a in task['actions']])}")


if __name__ == "__main__":
    main()
