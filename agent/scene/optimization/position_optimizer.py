"""
位置优化器：使用贪心算法优化物体位置并结合 LLM 语义约束。
支持：碰撞检测、椭圆范围约束、x轴遮挡优化及 LLM 语义布局。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import copy
import random
import os
import re
import math
from openai import OpenAI

class OptimizationMethod(Enum):
    MILP = "milp"  # 实际为带评分机制的贪心算法

@dataclass
class ObjectInfo:
    prim_path: str
    prim_name: str
    position: List[float]
    bounding_box: Dict[str, List[float]]
    
    @property
    def size(self) -> np.ndarray:
        return np.array(self.bounding_box.get("size", [0, 0, 0]))

@dataclass
class EllipseConstraint:
    """可达区域椭圆约束。"""
    center_x: float
    center_y: float
    semi_major: float
    semi_minor: float
    rotation: float
    
    def contains(self, x: float, y: float) -> bool:
        dx, dy = x - self.center_x, y - self.center_y
        cos_r, sin_r = np.cos(-self.rotation), np.sin(-self.rotation)
        x_rot = dx * cos_r - dy * sin_r
        y_rot = dx * sin_r + dy * cos_r
        return (x_rot / self.semi_major) ** 2 + (y_rot / self.semi_minor) ** 2 <= 1.0

class PositionOptimizer:
    def __init__(self, json_file_path: Path, z_height: float,
                 minimum_spacing: float,
                 ellipse_constraint: EllipseConstraint,
                 excluded_objects: Optional[Set[str]] = None):
        self.json_file_path = Path(json_file_path)
        self.z_height = z_height
        self.minimum_spacing = float(minimum_spacing)
        if (
            not math.isfinite(self.minimum_spacing)
            or self.minimum_spacing <= 0.0
        ):
            raise ValueError("minimum_spacing must be a positive finite number")
        self.ellipse_constraint = ellipse_constraint
        self.excluded_objects = {"table", "lounge_booth_table", "GroundPlane", "FumeHood"}
        if excluded_objects: self.excluded_objects.update(excluded_objects)
        self.excluded_patterns = ["lab_", "Lab_", "LAB_"]
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.scene_data = json.load(f)
        self.objects = self._parse_objects()

    @classmethod
    def from_profile(cls, json_file_path: Path, profile: dict) -> "PositionOptimizer":
        region = profile["reachable_region"]
        return cls(
            json_file_path=json_file_path,
            z_height=float(profile["surface_z"]),
            minimum_spacing=float(profile["minimum_spacing"]),
            ellipse_constraint=EllipseConstraint(
                center_x=float(region["center"][0]),
                center_y=float(region["center"][1]),
                semi_major=float(region["semi_axes"][0]),
                semi_minor=float(region["semi_axes"][1]),
                rotation=float(region["rotation"]),
            ),
        )

    def _parse_objects(self) -> List[ObjectInfo]:
        return [ObjectInfo(p, d["prim_name"], d.get("position", [0,0,0]), d.get("bounding_box", {}))
                for p, d in self.scene_data.items()
                if d.get("prim_name") not in self.excluded_objects and 
                not any(d.get("prim_name", "").startswith(pat) for pat in self.excluded_patterns)]

    def check_collision(self, obj1: ObjectInfo, pos1: np.ndarray, obj2: ObjectInfo, pos2: np.ndarray) -> bool:
        min1, max1 = pos1 - obj1.size/2, pos1 + obj1.size/2
        min2, max2 = pos2 - obj2.size/2, pos2 + obj2.size/2
        return not (np.any(max1 < min2) or np.any(min1 > max2))

    def compute_y_occlusion(self, positions: List[np.ndarray], sizes: List[np.ndarray]) -> float:
        """计算 Y 轴（横向）重叠度。Y 轴重叠意味着物体在前后方向可能互相遮挡。"""
        if len(positions) < 2: return 0.0
        score = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1, s1 = positions[i], sizes[i]
                p2, s2 = positions[j], sizes[j]
                # 计算 Y 轴区间的重叠
                overlap = max(0, min(p1[1]+s1[1]/2, p2[1]+s2[1]/2) - max(p1[1]-s1[1]/2, p2[1]-s2[1]/2))
                if overlap > 0:
                    score += overlap / min(s1[1], s2[1])
        return score

    def optimize_milp(self, grid_resolution: float) -> Optional[Dict[str, List[float]]]:
        if not self.objects: return {}
        print(f"开始贪心优化 {len(self.objects)} 个物体位置...")
        ec = self.ellipse_constraint
        
        # 1. 生成候选点：极度强化 X 轴约束，放宽 Y 轴
        candidates = []
        for x in np.arange(ec.center_x - ec.semi_major - 0.2, ec.center_x + ec.semi_major + 0.2, grid_resolution):
            for y in np.arange(ec.center_y - ec.semi_minor - 0.2, ec.center_y + ec.semi_minor + 0.2, grid_resolution):
                if ec.contains(x, y):
                    # X偏移惩罚极大(5.0)，Y偏移惩罚极小(0.1)，迫使物体沿 Y 轴线性分布
                    center_dist = 5.0 * (x - ec.center_x)**2 + 0.1 * (y - ec.center_y)**2
                    candidates.append(([x, y, self.z_height], center_dist))
        
        candidates.sort(key=lambda x: x[1])
        candidate_positions = [c[0] for c in candidates]

        solution, used_pos, failed = {}, [], []
        ideal_dist = self.minimum_spacing
        
        for obj in self.objects:
            best_pos, min_score = None, float('inf')
            
            for i in range(len(candidate_positions)):
                pos = np.array(candidate_positions[i])
                
                if any(self.check_collision(obj, pos, self.objects[j], up) 
                       for j, up in enumerate(used_pos) if self.objects[j].prim_path in solution):
                    continue
                min_d = None
                if used_pos:
                    min_d = min(
                        np.linalg.norm(pos[:2] - up[:2])
                        for up in used_pos
                    )
                    if min_d + 1e-12 < self.minimum_spacing:
                        continue
                
                # 2. 计算综合得分
                # 遮挡分：改为 Y 轴遮挡检查（这是防止前后重叠的关键）
                occ_score = self.compute_y_occlusion(used_pos + [pos], [o.size for o in self.objects[:len(used_pos)+1]])
                
                # 中心倾向分
                center_score = 5.0 * (pos[0] - ec.center_x)**2 + 0.1 * (pos[1] - ec.center_y)**2
                
                # 间距分
                if min_d is not None:
                    dist_penalty = (min_d - ideal_dist)**2
                else:
                    dist_penalty = 0.0
                
                # 3. 权重平衡：加大遮挡分权重，确保横向铺开
                score = 0.5 * occ_score + 0.3 * center_score + 0.2 * dist_penalty
                
                if score < min_score:
                    min_score, best_pos = score, pos
                
                if score < 0.01: break
            
            if best_pos is not None:
                solution[obj.prim_path] = list(best_pos)
                used_pos.append(best_pos)
            else:
                failed.append(obj.prim_name)
                default_pos = np.array([ec.center_x, ec.center_y, self.z_height])
                solution[obj.prim_path] = list(default_pos)
                used_pos.append(default_pos)
        
        if failed: print(f"警告：无法为 {failed} 找到理想无碰撞位置")
        return solution

    def optimize(self, method: OptimizationMethod = OptimizationMethod.MILP, **kwargs) -> Optional[Dict[str, List[float]]]:
        if method == OptimizationMethod.MILP: return self.optimize_milp(kwargs['grid_resolution'])
        raise ValueError(f"未知方法: {method}")

    def save_optimized_positions(self, positions: Dict[str, List[float]], output_path: Optional[Path] = None) -> Path:
        """保存优化后的位置到 JSON 文件，并同步更新内存中的数据。"""
        for p, pos in positions.items():
            if p in self.scene_data: 
                self.scene_data[p]["position"] = pos
        
        path = output_path or self.json_file_path
        with open(path, 'w', encoding='utf-8') as f: 
            json.dump(self.scene_data, f, indent=4, ensure_ascii=False)
        
        # 重新解析物体信息以保持同步
        self.objects = self._parse_objects()
        return path

    def apply_semantic_constraints(self, semantic_prompt: str, actions_file: Optional[Path] = None, **kwargs) -> Optional[Dict[str, List[float]]]:
        api_key = kwargs.get('api_key') or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY or api_key is required for semantic layout optimization")
        model = kwargs.get('model') or os.getenv("ACTION_AGENT_MODEL")
        if not model:
            raise ValueError("ACTION_AGENT_MODEL or model is required for semantic layout optimization")
        client_kwargs = {"api_key": api_key}
        base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        
        # 读取动作信息作为上下文
        actions_info = ""
        if actions_file and os.path.exists(actions_file):
            with open(actions_file, 'r', encoding='utf-8') as f:
                actions_info = f.read()

        ec = self.ellipse_constraint
        sys_p = f"""你是一个场景布局优化助手。我会给你一个已经经过基础位置优化（已处理碰撞和间距）的 JSON 数据，以及对应的实验动作流程。
你的任务是：**仅分析物体的语义角色并交换它们之间的坐标点，严禁微调数值**。

输入 JSON 中包含了核心物体（如加热板、固/液瓶）的候选坐标。你只需要识别出以下三类角色，并按照逻辑分配它们【现有的】坐标：

1. **放置类资产 (Placement-Type)**：识别逻辑为 HeatingPlate, Stirrer, Heater 等。
   - **分配逻辑**：将其分配到 Y 轴最接近中心点 ({ec.center_y}) 的那个坐标。
2. **待移入资产 (Interactive-Target)**：识别逻辑为 Solid Flask, Beaker (流程中先被 Pick 的物体) 等。
   - **分配逻辑**：将其分配到 Y 轴较小（左侧）的那个坐标。
3. **待放回资产 (Interactive-Return)**：识别逻辑为 Liquid Flask, Pouring vessel (流程中执行 Pour 后放回 TargetPlatform 的物体) 等。
   - **分配逻辑**：将其分配到 Y 轴较大（右侧）的那个坐标。

核心限制：
- **禁止修改坐标数值**：只能在物体间进行 position 数组的整体交换。
- **TargetPlatform 同步**：如果物体需要“放回原处”，将 TargetPlatform 的坐标设为与其【分配后】的坐标完全一致。
- 保持 z 轴高度固定。
- 环境资产（table, GroundPlane）禁止修改。
- 返回完整的 JSON 数据。"""

        user_content = f"待交换坐标的 JSON 数据：\n{json.dumps(self.scene_data, indent=4)}\n\n动作流程步骤：\n{actions_info}\n\n用户附加约束：\n{semantic_prompt}"

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_p}, 
                          {"role": "user", "content": user_content}],
                temperature=0.1)
            
            modified_data = self._parse_json_from_response(resp.choices[0].message.content)
            if not modified_data: return None
            
            new_pos = {p: d["position"] for p, d in modified_data.items() if isinstance(d, dict) and "position" in d}
            
            # 【最后一步】大模型交换完成后，再由程序物理确定中心物体并后移 15cm
            new_pos = self.shift_center_object_back(new_pos)
            
            self.save_optimized_positions(new_pos, kwargs.get('json_file_path'))
            return new_pos
        except Exception as e:
            print(f"语义约束出错: {e}"); return None

    def shift_center_object_back(self, positions: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        物理校正逻辑：
        1. 确保所有交互物体 X 轴对齐。
        2. 识别 Y 轴最居中的物体（此时应该是 LLM 交换后的加热板等）并将其向后移 15cm。
        """
        if not positions: return positions
        
        # 1. 识别交互物体（排除环境资产）
        movable_paths = [p for p in positions.keys() 
                         if p.split('/')[-1] not in self.excluded_objects and 
                         not any(p.split('/')[-1].startswith(pat) for pat in self.excluded_patterns)]
        
        if len(movable_paths) < 1: return positions
        
        ec = self.ellipse_constraint
        new_positions = copy.deepcopy(positions)
        
        # 2. 先将所有交互物体的 X 轴统一重置到椭圆中心基准线（防止 LLM 带来的数值漂移）
        for p in movable_paths:
            new_positions[p][0] = ec.center_x
            
        # 3. 在交换后的布局中，找到 Y 轴最居中的物体
        center_obj_path = min(movable_paths, key=lambda p: abs(positions[p][1] - ec.center_y))
        
        # 4. 执行后移 15cm
        new_positions[center_obj_path][0] = ec.center_x + 0.15
        
        print(f"[Step 3: Final Shift] 已将语义居中物体 {center_obj_path.split('/')[-1]} 物理后移 15cm")
        return new_positions

    def _parse_json_from_response(self, text: str) -> Optional[Dict]:
        for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```', r'(\{.*\})']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try: return json.loads(match.group(1))
                except: continue
        return None

    def _validate_semantic_positions(self, pos_dict: Dict, allow_coll: bool) -> bool:
        for p, pos in pos_dict.items():
            name = p.split('/')[-1]
            if name in self.excluded_objects or any(name.startswith(pat) for pat in self.excluded_patterns):
                continue
            if not self.ellipse_constraint.contains(pos[0], pos[1]): return False
        return True

def _load_layout_profile(
    profile_path: Optional[Path] = None,
    profile_name: str = "lab_table_franka",
) -> dict:
    import yaml

    path = profile_path or Path(__file__).resolve().parents[1] / "layout_profiles.yaml"
    profiles = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return profiles["profiles"][profile_name]


def _layout_profile_from_kwargs(kwargs: dict) -> dict:
    if kwargs.get("layout_profile"):
        profile = copy.deepcopy(kwargs["layout_profile"])
    else:
        profile_path = (
            Path(kwargs["layout_profile_path"])
            if kwargs.get("layout_profile_path")
            else None
        )
        profile = _load_layout_profile(
            profile_path,
            kwargs.get("layout_profile_name", "lab_table_franka"),
        )
    if kwargs.get("z_height") is not None:
        profile["surface_z"] = float(kwargs["z_height"])
    if kwargs.get("grid_resolution") is not None:
        profile["grid_resolution"] = float(kwargs["grid_resolution"])
    return profile


def optimize_scene_positions(json_file_path: str, **kwargs):
    profile = _layout_profile_from_kwargs(kwargs)
    opt = PositionOptimizer.from_profile(Path(json_file_path), profile)
    res = opt.optimize(
        OptimizationMethod(kwargs.get('method', 'milp')),
        grid_resolution=float(profile["grid_resolution"]),
    )
    
    # 步骤 1：此处不再进行后移，仅保存贪心摆放的结果
    if res: opt.save_optimized_positions(res, Path(kwargs['output']) if kwargs.get('output') else None)
    return res

def apply_semantic_constraints_to_scene(json_file_path: str, semantic_prompt: str, actions_file: Optional[str] = None, **kwargs):
    opt = PositionOptimizer.from_profile(
        Path(json_file_path),
        _layout_profile_from_kwargs(kwargs),
    )
    return opt.apply_semantic_constraints(semantic_prompt, actions_file=Path(actions_file) if actions_file else None, **kwargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="场景布局优化工具")
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--output")
    parser.add_argument("--method", default="milp") 
    parser.add_argument("--layout_profile_path")
    parser.add_argument("--layout_profile_name", default="lab_table_franka")
    parser.add_argument("--z_height", type=float)
    parser.add_argument("--grid_resolution", type=float)
    parser.add_argument("--semantic_prompt")
    parser.add_argument("--actions_file", help="动作流程描述文件路径")
    parser.add_argument("--semantic_only", action="store_true")
    parser.add_argument("--no_validate", action="store_true")
    parser.add_argument("--no_collision", action="store_true")
    
    args, unknown = parser.parse_known_args()
    prompt = args.semantic_prompt or ""
    params = {**vars(args), 'validate_constraints': not args.no_validate, 'allow_collision': not args.no_collision}
    
    if args.semantic_only:
        apply_semantic_constraints_to_scene(args.json_file, prompt, actions_file=args.actions_file, **params)
    else:
        res = optimize_scene_positions(args.json_file, **params)
        if res and (prompt or args.actions_file): 
            apply_semantic_constraints_to_scene(args.output or args.json_file, prompt, actions_file=args.actions_file, **params)
