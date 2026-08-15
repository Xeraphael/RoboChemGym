"""
连续优化器：根据日志文件分析场景问题，使用LLM优化场景JSON文件中的物体位置

功能：
1. 读取日志文件和场景JSON文件
2. 调用LLM API分析问题并优化物体位置
3. 将优化后的结果应用到JSON文件
"""

import os
import json
import math
import re
import shutil
import numpy as np
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from openai import OpenAI

# 导入约束相关的类
from agent.scene.optimization.position_optimizer import (
    EllipseConstraint,
    ObjectInfo
)


class ContinuousOptimizer:
    """连续优化器类：根据日志分析场景问题并优化物体位置"""
    
    def __init__(
        self,
        logs_dir: Optional[str] = None,
        scenes_dir: Optional[str] = None,
        prompt_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client=None,
        scene_json_path: Optional[Path] = None,
        scene_usd_path: Optional[Path] = None,
        layout_profile: Optional[Dict] = None,
        position_updater_factory=None,
    ):
        """
        初始化连续优化器
        
        Args:
            logs_dir: 日志文件目录（默认：agent/scene/optimization/logs）
            scenes_dir: 场景文件目录（默认：agent/scene/scenes）
            prompt_path: prompt文件路径（如果为None，使用默认prompt）
            api_key: OpenAI API密钥（如果为None，从环境变量获取）
            base_url: API基础URL（如果为None，使用默认值）
            model: LLM模型名称
        """
        # 设置默认路径
        base_dir = Path(__file__).parent.parent.parent.parent  # 项目根目录
        
        self.logs_dir = Path(logs_dir) if logs_dir else (
            base_dir / "agent" / "scene" / "optimization" / "logs"
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.scenes_dir = Path(scenes_dir) if scenes_dir else (
            base_dir / "agent" / "scene" / "scenes"
        )
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认prompt路径
        if prompt_path:
            self.prompt_path = Path(prompt_path)
        else:
            # 尝试查找默认prompt文件
            default_prompt = base_dir / "agent" / "scene" / "optimization" / "optimization_prompt.txt"
            if default_prompt.exists():
                self.prompt_path = default_prompt
            else:
                self.prompt_path = None
        
        self.scene_json_path = (
            Path(scene_json_path) if scene_json_path is not None else None
        )
        self.scene_usd_path = (
            Path(scene_usd_path) if scene_usd_path is not None else None
        )
        self.position_updater_factory = position_updater_factory

        if layout_profile is None:
            profile_path = base_dir / "agent" / "scene" / "layout_profiles.yaml"
            profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            layout_profile = profiles["profiles"]["lab_table_franka"]
        self.layout_profile = deepcopy(layout_profile)
        region = self.layout_profile["reachable_region"]
        self.z_height = float(self.layout_profile["surface_z"])
        self.ellipse_constraint = EllipseConstraint(
            center_x=float(region["center"][0]),
            center_y=float(region["center"][1]),
            semi_major=float(region["semi_axes"][0]),
            semi_minor=float(region["semi_axes"][1]),
            rotation=float(region["rotation"]),
        )

        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or api_key is required")
            base_url = base_url or os.getenv("OPENAI_BASE_URL")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)

        self.model = model or os.getenv("ACTION_AGENT_MODEL")
        if not self.model:
            raise ValueError("ACTION_AGENT_MODEL or model is required")
        self.excluded_objects = {
            "table",
            "lounge_booth_table",
            "GroundPlane",
            "FumeHood",
        }
    
    def read_text_file(self, path: Path) -> str:
        """读取文本文件"""
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def read_json_file(self, path: Path) -> Dict:
        """读取JSON文件"""
        if not path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_log_file(self, timestamp: Optional[str] = None) -> Optional[Path]:
        """
        查找日志文件
        
        Args:
            timestamp: 时间戳（格式：20251127_161552），如果为None，查找最新的日志文件
        
        Returns:
            日志文件路径，如果未找到则返回None
        """
        if timestamp:
            # 根据时间戳查找
            log_pattern = f"*{timestamp}*.txt"
            log_files = list(self.logs_dir.glob(log_pattern))
            if log_files:
                return log_files[0]
        else:
            # 查找最新的日志文件
            log_files = list(self.logs_dir.glob("*.txt"))
            if log_files:
                # 按修改时间排序，返回最新的
                return max(log_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def find_scene_json(self, timestamp: Optional[str] = None) -> Optional[Path]:
        """
        查找场景JSON文件
        
        Args:
            timestamp: 时间戳（格式：20251127_161552），如果为None，查找最新的JSON文件
        
        Returns:
            JSON文件路径，如果未找到则返回None
        """
        if timestamp:
            # 根据时间戳查找
            json_pattern = f"*{timestamp}*.json"
            json_files = list(self.scenes_dir.glob(json_pattern))
            if json_files:
                return json_files[0]
        else:
            # 查找最新的JSON文件
            json_files = list(self.scenes_dir.glob("*.json"))
            if json_files:
                # 按修改时间排序，返回最新的
                return max(json_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """
        从文件名中提取时间戳
        
        Args:
            filename: 文件名（如 equipment_20251127_161552.json）
        
        Returns:
            时间戳字符串（如 20251127_161552），如果未找到则返回None
        """
        # 匹配格式：YYYYMMDD_HHMMSS
        pattern = r'(\d{8}_\d{6})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def build_optimization_prompt(
        self,
        log_content: str,
        scene_json: Dict,
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        构建优化提示词
        
        Args:
            log_content: 日志文件内容
            scene_json: 场景JSON数据
            custom_prompt: 自定义提示词（如果为None，使用默认prompt文件）
        
        Returns:
            (system_prompt, user_message) 元组
        """
        # 读取系统prompt
        if custom_prompt:
            system_prompt = custom_prompt
        elif self.prompt_path and self.prompt_path.exists():
            system_prompt = self.read_text_file(self.prompt_path)
        else:
            # 使用默认prompt
            system_prompt = """你是一个场景布局优化助手。你的任务是根据执行日志分析场景中的物体摆放问题，并优化场景JSON文件中的物体位置。"""
        
        # 添加约束信息到prompt
        constraint_info = f"""
【重要约束条件 - 必须严格遵守】

1. 高度约束：
   - 所有物体的z坐标必须设置为 {self.z_height}（放置高度）

2. 椭圆可达性约束：
   - 所有物体的(x, y)坐标必须在椭圆范围内
   - 椭圆中心: ({self.ellipse_constraint.center_x:.6f}, {self.ellipse_constraint.center_y:.6f})
   - 长半轴: {self.ellipse_constraint.semi_major:.6f}m
   - 短半轴: {self.ellipse_constraint.semi_minor:.6f}m
   - 旋转角度: {self.ellipse_constraint.rotation:.6f}弧度（约{np.degrees(self.ellipse_constraint.rotation):.1f}度）
   - 椭圆方程（在椭圆坐标系中）: (x_rot / {self.ellipse_constraint.semi_major:.6f})² + (y_rot / {self.ellipse_constraint.semi_minor:.6f})² ≤ 1

3. 碰撞约束：
   - 物体之间不能发生碰撞（bounding_box不能重叠）
   - 注意：某些语义关系（如玻璃棒插入试管架）可能需要允许重叠，但应尽量避免

4. 遮挡约束：
   - 在x方向上尽量减少物体之间的遮挡
   - 尽量分散放置物体，避免在x方向上重叠

5. 修改限制：
   - 只能修改每个物体的"position"属性（一个包含3个浮点数的列表：[x, y, z]）
   - 不能修改其他任何属性（prim_path, prim_name, prim_type, bounding_box等）
   - 保持JSON结构完整
   - 返回完整的JSON对象，包含所有物体

6. 语义约束：
   - 满足语义要求（如目标平面放在烧杯下方等）
   - 但必须在满足上述物理约束的前提下

【要求】
1. 仔细分析日志文件中的错误、警告和失败信息
2. 识别物体位置相关的问题（如碰撞、不可达、遮挡等）
3. 优化后的位置必须满足所有约束条件
4. 如果无法同时满足所有约束，优先满足高度约束和椭圆约束

返回格式：直接返回修改后的完整JSON对象，不要添加任何解释文字。"""
        
        # 将约束信息添加到系统prompt
        if not custom_prompt:
            system_prompt = system_prompt + constraint_info
        
        # 构建用户消息
        scene_json_str = json.dumps(scene_json, indent=2, ensure_ascii=False)
        
        user_message = f"""【执行日志】

{log_content}

【当前场景JSON文件】

```json
{scene_json_str}
```

【任务】

请根据日志文件中的信息，分析场景中的物体摆放问题，修改JSON文件中的物体位置信息，优化场景布局。

重要：修改后的位置必须满足所有约束条件（高度约束、椭圆约束、碰撞约束等）。

只返回修改后的完整JSON对象，不要添加任何其他文字。"""
        
        return system_prompt, user_message
    
    def parse_json_from_response(self, response_text: str) -> Optional[Dict]:
        """
        从LLM响应中解析JSON对象
        
        Args:
            response_text: LLM返回的文本
        
        Returns:
            解析后的JSON字典，如果解析失败则返回None
        """
        # 尝试直接解析整个响应
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取代码块中的JSON
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # 尝试查找第一个 { 到最后一个 } 之间的内容
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = response_text[start_idx:end_idx + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _parse_objects(self, scene_data: Dict) -> List[ObjectInfo]:
        """
        解析物体信息
        
        Args:
            scene_data: 场景JSON数据
        
        Returns:
            物体信息列表
        """
        objects = []
        for prim_path, data in scene_data.items():
            prim_name = data.get("prim_name", "")
            if prim_name not in self.excluded_objects:
                obj = ObjectInfo(
                    prim_path=prim_path,
                    prim_name=prim_name,
                    position=data.get("position", [0, 0, 0]),
                    bounding_box=data.get("bounding_box", {})
                )
                objects.append(obj)
        return objects
    
    def check_collision(
        self, 
        obj1: ObjectInfo, 
        pos1: np.ndarray, 
        obj2: ObjectInfo, 
        pos2: np.ndarray
    ) -> bool:
        """
        检查两个物体是否碰撞（使用 AABB）
        
        Args:
            obj1: 第一个物体
            pos1: 第一个物体的位置
            obj2: 第二个物体
            pos2: 第二个物体的位置
        
        Returns:
            如果碰撞返回True，否则返回False
        """
        size1 = obj1.size
        size2 = obj2.size
        
        # AABB 碰撞检测
        min1 = pos1 - size1 / 2
        max1 = pos1 + size1 / 2
        min2 = pos2 - size2 / 2
        max2 = pos2 + size2 / 2
        
        return not (max1[0] < min2[0] or min1[0] > max2[0] or
                   max1[1] < min2[1] or min1[1] > max2[1] or
                   max1[2] < min2[2] or min1[2] > max2[2])
    
    def validate_constraints(
        self,
        optimized_scene_data: Dict,
        original_scene_data: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证优化后的场景数据是否满足所有约束
        
        Args:
            optimized_scene_data: 优化后的场景数据
            original_scene_data: 原始场景数据（用于对比，可选）
        
        Returns:
            (是否满足约束, 违反约束的详细信息列表)
        """
        violations = []
        objects = self._parse_objects(optimized_scene_data)
        
        # 检查每个物体
        for obj in objects:
            x, y, z = obj.position
            
            # 检查高度约束
            if abs(z - self.z_height) > 0.001:  # 允许小的浮点误差
                violations.append(
                    f"{obj.prim_name} ({obj.prim_path}): z坐标 {z:.6f} 不符合高度约束 {self.z_height}"
                )
            
            # 检查椭圆约束
            if not self.ellipse_constraint.contains(x, y):
                violations.append(
                    f"{obj.prim_name} ({obj.prim_path}): 位置 ({x:.6f}, {y:.6f}) 不在椭圆约束范围内"
                )
        
        # 检查碰撞约束
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1 = objects[i]
                obj2 = objects[j]
                pos1 = np.array(obj1.position)
                pos2 = np.array(obj2.position)
                
                if self.check_collision(obj1, pos1, obj2, pos2):
                    violations.append(
                        f"碰撞: {obj1.prim_name} ({obj1.prim_path}) 与 {obj2.prim_name} ({obj2.prim_path}) 发生碰撞"
                    )
        
        is_valid = len(violations) == 0
        return is_valid, violations

    @staticmethod
    def _reject_nonfinite_json_constant(value: str):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    @staticmethod
    def _is_position(value) -> bool:
        if not isinstance(value, list) or len(value) != 3:
            return False
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return False
            try:
                if not math.isfinite(item):
                    return False
            except (OverflowError, TypeError):
                return False
        return True

    @staticmethod
    def _candidate_path(path: Path) -> Path:
        return path.with_name(f"{path.stem}.candidate{path.suffix}")

    def _make_position_updater(self):
        factory = self.position_updater_factory
        if factory is None:
            from agent.scene.optimization.position_updater import PositionUpdater

            factory = PositionUpdater
        return factory(scenes_dir=str(self.scene_json_path.parent))

    def optimize_from_execution_report(self, report: dict) -> bool:
        """Apply a validated position-only recovery for one failed step."""
        if self.scene_json_path is None or self.scene_usd_path is None:
            raise ValueError(
                "structured scene optimization requires scene_json_path and "
                "scene_usd_path"
            )
        if not isinstance(report, dict):
            return False
        failed_step = report.get("failed_step")
        records = report.get("steps")
        if not isinstance(failed_step, str) or not failed_step:
            return False
        if not isinstance(records, list):
            return False
        failed_record = next(
            (
                item
                for item in reversed(records)
                if isinstance(item, dict) and item.get("step_id") == failed_step
            ),
            None,
        )
        if failed_record is None:
            return False
        verification = failed_record.get("verification")
        if not isinstance(verification, dict):
            return False
        if (
            failed_record.get("success") is not False
            or verification.get("success") is not False
        ):
            return False
        error_code = verification.get("code")
        measurements = verification.get("measurements")
        if not isinstance(error_code, str) or not isinstance(measurements, dict):
            return False

        try:
            original = self.read_json_file(self.scene_json_path)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        if not isinstance(original, dict) or not self.scene_usd_path.is_file():
            return False

        current_positions = {
            prim_path: data["position"]
            for prim_path, data in original.items()
            if isinstance(prim_path, str)
            and isinstance(data, dict)
            and data.get("prim_name") not in self.excluded_objects
            and self._is_position(data.get("position"))
        }
        if not current_positions:
            return False
        request = {
            "failed_step": failed_step,
            "verification": {
                "code": error_code,
                "measurements": measurements,
            },
            "current_positions": current_positions,
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return strict JSON with exactly one positions object "
                            "mapping existing movable prim paths to [x,y,z]. Do not "
                            "add prims or modify non-position data."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            request,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ),
                    },
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(
                content,
                parse_constant=self._reject_nonfinite_json_constant,
            )
        except Exception:
            return False

        if not isinstance(parsed, dict) or set(parsed) != {"positions"}:
            return False
        positions = parsed["positions"]
        if not isinstance(positions, dict) or not positions:
            return False
        if not set(positions).issubset(current_positions):
            return False

        candidate = deepcopy(original)
        changed = False
        for prim_path, position in positions.items():
            if not isinstance(prim_path, str) or not self._is_position(position):
                return False
            normalized = [float(value) for value in position]
            if normalized != candidate[prim_path]["position"]:
                changed = True
            candidate[prim_path]["position"] = normalized
        if not changed:
            return False

        try:
            valid, _ = self.validate_constraints(candidate, original)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not valid:
            return False

        candidate_json = self._candidate_path(self.scene_json_path)
        candidate_usd = self._candidate_path(self.scene_usd_path)
        try:
            original_json_bytes = self.scene_json_path.read_bytes()
        except OSError:
            return False

        committed = False
        json_replaced = False
        try:
            candidate_json.write_text(
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            shutil.copy2(self.scene_usd_path, candidate_usd)
            updater = self._make_position_updater()
            updated = updater.apply_positions_to_usd(
                candidate_json,
                candidate_usd,
                in_place=True,
                required_prim_paths=set(positions),
            )
            if updated is not True:
                return False

            candidate_json.replace(self.scene_json_path)
            json_replaced = True
            candidate_usd.replace(self.scene_usd_path)
            committed = True
            return True
        except Exception:
            return False
        finally:
            if not committed and json_replaced:
                try:
                    self.scene_json_path.write_bytes(original_json_bytes)
                except OSError as exc:
                    raise RuntimeError(
                        "failed to restore scene JSON after optimization"
                    ) from exc
            for candidate_path in (candidate_json, candidate_usd):
                try:
                    candidate_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def optimize_scene(
        self,
        log_file_path: Optional[Path] = None,
        scene_json_path: Optional[Path] = None,
        timestamp: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        temperature: float = 0.7,
        backup: bool = True,
        return_validation_info: bool = False
    ) -> Optional[Dict]:
        """
        优化场景
        
        Args:
            log_file_path: 日志文件路径（如果为None，根据timestamp或查找最新文件）
            scene_json_path: 场景JSON文件路径（如果为None，根据timestamp或查找最新文件）
            timestamp: 时间戳（格式：20251127_161552），用于匹配日志和JSON文件
            custom_prompt: 自定义提示词
            temperature: LLM温度参数
            backup: 是否在修改前备份原文件
            return_validation_info: 是否返回验证信息（如果为True，返回包含验证信息的字典）
        
        Returns:
            优化后的场景数据字典，如果失败则返回None
            如果return_validation_info=True，返回字典包含'scene_data', 'is_valid', 'violations'键
        """
        # 确定日志文件路径
        if log_file_path is None:
            log_file_path = self.find_log_file(timestamp)
            if log_file_path is None:
                raise FileNotFoundError(f"未找到日志文件（timestamp: {timestamp}）")
        else:
            log_file_path = Path(log_file_path)
            if not log_file_path.exists():
                raise FileNotFoundError(f"日志文件不存在: {log_file_path}")
        
        # 确定场景JSON文件路径
        if scene_json_path is None:
            # 尝试从日志文件名提取时间戳
            if timestamp is None:
                timestamp = self.extract_timestamp_from_filename(log_file_path.name)
            
            scene_json_path = self.find_scene_json(timestamp)
            if scene_json_path is None:
                raise FileNotFoundError(f"未找到场景JSON文件（timestamp: {timestamp}）")
        else:
            scene_json_path = Path(scene_json_path)
            if not scene_json_path.exists():
                raise FileNotFoundError(f"场景JSON文件不存在: {scene_json_path}")
        
        print(f"[ContinuousOptimizer] 日志文件: {log_file_path}")
        print(f"[ContinuousOptimizer] 场景JSON文件: {scene_json_path}")
        
        # 读取文件
        log_content = self.read_text_file(log_file_path)
        scene_data = self.read_json_file(scene_json_path)
        
        # 构建提示词
        system_prompt, user_message = self.build_optimization_prompt(
            log_content, scene_data, custom_prompt
        )
        
        # 调用LLM
        print(f"[ContinuousOptimizer] 调用LLM API优化场景...")
        print(f"[ContinuousOptimizer] 模型: {self.model}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature
            )
            
            # 提取响应内容
            if response and response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ''
                
                # 解析JSON
                optimized_scene_data = self.parse_json_from_response(content)
                
                if optimized_scene_data is None:
                    print(f"[ContinuousOptimizer] ✗ 无法从LLM响应中解析JSON")
                    print(f"[ContinuousOptimizer] 响应内容预览: {content[:500]}...")
                    return None
                
                print(f"[ContinuousOptimizer] ✓ 成功解析优化后的场景数据")
                
                # 验证约束
                print(f"[ContinuousOptimizer] 验证约束条件...")
                is_valid, violations = self.validate_constraints(
                    optimized_scene_data, 
                    scene_data
                )
                
                if not is_valid:
                    print(f"[ContinuousOptimizer] ⚠ 警告：优化后的场景不满足部分约束条件:")
                    for violation in violations:
                        print(f"  - {violation}")
                    print(f"[ContinuousOptimizer] 共发现 {len(violations)} 个约束违反")
                    
                    # 询问是否继续保存（在实际使用中，可以根据需要决定是否继续）
                    print(f"[ContinuousOptimizer] 将继续保存结果，但请注意约束违反问题")
                else:
                    print(f"[ContinuousOptimizer] ✓ 所有约束条件验证通过")
                
                # 备份原文件（如果需要）
                if backup:
                    backup_path = scene_json_path.with_suffix(
                        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(scene_data, f, indent=4, ensure_ascii=False)
                    print(f"[ContinuousOptimizer] 已备份原文件到: {backup_path}")
                
                # 保存优化后的JSON
                with open(scene_json_path, 'w', encoding='utf-8') as f:
                    json.dump(optimized_scene_data, f, indent=4, ensure_ascii=False)
                
                print(f"[ContinuousOptimizer] ✓ 优化完成，结果已保存到: {scene_json_path}")
                
                # 返回结果
                if return_validation_info:
                    return {
                        'scene_data': optimized_scene_data,
                        'is_valid': is_valid,
                        'violations': violations
                    }
                else:
                    # 返回优化后的场景数据（保持向后兼容）
                    return optimized_scene_data
            else:
                print(f"[ContinuousOptimizer] ✗ LLM返回无效响应")
                return None
                
        except Exception as e:
            print(f"[ContinuousOptimizer] ✗ 调用LLM时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def optimize_scene_from_logs(
    log_file_path: Optional[str] = None,
    scene_json_path: Optional[str] = None,
    timestamp: Optional[str] = None,
    logs_dir: Optional[str] = None,
    scenes_dir: Optional[str] = None,
    prompt_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    temperature: float = 0.7,
    backup: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict]:
    """
    便捷函数：根据日志文件优化场景
    
    Args:
        log_file_path: 日志文件路径
        scene_json_path: 场景JSON文件路径
        timestamp: 时间戳（格式：20251127_161552）
        logs_dir: 日志文件目录
        scenes_dir: 场景文件目录
        prompt_path: prompt文件路径
        custom_prompt: 自定义提示词
        temperature: LLM温度参数
        backup: 是否备份原文件
        api_key: OpenAI API密钥
        base_url: API基础URL
        model: LLM模型名称
    
    Returns:
        优化后的场景数据字典
    """
    optimizer = ContinuousOptimizer(
        logs_dir=logs_dir,
        scenes_dir=scenes_dir,
        prompt_path=prompt_path,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    
    return optimizer.optimize_scene(
        log_file_path=Path(log_file_path) if log_file_path else None,
        scene_json_path=Path(scene_json_path) if scene_json_path else None,
        timestamp=timestamp,
        custom_prompt=custom_prompt,
        temperature=temperature,
        backup=backup
    )


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="根据日志文件优化场景JSON文件中的物体位置"
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='日志文件路径（如果未指定，根据timestamp或查找最新文件）'
    )
    
    parser.add_argument(
        '--scene-json',
        type=str,
        default=None,
        help='场景JSON文件路径（如果未指定，根据timestamp或查找最新文件）'
    )
    
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='时间戳（格式：20251127_161552），用于匹配日志和JSON文件'
    )
    
    parser.add_argument(
        '--logs-dir',
        type=str,
        default=None,
        help='日志文件目录（默认：agent/scene/optimization/logs）'
    )
    
    parser.add_argument(
        '--scenes-dir',
        type=str,
        default=None,
        help='场景文件目录（默认：agent/scene/scenes）'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Prompt文件路径'
    )
    
    parser.add_argument(
        '--custom-prompt',
        type=str,
        default=None,
        help='自定义提示词（直接提供文本，不使用文件）'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM温度参数（默认：0.7）'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份原文件'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API密钥（默认从环境变量获取）'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='可选API基础URL（默认使用OpenAI客户端配置）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LLM模型名称（默认从 ACTION_AGENT_MODEL 读取）'
    )
    
    args = parser.parse_args()
    
    # 执行优化
    try:
        result = optimize_scene_from_logs(
            log_file_path=args.log_file,
            scene_json_path=args.scene_json,
            timestamp=args.timestamp,
            logs_dir=args.logs_dir,
            scenes_dir=args.scenes_dir,
            prompt_path=args.prompt,
            custom_prompt=args.custom_prompt,
            temperature=args.temperature,
            backup=not args.no_backup,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
        
        if result:
            print(f"\n✓ 优化完成")
            print(f"  修改的物体数量: {len(result)}")
        else:
            print(f"\n✗ 优化失败")
            exit(1)
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
