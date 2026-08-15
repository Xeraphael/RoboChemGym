"""
代码生成器：使用 LLM 生成控制器代码并自动注册

功能：
1. 从 prompt 文件和动作信息文件生成控制器代码
2. 将生成的代码保存到 controllers 目录
3. 在 controller_factory.py 中自动注册
4. 更新对应的 YAML 文件中的 controller_type
"""

import os
import re
import ast
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from openai import OpenAI


class CodeGenerator:
    """代码生成器类"""
    
    def __init__(
        self,
        prompt_path: Optional[str] = None,
        action_info_path: Optional[str] = None,
        controllers_dir: Optional[str] = None,
        factory_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化代码生成器
        
        Args:
            prompt_path: prompt 文件路径（默认：agent/action/prompts/code_prompt.txt）
            action_info_path: 动作信息文件路径
            controllers_dir: controllers 目录路径（默认：项目根目录/controllers）
            factory_path: controller_factory.py 路径（默认：项目根目录/factories/controller_factory.py）
            api_key: OpenAI API 密钥
            base_url: API 基础 URL
            model: LLM 模型名称
        """
        base_dir = Path(__file__).parent.parent.parent.parent
        
        self.prompt_path = Path(prompt_path) if prompt_path else (
            base_dir / "agent" / "action" / "prompts" / "code_prompt.txt"
        )
        
        self.action_info_path = Path(action_info_path) if action_info_path else None
        
        self.controllers_dir = Path(controllers_dir) if controllers_dir else (
            base_dir / "controllers"
        )
        
        self.factory_path = Path(factory_path) if factory_path else (
            base_dir / "factories" / "controller_factory.py"
        )
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        model = model or os.getenv("ACTION_AGENT_MODEL")
        if not model:
            raise ValueError("ACTION_AGENT_MODEL is required")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        
        self.model = model
    
    def read_text_file(self, path: Path) -> str:
        """读取文本文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_class_name(self, code: str) -> Optional[str]:
        """
        从生成的代码中提取控制器类名
        
        Args:
            code: 生成的 Python 代码
        
        Returns:
            类名，如果未找到则返回 None
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'BaseController':
                            return node.name
                        elif isinstance(base, ast.Attribute):
                            if base.attr == 'BaseController':
                                return node.name
        except SyntaxError:
            pass
        
        pattern = r'class\s+(\w+Controller)\s*\([^)]*BaseController'
        match = re.search(pattern, code)
        if match:
            return match.group(1)
        
        return None
    
    def generate_controller_name(self, class_name: str) -> str:
        """
        根据类名生成注册名称
        
        Args:
            class_name: 控制器类名（如 ExampleProtocolController）
        
        Returns:
            注册名称（如 example_protocol_experiment）
        """
        name = class_name.replace('Controller', '')
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        if not name.endswith('_experiment'):
            name = f"{name}_experiment"
        
        return name
    
    def generate_code(
        self,
        action_info_path: Optional[str] = None,
        temperature: float = 1.0
    ) -> str:
        """
        生成控制器代码
        
        Args:
            action_info_path: 动作信息文件路径（如果为 None，使用 self.action_info_path）
            temperature: LLM 温度参数
        
        Returns:
            生成的代码字符串
        """
        if action_info_path:
            action_info_path = Path(action_info_path)
        else:
            action_info_path = self.action_info_path
        
        if not action_info_path or not action_info_path.exists():
            raise FileNotFoundError(f"动作信息文件不存在: {action_info_path}")
        
        system_prompt = self.read_text_file(self.prompt_path)
        action_info = self.read_text_file(action_info_path)
        
        action_lines = action_info.strip().split('\n')
        action_count = len([line for line in action_lines if line.strip() and not line.strip().startswith('#')])
        
        if action_count > 20:
            tokens_per_action = 600
        else:
            tokens_per_action = 530
        estimated_code_tokens = 2000 + action_count * tokens_per_action
        
        if 'thinking' in self.model.lower():
            max_tokens = int(estimated_code_tokens * 2.0)
            max_tokens = max(20000, min(max_tokens, 80000))
        else:
            max_tokens = int(estimated_code_tokens * 1.3)
            max_tokens = max(16000, min(max_tokens, 32000))
        
        user_message = f"【动作信息】\n{action_info.strip()}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 提取生成的内容
        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ''
            
            finish_reason = None
            if hasattr(response.choices[0], 'finish_reason'):
                finish_reason = response.choices[0].finish_reason
            
            code = self._extract_code_from_response(content)
            is_complete, error_msg = self._validate_code_completeness(code)
            if finish_reason == 'length':
                raise RuntimeError(
                    f"代码因 token 限制被截断 (max_tokens={max_tokens}, finish_reason={finish_reason})。"
                    f"建议: 将 max_tokens 增加到至少 {int(max_tokens * 1.5)} 或使用非 thinking 模型。"
                )
            
            if not is_complete:
                raise RuntimeError(
                    f"生成的代码不完整: {error_msg}。"
                    f"代码长度: {len(code)} 字符, 行数: {code.count(chr(10)) + 1}。"
                    f"可能原因: 代码被截断或 LLM 未生成完整代码。"
                )
            
            return code
        else:
            raise RuntimeError("LLM 返回无效响应")
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """
        从 LLM 响应中提取代码
        
        Args:
            response_text: LLM 返回的文本
        
        Returns:
            提取的代码字符串
        """
        code_pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        lines = response_text.split('\n')
        code_lines = []
        code_start_idx = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('from ', 'import ')):
                code_start_idx = i
                break
            elif 'class' in stripped and 'Controller' in stripped:
                if code_start_idx == -1:
                    code_start_idx = i
                break
        
        if code_start_idx >= 0:
            code_lines = lines[code_start_idx:]
            return '\n'.join(code_lines).strip()
        
        return response_text.strip()
    
    def _validate_code_completeness(self, code: str) -> Tuple[bool, str]:
        """
        验证代码完整性
        
        Args:
            code: 要验证的代码字符串
        
        Returns:
            (是否完整, 错误信息)
        """
        if not code or not code.strip():
            return False, "代码为空"
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e.msg} (行 {e.lineno})"
        except Exception as e:
            return False, f"解析错误: {str(e)}"
        
        open_brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for i, char in enumerate(code):
            if char in open_brackets:
                stack.append((char, i))
            elif char in open_brackets.values():
                if not stack:
                    return False, f"多余的闭合括号 '{char}' (位置 {i})"
                open_char, open_pos = stack.pop()
                if open_brackets[open_char] != char:
                    return False, f"括号不匹配: '{open_char}' (位置 {open_pos}) 和 '{char}' (位置 {i})"
        
        if stack:
            open_char, open_pos = stack[-1]
            return False, f"未闭合的括号 '{open_char}' (位置 {open_pos})"
        
        class_pattern = r'class\s+(\w+Controller)'
        class_matches = list(re.finditer(class_pattern, code))
        if not class_matches:
            return False, "未找到控制器类定义"
        
        if 'def __init__' not in code:
            return False, "缺少 __init__ 方法"
        
        if 'def step' not in code:
            return False, "缺少 step 方法"
        
        code_start = code[:min(500, len(code))].strip()
        if not code_start.startswith(('from ', 'import ')):
            if code_start.startswith('class '):
                return False, "缺少导入语句（代码可能被截断，缺少文件开头的导入部分）"
        
        required_classes = ['BaseController', 'PickController', 'PlaceController', 'PressZController', 'PourController', 'ShakeController', 'StirController']
        used_classes = []
        for cls in required_classes:
            if cls in code:
                used_classes.append(cls)
        
        import_section = code[:min(1000, len(code))]
        missing_imports = []
        for cls in used_classes:
            found = False
            import_pattern = rf'from\s+[\w\.]+\s+import\s+.*{cls}'
            if re.search(import_pattern, import_section):
                found = True
            elif f'import {cls}' in import_section:
                found = True
            elif f', {cls}' in import_section or f'{cls},' in import_section:
                lines = import_section.split('\n')
                for line in lines:
                    if 'import' in line and cls in line:
                        found = True
                        break
            
            if not found:
                missing_imports.append(cls)
        
        if missing_imports:
            return False, f"缺少必要的导入语句: {', '.join(missing_imports)}（代码可能被截断）"
        
        lines = code.strip().split('\n')
        last_line = lines[-1].strip()
        if last_line and not last_line.endswith((':', ')', ']', '}', '"""', "'''")):
            if last_line.count('(') > last_line.count(')'):
                return False, f"代码末尾可能有未完成的函数调用: {last_line[:50]}..."
            if last_line.count('[') > last_line.count(']'):
                return False, f"代码末尾可能有未完成的列表/字典: {last_line[:50]}..."
        
        return True, "代码完整"
    
    def save_controller_code(
        self,
        code: str,
        controller_name: Optional[str] = None
    ) -> Path:
        """
        保存控制器代码到文件
        
        Args:
            code: 生成的代码
            controller_name: 控制器文件名（如果为 None，从代码中提取类名）
        
        Returns:
            保存的文件路径
        """
        if controller_name is None:
            class_name = self.extract_class_name(code)
            if not class_name:
                raise ValueError("无法从代码中提取类名")
            controller_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower() + '.py'
        
        if not controller_name.endswith('.py'):
            controller_name += '.py'
        
        output_path = self.controllers_dir / controller_name
        self.controllers_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return output_path
    
    def register_controller(
        self,
        class_name: str,
        register_name: Optional[str] = None
    ) -> str:
        """
        在 controller_factory.py 中注册控制器
        
        Args:
            class_name: 控制器类名（如 ExampleProtocolController）
            register_name: 注册名称（如果为 None，自动生成）
        
        Returns:
            注册名称
        """
        if register_name is None:
            register_name = self.generate_controller_name(class_name)
        
        factory_content = self.read_text_file(self.factory_path)
        filename = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        import_line = f"from controllers.{filename} import {class_name}"
        
        register_line = f'register_controller("{register_name}", {class_name})'
        
        if register_line in factory_content:
            return register_name
        
        if import_line not in factory_content:
            import_pattern = r'^from controllers\.\w+ import \w+'
            lines = factory_content.split('\n')
            last_import_idx = -1
            for i, line in enumerate(lines):
                if re.match(import_pattern, line.strip()):
                    last_import_idx = i
            
            if last_import_idx >= 0:
                lines.insert(last_import_idx + 1, import_line)
            else:
                lines.insert(0, import_line)
            
            factory_content = '\n'.join(lines)
        
        if register_line not in factory_content:
            register_pattern = r'^register_controller\('
            lines = factory_content.split('\n')
            last_register_idx = -1
            for i, line in enumerate(lines):
                if re.match(register_pattern, line.strip()):
                    last_register_idx = i
            
            if last_register_idx >= 0:
                lines.insert(last_register_idx + 1, register_line)
            else:
                lines.append(register_line)
            
            factory_content = '\n'.join(lines)
        
        with open(self.factory_path, 'w', encoding='utf-8') as f:
            f.write(factory_content)
        
        return register_name
    
    def update_yaml_file(
        self,
        yaml_path: str,
        controller_type: str
    ) -> bool:
        """
        更新 YAML 文件中的 controller_type
        
        Args:
            yaml_path: YAML 文件路径
            controller_type: 控制器类型（注册名称）
        
        Returns:
            是否成功更新
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return False
        
        content = self.read_text_file(yaml_path)
        
        pattern = r'controller_type:\s*["\']?([^"\'\n]*)["\']?'
        replacement = f'controller_type: "{controller_type}"'
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
        else:
            task_type_pattern = r'(task_type:\s*"[^"]*")'
            if re.search(task_type_pattern, content):
                content = re.sub(
                    task_type_pattern,
                    f'\\1\ncontroller_type: "{controller_type}"',
                    content
                )
            else:
                name_pattern = r'(name:\s*[^\n]+)'
                content = re.sub(
                    name_pattern,
                    f'\\1\ncontroller_type: "{controller_type}"',
                    content,
                    count=1
                )
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    def generate_and_register(
        self,
        action_info_path: str,
        yaml_path: Optional[str] = None,
        temperature: float = 1.0
    ) -> Tuple[Path, str, str]:
        """
        生成代码、保存、注册并更新 YAML 的完整流程
        
        Args:
            action_info_path: 动作信息文件路径
            yaml_path: YAML 文件路径（如果为 None，根据动作信息文件名查找）
            temperature: LLM 温度参数
        
        Returns:
            (控制器文件路径, 类名, 注册名称)
        """
        code = self.generate_code(action_info_path, temperature)
        class_name = self.extract_class_name(code)
        if not class_name:
            raise ValueError("无法从生成的代码中提取类名")
        
        controller_path = self.save_controller_code(code)
        register_name = self.register_controller(class_name)
        
        if yaml_path:
            self.update_yaml_file(yaml_path, register_name)
        else:
            action_info_name = Path(action_info_path).stem
            yaml_name = action_info_name.replace('actions_', 'equipment_') + '.yaml'
            yaml_path = Path(__file__).parent.parent.parent.parent / "config" / yaml_name
            if yaml_path.exists():
                self.update_yaml_file(str(yaml_path), register_name)
        
        return controller_path, class_name, register_name


def generate_controller_code(
    action_info_path: str,
    prompt_path: Optional[str] = None,
    yaml_path: Optional[str] = None,
    controllers_dir: Optional[str] = None,
    factory_path: Optional[str] = None,
    temperature: float = 1.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[Path, str, str]:
    """
    便捷函数：生成控制器代码并自动注册
    
    Args:
        action_info_path: 动作信息文件路径
        prompt_path: prompt 文件路径
        yaml_path: YAML 文件路径
        controllers_dir: controllers 目录路径
        factory_path: controller_factory.py 路径
        temperature: LLM 温度参数
        api_key: OpenAI API 密钥
        base_url: API 基础 URL
        model: LLM 模型名称
    
    Returns:
        (控制器文件路径, 类名, 注册名称)
    """
    generator = CodeGenerator(
        prompt_path=prompt_path,
        controllers_dir=controllers_dir,
        factory_path=factory_path,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    
    return generator.generate_and_register(
        action_info_path=action_info_path,
        yaml_path=yaml_path,
        temperature=temperature
    )


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成控制器代码并自动注册"
    )
    
    parser.add_argument(
        '--action-info',
        type=str,
        required=True,
        help='动作信息文件路径（如 agent/protocol/action_information/actions_20251127_161552.txt）'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Prompt 文件路径（默认：agent/action/prompts/code_prompt.txt）'
    )
    
    parser.add_argument(
        '--yaml',
        type=str,
        default=None,
        help='YAML 配置文件路径（如果未指定，根据动作信息文件名自动查找）'
    )
    
    parser.add_argument(
        '--controllers-dir',
        type=str,
        default=None,
        help='controllers 目录路径（默认：项目根目录/controllers）'
    )
    
    parser.add_argument(
        '--factory',
        type=str,
        default=None,
        help='controller_factory.py 路径（默认：项目根目录/factories/controller_factory.py）'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='LLM 温度参数（默认：1.0）'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API 密钥（默认从环境变量获取）'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='API 基础 URL（默认使用 OpenAI 客户端配置）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LLM 模型名称（默认从 ACTION_AGENT_MODEL 读取）'
    )
    
    args = parser.parse_args()
    
    # 执行生成和注册
    try:
        controller_path, class_name, register_name = generate_controller_code(
            action_info_path=args.action_info,
            prompt_path=args.prompt,
            yaml_path=args.yaml,
            controllers_dir=args.controllers_dir,
            factory_path=args.factory,
            temperature=args.temperature,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
    except Exception as e:
        print(f"✗ 错误: {e}")
        exit(1)


if __name__ == '__main__':
    main()
