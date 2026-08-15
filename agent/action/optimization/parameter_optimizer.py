"""
Parameter Optimizer - 参数优化器

根据执行日志中的失败信息，调用LLM优化controller代码中的可选参数
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from openai import OpenAI


class ParameterOptimizer:
    """参数优化器 - 优化controller中的可选参数"""
    
    def __init__(
        self,
        optimize_prompt_path: str = None,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        """
        初始化优化器
        
        Args:
            optimize_prompt_path: 优化提示词文件路径
            api_key: OpenAI API密钥
            base_url: API基础URL
        """
        # 设置优化提示词路径
        if optimize_prompt_path is None:
            base_dir = Path(__file__).parent.parent  # agent/
            optimize_prompt_path = base_dir / "prompts" / "optimize_prompt.txt"
        
        self.optimize_prompt_path = Path(optimize_prompt_path)
        
        # 读取优化提示词
        with open(self.optimize_prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        
        # 初始化OpenAI客户端
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("ACTION_AGENT_MODEL")
        if not self.model:
            raise ValueError("ACTION_AGENT_MODEL is required")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        
        print(f"[Optimizer] Initialized")
        print(f"[Optimizer] Prompt: {self.optimize_prompt_path}")
    
    def optimize(
        self,
        original_code: str,
        log_files: List[Path],
        iteration: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        优化controller代码
        
        Args:
            original_code: 原始controller代码
            log_files: 所有迭代的日志文件列表
            iteration: 当前迭代次数
            
        Returns:
            (分析文本, 优化后的代码) 元组，失败返回 (None, None)
        """
        print(f"\n[Optimizer] === Optimizing Code (Iteration {iteration}) ===")
        
        # 构建用户消息
        user_message = self._build_user_message(original_code, log_files)
        
        # 组装对话
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        print(f"[Optimizer] Calling LLM API...")
        
        try:
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1
            )
            
            # 提取生成的内容
            if response and response.choices and response.choices[0].message:
                full_response = response.choices[0].message.content or ''
                
                # 解析分析文本和代码
                analysis_text, optimized_code = self._parse_response(full_response)
                
                if optimized_code:
                    print(f"[Optimizer] ✓ Code optimized successfully")
                    if analysis_text:
                        print(f"[Optimizer] ✓ Analysis extracted ({len(analysis_text)} chars)")
                    return analysis_text, optimized_code
                else:
                    print(f"[Optimizer] ✗ Failed to extract code from response")
                    return None, None
            else:
                print(f"[Optimizer] ✗ Invalid response from LLM")
                return None, None
        
        except Exception as e:
            print(f"[Optimizer] ✗ Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _build_user_message(self, original_code: str, log_files: List[Path]) -> str:
        """构建用户消息"""
        # 读取所有日志文件
        all_logs = []
        for log_file in log_files:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    iteration_num = self._extract_iteration_number(log_file)
                    all_logs.append(f"=== Iteration {iteration_num} Log ===\n{log_content}\n")
        
        logs_text = "\n".join(all_logs)
        
        # 构建消息
        user_message = f"""# 原始Controller代码

```python
{original_code}
```

# 执行日志（按迭代顺序）

{logs_text}

# 任务

请根据以上日志中的失败信息，优化controller代码中的可选参数。

输出格式：
1. 先输出问题分析与解决思路（纯文本）
2. 然后输出分隔符 `===ANALYSIS_END===`
3. 最后输出完整的优化后的Python代码（无markdown标记）
"""
        
        return user_message
    
    def _extract_iteration_number(self, log_file: Path) -> int:
        """从日志文件路径提取迭代编号"""
        # 支持两种路径格式：
        # 1. 新格式: .../iterations/N/execution_log.txt (父目录是数字)
        # 2. 旧格式: .../iteration_N/execution_log.txt (父目录是 iteration_N)
        try:
            parent_name = log_file.parent.name
            
            # 新格式：父目录名直接是数字
            if parent_name.isdigit():
                return int(parent_name)
            
            # 旧格式：父目录名是 iteration_N
            if parent_name.startswith("iteration_"):
                return int(parent_name.split("_")[1])
        except:
            pass
        return 0
    
    def _parse_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析LLM响应，分离分析文本和代码
        
        Args:
            response: LLM的完整响应
            
        Returns:
            (分析文本, 代码) 元组
        """
        separator = "===ANALYSIS_END==="
        
        if separator in response:
            # 有分隔符，按分隔符分割
            parts = response.split(separator, 1)
            analysis_text = parts[0].strip() if len(parts) > 0 else None
            code_text = parts[1].strip() if len(parts) > 1 else None
        else:
            # 没有分隔符，尝试查找代码块
            # 如果包含```python或```，尝试提取代码部分
            if "```python" in response or "```" in response:
                # 尝试提取代码块
                code_text = self._extract_code_block(response)
                analysis_text = None
            else:
                # 没有代码块标记，整个响应作为代码
                analysis_text = None
                code_text = response.strip()
        
        # 清理代码
        if code_text:
            code_text = self._clean_code(code_text)
        
        return analysis_text, code_text
    
    def _extract_code_block(self, text: str) -> Optional[str]:
        """从文本中提取代码块"""
        lines = text.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            # 检测代码块标记
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 在代码块内的行
            if in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        return None
    
    def _clean_code(self, code: str) -> str:
        """清理LLM生成的代码，移除markdown标记"""
        lines = code.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            # 检测代码块标记
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 只保留代码行
            cleaned_lines.append(line)
        
        # 移除开头和结尾的空行
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)


def optimize_controller(
    original_code: str,
    log_files: List[Path],
    iteration: int,
    output_path: Optional[Path] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    便捷函数：优化controller代码
    
    Args:
        original_code: 原始controller代码
        log_files: 所有迭代的日志文件列表
        iteration: 当前迭代次数
        output_path: 可选的输出文件路径
        
    Returns:
        (分析文本, 优化后的代码) 元组，失败返回 (None, None)
    """
    optimizer = ParameterOptimizer()
    analysis_text, optimized_code = optimizer.optimize(original_code, log_files, iteration)
    
    if optimized_code and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(optimized_code, encoding='utf-8')
        print(f"[Optimizer] Saved to: {output_path}")
    
    return analysis_text, optimized_code
