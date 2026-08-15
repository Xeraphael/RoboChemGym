"""
Protocol Extractor - 化学协议提取模块

从化学实验协议文本中提取器材名称和原子动作序列
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI


class ProtocolExtractor:
    """化学协议提取器 - 从文本协议中提取器材和动作序列"""
    
    def __init__(
        self,
        prompt_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化协议提取器
        
        Args:
            prompt_path: 提示词文件路径，默认使用同目录下的 protocol_extract_prompt.txt
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
        """
        # 设置提示词路径
        if prompt_path is None:
            base_dir = Path(__file__).parent
            prompt_path = base_dir / "protocol_extract_prompt.txt"
        
        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        # 读取提示词
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        
        # 调试：打印提示词长度（不打印完整内容，太长）
        print(f"[ProtocolExtractor] Loaded prompt from {self.prompt_path} ({len(self.system_prompt)} chars)")
        
        # 初始化OpenAI客户端
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
    
    def extract(
        self,
        protocol_text: str,
        temperature: float = 0.7,
        output_dir: Optional[str] = None
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        从协议文本中提取器材和动作序列
        
        Args:
            protocol_text: 化学实验协议文本
            temperature: LLM温度参数
            output_dir: 输出目录，如果为None则不保存文件
            
        Returns:
            (equipment_text, actions_text, equipment_file_path, actions_file_path)
            - equipment_text: 器材名称表文本
            - actions_text: 动作序列文本
            - equipment_file_path: 器材名称表文件路径（如果保存了）
            - actions_file_path: 动作序列文件路径（如果保存了）
        """
        # 检查协议文本
        if not protocol_text or not protocol_text.strip():
            raise ValueError("Protocol text is empty or contains only whitespace")
        
        print(f"[ProtocolExtractor] Protocol text length: {len(protocol_text)} chars")
        print(f"[ProtocolExtractor] Protocol text preview: {protocol_text[:200]}...")
        
        # 组装消息
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": protocol_text
            }
        ]
        
        # 调用LLM
        print(f"[ProtocolExtractor] Calling LLM (model: {self.model}) to extract protocol...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        
        # 提取响应内容
        content = ''
        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ''
        
        # 调试：打印LLM响应的前1000个字符
        if content:
            preview = content[:1000] if len(content) > 1000 else content
            print(f"[ProtocolExtractor] LLM response preview ({len(content)} chars):")
            print("=" * 60)
            print(preview)
            if len(content) > 1000:
                print("... (truncated)")
            print("=" * 60)
        else:
            print("[ProtocolExtractor] Warning: Empty response from LLM")
        
        # 解析响应，提取器材和动作文本
        equipment_text, actions_text = self._parse_response(content)
        
        # 后处理：自动检测并补充试剂容器
        equipment_text = self._add_reagent_containers(equipment_text, protocol_text)
        
        # 保存文件（如果指定了输出目录）
        equipment_file_path = None
        actions_file_path = None
        
        if output_dir is not None:
            output_path = Path(output_dir)
            
            # 创建输出目录结构
            scene_info_dir = output_path / "scene_information"
            action_info_dir = output_path / "action_information"
            scene_info_dir.mkdir(parents=True, exist_ok=True)
            action_info_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存器材名称表到 scene_information 文件夹
            equipment_file_path = scene_info_dir / f"equipment_{timestamp}.txt"
            with open(equipment_file_path, 'w', encoding='utf-8') as f:
                f.write(equipment_text)
            
            # 保存动作序列到 action_information 文件夹
            actions_file_path = action_info_dir / f"actions_{timestamp}.txt"
            with open(actions_file_path, 'w', encoding='utf-8') as f:
                f.write(actions_text)
            
            print(f"[ProtocolExtractor] Equipment list saved to: {equipment_file_path}")
            print(f"[ProtocolExtractor] Actions sequence saved to: {actions_file_path}")
        
        return equipment_text, actions_text, str(equipment_file_path) if equipment_file_path else None, str(actions_file_path) if actions_file_path else None
    
    def _parse_response(self, content: str) -> Tuple[str, str]:
        """
        解析LLM响应，提取器材和动作文本
        
        Args:
            content: LLM返回的文本内容
            
        Returns:
            (equipment_text, actions_text) 元组
        """
        equipment_text = ""
        actions_text = ""
        
        try:
            # 提取器材部分
            if "===EQUIPMENT_START===" in content and "===EQUIPMENT_END===" in content:
                start = content.find("===EQUIPMENT_START===") + len("===EQUIPMENT_START===")
                end = content.find("===EQUIPMENT_END===")
                equipment_text = content[start:end].strip()
            else:
                print("[ProtocolExtractor] Warning: Equipment markers not found in response")
            
            # 提取动作部分
            if "===ACTIONS_START===" in content and "===ACTIONS_END===" in content:
                start = content.find("===ACTIONS_START===") + len("===ACTIONS_START===")
                end = content.find("===ACTIONS_END===")
                actions_text = content[start:end].strip()
            else:
                print("[ProtocolExtractor] Warning: Actions markers not found in response")
            
            # 如果标记未找到，尝试提取整个内容
            if not equipment_text and not actions_text:
                print("[ProtocolExtractor] Warning: No markers found in LLM response")
                print("[ProtocolExtractor] This usually means:")
                print("  1. LLM did not follow the output format requirements")
                print("  2. LLM returned an error message instead of extracted content")
                print("  3. The response format is different from expected")
                print("[ProtocolExtractor] Attempting to use full content as fallback...")
                # 检查内容是否看起来像错误消息
                if "需要" in content or "请提供" in content or "无法" in content:
                    print("[ProtocolExtractor] Warning: Response appears to be an error message, not extracted content")
                    print(f"[ProtocolExtractor] Response starts with: {content[:200]}")
                # 尝试智能分割（如果可能）
                equipment_text = content
                actions_text = content
            
        except Exception as e:
            print(f"[ProtocolExtractor] Error parsing response: {e}")
            print(f"[ProtocolExtractor] Raw content preview: {content[:500]}...")
            equipment_text = ""
            actions_text = ""
        
        return equipment_text, actions_text
    
    def _add_reagent_containers(
        self, 
        equipment_text: str, 
        protocol_text: str
    ) -> str:
        """
        后处理：自动检测协议中的试剂，并确保在器材列表中添加对应的容器
        
        Args:
            equipment_text: 提取的器材列表文本
            protocol_text: 原始协议文本
            
        Returns:
            补充后的器材列表文本
        """
        import re
        
        # 解析现有器材列表
        equipment_lines = [line.strip() for line in equipment_text.split('\n') if line.strip()]
        equipment_set = set(equipment_lines)
        
        solid_pattern = r'(\d+(?:\.\d+)?)\s*(?:g|kg|mg|毫克|克|千克)\s+'
        liquid_pattern = r'(\d+(?:\.\d+)?)\s*(?:ml|mL|L|毫升|升)\s+'
        
        solid_count = len(re.findall(solid_pattern, protocol_text, re.IGNORECASE))
        liquid_count = len(re.findall(liquid_pattern, protocol_text, re.IGNORECASE))
        
        added_containers = []
        
        for i in range(1, solid_count + 1):
            container_name = f"ErlenmeyerFlask_Solid{i}"
            if container_name not in equipment_set:
                added_containers.append(container_name)
                equipment_set.add(container_name)
        
        for i in range(1, liquid_count + 1):
            container_name = f"ErlenmeyerFlask_Liquid{i}"
            if container_name not in equipment_set:
                added_containers.append(container_name)
                equipment_set.add(container_name)
        
        # 如果有新添加的容器，更新器材列表
        if added_containers:
            print(f"[ProtocolExtractor] 自动添加试剂容器: {', '.join(added_containers)}")
            equipment_lines.extend(added_containers)
            equipment_text = '\n'.join(equipment_lines)
        
        return equipment_text


def extract_protocol(
    protocol_text: str,
    prompt_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    便捷函数：从协议文本中提取器材和动作序列
    
    Args:
        protocol_text: 化学实验协议文本
        prompt_path: 提示词文件路径
        output_dir: 输出目录
        api_key: API密钥
        base_url: API基础URL
        
    Returns:
        (equipment_text, actions_text, equipment_file_path, actions_file_path)
    """
    extractor = ProtocolExtractor(
        prompt_path=prompt_path,
        api_key=api_key,
        base_url=base_url
    )
    return extractor.extract(protocol_text, output_dir=output_dir)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="化学协议提取器 - 从协议文本中提取器材和动作序列")
    parser.add_argument(
        '--protocol',
        type=str,
        required=True,
        help='化学实验协议文本文件路径或直接输入文本'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='提示词文件路径（默认使用 protocol_extract_prompt.txt）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认与protocol文件同目录，或当前目录）'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API密钥（默认从环境变量读取）'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='API基础URL（默认使用配置的URL）'
    )
    
    args = parser.parse_args()
    
    # 读取协议文本
    protocol_path = Path(args.protocol)
    if protocol_path.exists():
        # 从文件读取
        with open(protocol_path, 'r', encoding='utf-8') as f:
            protocol_text = f.read()
        # 如果未指定输出目录，使用protocol目录下的protocol文件夹
        if args.output_dir is None:
            args.output_dir = str(protocol_path.parent / "protocol")
    else:
        # 直接使用输入作为协议文本
        protocol_text = args.protocol
        if args.output_dir is None:
            args.output_dir = str(Path.cwd() / "protocol")
    
    # 创建提取器并执行提取
    extractor = ProtocolExtractor(
        prompt_path=args.prompt,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    equipment_text, actions_text, equipment_file, actions_file = extractor.extract(
        protocol_text,
        output_dir=args.output_dir
    )
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("提取结果摘要")
    print("="*60)
    equipment_lines = [line.strip() for line in equipment_text.split('\n') if line.strip()]
    print(f"器材数量: {len(equipment_lines)}")
    actions_lines = [line.strip() for line in actions_text.split('\n') if line.strip()]
    print(f"动作描述行数: {len(actions_lines)}")
    
    if equipment_file:
        print(f"\n器材名称表: {equipment_file}")
    if actions_file:
        print(f"动作序列: {actions_file}")
    
    print("="*60)


if __name__ == '__main__':
    main()
