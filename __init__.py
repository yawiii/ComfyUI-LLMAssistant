import os
import sys

# 添加调试信息
print("Loading ComfyUI-LLMAssistant...")
print(f"Current directory: {os.path.dirname(os.path.realpath(__file__))}")

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 创建必要的目录
templates_dir = os.path.join(current_dir, "templates")
os.makedirs(templates_dir, exist_ok=True)

# 导入节点类
from llm_assistant import LLMPromptAssistant
from api_assistant import APIPromptAssistant

# 定义节点映射
NODE_CLASS_MAPPINGS = {
    "LLMPromptAssistant": LLMPromptAssistant,
    "APIPromptAssistant": APIPromptAssistant
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptAssistant": "LLM提示词助手（Ollama）",
    "APIPromptAssistant": "LLM提示词助手（API）"
}

# 声明插件的安全配置
NODE_CLASS_SECURITY_MAPPINGS = {
    "LLMPromptAssistant": {"level": "normal", "api_keys_required": ["OPENAI_API_KEY"]},
    "APIPromptAssistant": {"level": "normal", "api_keys_required": ["OPENAI_API_KEY"]}
}

print(f"NODE_CLASS_MAPPINGS: {NODE_CLASS_MAPPINGS}")

print("ComfyUI-LLMAssistant loaded successfully!")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'NODE_CLASS_SECURITY_MAPPINGS']
