import os
import re
import json
import requests
from pathlib import Path
from llm_assistant import LLMPromptAssistant

class APIPromptAssistant(LLMPromptAssistant):
    def __init__(self):
        super().__init__()
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "node_config.json")
        self.load_config()
    
    def load_config(self):
        """加载API配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self.config = {
                "api_key": "",
                "api_base": "https://api.openai.com/v1",
                "api_model": "gpt-3.5-turbo",
                "api_temperature": 0.7,
                "api_max_tokens": 1000
            }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "enable_enhancement": ("BOOLEAN", {"default": False}),
            }
        }
    
    def call_ollama(self, prompt, model):
        """重写调用方法，使用API替代Ollama"""
        try:
            print(f"[API Debug] 准备发送请求到: {self.config['api_base']}")
            print(f"[API Debug] 使用模型: {self.config['api_model']}")
            
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "model": self.config['api_model'],
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config['api_temperature'],
                "max_tokens": self.config['api_max_tokens']
            }
            
            print(f"[API Debug] 请求数据: {json.dumps(request_data, ensure_ascii=False)}")
            
            response = requests.post(
                f"{self.config['api_base']}/chat/completions",
                headers=headers,
                json=request_data
            )
            
            print(f"[API Debug] 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                print(f"[API Debug] 响应数据: {json.dumps(response_data, ensure_ascii=False)}")
                return response_data['choices'][0]['message']['content'].strip()
            else:
                print(f"[API Debug] 错误响应: {response.text}")
            return prompt
        except Exception as e:
            print(f"[API Debug] 发生错误: {str(e)}")
            return prompt
    
    def process_prompt(self, text, enable_enhancement):
        """重写处理方法，移除不需要的参数"""
        if enable_enhancement:
            optimized = self.enhance_prompt(text, None)
            think_match = re.search(r'<think>(.*?)</think>', optimized, re.DOTALL)
            think_content = think_match.group(0) if think_match else "<think>分析过程缺失</think>"
            enhanced_text = self.extract_enhanced_text(optimized)
            translated = self.translate_text(enhanced_text, None)
            preview_prompt = (
                f"原文：\n{text}\n\n"
                f"{think_content}\n\n"
                f"优化后的提示词：\n{enhanced_text}\n\n"
                f"译文：\n{translated}"
            )
            weighted = self.generate_tagger_prompt(translated)
        else:
            translated = self.translate_text(text, None)
            preview_prompt = f"原文：\n{text}\n\n译文：\n{translated}"
            weighted = self.generate_tagger_prompt(translated)
        
        return (preview_prompt, translated, weighted)