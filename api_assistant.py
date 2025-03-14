import os
import re
import json
import requests
from pathlib import Path

class APIPromptAssistant:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "api_config.json")
        self.load_config()
    
    def load_config(self):
        """加载API配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"错误：配置文件不存在 - {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"错误：配置文件格式无效 - {str(e)}")
            raise
        except Exception as e:
            print(f"错误：加载配置文件失败 - {str(e)}")
            raise
    
    def load_tags_config(self):
        """加载标签配置"""
        try:
            with open(self.tags_path, 'r', encoding='utf-8') as f:
                self.tags_config = json.load(f)
        except Exception as e:
            print(f"[Error] 加载标签配置失败: {str(e)}")
            self.tags_config = {"quality_tags": [], "style_mappings": {}}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "prompt_expansion": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_prompt", "output_prompt", "tagger_prompt")
    FUNCTION = "process_prompt"
    CATEGORY = "LLM-Assistant"

    def call_api(self, prompt):
        """调用API获取响应"""
        try:
            if not self.config['api_key']:
                raise ValueError("API密钥未配置")
                
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "model": self.config['api_model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config['api_temperature'],
                "max_tokens": self.config['api_max_tokens']
            }
            
            print(f"[API] 发送请求到: {self.config['api_base']}，使用模型: {self.config['api_model']}")
            
            response = requests.post(
                f"{self.config['api_base']}/chat/completions",
                headers=headers,
                json=request_data
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
                
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            print(f"[Error] API调用失败: {str(e)}")
            raise
    
    def process_prompt(self, text, prompt_expansion):
        if prompt_expansion:
            optimized = self.expansion_prompt(text)
            expansion_text = self.extract_expansion_text(optimized)
            translated = self.translate_text(expansion_text)
            
            # 根据是否存在think标签决定输出格式
            think_match = re.search(r'<think>(.*?)</think>', optimized, re.DOTALL)
            if (think_match):
                preview_prompt = (
                    f"原文：\n{text}\n\n"
                    f"{think_match.group(0)}\n\n"
                    f"优化后的提示词：\n{expansion_text}\n\n"
                    f"译文：\n{translated}"
                )
            else:
                preview_prompt = (
                    f"原文：\n{text}\n\n"
                    f"优化后的提示词：\n{expansion_text}\n\n"
                    f"译文：\n{translated}"
                )
            weighted = self.generate_tagger_prompt(translated)
        else:
            translated = self.translate_text(text)
            preview_prompt = f"原文：\n{text}\n\n译文：\n{translated}"
            weighted = self.generate_tagger_prompt(translated)
        
        return (preview_prompt, translated, weighted)

    def expansion_prompt(self, text):
        template = self.read_template("expansion_template.txt")
        if not template:
            return text
            
        response = self.call_api(template.format(text=text))
        return response.strip()

    def translate_text(self, text):
        template = self.read_template("translation_template.txt")
        if not template:
            return text
        
        if not text.strip():
            return ""
            
        response = self.call_api(template.format(text=text))
        return self.clean_translation(response)

    def clean_translation(self, text):
        """清理翻译结果"""
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)translation:.*?text:', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'Scene Description:', '', text, flags=re.IGNORECASE)
        
        prefixes = [
            "here's the translation:",
            "translated text:",
            "english version:",
            "translation:",
            "英文翻译：",
            "翻译结果：",
            "请将下面的中文文本翻译成英文",
            "直接输出英文结果："
        ]
        
        result = text.strip()
        for prefix in prefixes:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        
        result = ' '.join(result.split())
        result = re.sub(r'\s+([,.!?])', r'\1', result)
        
        return result.strip()

    def extract_expansion_text(self, text):
        """从优化结果中提取提示词"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
        
        match = re.search(r'优化后的提示词：\s*(.*?)(?:\n\n译文：|$)', text, re.DOTALL)
        if match:
            result = match.group(1).strip()
            result = re.sub(r'^[,，]', '', result)
            result = re.sub(r'(场景描述：|分析：|建议：)', '', result)
            return result.strip()
            
        return text.strip()

    def extract_prompt(self, text):
        """提取提示词内容，去除思考过程"""
        parts = re.split(r'<think>.*?</think>', text, flags=re.DOTALL)
        filtered = [part.strip() for part in parts if part.strip()]
        return filtered[-1] if filtered else ""

    def read_template(self, template_name):
        """读取模板文件"""
        template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates", template_name)
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading template {template_name}: {str(e)}")
            return None

    def generate_tagger_prompt(self, text):
        """生成符合CLIP模型理解的提示词"""
        if not text.strip():
            return ""
        
        template = self.read_template("tagger_template.txt")
        if not template:
            return ""
            
        try:
            response = self.call_api(template.format(text=text))
            
            # 移除思考过程
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            # 如果存在类似分析/思考的段落（以冒号结尾的行），只保留最后部分
            if re.search(r'.*:\s*\n', response, re.MULTILINE):
                sections = [s.strip() for s in re.split(r'.*:\s*\n', response) if s.strip()]
                response = sections[-1] if sections else response
            
            # 清理响应中的中文字符
            response = re.sub(r'[\u4e00-\u9fff]', '', response)
            # 清理并规范化标签
            tags = [tag.strip() for tag in re.split(r'[,，]', response) if tag.strip()]
            return ", ".join(tags)
            
        except Exception as e:
            print(f"[Debug] 生成标签时出错: {str(e)}")
            return ""