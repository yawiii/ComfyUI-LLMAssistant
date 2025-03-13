import os
import re
import json
import requests
from pathlib import Path

class OllamaPromptAssistant:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ollama_config.json")
        self.load_config()
        os.makedirs(self.template_dir, exist_ok=True)
        
    def load_config(self):
        """加载Ollama配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.ollama_host = config['ollama_host']
                self.template_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['template_dir'])
        except FileNotFoundError:
            print(f"[Error] 配置文件不存在: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"[Error] 配置文件格式无效: {str(e)}")
            raise
        except KeyError as e:
            print(f"[Error] 配置文件缺少必要字段: {str(e)}")
            raise
        except Exception as e:
            print(f"[Error] 加载配置文件失败: {str(e)}")
            raise
        
    def read_template(self, template_name):
        """读取模板文件"""
        template_path = os.path.join(self.template_dir, template_name)
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading template {template_name}: {str(e)}")
            return None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": (cls.list_models(), ),  # 注意这里添加了()
                "prompt_expansion": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_prompt", "output_prompt", "tagger_prompt")
    FUNCTION = "process_prompt"
    CATEGORY = "LLM-Assistant"

    @classmethod
    def list_models(cls):
        try:
            # 创建一个临时实例来获取配置
            instance = cls()
            response = requests.get(f"{instance.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models['models']]
        except:
            pass
        return ["llama2"]  # 默认返回值

    def process_prompt(self, text, model, prompt_expansion):
        if prompt_expansion:
            optimized = self.expansion_prompt(text, model)
            expansion_text = self.extract_expansion_text(optimized)
            translated = self.translate_text(expansion_text, model)
            
            # 根据是否存在think标签决定输出格式
            think_match = re.search(r'<think>(.*?)</think>', optimized, re.DOTALL)
            if think_match:
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
            weighted = self.generate_tagger_prompt(translated, model)
        else:
            translated = self.translate_text(text, model)
            preview_prompt = f"原文：\n{text}\n\n译文：\n{translated}"
            weighted = self.generate_tagger_prompt(translated, model)
        
        return (preview_prompt, translated, weighted)

    def expansion_prompt(self, text, model):
        template = self.read_template("expansion_template.txt")
        if not template:
            return text
            
        response = self.call_ollama(template.format(text=text), model)
        return response.strip()

    def translate_text(self, text, model):
        template = self.read_template("translation_template.txt")
        if not template:
            print("[Error] 无法读取翻译模板文件")
            return text
        
        # 确保文本不为空
        if not text.strip():
            return ""
            
        try:
            response = self.call_ollama(template.format(text=text), model)
            return self.clean_translation(response)
        except Exception as e:
            print(f"[Error] 翻译过程中发生错误: {str(e)}")
            return text

    def clean_translation(self, text):
        """清理翻译结果"""
        # 移除所有中文字符
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # 移除无关内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)translation:.*?text:', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'Scene Description:', '', text, flags=re.IGNORECASE)
        
        # 清理常见前缀
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
        
        # 清理格式
        result = ' '.join(result.split())
        result = re.sub(r'\s+([,.!?])', r'\1', result)  # 修复标点符号前的空格
        
        return result.strip()

    def extract_expansion_text(self, text):
        """从优化结果中提取提示词"""
        # 移除think标签及其内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 移除所有方括号标记及其内容
        text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
        
        # 提取"优化后的提示词："后面的内容
        match = re.search(r'优化后的提示词：\s*(.*?)(?:\n\n译文：|$)', text, re.DOTALL)
        if match:
            result = match.group(1).strip()
            # 移除开头可能的逗号和场景描述等标记
            result = re.sub(r'^[,，]', '', result)
            result = re.sub(r'(场景描述：|分析：|建议：)', '', result)
            return result.strip()
            
        return text.strip()

    def extract_prompt(self, text):
        """提取提示词内容，去除思考过程"""
        # 移除think标签及其内容，保留其他部分
        parts = re.split(r'<think>.*?</think>', text, flags=re.DOTALL)
        # 保留非空的部分
        filtered = [part.strip() for part in parts if part.strip()]
        # 如果没有有效内容，返回空字符串
        return filtered[-1] if filtered else ""

    def call_ollama(self, prompt, model):
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API请求失败: {response.status_code} - {response.text}")
                
            response_data = response.json()
            return response_data['response']
                
        except Exception as e:
            print(f"[Error] 调用Ollama API时发生错误: {str(e)}")
            raise

    def generate_tagger_prompt(self, text, model):
        """生成符合CLIP模型理解的提示词"""
        if not text.strip():
            return ""
        
        template = self.read_template("tagger_template.txt")
        if not template:
            return ""
            
        try:
            response = self.call_ollama(template.format(text=text), model)
            
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
