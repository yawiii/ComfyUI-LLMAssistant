import os
import re
import json
import requests
from pathlib import Path

class LLMPromptAssistant:
    def __init__(self):
        self.ollama_host = "http://192.168.10.111:11434"
        self.template_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")
        os.makedirs(self.template_dir, exist_ok=True)
        
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
                "ollama_host": ("STRING", {
                    "default": "http://192.168.10.111:11434",
                    "multiline": False
                }),
                "model": (cls.list_models(), ),  # 注意这里添加了()
                "enable_enhancement": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_prompt", "output_prompt", "tagger_prompt")
    FUNCTION = "process_prompt"
    CATEGORY = "LLM"

    @classmethod
    def list_models(cls):
        try:
            response = requests.get("http://192.168.10.111:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models['models']]
        except:
            pass
        return ["llama2"]  # 默认返回值

    def process_prompt(self, text, ollama_host, model, enable_enhancement):
        self.ollama_host = ollama_host
        
        if enable_enhancement:
            # 获取优化结果
            optimized = self.enhance_prompt(text, model)
            
            # 提取think标签内容
            think_match = re.search(r'<think>(.*?)</think>', optimized, re.DOTALL)
            think_content = think_match.group(0) if think_match else "<think>分析过程缺失</think>"
            
            # 提取优化后的提示词
            enhanced_text = self.extract_enhanced_text(optimized)
            
            # 翻译优化后的提示词
            translated = self.translate_text(enhanced_text, model)
            
            # 按固定格式组装输出
            preview_prompt = (
                f"原文：\n{text}\n\n"
                f"{think_content}\n\n"
                f"优化后的提示词：\n{enhanced_text}\n\n"
                f"译文：\n{translated}"
            )
            weighted = self.generate_tagger_prompt(translated)
        else:
            # 直接翻译
            translated = self.translate_text(text, model)
            preview_prompt = f"原文：\n{text}\n\n译文：\n{translated}"
            weighted = self.generate_tagger_prompt(translated)
            
        return (preview_prompt, translated, weighted)

    def enhance_prompt(self, text, model):
        template = self.read_template("enhancement_template.txt")
        if not template:
            return f"<think>无法读取优化模板文件</think>"
            
        # 发送请求并获取响应
        response = self.call_ollama(template.format(text=text), model)
        
        # 提取think标签内容和优化后的提示词
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        prompt_match = re.search(r'优化后的提示词：\s*(.*?)(?:\n\n|$)', response, re.DOTALL)
        
        # 如果响应不符合预期，尝试二次处理
        if not think_match or not prompt_match:
            # 尝试修复格式
            retry_prompt = """请将以下内容整理为规范格式：
1. 分析部分放在<think>标签内
2. 场景描述作为优化后的提示词输出
3. 去掉所有标题和分类
4. 确保描述自然流畅

内容：
{text}""".format(text=response)
            
            response = self.call_ollama(retry_prompt, model)
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            prompt_match = re.search(r'优化后的提示词：\s*(.*?)(?:\n\n|$)', response, re.DOTALL)
        
        # 组装最终输出
        analysis = think_match.group(1).strip() if think_match else "请分析场景的主体、环境、构图、光影和细节特征"
        prompt = prompt_match.group(1).strip() if prompt_match else response.strip()
        
        # 清理提示词格式
        prompt = re.sub(r'^\[.*?\]', '', prompt)  # 移除开头的说明文字
        prompt = re.sub(r'\[.*?\]', '', prompt)   # 移除所有方括号标记
        prompt = re.sub(r'^[,，]', '', prompt)    # 移除开头的逗号
        prompt = ' '.join(line.strip() for line in prompt.split('\n')) # 合并多行
        
        return f"<think>\n{analysis}\n</think>\n\n优化后的提示词：\n{prompt.strip()}"

    def translate_text(self, text, model):
        template = self.read_template("translation_template.txt")
        if not template:
            return text
        
        # 确保文本不为空
        if not text.strip():
            return ""
            
        response = self.call_ollama(template.format(text=text), model)
        return self.clean_translation(response)

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

    def extract_enhanced_text(self, text):
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
                    "stream": False,
                    "raw": False
                }
            )
            if response.status_code == 200:
                return response.json().get("response", prompt).strip()
            return prompt
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return prompt

    def generate_tagger_prompt(self, text):
        """生成符合CLIP模型理解的提示词"""
        # 基础质量标签
        quality_tags = [
            "(masterpiece:1.2)",
            "(best quality:1.2)", 
            "(detailed:1.1)",
            "(sharp focus:1.1)"
        ]
        
        # 如果输入为空，仅返回质量标签
        if not text.strip():
            return ", ".join(quality_tags)
            
        # 清理并提取关键词
        words = []
        text = text.lower()  # 转换为小写以便处理
        
        # 移除干扰词
        text = re.sub(r'\b(is|are|the|a|an|with|by|in|on|at|to|and|of)\b', '', text)
        
        # 分割文本
        parts = [p.strip() for p in re.split(r'[,.;，。；]', text) if p.strip()]
        
        # 处理每个部分
        for part in parts:
            # 清理多余空格
            clean_part = re.sub(r'\s+', ' ', part).strip()
            if clean_part:
                words.append(clean_part)
                
        # 添加常用效果标签
        style_tags = []
        if any(word in text for word in ['cinematic', 'movie', 'film']):
            style_tags.extend([
                "(cinematic lighting:1.2)",
                "(dramatic atmosphere:1.1)"
            ])
            
        if any(word in text for word in ['space', 'futuristic', 'sci-fi']):
            style_tags.extend([
                "(sci-fi:1.2)",
                "(futuristic:1.1)",
                "(high tech:1.1)"
            ])
            
        # 组合所有标签
        all_tags = quality_tags + style_tags + words
        
        # 去重并保持顺序
        seen = set()
        result = []
        for item in all_tags:
            if item not in seen:
                seen.add(item)
                result.append(item)
                
        return ", ".join(result)

    def default_tagger_prompt(self, text):
        """备用的提示词处理方法"""
        return self.generate_tagger_prompt(text)  # 使用主方法作为备用
