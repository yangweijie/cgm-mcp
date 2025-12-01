#!/usr/bin/env python3
"""
示例脚本：如何使用Ollama Cloud提供程序
"""

import os
from src.cgm_mcp.utils.config import LLMConfig
from src.cgm_mcp.utils.llm_client import LLMClient

def show_ollama_cloud_usage():
    """
    展示如何配置和使用Ollama Cloud提供程序
    """
    print("## 使用Ollama Cloud提供程序")
    print()
    print("### 环境变量配置:")
    print("```bash")
    print("# 设置Ollama Cloud提供程序")
    print("export CGM_LLM_PROVIDER=ollama_cloud")
    print("# 设置您的Ollama Cloud API密钥")
    print("export CGM_LLM_API_KEY=your-ollama-cloud-api-key")
    print("# 设置模型名称")
    print("export CGM_LLM_MODEL=llama3")
    print("# 可选：设置自定义API基础URL")
    print("# export CGM_LLM_API_BASE=https://api.ollama.example.com")
    print("```")
    print()
    print("### 代码配置示例:")
    print("```python")
    print("# 创建Ollama Cloud配置")
    print('config = LLMConfig(')
    print('    provider="ollama_cloud",')
    print('    model="llama3",')
    print('    api_key=os.getenv("CGM_LLM_API_KEY"),')
    print('    api_base=os.getenv("CGM_LLM_API_BASE", "https://api.ollama.example.com"),')
    print('    temperature=0.1,')
    print('    max_tokens=2000,')
    print('    timeout=60')
    print(')')
    print()
    print("# 创建LLM客户端")
    print("llm_client = LLMClient(config)")
    print("```")
    print()
    print("### 配置文件示例 (config.ollama_cloud.json):")
    print("```json")
    print('{')
    print('  "llm": {')
    print('    "provider": "ollama_cloud",')
    print('    "model": "llama3",')
    print('    "api_base": "https://api.ollama.example.com",  // 替换为实际的Ollama Cloud端点')
    print('    "temperature": 0.1,')
    print('    "max_tokens": 4000,')
    print('    "timeout": 60')
    print('  }')
    print('}')
    print("```")
    print()
    print("### 启动服务器:")
    print("```bash")
    print("# 使用配置文件启动")
    print("python main.py --config config.ollama_cloud.json")
    print()
    print("# 或使用环境变量启动")
    print("CGM_LLM_PROVIDER=ollama_cloud CGM_LLM_API_KEY=your-key python main.py")
    print("```")

if __name__ == "__main__":
    show_ollama_cloud_usage()