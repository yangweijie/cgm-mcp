#!/bin/bash
# 简化版 MCP 服务器启动脚本

# 设置项目路径
PROJECT_DIR="/Volumes/data/git/python/cgm-mcp"
cd "$PROJECT_DIR"

# 激活虚拟环境
source venv/bin/activate

# 设置 Python 路径
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# 从配置文件加载环境变量
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/config.ollama_cloud.json}"
if [ -f "$CONFIG_FILE" ]; then
    export CGM_LLM_PROVIDER=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('provider', 'ollama_cloud'))" 2>/dev/null || echo "ollama_cloud")
    export CGM_LLM_MODEL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('model', 'minimax-m2'))" 2>/dev/null || echo "minimax-m2")
    export CGM_LLM_API_KEY=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('api_key', ''))" 2>/dev/null || echo "")
    export CGM_LLM_API_BASE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('api_base', 'https://ollama.com'))" 2>/dev/null || echo "https://ollama.com")
    export CGM_LOG_LEVEL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server'].get('log_level', 'INFO'))" 2>/dev/null || echo "INFO")
fi

# 启动服务器
exec python "$PROJECT_DIR/run_mcp_server.py"