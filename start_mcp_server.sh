#!/bin/bash

# CGM MCP Server 启动脚本
# 用于启动 CGM Model Context Protocol 服务器

set -e  # 遇到错误时退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置项目根目录
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PATH="$PROJECT_ROOT/venv"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config.ollama_cloud.json}"

echo "CGM MCP Server 启动脚本"
echo "项目根目录: $PROJECT_ROOT"
echo "虚拟环境: $VENV_PATH"
echo "配置文件: $CONFIG_FILE"
echo

# 检查 Python 和虚拟环境
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "错误: 虚拟环境不存在，请先创建虚拟环境。"
    echo "运行: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export CGM_CONFIG_FILE="$CONFIG_FILE"

# 从配置文件加载环境变量
if [ -f "$CONFIG_FILE" ]; then
    echo "加载配置文件: $CONFIG_FILE"
    
    # 从 JSON 配置中提取值并设置环境变量
    if command -v python3 >/dev/null 2>&1; then
        export CGM_LLM_PROVIDER=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('provider', 'ollama_cloud'))" 2>/dev/null || echo "ollama_cloud")
        export CGM_LLM_MODEL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('model', 'minimax-m2'))" 2>/dev/null || echo "minimax-m2")
        export CGM_LLM_API_KEY=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('api_key', ''))" 2>/dev/null || echo "")
        export CGM_LLM_API_BASE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['llm'].get('api_base', 'https://ollama.com'))" 2>/dev/null || echo "https://ollama.com")
        export CGM_LOG_LEVEL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server'].get('log_level', 'INFO'))" 2>/dev/null || echo "INFO")
    fi
else
    echo "警告: 配置文件不存在，使用默认设置"
    export CGM_LLM_PROVIDER="ollama_cloud"
    export CGM_LLM_MODEL="minimax-m2"
    export CGM_LLM_API_KEY="5de86cef7c224fcea7d504ec55d72349.cMLIphFoOsGhbrJ5f85awiwG"
    export CGM_LLM_API_BASE="https://ollama.com"
    export CGM_LOG_LEVEL="INFO"
fi

echo "LLM 提供商: $CGM_LLM_PROVIDER"
echo "LLM 模型: $CGM_LLM_MODEL"
echo "日志级别: $CGM_LOG_LEVEL"
echo

# 检查必要的依赖
echo "检查依赖..."
python -c "import cgm_mcp.server; print('依赖检查通过')" || {
    echo "错误: 依赖检查失败，请确保已安装 requirements.txt 中的所有依赖。"
    exit 1
}
echo

# 启动 MCP 服务器
echo "正在启动 CGM MCP Server..."
echo "命令: python run_mcp_server.py"
echo

# 设置标准输入输出环境，适合 MCP 使用
exec python "$PROJECT_ROOT/run_mcp_server.py"