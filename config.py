import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# 读取 .env 环境变量
load_dotenv()

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newapi-proxy")

# 路径配置
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_PORT = int(os.getenv("PORT", 14300))

# 覆写规则持久化文件路径
OVERRIDE_STORE_PATH = BASE_DIR / "override_store.json"
# 自定义模型持久化文件路径
CUSTOM_MODELS_PATH = BASE_DIR / "custom_models.json"

# 上游 newapi 默认配置（可选）
NEWAPI_BASE_URL = os.getenv("NEWAPI_BASE_URL") or None
NEWAPI_API_KEY = os.getenv("NEWAPI_API_KEY") or None

# 代理侧 API Key（可选）
PROXY_API_KEY = os.getenv("PROXY_API_KEY")

# 环境变量中的模型覆写与渠道配置
MODEL_OVERRIDE_MAP_RAW = os.getenv("MODEL_OVERRIDE_MAP", "{}")
UPSTREAM_CHANNELS_RAW = os.getenv("UPSTREAM_CHANNELS", "{}")
# 环境变量中的自定义模型列表
CUSTOM_MODELS_RAW = os.getenv("CUSTOM_MODELS", "{}")

# 思考模式系统提示模板（由 thinking 模块注入）
THINKING_SYSTEM_TEMPLATE = (
    "<antml:thinking_mode>interleaved</antml:thinking_mode>"
    "<antml:max_thinking_length>{max_length}</antml:max_thinking_length>"
)
