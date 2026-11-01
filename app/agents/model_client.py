"""统一创建 DeepSeek 模型客户端；角色模块只声明各自的提示词和输出。"""

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

from app.config import PROJECT_ROOT


# 根据源码定位 .env，不依赖启动时的工作目录。
load_dotenv(PROJECT_ROOT / ".env")


def create_model(timeout: int = 60) -> ChatDeepSeek:
    """统一创建模型客户端；代码生成允许传入更长超时。"""

    return ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=timeout,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )
