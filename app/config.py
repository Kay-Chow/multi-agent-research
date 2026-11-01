"""命令行和 HTTP 服务共用的路径与 LangGraph 会话配置。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 当前工作流使用独立数据库；不会读取此前其他文件中的 checkpoint。
DB_PATH = PROJECT_ROOT / "data" / "agent_sessions.sqlite"


def task_config(task_id: str) -> dict:
    """同一个 ID 必须用于首次运行和恢复，才能读到同一份 checkpoint。"""

    return {"configurable": {"thread_id": task_id}}
