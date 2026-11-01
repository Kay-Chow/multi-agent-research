"""程序入口：打开持久化 checkpoint，启动终端聊天。"""

import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

# PyCharm 既可用模块名 app.main，也可直接运行此文件；两种方式都能找到 app 包。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chat_cli import chat
from app.config import DB_PATH
from app.workflow.graph import build_graph


def main() -> None:
    """只做启动配置；对话交互与工作流逻辑分别放在 chat_cli 和 graph。"""

    if len(sys.argv) > 1:
        raise SystemExit("此入口只支持交互式聊天；请不带参数运行 app.main。")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        chat(build_graph(saver))


if __name__ == "__main__":
    main()
