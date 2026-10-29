"""命令行入口：启动任务，或根据任务 ID 恢复人工审核。"""

import argparse
import re
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.graph import build_graph
from app.state import initial_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"


def show_result(result: dict, task_id: str) -> None:
    """展示待审核草稿，或展示已结束任务的结果。"""

    if "__interrupt__" in result:
        review = result["__interrupt__"][0].value
        print("\n===== 待审核报告 =====\n")
        print(review["draft_report"])
        print(f"\n任务已暂停。之后用 task_id 恢复：{task_id}")
        return

    print("\n审核状态：", result["review_status"])
    print("修改次数：", result["revision_count"])

    if result["output_path"]:
        print("报告文件：", PROJECT_ROOT / result["output_path"])

    print("执行路径：")
    for event in result["events"]:
        print("-", event)


def ask_review_decision() -> dict[str, str]:
    """从终端取得一次人工审核决定。"""

    while True:
        choice = input("\n批准 a / 退回 r / 终止 q：").strip().lower()
        if choice in {"a", "r", "q"}:
            break
        print("请输入 a、r 或 q")

    feedback = ""
    if choice == "r":
        while not feedback:
            feedback = input("请输入具体修改意见：").strip()

    action = {"a": "approve", "r": "reject", "q": "terminate"}[choice]
    return {"action": action, "feedback": feedback}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    start_parser = commands.add_parser("start", help="启动一个新任务")
    start_parser.add_argument("task")

    resume_parser = commands.add_parser("resume", help="恢复待审核任务")
    resume_parser.add_argument("task_id")

    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # SQLite 连接要覆盖 build_graph、invoke 和 get_state 的整个使用过程。
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)

        if args.command == "start":
            state = initial_state(args.task, False)
            task_id = state["task_id"]
            config = {"configurable": {"thread_id": task_id}}

            # 先打印 ID：即使之后终端关闭，也知道该用哪个 ID 恢复。
            print("任务 ID：", task_id)
            result = graph.invoke(state, config=config, durability="sync")
            show_result(result, task_id)
            return

        task_id = args.task_id
        if not re.fullmatch(r"[0-9a-f]{32}", task_id):
            parser.error("task_id 格式不正确")

        config = {"configurable": {"thread_id": task_id}}
        snapshot = graph.get_state(config)

        if snapshot.values.get("task_id") != task_id:
            parser.error("SQLite 中找不到这个任务 ID")

        if not snapshot.next:
            print("任务已经结束。")
            show_result(snapshot.values, task_id)
            return

        # 如果批准已经写入 checkpoint，但保存节点尚未完成，可继续执行保存。
        if snapshot.next == ("save",) and snapshot.values["review_status"] == "approved":
            result = graph.invoke(None, config=config, durability="sync")
            show_result(result, task_id)
            return

        if snapshot.next != ("review",):
            parser.error(f"任务当前不在人工审核节点：{snapshot.next}")

        interrupts = [
            item
            for task in snapshot.tasks
            for item in task.interrupts
        ]
        if len(interrupts) != 1:
            parser.error("找不到唯一的待处理审核中断")

        review = interrupts[0].value
        print("\n===== 待审核报告 =====\n")
        print(review["draft_report"])

        decision = ask_review_decision()
        result = graph.invoke(
            Command(resume=decision),
            config=config,  # 必须沿用原来的 thread_id。
            durability="sync",
        )
        show_result(result, task_id)


if __name__ == "__main__":
    main()
