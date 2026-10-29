"""产品工作流的命令行入口：启动任务或恢复产品方案审核。"""

import argparse
import re
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.product.graph import build_product_graph
from app.product.state import initial_product_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "product_checkpoints.sqlite"


def task_config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def show_snapshot(graph, task_id: str) -> None:
    """始终从 checkpoint 读当前状态，避免依赖 invoke 的返回形状。"""

    snapshot = graph.get_state(task_config(task_id))
    values = snapshot.values
    if values.get("task_id") != task_id:
        raise ValueError("找不到任务")

    waiting = (
        snapshot.next == ("review",)
        and any(task.interrupts for task in snapshot.tasks)
    )
    if waiting:
        print("\n===== 调研报告 =====\n")
        print(values["research_report"])
        print("\n===== 产品方案（待审核） =====\n")
        print(values["product_plan"])
        print(f"\n模式：{values['mode']}；已退回次数：{values['revision_count']}")
        print(f"任务暂停，可用 task_id 恢复：{task_id}")
        return

    print(f"\n审核状态：{values['review_status']}")
    print(f"工作流状态：{values['workflow_status']}")
    if values["research_path"]:
        print(f"调研报告：{PROJECT_ROOT / values['research_path']}")
    if values["proposal_path"]:
        print(f"产品方案：{PROJECT_ROOT / values['proposal_path']}")
    print("执行路径：")
    for event in values["events"]:
        print("-", event)


def ask_review() -> dict[str, str]:
    while True:
        choice = input("批准 a / 退回 r / 终止 q：").strip().lower()
        if choice in {"a", "r", "q"}:
            break
        print("请输入 a、r 或 q")

    feedback = ""
    if choice == "r":
        while not feedback:
            feedback = input("请输入修改意见：").strip()
    return {
        "action": {"a": "approve", "r": "reject", "q": "terminate"}[choice],
        "feedback": feedback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="启动产品调研/方案任务")
    start.add_argument("request")
    start.add_argument("--mode", choices=["proposal_only", "code_change"])
    resume = commands.add_parser("resume", help="恢复待审核的产品方案")
    resume.add_argument("task_id")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_product_graph(saver)
        if args.command == "start":
            state = initial_product_state(args.request, args.mode)
            task_id = state["task_id"]
            print("任务 ID：", task_id)
            graph.invoke(state, config=task_config(task_id), durability="sync")
            show_snapshot(graph, task_id)
            return

        task_id = args.task_id
        if not re.fullmatch(r"[0-9a-f]{32}", task_id):
            parser.error("task_id 格式不正确")
        snapshot = graph.get_state(task_config(task_id))
        if snapshot.values.get("task_id") != task_id:
            parser.error("SQLite 中找不到这个任务 ID")
        if not snapshot.next:
            show_snapshot(graph, task_id)
            return
        if snapshot.next == ("save",) and snapshot.values["review_status"] == "approved":
            # 审核已通过但保存阶段中断时，继续执行幂等的保存节点。
            graph.invoke(None, config=task_config(task_id), durability="sync")
            show_snapshot(graph, task_id)
            return
        if snapshot.next != ("review",) or not any(
            task.interrupts for task in snapshot.tasks
        ):
            parser.error(f"任务当前不在人工审核节点：{snapshot.next}")

        show_snapshot(graph, task_id)
        graph.invoke(
            Command(resume=ask_review()),
            config=task_config(task_id),
            durability="sync",
        )
        show_snapshot(graph, task_id)


if __name__ == "__main__":
    main()
