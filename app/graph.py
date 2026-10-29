"""阶段 1 的最小协作图：固定结果模拟 Worker，先验证控制流。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import re
from pathlib import Path

from app.state import AgentState, Route
from app.agents import decide_next, researcher_node, coder_node, writer_node

# 限制 Supervisor 的决策次数，防止路由错误导致无限循环。
MAX_STEPS = 8
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def supervisor_node(state: AgentState) -> dict:
    """调用模型规划，但由 Python 校验计划、依赖和最终路由。"""

    if state["step_count"] >= MAX_STEPS:
        return {
            "next_agent": "finish",
            "events": ["supervisor -> finish（达到步数上限）"],
        }

    decision = decide_next(state)

    # 先判断，首轮的时候不应该有计划，计划要在首轮的时候由大模型来制定
    if state["step_count"] > 0 and not state["plan"]:
        raise ValueError("状态不一致：已经执行过 Supervisor，但没有计划")

    needs_planning = not state["plan"]

    needs_code = decision.needs_code if needs_planning else state["needs_code"]
    expected_plan = (
        ["researcher", "coder", "writer"]
        if needs_code
        else ["researcher", "writer"]
    )

    if needs_planning:
        if decision.plan != expected_plan:
            raise ValueError(f"Supervisor 给出无效计划：{decision.plan}")
        plan = decision.plan
    else:
        plan = state["plan"]
        if decision.plan != plan or decision.needs_code != needs_code:
            raise ValueError("Supervisor 在执行过程中擅自改变了计划")

    # 按计划寻找第一项尚未完成的工作；plan 在这里真正参与路由。
    completed = {
        "researcher": state["research"] is not None,
        "coder": state["code"] is not None,
        "writer": state["draft_report"] is not None,
    }
    expected_next = next(
        (agent for agent in plan if not completed[agent]),
        "finish",
    )

    # 模型可以提议下一步，但不能跳过前置步骤或重复执行已完成步骤。
    if decision.next_agent != expected_next:
        raise ValueError(
            f"Supervisor 路由错误：提议 {decision.next_agent}，"
            f"当前应为 {expected_next}"
        )

    return {
        "plan": plan,
        "needs_code": needs_code,
        "next_agent": decision.next_agent,
        "instruction": decision.instruction,
        "step_count": state["step_count"] + 1,
        "events": [
            f"supervisor -> {decision.next_agent}：{decision.reason}"
        ],
    }


def route_after_supervisor(state: AgentState) -> Route:
    """条件边读取 Supervisor 写入的 next_agent，选择下一节点。"""

    return state["next_agent"]


MAX_REVISIONS = 2


def review_node(state: AgentState) -> dict:
    """暂停等待人工决定；暂停前不能保存文件等产生副作用。"""

    if state["review_status"] != "pending" or state["draft_report"] is None:
        raise ValueError("没有可审核的报告草稿")

    decision = interrupt({
        "task_id": state["task_id"],
        "draft_report": state["draft_report"],
        "revision_count": state["revision_count"],
        "options": ["approve", "reject", "terminate"],
    })

    if not isinstance(decision, dict):
        raise ValueError("审核结果必须是字典")

    action = decision.get("action")

    if action == "approve":
        return {
            "review_status": "approved",
            "final_report": state["draft_report"],
            "events": ["review：人工批准报告"],
        }

    if action == "reject":
        if state["revision_count"] >= MAX_REVISIONS:
            return {
                "review_status": "terminated",
                "events": ["review：修改次数已达上限，终止任务"],
            }

        feedback = str(decision.get("feedback") or "").strip()
        if not feedback:
            raise ValueError("拒绝报告时必须给出修改意见")

        return {
            "review_status": "rejected",
            "review_feedback": feedback,
            "revision_count": state["revision_count"] + 1,
            "events": ["review：人工退回，交给 Writer 修改"],
        }

    if action == "terminate":
        return {
            "review_status": "terminated",
            "events": ["review：人工终止任务"],
        }

    raise ValueError(f"未知审核操作：{action}")


def route_after_review(state: AgentState) -> str:
    """拒绝后返回 Writer；批准或终止则结束图。"""
    return state["review_status"]


def save_report_node(state: AgentState) -> dict:
    """只有人工批准后才保存；重复执行时不覆盖不同内容的旧报告。"""

    report = state["final_report"]
    if state["review_status"] != "approved" or not isinstance(report, str) or not report.strip():
        raise ValueError("报告未经批准，不能保存")

    task_id = state["task_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise ValueError("任务 ID 格式不正确")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{task_id}.md"
    content = report.rstrip() + "\n"

    if output_file.exists():
        # checkpoint 恢复或节点重试时，已有相同内容就视为保存成功。
        if output_file.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"同名报告已存在且内容不同：{output_file}")
    else:
        # 先写临时文件，再替换为最终文件，避免留下写了一半的报告。
        temp_file = output_dir / f".{task_id}.tmp"
        temp_file.write_text(content, encoding="utf-8")
        temp_file.replace(output_file)

    relative_path = f"outputs/{task_id}.md"
    return {
        "output_path": relative_path,
        "events": [f"save：报告已保存至 {relative_path}"],
    }


def build_graph(checkpointer):
    """组装并编译状态图；本阶段不配置 checkpoint 持久化。"""

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review", review_node)
    builder.add_node("save", save_report_node)

    # 入口固定到 Supervisor；其出口只使用条件边，避免意外并行执行。
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "finish": END,
        },
    )

    # Worker 完成后都回到 Supervisor，由它决定继续还是结束。
    for worker in ("researcher", "coder"):
        builder.add_edge(worker, "supervisor")

    # Writer 完成后进入人工审核；退回修改不再经过 Supervisor。
    builder.add_edge("writer", "review")
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {
            "rejected": "writer",
            "approved": "save",
            "terminated": END,
        },
    )
    builder.add_edge("save", END)

    return builder.compile(checkpointer=checkpointer)

