"""产品工作流阶段 1：调研、方案、人工审核与后续代码阶段交接。"""

import re
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.product.agents import classify_task, collect_research, create_product_plan
from app.product.state import ProductState, TaskMode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "product"
MAX_REVISIONS = 2


def classify_node(state: ProductState) -> dict:
    """任务模式只能二选一；用户显式指定的模式优先于模型判断。"""

    mode: TaskMode
    if state["requested_mode"] is not None:
        mode = state["requested_mode"]
        reason = "用户在命令行显式指定模式"
    else:
        decision = classify_task(state["request"])
        mode = decision.mode
        reason = decision.reason

    if mode not in ("proposal_only", "code_change"):
        raise ValueError(f"未知产品任务模式：{mode}")

    return {
        "mode": mode,
        "plan": ["researcher", "product_planner", "human_review"],
        "events": [f"classify：{mode}；{reason}"],
    }


def research_node(state: ProductState) -> dict:
    """只写研究产物，不修改产品方案或演示仓库。"""

    sources, findings, report = collect_research(state["request"])
    return {
        "sources": sources,
        "research_findings": findings,
        "research_report": report,
        "events": [f"researcher：{len(sources)} 条来源，{len(findings)} 条研究要点"],
    }


def planner_node(state: ProductState) -> dict:
    """人工退回时读取上一版和审核意见，再生成新方案。"""

    if state["mode"] is None or state["research_report"] is None:
        raise ValueError("产品方案缺少任务模式或研究结果")

    proposal, criteria, questions = create_product_plan(state)
    return {
        "product_plan": proposal,
        "acceptance_criteria": criteria,
        "open_questions": questions,
        "review_status": "pending",
        "events": [f"product_planner：生成方案，包含 {len(criteria)} 条验收标准"],
    }


def review_node(state: ProductState) -> dict:
    """先暂停，收到人工决定后再更新状态；这里不写文件。"""

    if state["review_status"] != "pending" or state["product_plan"] is None:
        raise ValueError("当前没有可审核的产品方案")

    decision = interrupt({
        "task_id": state["task_id"],
        "mode": state["mode"],
        "research_report": state["research_report"],
        "product_plan": state["product_plan"],
        "revision_count": state["revision_count"],
        "options": ["approve", "reject", "terminate"],
    })
    if not isinstance(decision, dict):
        raise ValueError("审核结果必须是字典")

    action = decision.get("action")
    if action == "approve":
        return {
            "review_status": "approved",
            "events": ["review：人工批准产品方案"],
        }
    if action == "reject":
        if state["revision_count"] >= MAX_REVISIONS:
            return {
                "review_status": "terminated",
                "workflow_status": "terminated",
                "events": ["review：退回次数达到上限，任务终止"],
            }
        feedback = str(decision.get("feedback") or "").strip()
        if not feedback:
            raise ValueError("退回产品方案时必须填写修改意见")
        return {
            "review_status": "rejected",
            "review_feedback": feedback,
            "revision_count": state["revision_count"] + 1,
            "events": ["review：人工退回产品方案"],
        }
    if action == "terminate":
        return {
            "review_status": "terminated",
            "workflow_status": "terminated",
            "events": ["review：人工终止任务"],
        }
    raise ValueError(f"未知审核操作：{action}")


def route_after_review(state: ProductState) -> str:
    return state["review_status"]


def _write_once(path: Path, content: str) -> None:
    """恢复/重试可再次写相同内容，但不能悄悄覆盖不同的批准版本。"""

    content = content.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"已存在不同内容的批准文件：{path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def save_node(state: ProductState) -> dict:
    """只有人工批准后才保存两份独立 Markdown 交付物。"""

    if state["review_status"] != "approved":
        raise ValueError("方案未经人工批准，不能保存")
    if not state["research_report"] or not state["product_plan"]:
        raise ValueError("调研报告或产品方案为空")

    task_id = state["task_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise ValueError("任务 ID 格式不正确")
    directory = OUTPUT_DIR / task_id
    directory.mkdir(parents=True, exist_ok=True)
    research_file = directory / "research.md"
    proposal_file = directory / "proposal.md"
    _write_once(research_file, state["research_report"])
    _write_once(proposal_file, state["product_plan"])

    return {
        "research_path": f"outputs/product/{task_id}/research.md",
        "proposal_path": f"outputs/product/{task_id}/proposal.md",
        "events": ["save：已保存调研报告与产品方案"],
    }


def finish_node(state: ProductState) -> dict:
    """阶段 1 不碰代码：代码类任务仅标记为等待下一阶段。"""

    if state["mode"] == "code_change":
        return {
            "workflow_status": "ready_for_repo",
            "events": ["handoff：方案已批准；等待阶段 2 的仓库只读分析"],
        }
    return {
        "workflow_status": "completed",
        "events": ["finish：仅调研/方案任务完成"],
    }


def build_product_graph(checkpointer):
    """独立于原技术报告图；后续只需替换代码类任务的末端。"""

    builder = StateGraph(ProductState)
    builder.add_node("classify", classify_node)
    builder.add_node("researcher", research_node)
    builder.add_node("product_planner", planner_node)
    builder.add_node("review", review_node)
    builder.add_node("save", save_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "researcher")
    builder.add_edge("researcher", "product_planner")
    builder.add_edge("product_planner", "review")
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"rejected": "product_planner", "approved": "save", "terminated": END},
    )
    builder.add_edge("save", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
