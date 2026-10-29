"""产品工作流在阶段 1 使用的共享 State。后续阶段按需扩展。"""

from operator import add
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from app.state import Source


TaskMode = Literal["proposal_only", "code_change"]
ReviewStatus = Literal["not_ready", "pending", "rejected", "approved", "terminated"]
WorkflowStatus = Literal["running", "completed", "ready_for_repo", "terminated"]


class ResearchFinding(TypedDict):
    claim: str
    source_id: int


class ProductState(TypedDict):
    """节点只返回本节点负责更新的字段，避免覆盖其他节点的结果。"""

    task_id: str
    request: str
    requested_mode: TaskMode | None
    mode: TaskMode | None
    plan: list[str]

    sources: list[Source]
    research_findings: list[ResearchFinding]
    research_report: str | None
    product_plan: str | None
    acceptance_criteria: list[str]
    open_questions: list[str]

    review_status: ReviewStatus
    review_feedback: str
    revision_count: int
    workflow_status: WorkflowStatus
    research_path: str | None
    proposal_path: str | None
    events: Annotated[list[str], add]


def initial_product_state(
    request: str, requested_mode: TaskMode | None = None
) -> ProductState:
    """一次新任务必须拥有全新 task_id；不复用其他任务的审核状态。"""

    request = request.strip()
    if not request or len(request) > 1000:
        raise ValueError("产品需求长度必须在 1～1000 字符之间")
    if requested_mode not in (None, "proposal_only", "code_change"):
        raise ValueError("未知任务模式")

    return {
        "task_id": uuid4().hex,
        "request": request,
        "requested_mode": requested_mode,
        "mode": None,
        "plan": [],
        "sources": [],
        "research_findings": [],
        "research_report": None,
        "product_plan": None,
        "acceptance_criteria": [],
        "open_questions": [],
        "review_status": "not_ready",
        "review_feedback": "",
        "revision_count": 0,
        "workflow_status": "running",
        "research_path": None,
        "proposal_path": None,
        "events": [],
    }
