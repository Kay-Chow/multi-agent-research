"""定义整张图共享的状态，以及每次运行的初始值。"""

from operator import add
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

# 限定路由字段可写入的角色名；真正的运行时校验会在后续阶段补充。
Route = Literal["researcher", "coder", "writer", "finish"]


class Source(TypedDict):
    source_id: int
    title: str
    url: str
    snippet: str
    retrieved_at: str


class AgentState(TypedDict):
    """各节点通过该状态交换结果；节点只需返回自己要更新的字段。"""

    # 当前任务及阶段 1 的手动开关；以后由 Supervisor 判断是否需要代码。
    task: str
    needs_code: bool

    instruction: str
    sources: list[Source]
    code_explanation: str | None
    code_syntax_ok: bool | None

    # Supervisor 记录计划和下一位执行者。
    plan: list[str]
    next_agent: Route

    # 三位 Worker 各写自己的结果，避免互相覆盖。
    research: str | None
    code: str | None
    draft_report: str | None

    # step_count 统计 Supervisor 已做的路由决策次数。
    step_count: int
    # reducer=add：节点返回的新事件会追加到旧列表，而不是替换旧列表。
    events: Annotated[list[str], add]

    task_id: str
    review_status: Literal["not_ready", "pending", "approved", "rejected", "terminated"]
    review_feedback: str
    revision_count: int
    final_report: str | None

    output_path: str | None


def initial_state(task: str, needs_code: bool = False) -> AgentState:
    """为一次全新的图运行构造完整状态，不沿用上次任务的结果。"""

    return {
        "task": task,
        "needs_code": needs_code,
        "instruction": "",
        "sources": [],
        "code_explanation": None,
        "code_syntax_ok": None,
        "plan": [],
        "next_agent": "researcher",  # 占位值；首次路由前会由 Supervisor 覆盖。
        "research": None,
        "code": None,
        "draft_report": None,
        "step_count": 0,
        "events": [],
        "task_id": uuid4().hex,
        "review_status": "not_ready",
        "review_feedback": "",
        "revision_count": 0,
        "final_report": None,
        "output_path": None,
    }
