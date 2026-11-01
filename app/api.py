"""产品 Agent 的本地 HTTP 入口；与命令行驱动同一张图。"""

import re
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.config import DB_PATH, task_config
from app.tools.repository import RepoAccessError, checked_root
from app.workflow.graph import build_graph
from app.workflow.state import RepoContext, initial_state


# 当前仍是可信本机的单进程演示；串行访问同一个 SQLite 文件。
# Lock 只协调同一进程里的 HTTP 请求，不等同于生产环境的分布式锁。
DB_LOCK = Lock()
app = FastAPI(title="Product Research & Delivery Agent")


class StartRequest(BaseModel):
    """创建任务的请求体；代码任务需要显式选择受限项目目录下的仓库。"""

    request: str = Field(min_length=1, max_length=1000)
    mode: Literal["proposal_only", "code_change"] | None = None
    repo_root: str | None = Field(default=None, max_length=500)


class ReviewRequest(BaseModel):
    """两个人工审核接口共用的决定格式；退回时必须附具体意见。"""

    action: Literal["approve", "reject", "terminate"]
    feedback: str = Field(default="", max_length=2000)


class ClarificationRequest(BaseModel):
    """调研缺资料时用户提供的说明；不会被当成人工审核决定。"""

    message: str = Field(min_length=1, max_length=1000)


class TaskView(BaseModel):
    """对外展示的任务快照，包含两次审核状态和阶段 3 的候选 diff。

    它由 checkpoint 映射而来，不是模型输出；`waiting_for_*` 表示图确实
    停在对应的 `interrupt` 节点，客户端据此显示补充输入或审核按钮。
    """

    task_id: str
    repo_root: str | None
    mode: str | None
    next_step: str | None
    supervisor_steps: int
    research_rounds: int
    review_status: str
    workflow_status: str
    revision_count: int
    waiting_for_review: bool
    waiting_for_diff_review: bool
    waiting_for_research_input: bool
    next_nodes: list[str]
    research_question: str | None
    research_report: str | None
    draft_plan: str | None
    final_plan: str | None
    acceptance_criteria: list[str]
    research_path: str | None
    proposal_path: str | None
    repo_context: RepoContext | None
    change_plan: str | None
    change_plan_path: str | None
    workspace_path: str | None
    diff_path: str | None
    diff: str | None
    file_hashes: list[dict[str, str]]
    code_summary: str | None
    code_review_status: str
    code_revision_count: int
    error: str | None
    events: list[str]


def validate_task_id(task_id: str) -> None:
    """只接受本项目生成的 32 位十六进制任务 ID。"""

    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise HTTPException(status_code=422, detail="task_id 格式不正确")


def load_snapshot(graph, task_id: str):
    """读取指定 thread 的 checkpoint，并确认其中确实是这项任务。"""

    snapshot = graph.get_state(task_config(task_id))
    if snapshot.values.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    return snapshot


def _interrupted_at(snapshot, node: str) -> bool:
    """必须真的停在该节点的 interrupt，不能只凭 next 节点名开放审核。"""

    return snapshot.next == (node,) and any(task.interrupts for task in snapshot.tasks)


def to_view(snapshot) -> TaskView:
    """把内部图状态转换为 API 响应，同时辨认当前暂停在哪个人工门禁。"""

    values = snapshot.values
    return TaskView(
        task_id=values["task_id"],
        repo_root=values.get("repo_root"),
        mode=values["mode"],
        next_step=values.get("next_step"),
        supervisor_steps=values.get("supervisor_steps", 0),
        research_rounds=values.get("research_rounds", 0),
        review_status=values["review_status"],
        workflow_status=values["workflow_status"],
        revision_count=values["revision_count"],
        waiting_for_review=_interrupted_at(snapshot, "review"),
        waiting_for_diff_review=_interrupted_at(snapshot, "diff_review"),
        waiting_for_research_input=_interrupted_at(snapshot, "await_research_input"),
        next_nodes=list(snapshot.next),
        research_question=values.get("research_question"),
        research_report=values["research_report"],
        draft_plan=values["draft_plan"],
        final_plan=values["final_plan"],
        acceptance_criteria=values["acceptance_criteria"],
        research_path=values["research_path"],
        proposal_path=values["proposal_path"],
        repo_context=values["repo_context"],
        change_plan=values["change_plan"],
        change_plan_path=values["change_plan_path"],
        workspace_path=values.get("workspace_path"),
        diff_path=values.get("diff_path"),
        diff=values.get("diff"),
        file_hashes=values.get("file_hashes", []),
        code_summary=values.get("code_summary"),
        code_review_status=values.get("code_review_status", "not_ready"),
        code_revision_count=values.get("code_revision_count", 0),
        error=values["error"],
        events=values["events"],
    )


def _resume_review(task_id: str, request: ReviewRequest, node: str) -> TaskView:
    """两个审核接口共用的恢复步骤；每次只允许恢复指定的审核节点。"""

    validate_task_id(task_id)
    if request.action == "reject" and not request.feedback.strip():
        detail = "退回 diff 时必须填写修改意见" if node == "diff_review" else "退回时必须填写修改意见"
        raise HTTPException(status_code=422, detail=detail)
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        if not _interrupted_at(load_snapshot(graph, task_id), node):
            detail = "任务当前不在代码审核状态" if node == "diff_review" else "任务当前不在待审核状态"
            raise HTTPException(status_code=409, detail=detail)
        graph.invoke(
            Command(resume={"action": request.action, "feedback": request.feedback.strip()}),
            config=task_config(task_id), durability="sync",
        )
        return to_view(load_snapshot(graph, task_id))


@app.post("/tasks", response_model=TaskView, status_code=201)
def start_task(request: StartRequest):
    """创建任务并运行到完成或第一个人工中断点。"""

    if request.mode == "code_change" and not request.repo_root:
        raise HTTPException(status_code=422, detail="代码任务必须提供 repo_root")
    try:
        repo_root = str(checked_root(request.repo_root)) if request.repo_root else None
    except RepoAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state = initial_state(request.request, request.mode, repo_root)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        graph.invoke(state, config=task_config(state["task_id"]), durability="sync")
        return to_view(load_snapshot(graph, state["task_id"]))


@app.get("/tasks/{task_id}", response_model=TaskView)
def get_task(task_id: str):
    """查询 checkpoint 快照；只读取状态，不推进工作流。"""

    validate_task_id(task_id)
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="任务不存在")
    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        return to_view(load_snapshot(graph, task_id))


@app.post("/tasks/{task_id}/clarify", response_model=TaskView)
def clarify_task(task_id: str, request: ClarificationRequest):
    """恢复调研补充中断，之后继续检索和准备产品方案。"""

    validate_task_id(task_id)
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="补充内容不能为空")
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="任务不存在")
    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        if not _interrupted_at(load_snapshot(graph, task_id), "await_research_input"):
            raise HTTPException(status_code=409, detail="任务当前不在等待调研补充状态")
        graph.invoke(
            Command(resume={"message": message}),
            config=task_config(task_id), durability="sync",
        )
        return to_view(load_snapshot(graph, task_id))


@app.post("/tasks/{task_id}/review", response_model=TaskView)
def review_task(task_id: str, request: ReviewRequest):
    """恢复产品方案审核；批准后代码类任务会继续进入后续阶段。"""

    return _resume_review(task_id, request, "review")


@app.post("/tasks/{task_id}/diff-review", response_model=TaskView)
def review_diff(task_id: str, request: ReviewRequest):
    """第二个人工门禁：审核隔离区 diff，不运行测试或提交。"""

    return _resume_review(task_id, request, "diff_review")
