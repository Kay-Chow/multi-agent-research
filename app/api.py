"""本地 HTTP 接口：启动任务、查询状态、提交人工审核。"""

import re
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.graph import build_graph
from app.state import initial_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"

# 当前版本只做单进程本地演示；串行访问 SQLite，避免并发写同一个库。
DB_LOCK = Lock()

app = FastAPI(title="Multi-Agent Research API")


class StartRequest(BaseModel):
    task: str = Field(min_length=1, max_length=1000)


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject", "terminate"]
    feedback: str = Field(default="", max_length=2000)


class TaskView(BaseModel):
    task_id: str
    review_status: str
    revision_count: int
    waiting_for_review: bool
    next_nodes: list[str]
    draft_report: str | None
    final_report: str | None
    output_path: str | None
    events: list[str]


def task_config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def validate_task_id(task_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise HTTPException(status_code=422, detail="task_id 格式不正确")


def load_snapshot(graph, task_id: str):
    snapshot = graph.get_state(task_config(task_id))
    if snapshot.values.get("task_id") != task_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    return snapshot


def to_view(snapshot) -> TaskView:
    """只返回 API 需要的字段，不直接暴露整个 checkpoint。"""
    values = snapshot.values
    waiting = (
        snapshot.next == ("review",)
        and any(task.interrupts for task in snapshot.tasks)
    )

    return TaskView(
        task_id=values["task_id"],
        review_status=values["review_status"],
        revision_count=values["revision_count"],
        waiting_for_review=waiting,
        next_nodes=list(snapshot.next),
        draft_report=values.get("draft_report"),
        final_report=values.get("final_report"),
        output_path=values.get("output_path"),
        events=values.get("events", []),
    )


@app.post("/tasks", response_model=TaskView, status_code=201)
def start_task(request: StartRequest):
    task = request.task.strip()
    if not task:
        raise HTTPException(status_code=422, detail="任务内容不能为空")

    state = initial_state(task, False)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        graph.invoke(
            state,
            config=task_config(state["task_id"]),
            durability="sync",
        )
        return to_view(load_snapshot(graph, state["task_id"]))


@app.get("/tasks/{task_id}", response_model=TaskView)
def get_task(task_id: str):
    validate_task_id(task_id)
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        return to_view(load_snapshot(graph, task_id))


@app.post("/tasks/{task_id}/review", response_model=TaskView)
def review_task(task_id: str, request: ReviewRequest):
    validate_task_id(task_id)
    if request.action == "reject" and not request.feedback.strip():
        raise HTTPException(status_code=422, detail="退回时必须填写修改意见")
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    with DB_LOCK, SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)
        snapshot = load_snapshot(graph, task_id)

        # 已结束的任务不能重复审核；必须确实停在 interrupt() 处。
        if snapshot.next != ("review",) or not any(
            task.interrupts for task in snapshot.tasks
        ):
            raise HTTPException(status_code=409, detail="任务当前不在待审核状态")

        graph.invoke(
            Command(resume={
                "action": request.action,
                "feedback": request.feedback.strip(),
            }),
            config=task_config(task_id),
            durability="sync",
        )
        return to_view(load_snapshot(graph, task_id))
