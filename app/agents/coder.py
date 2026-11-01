"""Coder 的结构化候选文件输出；实际文件校验与写入由隔离服务负责。"""

import json

from pydantic import BaseModel, Field

from app.agents.model_client import create_model
from app.agents.prompts import CODE_GENERATION_PROMPT
from app.workflow.state import AgentState


class CodeFile(BaseModel):
    """Coder 提出的单个候选文件，使用完整新内容而不是可执行命令。"""

    path: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=16_000)


class CodePatchDraft(BaseModel):
    """Coder 的结构化候选改动；程序仍需逐项校验后才写入隔离区。"""

    edits: list[CodeFile] = Field(min_length=1, max_length=3)
    summary: str = Field(min_length=1)
    risks: list[str]

def generate_code_patch(
    state: AgentState, contents: dict[str, str], planned_files: set[str]
) -> CodePatchDraft:
    """向 Coder 提供已批准的需求与仓库基线，取得候选文件全文。

    这里不写磁盘、不执行模型生成的代码。`planned_files` 只作为模型输入，
    真正的路径白名单、文件大小、语法及 diff 校验在 `apply_edits` 中完成。
    """

    if state["mode"] != "code_change" or state["review_status"] != "approved":
        raise ValueError("只有产品方案已批准的代码任务才能生成候选改动")
    if not state["final_plan"] or not state["change_plan"]:
        raise ValueError("缺少已批准的产品方案或文件级变更计划")
    return create_model(timeout=180).with_structured_output(
        CodePatchDraft, method="function_calling"
    ).invoke([
        ("system", CODE_GENERATION_PROMPT),
        ("human", json.dumps({
            "request": state["request"],
            "product_plan": state["final_plan"],
            "change_plan": state["change_plan"],
            "planned_files": sorted(planned_files),
            "baseline_files": {path: contents[path] for path in sorted(planned_files)},
            "previous_diff": state.get("diff") if state.get("code_review_status") == "rejected" else None,
            "feedback": state.get("code_review_feedback", ""),
        }, ensure_ascii=False)),
    ])
