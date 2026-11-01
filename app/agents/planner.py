"""产品与文件级变更计划：结构化起草、引用校验和 Markdown 展示。"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.agents.model_client import create_model
from app.agents.prompts import CHANGE_PLAN_PROMPT, PRODUCT_PLAN_PROMPT
from app.tools.repository import validate_change_path
from app.workflow.state import AgentState


class ProductPlanDraft(BaseModel):
    """Product Planner 的结构化方案草稿，后续渲染为待人工审核的 Markdown。"""

    problem: str = Field(min_length=1)
    target_users: list[str] = Field(min_length=1)
    goal: str = Field(min_length=1)
    in_scope: list[str] = Field(min_length=1)
    out_of_scope: list[str]
    rules: list[str] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    risks: list[str]
    open_questions: list[str]
    source_ids: list[int]


class FileChange(BaseModel):
    """文件级变更计划的一项：目标路径、拟修改内容和修改原因。"""

    path: str = Field(min_length=1)
    change: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ChangePlanDraft(BaseModel):
    """Change Planner 的输出；只描述改动与测试，不包含实际代码。"""

    summary: str = Field(min_length=1)
    file_changes: list[FileChange] = Field(min_length=1, max_length=4)
    tests: list[str] = Field(min_length=1)
    risks: list[str]
    open_questions: list[str]

@dataclass(frozen=True)
class ProductPlanResult:
    """已校验并排版的产品方案；由图节点写入共享 State。"""

    markdown: str
    acceptance_criteria: list[str]
    open_questions: list[str]

def build_product_plan(state: AgentState) -> ProductPlanResult:
    """让模型起草并校验产品方案；不写共享状态、不选择下一节点。"""

    if state["mode"] is None or state["research"] is None:
        raise ValueError("产品方案缺少任务模式或研究结果")

    draft = create_model().with_structured_output(
        ProductPlanDraft, method="function_calling"
    ).invoke([
        ("system", PRODUCT_PLAN_PROMPT),
        ("human", json.dumps({
            "request": state["request"],
            "mode": state["mode"],
            "research": state["research"],
            "sources": state["sources"],
            "previous_plan": state["draft_plan"],
            "review_feedback": state["review_feedback"],
        }, ensure_ascii=False)),
    ])

    valid_ids = {source["source_id"] for source in state["sources"]}
    invalid_ids = set(draft.source_ids) - valid_ids
    if invalid_ids:
        raise ValueError(f"产品方案引用了不存在的来源：{sorted(invalid_ids)}")
    for name, items in {
        "目标用户": draft.target_users,
        "范围": draft.in_scope,
        "规则": draft.rules,
        "验收标准": draft.acceptance_criteria,
    }.items():
        if any(not item.strip() for item in items):
            raise ValueError(f"产品方案的{name}包含空白项")
    if not any(source["source_type"] in {"knowledge_base", "local_file", "user"} for source in state["sources"]):
        if not any(item.strip() for item in draft.open_questions):
            raise ValueError("知识库没有相关来源时，产品方案必须列出待确认问题")

    source_text = (
        "、".join(f"[{number}]" for number in sorted(set(draft.source_ids)))
        if draft.source_ids else "无；以下主要依据用户需求，背景事实待核验"
    )
    sections = [
        "# 产品方案",
        f"## 问题\n{draft.problem.strip()}",
        f"## 目标用户\n{_markdown_bullets(draft.target_users)}",
        f"## 目标\n{draft.goal.strip()}",
        f"## 本次范围\n{_markdown_bullets(draft.in_scope)}",
        f"## 非本次范围\n{_markdown_bullets(draft.out_of_scope)}",
        f"## 行为与接口规则\n{_markdown_bullets(draft.rules)}",
        f"## 验收标准\n{_markdown_bullets(draft.acceptance_criteria)}",
        f"## 风险\n{_markdown_bullets(draft.risks)}",
        f"## 待确认问题\n{_markdown_bullets(draft.open_questions)}",
        f"## 调研依据\n{source_text}；详见同任务调研报告中的知识库/网页来源分类。",
    ]
    return ProductPlanResult(
        markdown="\n\n".join(sections) + "\n",
        acceptance_criteria=[item.strip() for item in draft.acceptance_criteria],
        open_questions=[item.strip() for item in draft.open_questions if item.strip()],
    )

def _markdown_bullets(items: list[str]) -> str:
    """把非空文本渲染为 Markdown 列表；全为空时明确显示“暂无”。"""

    lines = [f"- {item.strip()}" for item in items if item.strip()]
    return "\n".join(lines) or "- 暂无"

def propose_change_plan(state: AgentState, contents: dict[str, str]) -> str:
    """根据已批准的产品方案与只读源码，提出文件级修改计划。

    此处仍不生成、更不写入代码。程序会核对每个目标路径属于已检查
    文件和变更白名单，再把计划交给阶段 3 作为候选改动的范围约束。
    """

    if state["mode"] != "code_change" or state["review_status"] != "approved":
        raise ValueError("只有已批准的代码类产品任务才能规划仓库变更")
    if not state["final_plan"] or not state["repo_context"]:
        raise ValueError("缺少已批准的产品方案或仓库摘要")

    draft = create_model().with_structured_output(
        ChangePlanDraft, method="function_calling"
    ).invoke([
        ("system", CHANGE_PLAN_PROMPT),
        ("human", json.dumps({
            "request": state["request"],
            "product_plan": state["final_plan"],
            "acceptance_criteria": state["acceptance_criteria"],
            "repo_context": state["repo_context"],
            "read_only_files": contents,
        }, ensure_ascii=False)),
    ])

    seen: set[str] = set()
    known_paths = {item["path"] for item in state["repo_context"]["files"]}
    file_lines = []
    for item in draft.file_changes:
        validate_change_path(item.path)
        if item.path not in known_paths:
            raise ValueError(f"变更文件不在已检查的仓库文件中：{item.path}")
        if item.path in seen:
            raise ValueError(f"变更计划重复列出文件：{item.path}")
        seen.add(item.path)
        if not item.change.strip() or not item.reason.strip():
            raise ValueError(f"文件 {item.path} 的变更说明不能为空")
        file_lines.append(f"- `{item.path}`：{item.change.strip()}。原因：{item.reason.strip()}")
    if any(not item.strip() for item in draft.tests):
        raise ValueError("测试计划包含空白项")

    sections = [
        "# 文件级变更计划（尚未修改代码）",
        f"## 变更概述\n{draft.summary.strip()}",
        "## 目标文件\n" + "\n".join(file_lines),
        f"## 测试计划\n{_markdown_bullets(draft.tests)}",
        f"## 风险\n{_markdown_bullets(draft.risks)}",
        f"## 待确认\n{_markdown_bullets(draft.open_questions)}",
        f"## 仓库基线\n`{state['repo_context']['head']}`；如仓库变化，须重新分析。",
    ]
    return "\n\n".join(sections) + "\n"
