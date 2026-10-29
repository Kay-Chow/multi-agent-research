"""阶段 1 的模型工作：分类、搜索摘要整理、产品方案。"""

import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from app.product.prompts import FINDING_PROMPT, MODE_PROMPT, PLAN_PROMPT, QUERY_PROMPT
from app.product.state import ProductState, ResearchFinding, TaskMode
from app.state import Source
from app.tools import search_web


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _model() -> ChatDeepSeek:
    """沿用旧项目已经验证过的 DeepSeek 配置，不新引入模型依赖。"""

    return ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )


class ModeDecision(BaseModel):
    mode: TaskMode
    reason: str = Field(min_length=1)


class QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=2)


class Finding(BaseModel):
    claim: str = Field(min_length=1)
    source_id: int


class ResearchNotes(BaseModel):
    findings: list[Finding] = Field(max_length=5)


class ProductPlanDraft(BaseModel):
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


def classify_task(request: str) -> ModeDecision:
    """模型只提出任务模式；图中仍用固定的两种模式控制路由。"""

    return _model().with_structured_output(
        ModeDecision, method="function_calling"
    ).invoke([("system", MODE_PROMPT), ("human", request)])


def collect_research(request: str) -> tuple[list[Source], list[ResearchFinding], str]:
    """先获得真实搜索结果，再让模型根据编号提炼研究点。"""

    model = _model()
    query_plan = model.with_structured_output(
        QueryPlan, method="function_calling"
    ).invoke([("system", QUERY_PROMPT), ("human", request)])

    sources: list[Source] = []
    seen_queries: set[str] = set()
    seen_urls: set[str] = set()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for raw_query in query_plan.queries:
        query = raw_query.strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        for hit in search_web(query):
            if hit["url"] in seen_urls or not hit["snippet"].strip():
                continue
            seen_urls.add(hit["url"])
            sources.append({
                "source_id": len(sources) + 1,
                "title": hit["title"],
                "url": hit["url"],
                "snippet": hit["snippet"],
                "retrieved_at": retrieved_at,
            })
            if len(sources) >= 6:
                break
        if len(sources) >= 6:
            break

    if not sources:
        report = (
            "# 调研报告\n\n"
            "没有检索到可用的外部资料摘要。外部背景与可行性尚无依据，"
            "不能据此声称已经完成事实核验。\n"
        )
        return [], [], report

    notes = model.with_structured_output(
        ResearchNotes, method="function_calling"
    ).invoke([
        ("system", FINDING_PROMPT),
        ("human", json.dumps({"request": request, "sources": sources}, ensure_ascii=False)),
    ])

    valid_ids = {source["source_id"] for source in sources}
    findings: list[ResearchFinding] = []
    for item in notes.findings:
        if item.source_id not in valid_ids:
            raise ValueError(f"Researcher 引用了不存在的来源：{item.source_id}")
        claim = item.claim.strip()
        if claim:
            findings.append({"claim": claim, "source_id": item.source_id})

    lines = ["# 调研报告", "", "## 研究要点（仅依据搜索摘要）", ""]
    lines.extend(
        f"- {item['claim']} [{item['source_id']}]" for item in findings
    )
    if not findings:
        lines.append("- 搜索摘要不足以支持明确结论；具体事实仍待核查。")
    lines.extend(["", "## 来源", ""])
    for source in sources:
        # 来源标题来自外部搜索结果，压成单行，避免破坏报告结构。
        title = source["title"].replace("\n", " ").replace("\r", " ")
        lines.append(
            f"- [{source['source_id']}] {title}：<{source['url']}> "
            f"（检索时间 UTC：{source['retrieved_at']}）"
        )
    lines.extend([
        "", "## 局限性", "",
        "以上结论只依据搜索服务返回的摘要，尚未逐篇核对网页原文；"
        "产品决策还需结合实际业务与人工确认。",
    ])
    return sources, findings, "\n".join(lines) + "\n"


def create_product_plan(state: ProductState) -> tuple[str, list[str], list[str]]:
    """生成可审核方案；引用编号与验收标准由 Python 再校验一遍。"""

    draft = _model().with_structured_output(
        ProductPlanDraft, method="function_calling"
    ).invoke([
        ("system", PLAN_PROMPT),
        ("human", json.dumps({
            "request": state["request"],
            "mode": state["mode"],
            "research_findings": state["research_findings"],
            "sources": state["sources"],
            "previous_plan": state["product_plan"] if state["review_status"] == "rejected" else None,
            "review_feedback": state["review_feedback"],
        }, ensure_ascii=False)),
    ])

    valid_ids = {source["source_id"] for source in state["sources"]}
    if set(draft.source_ids) - valid_ids:
        raise ValueError(f"产品方案引用了不存在的来源：{sorted(set(draft.source_ids) - valid_ids)}")

    # Pydantic 的 min_length 只管列表元素数量；空白字符串仍需业务校验。
    required_lists = {
        "目标用户": draft.target_users,
        "范围": draft.in_scope,
        "规则": draft.rules,
        "验收标准": draft.acceptance_criteria,
    }
    for name, items in required_lists.items():
        if any(not item.strip() for item in items):
            raise ValueError(f"产品方案的{name}包含空白项")
    if not state["sources"] and not any(item.strip() for item in draft.open_questions):
        raise ValueError("没有外部来源时，方案必须列出待确认问题")

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item.strip()}" for item in items) if items else "- 暂无"

    source_text = (
        "、".join(f"[{source_id}]" for source_id in sorted(set(draft.source_ids)))
        if draft.source_ids else "无；以下主要依据用户需求，外部事实仍待核验"
    )
    sections = [
        "# 产品方案",
        f"## 问题\n{draft.problem.strip()}",
        f"## 目标用户\n{bullets(draft.target_users)}",
        f"## 目标\n{draft.goal.strip()}",
        f"## 本次范围\n{bullets(draft.in_scope)}",
        f"## 非本次范围\n{bullets(draft.out_of_scope)}",
        f"## 行为与接口规则\n{bullets(draft.rules)}",
        f"## 验收标准\n{bullets(draft.acceptance_criteria)}",
        f"## 风险\n{bullets(draft.risks)}",
        f"## 待确认问题\n{bullets(draft.open_questions)}",
        f"## 外部研究依据\n{source_text}；详见同任务的调研报告。",
    ]
    return (
        "\n\n".join(sections) + "\n",
        [item.strip() for item in draft.acceptance_criteria],
        [item.strip() for item in draft.open_questions if item.strip()],
    )
