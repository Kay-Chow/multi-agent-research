"""Researcher：规划检索、核对来源、判断证据缺口并生成调研报告。"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.agents.model_client import create_model
from app.agents.prompts import RESEARCH_QUERY_PROMPT, RESEARCH_SUMMARY_PROMPT
from app.tools.knowledge_base import KnowledgeBaseError, search_knowledge_base
from app.tools.web_search import WebSearchError, search_web
from app.workflow.state import AgentState, Source


class SearchQueries(BaseModel):
    """Researcher 的检索计划；标记是否仍缺任务必需的内部事实。"""

    knowledge_queries: list[str] = Field(max_length=2)
    web_queries: list[str] = Field(max_length=2)
    internal_facts_required: bool = Field(
        default=False,
        description="现有资料不足以确认任务必需的内部规则或背景时为 true",
    )


class Finding(BaseModel):
    """单条研究结论及其来源编号；编号必须在实际检索结果中存在。"""

    claim: str = Field(min_length=1)
    source_id: int


class ResearchBrief(BaseModel):
    """Researcher 对检索片段的精简归纳，限制结论数量以便人工核对。"""

    findings: list[Finding] = Field(max_length=5)

@dataclass(frozen=True)
class _ResearchEvidence:
    """检索计划的中间结果，供调研业务逻辑判断证据是否充分。

    findings=None 表示尚不应调用归纳模型；空列表表示调用后仍无有效要点。
    """

    sources: list[Source]
    knowledge_queried: bool
    web_queried: bool
    kb_error: str | None
    web_error: str | None
    missing_internal: bool
    findings: list[str] | None


@dataclass(frozen=True)
class ResearchNeedsInput:
    """关键依据不足：请用户补充，再继续本次调研。"""

    sources: list[Source]
    question: str


@dataclass(frozen=True)
class ResearchCompleted:
    """调研完成：Graph 只需写入结果并转往下一节点。"""

    sources: list[Source]
    research: str
    report: str
    event_detail: str

def _collect_research_evidence(state: AgentState) -> _ResearchEvidence:
    """让模型规划检索、收集来源并归纳要点，供调研业务逻辑判断。

    知识库和网页搜索都不是硬依赖；失败原因随结果返回。
    来源在这里去重并连续编号，不把公网摘要当作内部依据。
    """

    model = create_model()
    sources: list[Source] = list(state.get("sources", []))
    query_plan = model.with_structured_output(
        SearchQueries, method="function_calling"
    ).invoke([("system", RESEARCH_QUERY_PROMPT), ("human", json.dumps({
        "request": state["request"], "existing_sources": sources,
        "research_rounds": state.get("research_rounds", 0),
        "previous_research": state.get("research"),
    }, ensure_ascii=False))])

    seen_locators = {item["locator"] for item in sources}
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kb_error = None
    try:
        if query_plan.knowledge_queries:
            _collect_knowledge_sources(
                query_plan.knowledge_queries, sources, seen_locators, retrieved_at
            )
    except KnowledgeBaseError as exc:
        kb_error = str(exc)[:240]

    web_error = None
    if query_plan.web_queries:
        web_error = _collect_web_sources(
            query_plan.web_queries, sources, seen_locators, retrieved_at
        )

    missing_internal = query_plan.internal_facts_required and not any(
        source["source_type"] in {"knowledge_base", "user"} for source in sources
    )
    findings = None
    # 如果关键依据还需要用户补充，就不额外调用摘要模型；用户已补充过则继续。
    # 没有任何来源时，也没有可供摘要模型引用的材料。
    needs_clarification = (missing_internal or not sources) and not state.get("research_clarified", False)
    if sources and not needs_clarification:
        brief = model.with_structured_output(
            ResearchBrief, method="function_calling"
        ).invoke([
            ("system", RESEARCH_SUMMARY_PROMPT),
            ("human", json.dumps({"request": state["request"], "sources": sources}, ensure_ascii=False)),
        ])
        valid_ids = {source["source_id"] for source in sources}
        findings = []
        for finding in brief.findings:
            if finding.source_id not in valid_ids:
                raise ValueError(f"Researcher 引用了不存在的来源：{finding.source_id}")
            if finding.claim.strip():
                findings.append(f"- {finding.claim.strip()} [{finding.source_id}]")

    return _ResearchEvidence(
        sources=sources,
        knowledge_queried=bool(query_plan.knowledge_queries),
        web_queried=bool(query_plan.web_queries),
        kb_error=kb_error,
        web_error=web_error,
        missing_internal=missing_internal,
        findings=findings,
    )


def conduct_research(state: AgentState) -> ResearchNeedsInput | ResearchCompleted:
    """完成 Researcher 的业务判断，但不读写 LangGraph 的路由字段。

    返回两种明确结果：资料缺口需要用户补充，或调研已完成。
    这里决定追问内容、研究要点和报告文本；Graph 只把结果映射回 State。
    """

    evidence = _collect_research_evidence(state)
    sources = evidence.sources
    kb_error = evidence.kb_error
    web_error = evidence.web_error
    clarified = state.get("research_clarified", False)

    if not clarified and evidence.missing_internal:
        reason = "本地知识库暂不可用" if kb_error else "现有资料无法确认内部规则"
        return ResearchNeedsInput(
            sources=sources,
            question=f"{reason}。请补充相关规则或文档；如果暂时没有，也可以说‘先按待确认项出方案’。",
        )

    if not clarified and not sources:
        failures = []
        if kb_error:
            failures.append("本地知识库暂不可用")
        if web_error:
            failures.append("公开搜索暂不可用")
        reason = "、".join(failures) or "现有渠道没有找到可用依据"
        return ResearchNeedsInput(
            sources=sources,
            question=f"{reason}。请补充需求背景或相关文档；如果暂时没有，也可以说‘先按待确认项出方案’。",
        )

    if not sources:
        kb_note = (
            "知识库不可用" if kb_error else
            "知识库没有相关命中" if evidence.knowledge_queried else "未查询知识库"
        )
        web_note = (
            "网页检索失败" if web_error else
            "网页没有可用摘要" if evidence.web_queried else "未查询公网"
        )
        research = f"证据不足：{kb_note}；{web_note}。"
        return ResearchCompleted(
            sources=[],
            research=research,
            report=_render_research_report(research, [], kb_error, web_error),
            event_detail="没有可用来源，已标注证据缺口"
                         + ("；知识库不可用" if kb_error else ""),
        )

    lines = evidence.findings
    if lines is None:
        raise ValueError("研究来源存在，但未生成归纳结果")
    if not lines and not clarified:
        # 来源有命中，但模型无法从中提炼相关结论，不能硬凑产品方案。
        return ResearchNeedsInput(
            sources=sources,
            question="目前查到的资料不足以支持明确结论。请补充需求背景或相关规则；"
                     "如果暂时没有，也可以说‘先按待确认项出方案’。",
        )

    research = "\n".join(lines) if lines else "证据不足：检索片段无法支持明确结论。"
    if kb_error:
        research += "\n- 本地知识库暂不可用；内部背景待核验。"
    if web_error:
        research += "\n- 网页检索暂不可用；公开背景待核验。"
    if evidence.missing_internal:
        research += "\n- 内部知识库没有可用依据；用户补充需在方案审核时确认。"
    counts = {
        kind: sum(source["source_type"] == kind for source in sources)
        for kind in ("local_file", "knowledge_base", "web", "user")
    }
    event_detail = (
        f"仓库 {counts['local_file']} 条、知识库 {counts['knowledge_base']} 条、"
        f"网页 {counts['web']} 条、用户补充 {counts['user']} 条，形成 {len(lines)} 条要点"
        + ("；知识库不可用，已标注局限" if kb_error else "")
        + ("；网页检索失败，已标注局限" if web_error else "")
    )
    return ResearchCompleted(
        sources=sources,
        research=research,
        report=_render_research_report(research, sources, kb_error, web_error),
        event_detail=event_detail,
    )

def _unique_queries(raw_queries: list[str]) -> list[str]:
    """按模型给出的顺序保留非空查询；同一检索渠道不重复请求。"""

    queries: list[str] = []
    seen: set[str] = set()
    for raw in raw_queries:
        query = raw.strip()
        if query and query not in seen:
            queries.append(query)
            seen.add(query)
    return queries

def _collect_knowledge_sources(
    queries: list[str], sources: list[Source], seen_locators: set[str], retrieved_at: str
) -> None:
    """累计最多六条知识库证据；补查时保留旧来源并按 locator 去重。"""

    kb_count = sum(item["source_type"] == "knowledge_base" for item in sources)
    for query in _unique_queries(queries):
        if kb_count >= 6:
            return
        for hit in search_knowledge_base(query):
            locator = hit["locator"]
            if locator in seen_locators or not hit["snippet"].strip():
                continue
            seen_locators.add(locator)
            sources.append({
                "source_id": len(sources) + 1,
                "source_type": "knowledge_base",
                "title": hit["title"],
                "locator": locator,
                "snippet": hit["snippet"],
                "retrieved_at": retrieved_at,
            })
            kb_count += 1
            if kb_count >= 6:
                return


def _collect_web_sources(
    queries: list[str], sources: list[Source], seen_locators: set[str], retrieved_at: str
) -> str | None:
    """补充最多六条网页摘要；网页失败只记录局限，不丢弃已有本地证据。"""

    web_count = sum(item["source_type"] == "web" for item in sources)
    for query in _unique_queries(queries):
        if web_count >= 6:
            return None
        try:
            hits = search_web(query)
        except WebSearchError as exc:
            return str(exc)[:240]
        for hit in hits:
            locator = hit["url"]
            if locator in seen_locators or not hit["snippet"].strip():
                continue
            seen_locators.add(locator)
            sources.append({
                "source_id": len(sources) + 1,
                "source_type": "web",
                "title": hit["title"],
                "locator": locator,
                "snippet": hit["snippet"],
                "retrieved_at": retrieved_at,
            })
            web_count += 1
            if web_count >= 6:
                return None
    return None


def _render_research_report(
    research: str, sources: list[Source], kb_error: str | None, web_error: str | None
) -> str:
    """分开排版仓库、知识库、网页和用户补充，标出不可用渠道。"""

    report_lines = ["# 调研报告", "", "## 研究要点（按编号核对来源）", "", research,
                    "", "## 已选仓库文件（只读）", ""]
    local_sources = [item for item in sources if item["source_type"] == "local_file"]
    kb_sources = [item for item in sources if item["source_type"] == "knowledge_base"]
    web_sources = [item for item in sources if item["source_type"] == "web"]
    user_sources = [item for item in sources if item["source_type"] == "user"]
    if not local_sources:
        report_lines.append("- 未读取仓库文件；不能推断当前代码实现。")
    for source in local_sources:
        report_lines.append(f"- [{source['source_id']}] `{source['locator']}`（读取时间 UTC：{source['retrieved_at']}）")
    report_lines.extend(["", "## 本地知识库片段（演示库）", ""])
    if not kb_sources:
        report_lines.append("- 没有可用知识库证据；内部背景仍待确认。")
    for source in kb_sources:
        report_lines.append(
            f"- [{source['source_id']}] `{source['locator']}` "
            f"（检索时间 UTC：{source['retrieved_at']}）"
        )
    if kb_error:
        report_lines.append(f"- 本轮知识库检索失败：{kb_error}；内部背景未核验。")
    report_lines.extend(["", "## 外部网页搜索摘要", ""])
    if not web_sources:
        report_lines.append("- 无可用网页摘要。")
    if web_error:
        report_lines.append(f"- 本次网页检索失败：{web_error}")
    for source in web_sources:
        title = source["title"].replace("\n", " ").replace("\r", " ")
        report_lines.append(
            f"- [{source['source_id']}] {title}：<{source['locator']}> "
            f"（检索时间 UTC：{source['retrieved_at']}）"
        )
    report_lines.extend(["", "## 用户补充（未经独立核验）", ""])
    if not user_sources:
        report_lines.append("- 本轮没有额外的用户补充。")
    for source in user_sources:
        report_lines.append(
            f"- [{source['source_id']}] {source['snippet']} "
            f"（提供时间 UTC：{source['retrieved_at']}）"
        )
    report_lines.extend([
        "", "## 局限性", "",
        "仓库文件只证明已读取片段，知识库命中只是本地演示资料，可能过期或与需求无关；"
        "网页内容只依据搜索摘要，尚未逐篇核对原文。"
        "用户补充可作为需求依据，但尚未经其他渠道独立核验。"
        "产品决策仍需结合实际业务与人工确认。",
    ])
    return "\n".join(report_lines) + "\n"
