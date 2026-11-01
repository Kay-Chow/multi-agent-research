"""各模型 Worker 的结构化输出与调用：接待、派工、调研、方案和候选代码。"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.prompts import (
    CHANGE_PLAN_PROMPT, CODE_GENERATION_PROMPT, PRODUCT_PLAN_PROMPT,
    INTAKE_PROMPT, RESEARCH_QUERY_PROMPT, RESEARCH_SUMMARY_PROMPT,
    INTAKE_MODE_PROMPT, SUPERVISOR_ROUTE_PROMPT,
)
from app.state import AgentState, Source, SupervisorTarget, TaskMode
from app.repo_tools import validate_change_path
from app.kb_tools import KnowledgeBaseError, search_knowledge_base
from app.tools import WebSearchError, search_web


# 根据源码定位 .env，不依赖启动时的工作目录。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)
MAX_INTAKE_DECISION_ATTEMPTS = 2


class IntakeDecisionError(Exception):
    """接待模型连续给出无效结构化决定；由图节点转成可恢复的澄清。"""


def _model(timeout: int = 60) -> ChatDeepSeek:
    """统一创建模型客户端；代码生成允许传入更长超时。"""

    return ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=timeout,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )


class ModeDecision(BaseModel):
    """Intake 的结构化模式判断：是否需要进入代码修改流程。"""

    mode: TaskMode
    reason: str = Field(min_length=1)


class SupervisorDecision(BaseModel):
    """Supervisor 的一次派工决定；实际允许集合仍由图根据状态计算。"""

    next_agent: SupervisorTarget
    reason: str = Field(min_length=1, max_length=300)


class IntakeDecision(BaseModel):
    """探索接待的单步决定：回答/澄清、请求只读工具，或开始正式任务。

    字段类型由 Pydantic 校验；model_validator 再检查字段之间的关系。
    这样模型返回矛盾决定时，错误在入口处集中暴露，而不是留给各节点猜测。
    """

    ready: bool
    reply: str = Field(default="", max_length=600, description="回复用户的文本；开始正式任务时可为空")
    request: str | None = Field(default=None, max_length=1000)
    tool_name: Literal[
        "list_projects", "inspect_project", "read_file", "search_kb", "search_web"
    ] | None = Field(default=None, description="需要查资料时选择一个只读工具；否则为空")
    tool_query: str | None = Field(default=None, max_length=200, description="知识库或网页检索词")
    tool_path: str | None = Field(default=None, max_length=500, description="仓库名/路径或已选仓库内相对文件路径")

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        """一个决定只能选工具、开始任务或回复用户三条路径中的一条。"""

        if self.tool_name is not None:
            # tool_name 的可选值已由 Literal 限定，这里只检查跨字段约束。
            if self.ready or self.request:
                raise ValueError("调用工具时不能同时开始正式任务")
            if self.tool_name in {"search_kb", "search_web"} and not (self.tool_query or "").strip():
                raise ValueError("检索工具缺少查询词")
            if self.tool_name in {"inspect_project", "read_file"} and not (self.tool_path or "").strip():
                raise ValueError("仓库工具缺少路径")
            return self

        if self.ready:
            # 已有可执行需求就能开始；纯提示性的 reply 由图节点补默认文案。
            if not (self.request or "").strip():
                raise ValueError("开始正式任务时必须给出可执行的任务描述")
            return self
        if not self.reply.strip():
            raise ValueError("未调用工具且未开始任务时必须给出用户可见的回复")
        if self.request:
            raise ValueError("仍需澄清时不能提前生成任务")
        return self


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


class CodeFile(BaseModel):
    """Coder 提出的单个候选文件，使用完整新内容而不是可执行命令。"""

    path: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=16_000)


class CodePatchDraft(BaseModel):
    """Coder 的结构化候选改动；程序仍需逐项校验后才写入隔离区。"""

    edits: list[CodeFile] = Field(min_length=1, max_length=3)
    summary: str = Field(min_length=1)
    risks: list[str]


def intake_requirement(state: AgentState) -> IntakeDecision:
    """根据对话选择下一步；无效模型输出只重试一次，不执行任何工具。

    只捕获结构化解析/校验错误。网络超时、额度耗尽等调用故障不应伪装成
    “用户没说清楚”；真正的工具调用仍由图内的 discovery_tool 节点负责。
    """

    decision_model = _model().with_structured_output(
        IntakeDecision, method="function_calling"
    )
    messages = [
        ("system", INTAKE_PROMPT),
        ("human", json.dumps({
            "conversation": state["conversation"][-12:],
            "observations": state.get("discovery_observations", [])[-5:],
            "selected_repo": state.get("repo_root"),
            "remaining_lookups": max(0, 4 - state.get("discovery_lookup_count", 0)),
        }, ensure_ascii=False)),
    ]
    last_error: ValidationError | OutputParserException | None = None
    for attempt in range(1, MAX_INTAKE_DECISION_ATTEMPTS + 1):
        try:
            # with_structured_output 通常已返回 IntakeDecision；再校验一次可防御
            # 返回 dict/None 的模型适配器，且不依赖某个服务商的解析实现。
            return IntakeDecision.model_validate(decision_model.invoke(messages))
        except (ValidationError, OutputParserException) as exc:
            last_error = exc
            # 只记录出错字段与通用错误说明，不记录原始模型输出、
            # 用户对话或文件内容。自定义校验错误的说明均为固定文案。
            if isinstance(exc, ValidationError):
                error_codes = "; ".join(
                    f"{'.'.join(map(str, error['loc'])) or 'decision'}:"
                    f"{error['type']}({error['msg']})"
                    for error in exc.errors(include_input=False)
                )[:300]
            else:
                error_codes = type(exc).__name__
            logger.warning(
                "接待模型决定无效：task_id=%s，尝试=%s/%s，校验=%s",
                state["task_id"], attempt, MAX_INTAKE_DECISION_ATTEMPTS,
                error_codes,
            )
            if attempt < MAX_INTAKE_DECISION_ATTEMPTS:
                # 下次调用时给出纠错指令；不无限重试，也不执行未校验的工具决定。
                messages.append((
                    "human",
                    "上一条决定不符合结构化字段或字段关系。请重新给出一个决定："
                    "调用工具时 ready=false、request=null，并填写该工具所需参数；"
                    "不调用工具且 ready=false 时填写 reply；"
                    "ready=true 时必须填写 request，reply 可为空。",
                ))

    # 循环只会因决定有效而 return，或因达到重试上限而走到这里。
    raise IntakeDecisionError("接待模型连续返回无效决定") from last_error


def decide_mode(request: str) -> ModeDecision:
    """在 Intake 确认需求时判断模式；Supervisor 只读取已确定的结果。"""

    return _model().with_structured_output(
        ModeDecision, method="function_calling"
    ).invoke([("system", INTAKE_MODE_PROMPT), ("human", request)])


def decide_next(state: AgentState, allowed: list[SupervisorTarget]) -> SupervisorDecision:
    """在程序给定的安全候选中选择下一位执行者；无效输出回退到安全首选。

    单候选时无需消耗模型调用；多候选只出现在研究后、方案后两个
    补查决策点。模型只能在当前候选中选择，不能越过人工审核。
    """

    if not allowed:
        raise ValueError("Supervisor 没有可执行的下一步")
    if len(allowed) == 1:
        return SupervisorDecision(next_agent=allowed[0], reason="当前阶段只有一项允许的后续操作")

    messages = [
        ("system", SUPERVISOR_ROUTE_PROMPT),
        ("human", json.dumps({
            "request": state["request"],
            "decision_point": state["next_step"],
            "allowed_next_agents": allowed,
            "research_rounds": state.get("research_rounds", 0),
            "source_count": len(state.get("sources", [])),
            "research": (state.get("research") or "")[:1200],
            "draft_plan": (state.get("draft_plan") or "")[:1200],
            "open_questions": state.get("open_questions", [])[:5],
        }, ensure_ascii=False)),
    ]
    model = _model().with_structured_output(SupervisorDecision, method="function_calling")
    for attempt in range(2):
        try:
            decision = SupervisorDecision.model_validate(model.invoke(messages))
            if decision.next_agent in allowed:
                return decision
        except (ValidationError, OutputParserException):
            pass
        # 模型可能给出合法节点名，却不在当前状态允许的集合里。
        logger.warning("Supervisor 路由无效：task_id=%s，尝试=%s/2", state["task_id"], attempt + 1)
        messages.append(("human", f"上次路由无效。只能从 {allowed} 中选择一个 next_agent。"))
    return SupervisorDecision(next_agent=allowed[0], reason="模型路由无效，采用安全默认步骤")


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


def _markdown_bullets(items: list[str]) -> str:
    """把非空文本渲染为 Markdown 列表；全为空时明确显示“暂无”。"""

    lines = [f"- {item.strip()}" for item in items if item.strip()]
    return "\n".join(lines) or "- 暂无"


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


def _ask_research_question(state: AgentState, sources: list[Source], question: str) -> dict:
    """把缺口留在 State，交给图内的用户输入节点，而不是终止任务。"""

    return {
        "sources": sources,
        "research": None,
        "research_report": None,
        "research_question": question,
        "next_step": "await_research_input",
        "conversation": state["conversation"] + [{"role": "assistant", "text": question}],
        "events": ["researcher：关键依据不足，暂停并请用户补充"],
    }


def researcher_node(state: AgentState) -> dict:
    """收集需求证据，并生成带可核对引用编号的调研摘要。

    模型可按需提出本地知识库和公网检索词，也可在已有证据足够时
    不新增查询。知识库用于查业务背景，公网搜索只提供公开资料摘要；
    程序负责来源去重、编号及引用校验，不把网页摘要冒充内部依据。
    """

    model = _model()
    sources: list[Source] = list(state.get("sources", []))
    query_plan = model.with_structured_output(
        SearchQueries, method="function_calling"
    ).invoke([("system", RESEARCH_QUERY_PROMPT), ("human", json.dumps({
        "request": state["request"], "existing_sources": sources,
        "research_rounds": state.get("research_rounds", 0),
        "previous_research": state.get("research"),
    }, ensure_ascii=False))])

    # 本地与网页来源按进入列表的顺序连续编号；方案只能引用这些真实编号。
    seen_locators: set[str] = {item["locator"] for item in sources}
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 知识库不可用不直接终止：先尝试其他渠道；若仍缺关键依据，
    # 把问题交给用户。用户回答后再整理报告，不伪称检索成功。
    # Supervisor 补查可再发新查询；证据按 locator 去重、总数受限。
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
    if (missing_internal or not sources) and not state.get("research_clarified", False):
        # 一次澄清后不再反复追问。用户可以提供规则，也可以明确要求
        # 先带着待确认项出方案；两种回答都由下一轮 Researcher 如实记录。
        if missing_internal:
            reason = "本地知识库暂不可用" if kb_error else "现有资料无法确认内部规则"
            question = f"{reason}。请补充相关规则或文档；如果暂时没有，也可以说‘先按待确认项出方案’。"
        else:
            failures = []
            if kb_error:
                failures.append("本地知识库暂不可用")
            if web_error:
                failures.append("公开搜索暂不可用")
            reason = "、".join(failures) or "现有渠道没有找到可用依据"
            question = f"{reason}。请补充需求背景或相关文档；如果暂时没有，也可以说‘先按待确认项出方案’。"
        return _ask_research_question(state, sources, question)

    if not sources:
        kb_note = (
            "知识库不可用" if kb_error else
            "知识库没有相关命中" if query_plan.knowledge_queries else "未查询知识库"
        )
        web_note = (
            "网页检索失败" if web_error else
            "网页没有可用摘要" if query_plan.web_queries else "未查询公网"
        )
        research = f"证据不足：{kb_note}；{web_note}。"
        return {
            "sources": [],
            "research": research,
            "research_report": _render_research_report(research, [], kb_error, web_error),
            "next_step": "assess_research",
            "events": ["researcher：没有可用来源，已标注证据缺口"
                       + ("；知识库不可用" if kb_error else "")],
        }

    brief = model.with_structured_output(
        ResearchBrief, method="function_calling"
    ).invoke([
        ("system", RESEARCH_SUMMARY_PROMPT),
        ("human", json.dumps({"request": state["request"], "sources": sources}, ensure_ascii=False)),
    ])

    # 不能让模型凭空编造引用；即使文字看似合理，也必须有真实检索来源。
    valid_ids = {source["source_id"] for source in sources}
    lines = []
    for finding in brief.findings:
        if finding.source_id not in valid_ids:
            raise ValueError(f"Researcher 引用了不存在的来源：{finding.source_id}")
        if finding.claim.strip():
            lines.append(f"- {finding.claim.strip()} [{finding.source_id}]")

    if not lines and not state.get("research_clarified", False):
        # 检索有结果但与任务无关，也不能仅凭“来源列表非空”继续编方案。
        return _ask_research_question(
            state, sources,
            "目前查到的资料不足以支持明确结论。请补充需求背景或相关规则；"
            "如果暂时没有，也可以说‘先按待确认项出方案’。",
        )

    research = "\n".join(lines) if lines else "证据不足：检索片段无法支持明确结论。"
    if kb_error:
        research += "\n- 本地知识库暂不可用；内部背景待核验。"
    if web_error:
        research += "\n- 网页检索暂不可用；公开背景待核验。"
    if missing_internal:
        research += "\n- 内部知识库没有可用依据；用户补充需在方案审核时确认。"
    kb_sources = [item for item in sources if item["source_type"] == "knowledge_base"]
    web_sources = [item for item in sources if item["source_type"] == "web"]
    local_sources = [item for item in sources if item["source_type"] == "local_file"]
    user_sources = [item for item in sources if item["source_type"] == "user"]
    return {
        "sources": sources,
        "research": research,
        "research_report": _render_research_report(research, sources, kb_error, web_error),
        "next_step": "assess_research",
        "events": [f"researcher：仓库 {len(local_sources)} 条、知识库 {len(kb_sources)} 条、网页 {len(web_sources)} 条、用户补充 {len(user_sources)} 条，形成 {len(lines)} 条要点"
                   + ("；知识库不可用，已标注局限" if kb_error else "")
                   + ("；网页检索失败，已标注局限" if web_error else "")],
    }


def product_planner_node(state: AgentState) -> dict:
    """把用户需求与调研证据整理成待人工批准的产品方案。

    模型输出先经过字段、引用编号和验收项校验，再渲染为 Markdown。
    人工退回时会把旧方案与反馈重新交给模型，但不能据此编造新证据。
    """

    if state["mode"] is None or state["research"] is None:
        raise ValueError("产品方案缺少任务模式或研究结果")

    draft = _model().with_structured_output(
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
    # 列表非空不代表内容有效；还要排除空白的验收项和规则。
    for name, items in {
        "目标用户": draft.target_users,
        "范围": draft.in_scope,
        "规则": draft.rules,
        "验收标准": draft.acceptance_criteria,
    }.items():
        if any(not item.strip() for item in items):
            raise ValueError(f"产品方案的{name}包含空白项")
    # 用户补充也可作为需求依据；仅有网页来源时，
    # 必须留下待确认问题，不能把公开资料当作公司内部规则。
    if not any(source.get("source_type") in {"knowledge_base", "local_file", "user"} for source in state["sources"]):
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
    return {
        "draft_plan": "\n\n".join(sections) + "\n",
        "acceptance_criteria": [item.strip() for item in draft.acceptance_criteria],
        "open_questions": [item.strip() for item in draft.open_questions if item.strip()],
        "review_status": "pending",
        "next_step": "assess_plan",
        "events": [f"product_planner：生成方案，包含 {len(draft.acceptance_criteria)} 条验收标准"],
    }


def propose_change_plan(state: AgentState, contents: dict[str, str]) -> str:
    """根据已批准的产品方案与只读源码，提出文件级修改计划。

    此处仍不生成、更不写入代码。程序会核对每个目标路径属于已检查
    文件和变更白名单，再把计划交给阶段 3 作为候选改动的范围约束。
    """

    if state["mode"] != "code_change" or state["review_status"] != "approved":
        raise ValueError("只有已批准的代码类产品任务才能规划仓库变更")
    if not state["final_plan"] or not state["repo_context"]:
        raise ValueError("缺少已批准的产品方案或仓库摘要")

    draft = _model().with_structured_output(
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
    return _model(timeout=180).with_structured_output(
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
