import json
from typing import Literal, Any
import ast
import re

from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from app.prompts import SUPERVISOR_PROMPT, RESEARCH_QUERY_PROMPT, RESEARCH_SUMMARY_PROMPT, CODER_PROMPT, WRITER_PROMPT
from app.state import AgentState, Source
from app.tools import search_web

from pathlib import Path
from dotenv import load_dotenv

from datetime import datetime, timezone


# 根据本文件位置定位项目根目录，不依赖 PowerShell 当前目录。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

Worker = Literal["researcher", "coder", "writer"]


class SupervisorDecision(BaseModel):
    """模型必须按此结构返回决策；字段不合要求会触发校验错误。"""

    needs_code: bool
    plan: list[Worker] = Field(min_length=2, max_length=3)
    next_agent: Literal["researcher", "coder", "writer", "finish"]
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def decide_next(state: AgentState) -> dict[str, Any] | BaseModel:
    """让模型读取任务进度，返回一次结构化的 Supervisor 决策。"""

    # 首次调用时 needs_code 还只是阶段 1 的占位值，不应影响模型判断。
    snapshot = {
        "task": state["task"],
        "existing_plan": state["plan"],
        "known_needs_code": state["needs_code"] if state["plan"] else None,
        "research_done": state["research"] is not None,
        "code_done": state["code"] is not None,
        "draft_report_done": state["draft_report"] is not None,
    }

    llm = ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # function_calling 负责引导模型按 schema 输出；
    # Pydantic 负责校验字段类型，但业务规则仍需下一步在 Python 中检查。
    structured_llm = llm.with_structured_output(
        SupervisorDecision,
        method="function_calling",
    )
    return structured_llm.invoke([
        ("system", SUPERVISOR_PROMPT),
        ("human", json.dumps(snapshot, ensure_ascii=False)),
    ])


class SearchQueries(BaseModel):
    """Researcher 先决定查什么，而不是把整段任务直接丢给搜索 API。"""

    queries: list[str] = Field(min_length=1, max_length=2)


class Finding(BaseModel):
    claim: str = Field(min_length=1)
    source_id: int


class ResearchBrief(BaseModel):
    findings: list[Finding] = Field(max_length=5)


def researcher_node(state: AgentState) -> dict:
    """检索真实网页，并把可追溯的研究要点写入共享 State。"""

    llm = ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )

    query_plan = llm.with_structured_output(
        SearchQueries,
        method="function_calling",
    ).invoke([
        ("system", RESEARCH_QUERY_PROMPT),
        ("human", json.dumps({
            "task": state["task"],
            "instruction": state["instruction"],
        }, ensure_ascii=False)),
    ])

    sources: list[Source] = []
    seen_queries = set()
    seen_urls = set()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for query in query_plan.queries:
        query = query.strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)

        # search_web 只返回真实搜索结果；跨查询再做一次 URL 去重。
        for hit in search_web(query):
            if hit["url"] in seen_urls or not hit["snippet"].strip():
                continue
            seen_urls.add(hit["url"])

            sources.append({
                "source_id": len(sources) + 1,  # 编号由程序分配，不由模型编造。
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
        return {
            "research": "证据不足：没有检索到可用的网页摘要。",
            "sources": [],
            "events": ["researcher：搜索完成，但没有可用来源"],
        }

    brief = llm.with_structured_output(
        ResearchBrief,
        method="function_calling",
    ).invoke([
        ("system", RESEARCH_SUMMARY_PROMPT),
        ("human", json.dumps({
            "task": state["task"],
            "sources": sources,
        }, ensure_ascii=False)),
    ])

    valid_ids = {source["source_id"] for source in sources}
    lines = []
    for finding in brief.findings:
        # 结构化输出只能保证字段类型；来源编号还需要程序核验。
        if finding.source_id not in valid_ids:
            raise ValueError(f"Researcher 引用了不存在的来源：{finding.source_id}")
        lines.append(f"- {finding.claim} [{finding.source_id}]")

    research = "\n".join(lines) if lines else "证据不足：检索摘要无法支持明确结论。"
    return {
        "research": research,
        "sources": sources,
        "events": [f"researcher：检索到 {len(sources)} 条来源，形成 {len(lines)} 条要点"],
    }


class CodeDraft(BaseModel):
    code: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


def coder_node(state: AgentState) -> dict:
    """生成代码并做静态语法检查；不运行代码。"""

    if not state["needs_code"] or state["research"] is None:
        raise ValueError("Coder 缺少执行前提：任务不需要代码，或研究尚未完成")

    llm = ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )
    structured_llm = llm.with_structured_output(
        CodeDraft,
        method="function_calling",
    )

    previous_code = ""
    syntax_error = ""

    # 最多生成两次：首次生成；若语法错误，再携带错误信息修正一次。
    for attempt in range(2):
        draft = structured_llm.invoke([
            ("system", CODER_PROMPT),
            ("human", json.dumps({
                "task": state["task"],
                "instruction": state["instruction"],
                "research": state["research"],
                "sources": state["sources"],
                "previous_code": previous_code,
                "syntax_error": syntax_error,
            }, ensure_ascii=False)),
        ])

        if not draft.code.strip():
            syntax_error = "代码为空"
            previous_code = draft.code
            continue

        try:
            ast.parse(draft.code)
        except SyntaxError as exc:
            syntax_error = f"{exc.msg}（第 {exc.lineno} 行）"
            previous_code = draft.code
            continue

        return {
            "code": draft.code,
            "code_explanation": draft.explanation,
            "code_syntax_ok": True,
            "events": [f"coder：生成代码，语法检查通过（第 {attempt + 1} 次）"],
        }

    raise ValueError(f"Coder 两次生成后仍有语法错误：{syntax_error}")

class ReportDraft(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    body: str = Field(min_length=1)
    limitations: str = Field(min_length=1)


def writer_node(state: AgentState) -> dict:
    """生成报告正文，校验引用，再由程序拼入代码和来源表。"""

    if state["research"] is None:
        raise ValueError("Writer 不能在 Researcher 完成前执行")
    if state["needs_code"] and (
        state["code"] is None or state["code_syntax_ok"] is not True
    ):
        raise ValueError("任务需要代码，但尚无通过语法检查的代码")

    llm = ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )
    structured_llm = llm.with_structured_output(
        ReportDraft,
        method="function_calling",
    )

    valid_ids = {source["source_id"] for source in state["sources"]}
    validation_error = ""
    previous_draft = None

    # 引用不合格时让模型修正一次；不能无限重试。
    for _ in range(2):
        draft = structured_llm.invoke([
            ("system", WRITER_PROMPT),
            ("human", json.dumps({
                "task": state["task"],
                "instruction": state["instruction"],
                "research": state["research"],
                "sources": state["sources"],
                "needs_code": state["needs_code"],
                "code_explanation": state["code_explanation"],
                "validation_error": validation_error,
                "previous_draft": previous_draft,
                "review_feedback": state["review_feedback"],
                "prior_report": (
                    state["draft_report"]
                    if state["review_status"] == "rejected"
                    else None
                ),
            }, ensure_ascii=False)),
        ])

        model_text = "\n".join([
            draft.title, draft.summary, draft.body, draft.limitations
        ])
        cited_ids = {
            int(number) for number in re.findall(r"\[(\d+)\]", model_text)
        }

        # 只校验模型写的部分；程序生成的“来源列表”不参与此检查。
        if cited_ids - valid_ids:
            validation_error = f"引用了不存在的编号：{sorted(cited_ids - valid_ids)}"
        elif re.search(r"https?://", model_text):
            validation_error = "正文不要直接写 URL，由程序统一生成来源列表"
        elif (
            valid_ids
            and not state["research"].startswith("证据不足")
            and not cited_ids
        ):
            validation_error = "已有研究来源，正文至少需要一个 [编号] 引用"
        else:
            break

        previous_draft = draft.model_dump()
    else:
        raise ValueError(f"Writer 两次输出后仍未通过引用校验：{validation_error}")

    parts = [
        f"# {draft.title}",
        f"## 摘要\n{draft.summary}",
        f"## 研究结论\n{draft.body}",
    ]

    if state["needs_code"]:
        # 代码取自 Coder 的 State，不让 Writer 重新生成或悄悄改写。
        parts.append(
            "## Python 示例\n"
            f"```python\n{state['code'].rstrip()}\n```\n\n"
            f"{state['code_explanation'] or ''}\n\n"
            "验证范围：仅通过 Python 语法解析，未执行代码。"
        )

    if state["sources"]:
        source_lines = [
            f"[{source['source_id']}] "
            f"[{source['title']}]({source['url']}) "
            f"（检索时间 UTC：{source['retrieved_at']}）"
            for source in state["sources"]
        ]
        parts.append("## 来源\n" + "\n".join(source_lines))
    else:
        parts.append("## 来源\n没有可用来源；上述内容不得视为已核实结论。")

    parts.append(f"## 局限性\n{draft.limitations}")

    return {
        "draft_report": "\n\n".join(parts) + "\n",
        "events": [f"writer：完成草稿，校验了 {len(cited_ids)} 个引用编号"],
        "review_status": "pending",
    }