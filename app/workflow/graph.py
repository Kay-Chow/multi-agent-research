"""产品 Agent 的唯一 LangGraph 工作流；阶段 3 只生成隔离 diff 并等待审核。"""

from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.intake import IntakeDecisionError, decide_mode, intake_requirement
from app.agents.planner import build_product_plan, propose_change_plan
from app.agents.researcher import ResearchNeedsInput, conduct_research
from app.agents.supervisor import decide_next
from app.config import PROJECT_ROOT
from app.services.artifact_store import save_change_plan, save_reports, verify_diff_candidate
from app.services.candidate_delivery import generate_candidate
from app.services.evidence import merge_discovery_sources, user_research_source
from app.tools.discovery import run_discovery_tool
from app.tools.knowledge_base import KnowledgeBaseError
from app.tools.repository import RepoAccessError, inspect_repository, verify_repository
from app.tools.web_search import WebSearchError
from app.tools.workspace import (
    WorkspaceError, planned_paths, prepare_workspace,
)
from app.workflow.state import AgentState, SupervisorTarget


OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAX_REVISIONS = 2
MAX_CODE_REVISIONS = 2
MAX_RESEARCH_ROUNDS = 2
MAX_SUPERVISOR_STEPS = 30
WORKER_STEPS: tuple[SupervisorTarget, ...] = (
    "intake", "discovery_tool", "await_user", "await_research_input",
    "researcher", "product_planner", "review", "save", "finish",
    "repo_inspector", "change_planner", "save_change_plan",
    "workspace_prepare", "coder", "diff_review", "finish_code",
)


# 1. 工作流节点：按实际执行顺序集中排列。
def supervisor_node(state: AgentState) -> dict:
    """状态机的唯一派工入口。

    `next_step` 由上一个节点显式写入。Intake 已确定任务模式，Supervisor
    不再重新解读用户需求；它只在研究后、方案后决定是否补查。
    接待阶段可跨多轮，因此不占正式任务的派工预算。
    """

    step = state["next_step"]
    if step in {"intake", "discovery_tool", "await_user"}:
        return {"events": [f"supervisor -> {step}：接待阶段"]}

    steps = state.get("supervisor_steps", 0) + 1
    if steps > MAX_SUPERVISOR_STEPS:
        return {
            "workflow_status": "blocked", "error": "Supervisor 派工次数达到上限",
            "next_step": None,
            "events": ["supervisor：达到派工上限，停止任务"],
        }

    mode = state["mode"]
    if mode not in ("proposal_only", "code_change"):
        raise ValueError("任务模式尚未由 Intake 确定")

    allowed = _allowed_supervisor_targets(state)
    chosen = decide_next(state, allowed)
    if chosen.next_agent not in allowed:
        raise ValueError(f"Supervisor 选择了当前不允许的节点：{chosen.next_agent}")
    update = {
        "supervisor_steps": steps,
        "next_step": chosen.next_agent,
        "events": [f"supervisor -> {chosen.next_agent}：{chosen.reason}"],
    }
    if chosen.next_agent == "researcher":
        update["research_rounds"] = state.get("research_rounds", 0) + 1
        if state["review_status"] == "pending":
            # 已有方案尚未交人审核时，先作废待审核标记；保留草稿供重写参考。
            update["review_status"] = "not_ready"
    return update


def intake_node(state: AgentState) -> dict:
    """理解一轮输入，返回接待状态；下一位 Worker 由 Supervisor 统一派发。

    对话入口把历史消息和已有观察交给模型。它可以请求只读工具、
    向用户澄清或提炼正式需求，但不直接连到工具、用户中断或 Researcher。
    显式创建任务已经给出完整 request；本节点只确认模式，不调用澄清模型。
    """

    if state["intake_status"] == "ready" and not state["conversation"]:
        return _finish_intake(state, state["request"], None)
    if state["intake_status"] != "collecting" or not state["conversation"]:
        raise ValueError("需求澄清状态或对话历史无效")

    try:
        decision = intake_requirement(state)
    except IntakeDecisionError:
        # 连续两次无效输出是模型问题，不把它当成用户已确认的需求；
        # 保留当前对话和已查到的证据，等待用户下一句话重新尝试。
        reply = "抱歉，我刚才没能完成这一步判断，尚未开始正式任务。你可以回复‘重试’，或补充当前需求；已查到的资料会保留。"
        return {
            "conversation": state["conversation"] + [{"role": "assistant", "text": reply}],
            "intake_status": "clarify", "intake_reply": reply,
            "next_step": "await_user",
            "events": ["intake：模型决定校验失败，等待用户重述"],
        }
    if decision.tool_name:
        if state.get("discovery_lookup_count", 0) >= 4:
            reply = "这轮已查阅多项资料。请提供更具体的问题或仓库路径，我再继续。"
            return {
                "conversation": state["conversation"] + [{"role": "assistant", "text": reply}],
                "intake_status": "clarify", "intake_reply": reply,
                "next_step": "await_user",
                "events": ["intake：本轮只读查找达到上限，等待用户补充"],
            }
        return {
            "intake_status": "lookup",
            "next_step": "discovery_tool",
            "discovery_tool": decision.tool_name,
            "discovery_query": decision.tool_query,
            "discovery_path": decision.tool_path,
            "events": [f"intake：请求只读工具 {decision.tool_name}"],
        }
    # ready=True 时 request 才是必需产物；如果模型省略提示语，就由程序补一句。
    reply = decision.reply.strip() or "需求已明确，我先调研并整理产品方案。"
    if decision.ready:
        return _finish_intake(state, decision.request.strip(), reply)
    return {
        "conversation": state["conversation"] + [{"role": "assistant", "text": reply}],
        "intake_status": "clarify",
        "next_step": "await_user",
        "intake_reply": reply,
        "events": ["intake：信息不足，等待用户补充"],
    }


def discovery_tool_node(state: AgentState) -> dict:
    """执行一个受限只读工具，合并来源后回 Supervisor，再由它派发 Intake。"""

    name = state.get("discovery_tool")
    if state.get("intake_status") != "lookup" or not name:
        raise ValueError("当前没有待执行的接待工具")
    try:
        result = run_discovery_tool(
            name, query=state.get("discovery_query"), path=state.get("discovery_path"),
            repo_root=state.get("repo_root"),
        )
        observation = result.observation[:5000]
        repo_root = result.repo_root or state.get("repo_root")
        sources = merge_discovery_sources(state["sources"], result.sources)
        event = f"discovery_tool：{name} 已返回只读观察"
    except (RepoAccessError, KnowledgeBaseError, WebSearchError, ValueError) as exc:
        # 可选的探索查询失败不立即销毁整条会话；模型可以解释或请用户给路径。
        observation = f"工具 {name} 失败：{exc}"
        repo_root = state.get("repo_root")
        sources = state["sources"]
        event = f"discovery_tool：{name} 失败，交 Intake 说明或澄清"
    return {
        "intake_status": "collecting",
        "next_step": "intake",
        "discovery_tool": None, "discovery_query": None, "discovery_path": None,
        "discovery_lookup_count": state.get("discovery_lookup_count", 0) + 1,
        "discovery_observations": (state.get("discovery_observations", []) + [f"{name}: {observation}"])[-8:],
        "repo_root": repo_root,
        "sources": sources,
        "events": [event],
    }


def await_user_node(state: AgentState) -> dict:
    """用 LangGraph interrupt 暂停会话，并把下一句用户输入存进 State。

    恢复时这个节点会从头重跑，因此 interrupt 之前只做无副作用校验；
    CLI 用同一 thread_id 和 Command(resume=...) 送回下一句话。
    """

    if state["intake_status"] != "clarify" or not state["intake_reply"]:
        raise ValueError("当前没有待用户回答的澄清问题")
    response = interrupt({
        "kind": "intake",
        "session_id": state["task_id"],
        "question": state["intake_reply"],
    })
    if not isinstance(response, dict) or not isinstance(response.get("message"), str):
        raise ValueError("澄清回复必须包含文本 message")
    message = response["message"].strip()
    if not message or len(message) > 1000:
        raise ValueError("单条聊天消息长度必须在 1～1000 字符之间")
    return {
        "conversation": state["conversation"] + [{"role": "user", "text": message}],
        "intake_status": "collecting",
        "next_step": "intake",
        "intake_reply": None,
        "discovery_lookup_count": 0,
        "events": ["await_user：收到用户补充，交回 Supervisor 继续派工"],
    }


def researcher_node(state: AgentState) -> dict:
    """接收调研结果，只负责更新 State 和选择下一步。

    取证、证据充分性判断、追问文案与报告生成都在 agents/researcher.py。
    本节点不直接调用模型或搜索服务，也不拼接业务结论。
    """

    result = conduct_research(state)
    if isinstance(result, ResearchNeedsInput):
        return {
            "sources": result.sources,
            "research": None,
            "research_report": None,
            "research_question": result.question,
            "next_step": "await_research_input",
            "conversation": state["conversation"] + [{"role": "assistant", "text": result.question}],
            "events": ["researcher：关键依据不足，暂停并请用户补充"],
        }
    return {
        "sources": result.sources,
        "research": result.research,
        "research_report": result.report,
        "next_step": "assess_research",
        "events": [f"researcher：{result.event_detail}"],
    }


def await_research_input_node(state: AgentState) -> dict:
    """调研缺关键依据时暂停，收到用户说明后保留为可引用的用户来源。

    Researcher 先把问题写进 State；本节点只负责 interrupt 和接收回复。
    恢复后仍回 Supervisor，再由它派 Researcher 重新整理证据，而不是
    把用户回答直接当成已完成的调研报告。
    """

    question = state.get("research_question")
    if not question:
        raise ValueError("当前没有待用户补充的调研问题")
    response = interrupt({
        "kind": "research",
        "session_id": state["task_id"],
        "question": question,
    })
    if not isinstance(response, dict) or not isinstance(response.get("message"), str):
        raise ValueError("调研补充必须包含文本 message")
    message = response["message"].strip()
    if not message or len(message) > 1000:
        raise ValueError("单条补充消息长度必须在 1～1000 字符之间")
    user_source = user_research_source(state["sources"], state["task_id"], message)
    return {
        "conversation": state["conversation"] + [{"role": "user", "text": message}],
        "sources": state["sources"] + [user_source],
        "research_question": None,
        "research_clarified": True,
        "next_step": "researcher",
        "events": ["await_research_input：收到用户补充，交回 Supervisor 继续调研"],
    }


def product_planner_node(state: AgentState) -> dict:
    """把校验后的产品方案写入 State，交由 Supervisor 派发人工审核。"""

    plan = build_product_plan(state)
    return {
        "draft_plan": plan.markdown,
        "acceptance_criteria": plan.acceptance_criteria,
        "open_questions": plan.open_questions,
        "review_status": "pending",
        "next_step": "assess_plan",
        "events": [f"product_planner：生成方案，包含 {len(plan.acceptance_criteria)} 条验收标准"],
    }


# 第一道人审与方案落盘。
def review_node(state: AgentState) -> dict:
    """只暂停和接收决定；写文件等副作用必须放到后续节点。"""

    if state["review_status"] != "pending" or state["draft_plan"] is None:
        raise ValueError("当前没有可审核的产品方案")

    decision = interrupt({
        "task_id": state["task_id"],
        "mode": state["mode"],
        "research_report": state["research_report"],
        "draft_plan": state["draft_plan"],
        "revision_count": state["revision_count"],
        "options": ["approve", "reject", "terminate"],
    })
    if not isinstance(decision, dict):
        raise ValueError("审核结果必须是字典")

    action = decision.get("action")
    if action == "approve":
        return {
            "review_status": "approved",
            "final_plan": state["draft_plan"],
            "next_step": "save",
            "events": ["review：人工批准产品方案"],
        }
    if action == "reject":
        if state["revision_count"] >= MAX_REVISIONS:
            return {
                "review_status": "terminated",
                "workflow_status": "terminated",
                "next_step": None,
                "events": ["review：退回次数达到上限，任务终止"],
            }
        feedback = str(decision.get("feedback") or "").strip()
        if not feedback:
            raise ValueError("退回方案时必须填写修改意见")
        return {
            "review_status": "rejected",
            "review_feedback": feedback,
            "revision_count": state["revision_count"] + 1,
            "next_step": "product_planner",
            "events": ["review：人工退回产品方案"],
        }
    if action == "terminate":
        return {
            "review_status": "terminated",
            "workflow_status": "terminated",
            "next_step": None,
            "events": ["review：人工终止任务"],
        }
    raise ValueError(f"未知审核操作：{action}")


def save_node(state: AgentState) -> dict:
    """方案批准后保存两份独立 Markdown；重复恢复不会产生第二份。"""

    if state["review_status"] != "approved":
        raise ValueError("方案未经人工批准，不能保存")
    research = state["research_report"]
    proposal = state["final_plan"]
    if not research or not proposal:
        raise ValueError("调研报告或最终方案为空")

    research_path, proposal_path = save_reports(
        OUTPUT_DIR, state["task_id"], research, proposal,
    )
    return {
        "research_path": research_path,
        "proposal_path": proposal_path,
        "next_step": "finish" if state["mode"] == "proposal_only" else "repo_inspector",
        "events": ["save：已保存调研报告与产品方案"],
    }


def finish_node(state: AgentState) -> dict:
    """只出方案任务在人工批准并保存后完成。"""

    if state["mode"] != "proposal_only":
        raise ValueError("代码任务不能在方案保存后提前结束")
    if state["review_status"] != "approved" or not state["research_path"] or not state["proposal_path"]:
        raise ValueError("产品方案尚未批准或保存，不能结束任务")
    return {
        "workflow_status": "completed",
        "next_step": None,
        "events": ["finish：仅调研/方案任务完成"],
    }


# 代码阶段：仓库检查、隔离候选和第二道人审。
def repo_inspector_node(state: AgentState) -> dict:
    """只读检查会话中选定的仓库；代码阶段仍只取小范围文件元数据。"""

    try:
        if state["review_status"] != "approved":
            raise RepoAccessError("产品方案尚未批准，不能进入仓库阶段")
        if not state.get("repo_root"):
            raise RepoAccessError("尚未选定仓库；请在需求中给出路径并重新开始任务")
        context, _ = inspect_repository(state["repo_root"])
    except RepoAccessError as exc:
        return {
            "workflow_status": "blocked",
            "error": str(exc),
            "next_step": None,
            "events": [f"repo_inspector：停止，{exc}"],
        }
    return {
        "repo_context": context,
        "next_step": "change_planner",
        "events": [f"repo_inspector：只读检查 {len(context['files'])} 个文件，HEAD={context['head'][:8]}"],
    }


def change_planner_node(state: AgentState) -> dict:
    """仓库未变化才让模型写文件级计划；任何校验失败都不改仓库。"""

    context = state["repo_context"]
    if context is None:
        raise ValueError("缺少仓库只读检查结果")
    try:
        contents = verify_repository(context)
        plan = propose_change_plan(state, contents)
    except (RepoAccessError, ValueError) as exc:
        return {
            "workflow_status": "blocked",
            "error": str(exc),
            "next_step": None,
            "events": [f"change_planner：停止，{exc}"],
        }
    return {
        "change_plan": plan,
        "next_step": "save_change_plan",
        "events": ["change_planner：形成文件级改动和测试计划；尚未修改代码"],
    }


def save_change_plan_node(state: AgentState) -> dict:
    """将文件级计划写入 Agent 输出目录；不修改已选原仓库。"""

    plan = state["change_plan"]
    if state["review_status"] != "approved" or not plan:
        raise ValueError("没有可保存的文件级变更计划")
    path = save_change_plan(OUTPUT_DIR, state["task_id"], plan)
    return {
        "change_plan_path": path,
        "next_step": "workspace_prepare",
        "events": ["save：已保存文件级变更计划"],
    }


def workspace_prepare_node(state: AgentState) -> dict:
    """建立一次代码尝试对应的隔离工作区。

    首次生成和人工退回后重新生成都经过此节点；每次先验证原仓库
    与先前读取的基线一致，再从该基线复制固定白名单文件。单独成节点
    是为了在 checkpoint 中明确记录准备完成；Coder 重跑时会再次复位
    工作区，避免沿用一次中断后留下的不完整候选文件。
    """

    context = state["repo_context"]
    plan = state["change_plan"]
    if state["review_status"] != "approved" or context is None or not plan or not state["change_plan_path"]:
        raise ValueError("方案未批准，或缺少仓库快照、文件级计划及其保存路径")
    attempt = state.get("code_revision_count", 0)
    try:
        # 即使此节点尚不写候选代码，也先确认计划确实包含合法的目标文件。
        planned_paths(plan)
        workspace, _ = prepare_workspace(context, state["task_id"], attempt)
    except (RepoAccessError, WorkspaceError) as exc:
        return {
            "workflow_status": "blocked", "error": str(exc), "next_step": None,
            "events": [f"workspace_prepare：停止，{exc}"],
        }
    return {
        "workspace_path": workspace,
        "code_review_status": "not_ready",
        "next_step": "coder",
        "events": [f"workspace_prepare：建立第 {attempt + 1} 次隔离工作区；基线仓库未改动"],
    }


def coder_node(state: AgentState) -> dict:
    """调用隔离候选服务，把产物写入 State 后交给第二个人工门禁。

    模型调用、工作区复位、文件校验和 diff 保存由 candidate_delivery
    完成；本节点只检查进入门槛、处理失败和决定下一步。
    """

    context = state["repo_context"]
    plan = state["change_plan"]
    if state["review_status"] != "approved" or context is None or not plan or not state["workspace_path"]:
        raise ValueError("方案未批准，或缺少仓库快照、文件级计划及隔离工作区")
    try:
        candidate = generate_candidate(state, OUTPUT_DIR)
    except (RepoAccessError, WorkspaceError, ValueError) as exc:
        return {
            "workflow_status": "blocked", "error": str(exc), "next_step": None,
            "events": [f"coder：停止，{exc}"],
        }
    return {
        "workspace_path": candidate.workspace_path,
        "diff_path": candidate.diff_path,
        "diff": candidate.diff,
        "file_hashes": candidate.file_hashes,
        "code_summary": candidate.summary,
        "code_review_status": "pending",
        "workflow_status": "diff_pending",
        "next_step": "diff_review",
        "error": None,
        "events": [f"coder：生成 {len(candidate.file_hashes)} 个候选文件和 diff；未运行测试、未提交"],
    }


def diff_review_node(state: AgentState) -> dict:
    """暂停等待人工审核隔离区 diff，而非直接执行候选代码。

    批准时重新核对 diff 文件、原仓库和隔离区哈希，防止等待期间被调包；
    退回时记录反馈并进入下一次独立尝试；终止时不碰原仓库。
    """

    diff = state["diff"]
    if state.get("code_review_status") != "pending" or not diff:
        raise ValueError("当前没有可审核的代码 diff")
    decision = interrupt({
        "task_id": state["task_id"],
        "diff": diff,
        "diff_path": state["diff_path"],
        "workspace_path": state["workspace_path"],
        "file_hashes": state["file_hashes"],
        "code_revision_count": state["code_revision_count"],
        "options": ["approve", "reject", "terminate"],
    })
    if not isinstance(decision, dict):
        raise ValueError("代码审核结果必须是字典")
    action = decision.get("action")
    if action == "approve":
        try:
            verify_diff_candidate(
                OUTPUT_DIR, state["task_id"], state["code_revision_count"],
                diff, state["repo_context"], state["file_hashes"],
            )
        except (RepoAccessError, WorkspaceError) as exc:
            return {
                "workflow_status": "blocked", "error": str(exc), "next_step": None,
                "events": [f"diff_review：审核时文件变化，停止；{exc}"],
            }
        return {
            "code_review_status": "approved",
            "next_step": "finish_code",
            "events": ["diff_review：人工批准 diff；尚未运行测试或提交"],
        }
    if action == "reject":
        # 退回不在现有候选文件上继续修改；新 attempt 从同一基线重新生成。
        if state["code_revision_count"] >= MAX_CODE_REVISIONS:
            return {
                "code_review_status": "terminated", "workflow_status": "terminated",
                "next_step": None,
                "events": ["diff_review：退回次数达到上限，任务终止"],
            }
        feedback = str(decision.get("feedback") or "").strip()
        if not feedback:
            raise ValueError("退回 diff 时必须填写修改意见")
        return {
            "code_review_status": "rejected",
            "code_review_feedback": feedback,
            "code_revision_count": state["code_revision_count"] + 1,
            "next_step": "workspace_prepare",
            "events": ["diff_review：人工退回 diff，进入新隔离工作区重做"],
        }
    if action == "terminate":
        return {
            "code_review_status": "terminated", "workflow_status": "terminated",
            "next_step": None,
            "events": ["diff_review：人工终止代码任务；无 Git 提交"],
        }
    raise ValueError(f"未知代码审核操作：{action}")


def finish_code_node(state: AgentState) -> dict:
    """阶段 3 的完成线只是 diff 获批，不伪称已测试或提交。"""

    if state["code_review_status"] != "approved":
        raise ValueError("代码 diff 未经人工批准")
    return {
        "workflow_status": "diff_approved",
        "next_step": None,
        "events": ["finish_code：diff 已批准；当前未执行测试、提交或发布"],
    }


# 2. 辅助函数：供上述节点和图装配调用，本身不是 LangGraph 节点。
def _allowed_supervisor_targets(state: AgentState) -> list[SupervisorTarget]:
    """把显式下一步转成合法候选；仅两个补查点允许模型二选一。"""

    if state["review_status"] == "terminated" or state["code_review_status"] == "terminated":
        raise ValueError("任务已终止，不允许继续派工")
    step = state["next_step"]
    if step == "assess_research":
        allowed: list[SupervisorTarget] = ["product_planner"]
    elif step == "assess_plan":
        allowed = ["review"]
    elif step in WORKER_STEPS:
        return [cast(SupervisorTarget, step)]
    else:
        raise ValueError(f"未知或已结束的下一步：{step}")
    if state["research_rounds"] < MAX_RESEARCH_ROUNDS:
        allowed.append("researcher")
    return allowed


def route_after_supervisor(state: AgentState) -> str:
    """Supervisor 已把决策写成具体 Worker；阻断则直接结束。"""

    if state["workflow_status"] == "blocked":
        return "blocked"
    target = state["next_step"]
    if target not in WORKER_STEPS:
        raise ValueError(f"Supervisor 未设置有效的下一位执行者：{target}")
    return target


def route_after_worker(state: AgentState) -> str:
    """所有非完成节点共用这一条回边；终止或阻断时直接结束。"""

    return "stop" if state["workflow_status"] in {"blocked", "terminated"} else "continue"


def _finish_intake(state: AgentState, request: str, reply: str | None) -> dict:
    """需求明确后确定模式；显式任务和聊天任务共用这一个出口。

    聊天模式仅依据用户原话分类，避免仓库资料影响模型整理出的 request；
    显式任务则优先使用调用方指定的模式，没有指定才依据 request 判断。
    代码任务若尚无仓库，先澄清或阻断，不进入调研。
    """

    user_words = "\n".join(
        item["text"] for item in state["conversation"][-12:] if item["role"] == "user"
    )
    if state["requested_mode"] is not None:
        mode, reason = state["requested_mode"], "调用方指定模式"
    else:
        decision = decide_mode(user_words or request)
        mode, reason = decision.mode, decision.reason
    if mode not in ("proposal_only", "code_change"):
        raise ValueError(f"未知任务模式：{mode}")

    if mode == "code_change" and not state.get("repo_root"):
        if not state["conversation"]:
            return {
                "mode": mode, "next_step": None,
                "workflow_status": "blocked",
                "error": "代码任务缺少 repo_root；HTTP 创建任务需显式提供仓库路径",
                "events": ["intake：缺少仓库路径，任务停止"],
            }
        question = "要生成候选代码，请先告诉我仓库路径；也可以让我在已授权的项目目录中查找。"
        return {
            "request": request,
            "conversation": state["conversation"] + [{"role": "assistant", "text": question}],
            "intake_status": "clarify", "intake_reply": question,
            "next_step": "await_user",
            "events": ["intake：等待选定代码仓库"],
        }

    plan = ["researcher", "product_planner", "human_review"]
    if mode == "code_change":
        plan += ["repo_inspector", "change_planner", "workspace_prepare", "coder", "diff_review"]
    conversation = state["conversation"]
    if reply is not None:
        conversation = conversation + [{"role": "assistant", "text": reply}]
    return {
        "request": request,
        "conversation": conversation,
        "mode": mode,
        "plan": plan,
        "intake_status": "ready",
        "intake_reply": reply,
        "next_step": "researcher",
        "events": [f"intake：需求已明确，模式 {mode}；{reason}"],
    }


# 3. 图装配：注册节点并连接所有派工与停止边。
def build_graph(checkpointer):
    """组装唯一的产品交付工作流，并绑定可恢复的 checkpointer。

    START 先到 Supervisor，再由它派发 Intake；接待、只读工具和等待用户
    完成后都返回 Supervisor。正式任务的每个专职节点也回到 Supervisor
    重新派工；资料缺口会暂停向用户求证，再继续调研。Supervisor 只能在程序允许的节点中
    选择下一位执行者；两道审核仍由 `interrupt` 暂停，不能由模型批准。
    """

    builder = StateGraph(AgentState)
    # 注册顺序按上方职责分区排列；真正的执行顺序由下方边决定。
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("intake", intake_node)
    builder.add_node("discovery_tool", discovery_tool_node)
    builder.add_node("await_user", await_user_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("await_research_input", await_research_input_node)
    builder.add_node("product_planner", product_planner_node)
    builder.add_node("review", review_node)
    builder.add_node("save", save_node)
    builder.add_node("finish", finish_node)
    builder.add_node("repo_inspector", repo_inspector_node)
    builder.add_node("change_planner", change_planner_node)
    builder.add_node("save_change_plan", save_change_plan_node)
    builder.add_node("workspace_prepare", workspace_prepare_node)
    builder.add_node("coder", coder_node)
    builder.add_node("diff_review", diff_review_node)
    builder.add_node("finish_code", finish_code_node)

    # 入口和唯一派工点。具体目标直接取 next_step，不再手写第二份节点名单。
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {**{worker: worker for worker in WORKER_STEPS}, "blocked": END},
    )
    # 所有非完成 Worker 都只写 next_step，然后回 Supervisor；阻断/终止直接结束。
    # 因此图的回边只有一条规则，具体业务转移写在产生结果的节点旁边。
    for worker in WORKER_STEPS:
        if worker in {"finish", "finish_code"}:
            continue
        builder.add_conditional_edges(
            worker, route_after_worker,
            {"continue": "supervisor", "stop": END},
        )
    builder.add_edge("finish_code", END)
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
