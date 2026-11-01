"""产品调研与研发交付 Agent 的共享状态。"""

from operator import add
from typing import Annotated, Literal, TypedDict
from uuid import uuid4


TaskMode = Literal["proposal_only", "code_change"]
SupervisorTarget = Literal[
    "intake", "discovery_tool", "await_user", "await_research_input",
    "researcher", "product_planner",
    "review", "save", "repo_inspector",
    "change_planner", "save_change_plan", "workspace_prepare", "coder",
    "diff_review", "finish", "finish_code",
]
# `next_step` 是状态机唯一的路由依据；assess_* 是 Supervisor 的补查决策点，
# 其余非空值直接对应下一位 Worker。任务模式由 Intake 确定。
NextStep = SupervisorTarget | Literal["assess_research", "assess_plan"] | None
IntakeStatus = Literal["collecting", "lookup", "clarify", "ready"]
ReviewStatus = Literal["not_ready", "pending", "rejected", "approved", "terminated"]
WorkflowStatus = Literal[
    "running", "completed", "diff_pending", "diff_approved",
    "blocked", "terminated",
]
CodeReviewStatus = Literal["not_ready", "pending", "rejected", "approved", "terminated"]


class Source(TypedDict):
    """一条可追溯的研究依据；用户补充与检索结果一样有独立编号。"""

    source_id: int
    source_type: Literal["local_file", "knowledge_base", "web", "user"]
    title: str
    locator: str
    snippet: str
    retrieved_at: str


class RepoFile(TypedDict):
    """只读仓库快照中的单文件元数据，不保存源码正文。"""

    path: str
    sha256: str
    bytes: int
    lines: int


class RepoContext(TypedDict):
    """仓库基线：固定根目录、Git HEAD 和白名单文件的哈希摘要。"""

    root: str
    head: str
    files: list[RepoFile]


class FileHash(TypedDict):
    """候选改动前后的哈希，供人工批准 diff 时防止文件被调包。"""

    path: str
    before_sha256: str
    after_sha256: str


class ChatMessage(TypedDict):
    """接待或调研澄清中的一句话；只保存角色和文本。"""

    role: Literal["user", "assistant"]
    text: str


class AgentState(TypedDict):
    """LangGraph 的共享状态，也是跨进程恢复任务的 checkpoint 内容。

    各节点只返回自己负责更新的字段：普通字段会覆盖旧值，`events`
    则因下方的 `Annotated[..., add]` 而把新列表追加到旧列表。
    产品方案审核和代码 diff 审核是两个独立门禁，分别使用 `review_status`
    与 `code_review_status`，避免一次批准被误当成另一次批准。
    """

    # 图内探索环节：模型可请求受限只读工具；每轮最多查少量资料，用户回答后重新计数。
    task_id: str
    intake_status: IntakeStatus
    intake_reply: str | None
    conversation: list[ChatMessage]
    repo_root: str | None
    discovery_tool: str | None
    discovery_query: str | None
    discovery_path: str | None
    discovery_observations: list[str]
    discovery_lookup_count: int

    # 正式需求与状态机的下一步。澄清中 request 暂时为空。
    request: str
    requested_mode: TaskMode | None
    mode: TaskMode | None
    plan: list[str]
    next_step: NextStep
    supervisor_steps: int  # 正式任务派工次数；对话澄清可跨多轮，不占该预算。
    research_rounds: int

    # 阶段 1：调研证据、产品方案及第一次人工审核。
    sources: list[Source]
    research_question: str | None
    research_clarified: bool  # 缺证据时最多主动追问一次，避免反复打断用户。
    research: str | None
    research_report: str | None
    draft_plan: str | None
    final_plan: str | None
    acceptance_criteria: list[str]
    open_questions: list[str]

    review_status: ReviewStatus
    review_feedback: str
    revision_count: int
    workflow_status: WorkflowStatus
    # 阶段 2：批准后的报告文件与已选仓库的只读分析。
    research_path: str | None
    proposal_path: str | None
    repo_context: RepoContext | None
    change_plan: str | None
    change_plan_path: str | None
    # 阶段 3：隔离候选代码、统一 diff 和第二次人工审核。
    # 文件位置均相对 Agent 项目根目录；已选原仓库始终只读。
    workspace_path: str | None
    diff_path: str | None
    diff: str | None
    file_hashes: list[FileHash]
    code_summary: str | None
    code_review_status: CodeReviewStatus
    code_review_feedback: str
    code_revision_count: int
    # 跨节点诊断与追加式执行轨迹。
    error: str | None
    events: Annotated[list[str], add]


def _empty_state() -> AgentState:
    """只在这里列出一次状态默认值，供正式任务与聊天会话共用。

    聊天会话先处于 collecting，且 request 为空；显式 start/API 会在
    `initial_state` 中改为 ready。这样不必先伪造一个“正式需求”再清空。
    """

    return {
        "task_id": uuid4().hex,
        "intake_status": "collecting",
        "intake_reply": None,
        "conversation": [],
        "repo_root": None,
        "discovery_tool": None,
        "discovery_query": None,
        "discovery_path": None,
        "discovery_observations": [],
        "discovery_lookup_count": 0,
        "request": "",
        "requested_mode": None,
        "mode": None,
        "plan": [],
        "next_step": "intake",
        "supervisor_steps": 0,
        "research_rounds": 0,
        "sources": [],
        "research_question": None,
        "research_clarified": False,
        "research": None,
        "research_report": None,
        "draft_plan": None,
        "final_plan": None,
        "acceptance_criteria": [],
        "open_questions": [],
        "review_status": "not_ready",
        "review_feedback": "",
        "revision_count": 0,
        "workflow_status": "running",
        "research_path": None,
        "proposal_path": None,
        "repo_context": None,
        "change_plan": None,
        "change_plan_path": None,
        "workspace_path": None,
        "diff_path": None,
        "diff": None,
        "file_hashes": [],
        "code_summary": None,
        "code_review_status": "not_ready",
        "code_review_feedback": "",
        "code_revision_count": 0,
        "error": None,
        "events": [],
    }


def initial_state(
    request: str, requested_mode: TaskMode | None = None, repo_root: str | None = None
) -> AgentState:
    """显式 start/API 入口：完整需求进入 Intake，仅作模式确认，不再澄清。"""

    request = request.strip()
    if not request or len(request) > 1000:
        raise ValueError("产品需求长度必须在 1～1000 字符之间")
    if requested_mode not in (None, "proposal_only", "code_change"):
        raise ValueError("未知任务模式")
    state = _empty_state()
    state["request"] = request
    state["requested_mode"] = requested_mode
    state["repo_root"] = repo_root
    state["intake_status"] = "ready"
    return state


def initial_chat_state(message: str) -> AgentState:
    """从第一句聊天创建图内会话，而不是立即把这句话当成正式需求。

    内部 ID 用作 SQLite checkpoint 的 thread_id；在需求明确前，
    `request` 保持空字符串，Researcher 等后续节点不会运行。
    """

    message = message.strip()
    if not message or len(message) > 1000:
        raise ValueError("单条聊天消息长度必须在 1～1000 字符之间")
    state = _empty_state()
    state["conversation"] = [{"role": "user", "text": message}]
    return state
