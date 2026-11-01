"""终端聊天界面：显示状态、接收用户消息并恢复对应的图线程。"""

import re
from dataclasses import dataclass

from langgraph.types import Command

from app.config import PROJECT_ROOT, task_config
from app.state import initial_chat_state, initial_state


INTRO = (
    "你好，我是产品调研与研发交付 Agent。你可以直接用日常语言描述需求，"
    "最好说清楚当前问题、目标用户、希望实现的行为、限制条件，"
    "以及是否需要修改代码；暂时不确定的部分也可以直接说明。\n"
    "我会按需查授权项目目录、已选仓库文件、本地知识库或公开资料；也可以先回答问题。\n"
    "如果你明确要求修改代码，我会先确认仓库，在你批准产品方案后"
    "生成隔离区候选改动和 diff，再请你审核。当前不会自动运行候选代码、提交或发布。\n"
    "例如：‘调研反馈列表优先级筛选，给出产品方案，不改代码’；"
    "或‘给演示项目增加优先级筛选，需要修改代码’。\n"
    "想接着聊旧任务时，输入‘继续上次对话’或 /recent 查看待继续会话，"
    "再用 /resume 会话ID 返回原会话。"
)


@dataclass(frozen=True)
class ResumeOption:
    """供终端展示的一条待继续会话；不把完整报告或对话历史打印出来。"""

    task_id: str
    stage: str
    preview: str


@dataclass
class ChatSession:
    """终端当前所在会话；SQLite checkpoint 才是持久化的任务状态。"""

    active_id: str | None = None
    retry_request: str | None = None
    retry_repo_root: str | None = None


def _recent_resumable(graph, limit: int = 5) -> list[ResumeOption]:
    """从 checkpoint 找最近暂停在用户输入处的会话，不创建新任务。

    saver 的 list 按最新 checkpoint 先返回；一个会话有多个 checkpoint，
    所以先按 thread_id 去重，再用图的最新状态判断是否真的在 interrupt。
    最多扫描 500 条记录，避免历史很多时让聊天启动变慢。
    """

    thread_ids: list[str] = []
    seen: set[str] = set()
    for checkpoint in graph.checkpointer.list(None, limit=500):
        task_id = checkpoint.config["configurable"]["thread_id"]
        if task_id not in seen:
            seen.add(task_id)
            thread_ids.append(task_id)

    options: list[ResumeOption] = []
    stages = {
        "await_user": "需求澄清", "await_research_input": "调研补充",
        "review": "方案审核", "diff_review": "代码 diff 审核",
    }
    for task_id in thread_ids:
        snapshot = graph.get_state(task_config(task_id))
        if snapshot.values.get("task_id") != task_id or not _waiting_input(snapshot):
            continue
        stage = stages[snapshot.next[0]]
        conversation = snapshot.values.get("conversation") or []
        first_message = conversation[0]["text"] if conversation else ""
        summary = snapshot.values.get("request") or first_message or "未命名会话"
        preview = " ".join(summary.split())[:60]
        options.append(ResumeOption(task_id, stage, preview))
        if len(options) >= limit:
            break
    return options


def _show_recent_sessions(graph) -> None:
    """只展示可继续的会话，并给出显式恢复命令，避免猜错历史任务。"""

    options = _recent_resumable(graph)
    if not options:
        print("Agent：没有待继续的会话。可以直接告诉我新的需求。")
        return
    print("Agent：找到以下待继续会话：")
    for index, option in enumerate(options, 1):
        print(f"{index}. [{option.stage}] {option.preview}（ID：{option.task_id}）")
    print("Agent：输入 /resume 会话ID 恢复；不确定时先看摘要，不会自动替你选择。")


def _wants_previous_chat(message: str) -> bool:
    """只匹配独立的恢复请求，避免误拦截“开发恢复会话功能”等需求。"""

    short = message.strip(" 。！？!?.,，")
    return re.fullmatch(
        r"(?:请)?(?:我想|我要|我们|帮我)?(?:继续|恢复|接着)"
        r"(?:上次|之前)(?:的|那个)?(?:对话|会话|聊天|任务|需求|聊)?(?:吧)?",
        short,
    ) is not None


def _print_snapshot(snapshot, task_id: str) -> None:
    """只负责展示状态；澄清、产品审核、diff 审核各有不同输出。"""

    values = snapshot.values
    if values.get("task_id") != task_id:
        raise ValueError("找不到这个任务")

    waiting = _waiting_input(snapshot)
    if waiting and snapshot.next == ("await_user",):
        print("Agent：" + values["intake_reply"])
        stage = "需求尚未建立" if not values.get("request") else "等待补充信息"
        print(f"会话 ID：{task_id}（{stage}；重启后可用 /resume 会话ID 继续）")
        return

    if waiting and snapshot.next == ("await_research_input",):
        print("Agent：" + values["research_question"])
        print(f"会话 ID：{task_id}（等待调研补充；重启后可用 /resume 会话ID 继续）")
        return

    if values.get("conversation") and values.get("intake_status") == "ready":
        # 接待阶段可能刚回答了用户的项目问题；不能只显示整理后的任务而吞掉回复。
        if values.get("intake_reply") and values.get("review_status") == "pending":
            print("Agent：" + values["intake_reply"])
        print("\n接待环节整理的需求：", values["request"])

    if waiting and snapshot.next == ("review",):
        print("\n===== 调研报告 =====\n")
        print(values["research_report"])
        print("\n===== 产品方案（待审核） =====\n")
        print(values["draft_plan"])
        print(f"\n模式：{values['mode']}；已退回次数：{values['revision_count']}")
        print(f"任务暂停，可用 task_id 恢复：{task_id}")
        return

    if waiting and snapshot.next == ("diff_review",):
        print("\n===== 代码变更 diff（仅隔离工作区，待审核） =====\n")
        print(values["diff"])
        print("修改摘要：", values.get("code_summary"))
        print("隔离工作区：", PROJECT_ROOT / values["workspace_path"])
        print("diff 文件：", PROJECT_ROOT / values["diff_path"])
        print(f"\n已退回次数：{values.get('code_revision_count', 0)}")
        print(f"任务暂停，可用 task_id 恢复代码审核：{task_id}")
        return

    print("\n审核状态：", values["review_status"])
    print("工作流状态：", values["workflow_status"])
    if values["error"]:
        print("停止原因：", values["error"])
    if values["research_path"]:
        print("调研报告：", PROJECT_ROOT / values["research_path"])
    if values["proposal_path"]:
        print("产品方案：", PROJECT_ROOT / values["proposal_path"])
    if values["change_plan_path"]:
        print("文件级变更计划：", PROJECT_ROOT / values["change_plan_path"])
        print("\n===== 文件级变更计划（只读分析） =====\n")
        print(values["change_plan"])
    if values.get("diff_path"):
        print("代码 diff：", PROJECT_ROOT / values["diff_path"])
        print("隔离工作区：", PROJECT_ROOT / values["workspace_path"])
    print("执行路径：")
    for event in values["events"]:
        print("-", event)


def _waiting_input(snapshot) -> bool:
    """只有图确实停在 interrupt 上，聊天输入才可以恢复对应节点。"""

    return snapshot.next in {
        ("await_user",), ("await_research_input",), ("review",), ("diff_review",),
    } and any(
        task.interrupts for task in snapshot.tasks
    )


def _resume_pending_node(graph, task_id: str, snapshot):
    """重启后推进尚未执行的普通节点；人工 interrupt 必须等待用户决定。"""

    ordinary_nodes = {
        "intake", "discovery_tool", "supervisor", "researcher", "product_planner",
        "save", "repo_inspector", "change_planner", "save_change_plan",
        "workspace_prepare", "coder", "finish", "finish_code",
    }
    if (
        len(snapshot.next) == 1 and snapshot.next[0] in ordinary_nodes
        and snapshot.values.get("workflow_status") not in {"blocked", "completed", "terminated", "diff_approved"}
    ):
        graph.invoke(None, config=task_config(task_id), durability="sync")
        return graph.get_state(task_config(task_id))
    return snapshot


def _blocked_request(snapshot) -> str | None:
    """只有失败的正式需求才允许 /retry；普通结束和澄清中都不重开任务。"""

    if snapshot.values.get("workflow_status") == "blocked":
        return snapshot.values.get("request") or None
    return None


def _review_reply(message: str) -> dict[str, str] | None:
    """只把明确的批准、退回或终止指令转换成 LangGraph 审核决定。

    普通的“好的”“继续”等聊天内容不等于批准；退回必须带具体意见。
    """

    if message in {"批准", "/approve"}:
        return {"action": "approve", "feedback": ""}
    if message in {"终止", "/terminate"}:
        return {"action": "terminate", "feedback": ""}
    for prefix in ("退回 ", "退回：", "退回:", "退回，", "/reject "):
        if message.startswith(prefix) and message[len(prefix):].strip():
            return {"action": "reject", "feedback": message[len(prefix):].strip()}
    return None


def _show_chat_result(snapshot, task_id: str) -> bool:
    """展示一次图运行的结果；返回是否还在等待用户输入。"""

    _print_snapshot(snapshot, task_id)
    if _waiting_input(snapshot):
        if snapshot.next not in {("await_user",), ("await_research_input",)}:
            stage = "产品方案" if snapshot.next == ("review",) else "代码 diff"
            print(f"Agent：请审核上面的{stage}。输入‘批准’、‘退回 修改意见’或‘终止’。")
        return True
    if snapshot.next:
        print(f"Agent：任务停在 {snapshot.next}，请用 /resume 会话ID 继续。")
    elif snapshot.values.get("workflow_status") == "blocked":
        print("Agent：本次任务因上面的错误被阻断。修复后可输入 /retry，按同一需求创建新任务。")
    else:
        print("Agent：本次任务已结束。可以继续输入一个新需求。")
    return False


def _remember_result(graph, session: ChatSession, task_id: str) -> None:
    """只读一次最新 checkpoint，更新终端会话并打印结果。"""

    snapshot = graph.get_state(task_config(task_id))
    session.retry_request = _blocked_request(snapshot)
    session.retry_repo_root = snapshot.values.get("repo_root") if session.retry_request else None
    session.active_id = task_id if _show_chat_result(snapshot, task_id) else None


def _retry_chat_task(graph, session: ChatSession) -> None:
    """用同一需求创建新任务；若仍被阻断，保留再次 /retry 的机会。"""

    if session.retry_request is None:
        raise ValueError("当前没有可重试的正式需求")
    state = initial_state(session.retry_request, repo_root=session.retry_repo_root)
    task_id = state["task_id"]
    print(f"Agent：需求已明确，任务 ID：{task_id}。开始调研并整理方案……")
    graph.invoke(state, config=task_config(task_id), durability="sync")
    # 统一走结果处理函数：成功时清空重试上下文，再次 blocked 时保留它。
    _remember_result(graph, session, task_id)


def _handle_terminal_command(graph, session: ChatSession, message: str) -> bool:
    """处理终端控制命令；返回 True 表示已处理，不作为用户消息送入图。"""

    if message in {"/help", "帮助"}:
        print(INTRO)
        print("Agent：审核时输入‘批准’、‘退回 修改意见’或‘终止’；"
              "其他命令：/recent、/status、/resume 会话ID、/retry、/new、/exit。")
        return True
    # 切换旧 thread_id 必须由入口处理，不能在新图的 Intake 中完成。
    if message == "/recent" or _wants_previous_chat(message):
        _show_recent_sessions(graph)
        return True
    if message == "/status":
        if session.active_id:
            snapshot = graph.get_state(task_config(session.active_id))
            _show_chat_result(snapshot, session.active_id)
        else:
            print("Agent：当前没有进行中的对话任务，可以直接输入需求。")
        return True
    if message == "/new":
        if session.active_id:
            print(f"Agent：已离开会话 {session.active_id}；它没有被终止，可用 /resume ID 返回。")
        session.active_id = None
        session.retry_request = None
        session.retry_repo_root = None
        print("Agent：请描述新需求。")
        return True
    if message == "/retry":
        if session.active_id:
            print("Agent：当前任务仍在等待你的输入，不能重试。请先回复，或输入 /new。")
        elif session.retry_request is None:
            print("Agent：没有可重试的需求，请先描述你想做什么。")
        else:
            _retry_chat_task(graph, session)
        return True
    if message.startswith("/resume "):
        task_id = message[len("/resume "):].strip()
        if not re.fullmatch(r"[0-9a-f]{32}", task_id):
            print("Agent：任务 ID 应为 32 位小写十六进制字符。")
            return True
        snapshot = graph.get_state(task_config(task_id))
        if snapshot.values.get("task_id") != task_id:
            print("Agent：找不到这个任务。")
            return True
        snapshot = _resume_pending_node(graph, task_id, snapshot)
        session.retry_request = _blocked_request(snapshot)
        session.retry_repo_root = snapshot.values.get("repo_root") if session.retry_request else None
        session.active_id = task_id if _show_chat_result(snapshot, task_id) else None
        return True
    if message.startswith("/"):
        print("Agent：未知命令。输入 /help 查看用法，或直接描述需求。")
        return True
    return False


def _handle_graph_input(graph, session: ChatSession, message: str) -> None:
    """把用户消息交给 LangGraph：恢复当前 interrupt，或创建新的图会话。"""

    if len(message) > 1000:
        print("Agent：这条消息较长，请先用不超过 1000 字概括核心需求。")
        return
    if session.active_id:
        snapshot = graph.get_state(task_config(session.active_id))
        if not _waiting_input(snapshot):
            print("Agent：这项任务当前不在等待输入；请查看状态后再继续。")
            _show_chat_result(snapshot, session.active_id)
            session.active_id = None
            return
        if snapshot.next in {("await_user",), ("await_research_input",)}:
            # 两类澄清都是普通文本，不是审核决定或另一项新需求。
            response = {"message": message}
            if snapshot.next == ("await_research_input",):
                print("Agent：收到补充，我继续核对资料并整理方案……")
            else:
                print("Agent：收到，我继续梳理需求；如果已经明确，就开始调研……")
        else:
            response = _review_reply(message)
            if response is None:
                print("Agent：当前任务正在等待审核。请输入‘批准’、‘退回 修改意见’或‘终止’；"
                      "也可以输入 /new 开始另一项任务。")
                return
        graph.invoke(Command(resume=response), config=task_config(session.active_id), durability="sync")
        _remember_result(graph, session, session.active_id)
        return

    # 新会话先进入图内 Intake；只有模型认为需求明确后才会开始调研。
    state = initial_chat_state(message)
    task_id = state["task_id"]
    print("Agent：我先理解一下你想做什么……")
    graph.invoke(state, config=task_config(task_id), durability="sync")
    _remember_result(graph, session, task_id)


def chat(graph) -> None:
    """终端循环只做三件事：读取输入、处理终端命令、把消息交给图。"""

    session = ChatSession()
    print("Agent：已进入对话。可以直接描述需求；输入 /help 查看操作。")
    while True:
        try:
            message = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent：对话已退出；已有任务的 checkpoint 仍会保留。")
            return
        if not message:
            continue
        # 退出命令直接结束外层循环；其他终端命令交给下方的命令处理函数。
        if message in {"/exit", "退出"}:
            if session.active_id:
                print(f"Agent：当前会话/任务 ID：{session.active_id}；稍后可用 /resume ID 继续。")
            print("Agent：对话已退出；已有任务的 checkpoint 仍会保留。")
            return
        if not _handle_terminal_command(graph, session, message):
            _handle_graph_input(graph, session, message)
