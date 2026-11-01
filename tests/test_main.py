"""命令行输入转换测试：普通聊天不能被误当成人工批准。"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver

from app import chat_cli, main
from app import graph as agent_graph
from app.agents import IntakeDecision, ModeDecision, SupervisorDecision
from app.chat_cli import _blocked_request, _recent_resumable, _review_reply, _wants_previous_chat
from app.config import task_config


class MainInputTests(unittest.TestCase):
    """不连接真实模型或知识库；入口测试使用临时 SQLite 或内存 checkpoint。"""

    def test_main_only_enters_chat(self):
        """不再走旧 CLI 子命令；用测试指定的临时数据库构建图。"""

        with tempfile.TemporaryDirectory() as directory, patch.object(
            main, "DB_PATH", Path(directory) / "checkpoints.sqlite"
        ), patch.object(main, "chat") as chat, patch.object(main.sys, "argv", ["app.main"]):
            main.main()
        chat.assert_called_once()

    def test_positional_arguments_are_rejected(self):
        """入口只支持交互式聊天；任何位置参数都不能被悄悄忽略。"""

        with patch.object(main.sys, "argv", ["app.main", "start", "需求"]):
            with self.assertRaises(SystemExit) as error:
                main.main()
        self.assertIn("只支持交互式聊天", str(error.exception))

    def test_chat_startup_only_shows_short_prompt(self):
        """启动时不主动列历史；问候和恢复意图留给后续输入处理。"""

        output = StringIO()
        with patch("builtins.input", side_effect=["/exit"]), patch.object(
            chat_cli, "_recent_resumable", side_effect=AssertionError("启动时不应查旧会话")
        ), redirect_stdout(output):
            chat_cli.chat(object())
        self.assertEqual(
            output.getvalue().splitlines()[0],
            "Agent：已进入对话。可以直接描述需求；输入 /help 查看操作。",
        )

    def test_review_requires_explicit_decision(self):
        """“好的”不能批准方案；退回必须提供实际修改意见。"""

        self.assertIsNone(_review_reply("好的"))
        self.assertIsNone(_review_reply("退回"))
        self.assertEqual(_review_reply("批准"), {"action": "approve", "feedback": ""})
        self.assertEqual(
            _review_reply("退回 请补充验收条件"),
            {"action": "reject", "feedback": "请补充验收条件"},
        )

    def test_only_blocked_formal_request_can_retry(self):
        """已完成任务和尚在澄清的聊天不应被 /retry 重新提交。"""

        blocked = SimpleNamespace(values={"workflow_status": "blocked", "request": "筛选反馈"})
        completed = SimpleNamespace(values={"workflow_status": "completed", "request": "筛选反馈"})
        clarifying = SimpleNamespace(values={"workflow_status": "blocked", "request": ""})
        self.assertEqual(_blocked_request(blocked), "筛选反馈")
        self.assertIsNone(_blocked_request(completed))
        self.assertIsNone(_blocked_request(clarifying))

    def test_retry_can_be_repeated_if_dependency_is_still_down(self):
        """第一次重试仍 blocked 时保留需求；第二次成功后才清空。"""

        session = chat_cli.ChatSession(
            retry_request="调研反馈筛选", retry_repo_root="C:/demo"
        )
        blocked = SimpleNamespace(values={
            "workflow_status": "blocked", "request": "调研反馈筛选", "repo_root": "C:/demo",
        })
        completed = SimpleNamespace(values={
            "workflow_status": "completed", "request": "调研反馈筛选", "repo_root": "C:/demo",
        })
        graph = SimpleNamespace(invoke=Mock(), get_state=Mock(side_effect=[blocked, completed]))
        with patch.object(chat_cli, "_show_chat_result", return_value=False), redirect_stdout(StringIO()):
            self.assertTrue(chat_cli._handle_terminal_command(graph, session, "/retry"))
            self.assertEqual(session.retry_request, "调研反馈筛选")
            self.assertEqual(session.retry_repo_root, "C:/demo")
            self.assertTrue(chat_cli._handle_terminal_command(graph, session, "/retry"))
        self.assertIsNone(session.retry_request)
        self.assertIsNone(session.retry_repo_root)
        self.assertEqual(graph.invoke.call_count, 2)

    def test_recent_sessions_only_lists_paused_threads(self):
        """同一 thread 的多个 checkpoint 只显示一次；已结束任务不显示。"""

        waiting_id, done_id = "a" * 32, "b" * 32

        def checkpoint(task_id):
            return SimpleNamespace(config={"configurable": {"thread_id": task_id}})

        paused = SimpleNamespace(
            values={"task_id": waiting_id, "request": "", "conversation": [
                {"role": "user", "text": "我想做一个需求"}
            ]},
            next=("await_user",),
            tasks=[SimpleNamespace(interrupts=[object()])],
        )
        finished = SimpleNamespace(
            values={"task_id": done_id, "request": "已完成任务"}, next=(), tasks=[],
        )
        graph = SimpleNamespace(
            checkpointer=SimpleNamespace(list=lambda *args, **kwargs: iter([
                checkpoint(waiting_id), checkpoint(waiting_id), checkpoint(done_id)
            ])),
            get_state=lambda config: {
                waiting_id: paused, done_id: finished
            }[config["configurable"]["thread_id"]],
        )
        options = _recent_resumable(graph)
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].task_id, waiting_id)
        self.assertEqual(options[0].stage, "需求澄清")
        self.assertEqual(options[0].preview, "我想做一个需求")

    def test_resume_intent_does_not_create_new_task(self):
        """“继续上次对话”先显示恢复入口，不送进新任务的 Intake。"""

        self.assertTrue(_wants_previous_chat("我想继续上次对话"))
        self.assertFalse(_wants_previous_chat("实现一个继续上次对话的功能，并修改演示仓库代码"))
        option = chat_cli.ResumeOption("a" * 32, "需求澄清", "我想做一个需求")
        output = StringIO()
        with patch.object(chat_cli, "_recent_resumable", return_value=[option]), patch(
            "builtins.input", side_effect=["继续上次对话", "/exit"]
        ), patch.object(chat_cli, "initial_chat_state") as create_state, redirect_stdout(output):
            chat_cli.chat(object())
        create_state.assert_not_called()
        self.assertIn("/resume 会话ID", output.getvalue())

    def test_resume_advances_ordinary_node_but_not_human_interrupt(self):
        """重启若停在 save 等普通节点，/resume 应真正推进；审核仍等人操作。"""

        task_id = "a" * 32
        pending = SimpleNamespace(next=("save",), values={"workflow_status": "running"}, tasks=[])
        finished = SimpleNamespace(next=(), values={"workflow_status": "completed"}, tasks=[])
        graph = SimpleNamespace(invoke=Mock(), get_state=Mock(return_value=finished))
        self.assertIs(chat_cli._resume_pending_node(graph, task_id, pending), finished)
        graph.invoke.assert_called_once_with(
            None, config=task_config(task_id), durability="sync",
        )

        waiting = SimpleNamespace(
            next=("review",), values={"workflow_status": "running"},
            tasks=[SimpleNamespace(interrupts=[object()])],
        )
        graph.invoke.reset_mock()
        self.assertIs(chat_cli._resume_pending_node(graph, task_id, waiting), waiting)
        graph.invoke.assert_not_called()

    def test_research_question_is_shown_as_normal_chat_input(self):
        """调研补充是用户消息，不应显示成产品方案的批准按钮。"""

        task_id = "a" * 32
        paused = SimpleNamespace(
            next=("await_research_input",),
            values={"task_id": task_id, "research_question": "请补充内部规则。"},
            tasks=[SimpleNamespace(interrupts=[object()])],
        )
        output = StringIO()
        with redirect_stdout(output):
            chat_cli._print_snapshot(paused, task_id)
        self.assertTrue(chat_cli._waiting_input(paused))
        self.assertIn("请补充内部规则", output.getvalue())
        self.assertIn("等待调研补充", output.getvalue())

    def test_chat_clarification_stays_in_one_thread(self):
        """终端两轮输入仍恢复同一图会话，不会把补充说明建成第二项任务。"""

        decisions = [
            IntakeDecision(ready=False, reply="你想解决什么问题？"),
            IntakeDecision(ready=True, reply="开始调研。", request="反馈列表增加筛选，仅写方案"),
        ]
        research = lambda state: {
            "sources": [], "research": "证据不足", "research_report": "# 调研报告\n",
            "next_step": "assess_research",
            "events": ["researcher：模拟研究"],
        }
        planner = lambda state: {
            "draft_plan": "# 产品方案\n", "acceptance_criteria": ["按条件筛选"],
            "open_questions": ["背景待核验"], "review_status": "pending",
            "next_step": "assess_plan",
            "events": ["product_planner：模拟方案"],
        }
        with patch.object(agent_graph, "intake_requirement", side_effect=decisions) as intake, patch.object(
            agent_graph, "decide_mode", return_value=ModeDecision(mode="proposal_only", reason="只写方案")
        ), patch.object(
            agent_graph, "decide_next", side_effect=lambda _state, allowed: SupervisorDecision(
                next_agent=allowed[0], reason="终端测试派工"
            )
        ), patch.object(agent_graph, "researcher_node", research), patch.object(
            agent_graph, "product_planner_node", planner
        ), patch("builtins.input", side_effect=[
            "我想做一个需求", "反馈列表增加筛选，仅写方案", "/exit"
        ]), redirect_stdout(StringIO()):
            graph = agent_graph.build_graph(InMemorySaver())
            chat_cli.chat(graph)

        thread_ids = {item.config["configurable"]["thread_id"] for item in graph.checkpointer.list(None)}
        self.assertEqual(len(thread_ids), 1)
        snapshot = graph.get_state(task_config(thread_ids.pop()))
        self.assertEqual(snapshot.next, ("review",))
        self.assertEqual(snapshot.values["request"], "反馈列表增加筛选，仅写方案")
        self.assertEqual(intake.call_count, 2)


if __name__ == "__main__":
    unittest.main()
