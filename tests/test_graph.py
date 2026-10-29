"""图的确定性测试：用假模型和假 Worker，不调用外部 API。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph import MAX_REVISIONS
from app.graph import MAX_STEPS, build_graph
from app.state import initial_state


def fake_decide_next(state):
    """模拟 Supervisor；仍让 graph.py 的业务约束负责校验路由。"""
    needs_code = (
        "需要代码" in state["task"]
        if not state["plan"]
        else state["needs_code"]
    )
    plan = (
        ["researcher", "coder", "writer"]
        if needs_code
        else ["researcher", "writer"]
    )

    if state["research"] is None:
        next_agent = "researcher"
    elif needs_code and state["code"] is None:
        next_agent = "coder"
    elif state["draft_report"] is None:
        next_agent = "writer"
    else:
        next_agent = "finish"

    return SimpleNamespace(
        needs_code=needs_code,
        plan=plan,
        next_agent=next_agent,
        instruction="执行测试任务",
        reason="测试用固定决策",
    )


def fake_researcher(state):
    return {
        "research": "模拟研究结论",
        "sources": [],
        "events": ["researcher：完成模拟研究"],
    }


def fake_coder(state):
    return {
        "code": "print('demo')",
        "code_explanation": "模拟代码",
        "code_syntax_ok": True,
        "events": ["coder：完成模拟代码"],
    }


def fake_writer(state):
    return {
        "draft_report": "# 测试报告",
        "review_status": "pending",
        "events": ["writer：完成模拟报告"],
    }


class GraphTests(unittest.TestCase):
    def setUp(self):
        # 每个测试使用独立 checkpoint 和临时输出目录。
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name) / "outputs"

        patches = [
            patch("app.graph.OUTPUT_DIR", self.output_dir),
            patch("app.graph.decide_next", new=fake_decide_next),
            patch("app.graph.researcher_node", new=fake_researcher),
            patch("app.graph.coder_node", new=fake_coder),
            patch("app.graph.writer_node", new=fake_writer),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

        self.graph = build_graph(InMemorySaver())

    @staticmethod
    def config_for(state):
        return {"configurable": {"thread_id": state["task_id"]}}

    def test_no_code_approve_then_save(self):
        state = initial_state("只写简报")
        config = self.config_for(state)
        report_file = self.output_dir / f"{state['task_id']}.md"

        paused = self.graph.invoke(state, config=config)
        self.assertIn("__interrupt__", paused)
        self.assertIsNone(paused["code"])
        self.assertFalse(report_file.exists())  # 审核前绝不保存

        finished = self.graph.invoke(
            Command(resume={"action": "approve"}),
            config=config,
        )
        self.assertEqual(finished["review_status"], "approved")
        self.assertEqual(report_file.read_text(encoding="utf-8"), "# 测试报告\n")

    def test_code_task_can_terminate_without_saving(self):
        state = initial_state("需要代码示例")
        config = self.config_for(state)

        paused = self.graph.invoke(state, config=config)
        self.assertIn("__interrupt__", paused)
        self.assertEqual(paused["code"], "print('demo')")

        finished = self.graph.invoke(
            Command(resume={"action": "terminate"}),
            config=config,
        )
        self.assertEqual(finished["review_status"], "terminated")
        self.assertFalse((self.output_dir / f"{state['task_id']}.md").exists())

    def test_reject_returns_to_writer_and_pauses_again(self):
        state = initial_state("只写简报")
        config = self.config_for(state)
        self.graph.invoke(state, config=config)

        paused_again = self.graph.invoke(
            Command(resume={
                "action": "reject",
                "feedback": "不要夸大结论",
            }),
            config=config,
        )
        self.assertIn("__interrupt__", paused_again)
        self.assertEqual(paused_again["revision_count"], 1)
        self.assertEqual(paused_again["review_status"], "pending")

    def test_invalid_supervisor_route_is_rejected(self):
        state = initial_state("只写简报")
        config = self.config_for(state)

        # 首轮还没研究，Supervisor 却要求直接去 Writer。
        def invalid_decision(_state):
            return SimpleNamespace(
                needs_code=False,
                plan=["researcher", "writer"],
                next_agent="writer",
                instruction="跳过研究",
                reason="模拟错误路由",
            )

        with patch("app.graph.decide_next", new=invalid_decision):
            with self.assertRaisesRegex(ValueError, "路由错误"):
                self.graph.invoke(state, config=config)

    def test_step_limit_stops_graph(self):
        state = initial_state("只写简报")
        state["step_count"] = MAX_STEPS

        result = self.graph.invoke(state, config=self.config_for(state))
        self.assertEqual(result["next_agent"], "finish")
        self.assertIsNone(result["draft_report"])

    def test_sqlite_reopen_and_resume(self):
        """关闭数据库连接再打开，仍能从人工审核处继续。"""
        state = initial_state("只写简报")
        config = self.config_for(state)
        db_path = Path(self.temp_dir.name) / "checkpoints.sqlite"
        report_file = self.output_dir / f"{state['task_id']}.md"

        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = build_graph(saver)
            paused = graph.invoke(state, config=config, durability="sync")
            self.assertIn("__interrupt__", paused)

        self.assertFalse(report_file.exists())

        # 新连接、新图，相当于应用重新启动。
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = build_graph(saver)
            snapshot = graph.get_state(config)
            self.assertEqual(snapshot.next, ("review",))
            self.assertEqual(snapshot.values["task_id"], state["task_id"])

            finished = graph.invoke(
                Command(resume={"action": "approve"}),
                config=config,
                durability="sync",
            )

        self.assertEqual(finished["review_status"], "approved")
        self.assertTrue(report_file.exists())

    def test_revision_limit_terminates(self):
        """超过允许的退回次数后结束，不再调用 Writer。"""
        state = initial_state("只写简报")
        config = self.config_for(state)
        self.graph.invoke(state, config=config)

        for count in range(1, MAX_REVISIONS + 1):
            paused = self.graph.invoke(
                Command(resume={
                    "action": "reject",
                    "feedback": f"第 {count} 次修改意见",
                }),
                config=config,
            )
            self.assertIn("__interrupt__", paused)
            self.assertEqual(paused["revision_count"], count)

        finished = self.graph.invoke(
            Command(resume={
                "action": "reject",
                "feedback": "仍不批准",
            }),
            config=config,
        )
        self.assertEqual(finished["review_status"], "terminated")
        self.assertFalse((self.output_dir / f"{state['task_id']}.md").exists())


if __name__ == "__main__":
    unittest.main()
