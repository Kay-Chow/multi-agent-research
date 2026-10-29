"""阶段 1 图测试：两种模式、人工退回/批准、SQLite 重启恢复。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.product import graph as product_graph
from app.product.agents import ModeDecision
from app.product.state import initial_product_state


class ProductGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_patch = patch.object(
            product_graph, "OUTPUT_DIR", Path(self.temp_dir.name) / "outputs"
        )
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)

        self.research_patch = patch.object(
            product_graph,
            "collect_research",
            return_value=(
                [{
                    "source_id": 1,
                    "title": "参考",
                    "url": "https://example.com/docs",
                    "snippet": "摘要",
                    "retrieved_at": "2026-10-29T00:00:00+00:00",
                }],
                [{"claim": "一条研究要点", "source_id": 1}],
                "# 调研报告\n\n- 一条研究要点 [1]\n",
            ),
        )
        self.research_patch.start()
        self.addCleanup(self.research_patch.stop)

        def fake_plan(state):
            suffix = state["review_feedback"] if state["review_status"] == "rejected" else "初稿"
            return f"# 产品方案\n\n{suffix}\n", ["传 high 时只返回 high 反馈"], []

        self.plan_patch = patch.object(
            product_graph, "create_product_plan", side_effect=fake_plan
        )
        self.plan_patch.start()
        self.addCleanup(self.plan_patch.stop)

    @staticmethod
    def config(task_id):
        return {"configurable": {"thread_id": task_id}}

    def test_proposal_only_approve_saves_two_documents(self) -> None:
        with patch.object(product_graph, "classify_task", return_value=ModeDecision(
            mode="proposal_only", reason="只要方案"
        )):
            graph = product_graph.build_product_graph(InMemorySaver())
            state = initial_product_state("只写反馈筛选方案，不要代码")
            graph.invoke(state, config=self.config(state["task_id"]))

        waiting = graph.get_state(self.config(state["task_id"]))
        self.assertEqual(waiting.next, ("review",))
        self.assertEqual(waiting.values["mode"], "proposal_only")
        self.assertFalse((Path(self.temp_dir.name) / "outputs").exists())

        result = graph.invoke(
            Command(resume={"action": "approve"}),
            config=self.config(state["task_id"]),
        )
        self.assertEqual(result["workflow_status"], "completed")
        self.assertEqual(result["review_status"], "approved")
        output_dir = Path(self.temp_dir.name) / "outputs" / state["task_id"]
        self.assertTrue((output_dir / "research.md").exists())
        self.assertTrue((output_dir / "proposal.md").exists())

    def test_code_change_stops_at_repo_handoff_without_touching_repo(self) -> None:
        graph = product_graph.build_product_graph(InMemorySaver())
        state = initial_product_state("实现反馈列表优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state["task_id"]))
        result = graph.invoke(
            Command(resume={"action": "approve"}),
            config=self.config(state["task_id"]),
        )
        self.assertEqual(result["mode"], "code_change")
        self.assertEqual(result["workflow_status"], "ready_for_repo")
        self.assertIn("等待阶段 2", result["events"][-1])

    def test_reject_returns_to_planner_and_pauses_again(self) -> None:
        graph = product_graph.build_product_graph(InMemorySaver())
        state = initial_product_state("只要方案", "proposal_only")
        graph.invoke(state, config=self.config(state["task_id"]))
        graph.invoke(
            Command(resume={"action": "reject", "feedback": "补充兼容性要求"}),
            config=self.config(state["task_id"]),
        )
        snapshot = graph.get_state(self.config(state["task_id"]))
        self.assertEqual(snapshot.next, ("review",))
        self.assertEqual(snapshot.values["revision_count"], 1)
        self.assertIn("补充兼容性要求", snapshot.values["product_plan"])
        self.assertFalse((Path(self.temp_dir.name) / "outputs").exists())

    def test_sqlite_reopen_then_approve(self) -> None:
        db_path = Path(self.temp_dir.name) / "product.sqlite"
        state = initial_product_state("只写方案", "proposal_only")
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = product_graph.build_product_graph(saver)
            graph.invoke(state, config=self.config(state["task_id"]), durability="sync")

        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = product_graph.build_product_graph(saver)
            self.assertEqual(graph.get_state(self.config(state["task_id"])).next, ("review",))
            result = graph.invoke(
                Command(resume={"action": "approve"}),
                config=self.config(state["task_id"]),
                durability="sync",
            )
        self.assertEqual(result["workflow_status"], "completed")


if __name__ == "__main__":
    unittest.main()
