"""同一个 HTTP 入口创建、审核并查询产品任务。"""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import api
from app.services import artifact_store, candidate_delivery
from app.workflow import graph
from app.agents.coder import CodePatchDraft
from app.agents.supervisor import SupervisorDecision


class ApiTests(unittest.TestCase):
    """验证 HTTP 创建、资料补充及两个独立人工审核接口的状态流转。"""

    def test_create_review_and_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def fake_researcher(state):
                return {
                    "sources": [], "research": "证据不足",
                    "research_report": "# 调研报告\n\n证据不足。\n",
                    "next_step": "assess_research",
                    "events": ["researcher：模拟研究"],
                }

            def fake_planner(state):
                return {
                    "draft_plan": "# 产品方案\n\n增加优先级筛选。\n",
                    "acceptance_criteria": ["仅返回匹配优先级的反馈"],
                    "open_questions": ["外部依据待核验"],
                    "review_status": "pending",
                    "next_step": "assess_plan",
                    "events": ["product_planner：模拟方案"],
                }

            with patch.object(api, "DB_PATH", root / "checkpoint.sqlite"), patch.object(
                graph, "OUTPUT_DIR", root / "outputs"
            ), patch.object(graph, "researcher_node", fake_researcher), patch.object(
                graph, "product_planner_node", fake_planner
            ), patch.object(
                graph, "decide_next", side_effect=lambda _state, allowed: SupervisorDecision(
                    next_agent=allowed[0], reason="API 测试派工"
                )
            ):
                client = TestClient(api.app)
                created = client.post("/tasks", json={
                    "request": "只写反馈筛选方案", "mode": "proposal_only"
                })
                self.assertEqual(created.status_code, 201)
                body = created.json()
                self.assertTrue(body["waiting_for_review"])
                self.assertEqual(body["mode"], "proposal_only")
                self.assertEqual(client.post(f"/tasks/{body['task_id']}/develop").status_code, 404)

                missing_feedback = client.post(
                    f"/tasks/{body['task_id']}/review", json={"action": "reject"}
                )
                self.assertEqual(missing_feedback.status_code, 422)

                approved = client.post(
                    f"/tasks/{body['task_id']}/review", json={"action": "approve"}
                )
                self.assertEqual(approved.status_code, 200)
                self.assertEqual(approved.json()["workflow_status"], "completed")
                self.assertEqual(
                    client.get(f"/tasks/{body['task_id']}").json()["review_status"],
                    "approved",
                )
                self.assertTrue((root / "outputs" / body["task_id"] / "proposal.md").exists())

    def test_research_clarification_has_separate_input_endpoint(self):
        """HTTP 任务缺资料时返回问题；用户补充后继续到产品方案审核。"""

        def researcher(state):
            """用可重复的假研究结果替代真实知识库与模型调用。"""

            if not state["research_clarified"]:
                question = "请补充内部规则。"
                return {
                    "research_question": question,
                    "research": None,
                    "research_report": None,
                    "next_step": "await_research_input",
                    "events": ["researcher：等待用户补充"],
                }
            return {
                "research": "用户补充了规则 [1]",
                "research_report": "# 调研报告\n\n用户补充了规则 [1]\n",
                "next_step": "assess_research",
                "events": ["researcher：已整理用户补充"],
            }

        def planner(state):
            return {
                "draft_plan": "# 产品方案\n\n按用户补充的规则实施。\n",
                "acceptance_criteria": ["按规则筛选"],
                "open_questions": [],
                "review_status": "pending",
                "next_step": "assess_plan",
                "events": ["planner：完成方案"],
            }

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            api, "DB_PATH", Path(temp_dir) / "checkpoint.sqlite"
        ), patch.object(graph, "researcher_node", researcher), patch.object(
            graph, "product_planner_node", planner
        ), patch.object(
            graph, "decide_next", side_effect=lambda _state, allowed: SupervisorDecision(
                next_agent=allowed[0], reason="API 测试派工"
            )
        ):
            client = TestClient(api.app)
            created = client.post("/tasks", json={
                "request": "依据内部规则设计筛选", "mode": "proposal_only",
            })
            self.assertEqual(created.status_code, 201)
            task_id = created.json()["task_id"]
            self.assertTrue(created.json()["waiting_for_research_input"])
            self.assertEqual(created.json()["research_question"], "请补充内部规则。")
            self.assertEqual(client.post(
                f"/tasks/{task_id}/review", json={"action": "approve"}
            ).status_code, 409)
            self.assertEqual(client.post(
                f"/tasks/{task_id}/clarify", json={"message": "   "}
            ).status_code, 422)

            continued = client.post(
                f"/tasks/{task_id}/clarify", json={"message": "只允许 high 和 low"}
            )
            self.assertEqual(continued.status_code, 200)
            self.assertTrue(continued.json()["waiting_for_review"])
            self.assertFalse(continued.json()["waiting_for_research_input"])
            self.assertIn("用户补充了规则", continued.json()["research_report"])
            self.assertEqual(client.post(
                f"/tasks/{task_id}/clarify", json={"message": "再补一次"}
            ).status_code, 409)

    def test_code_task_has_separate_diff_review_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir, ExitStack() as stack:
            root = Path(temp_dir)
            repo_context = {"root": "C:/demo", "head": "a" * 40, "files": []}
            stack.enter_context(patch.object(api, "DB_PATH", root / "checkpoint.sqlite"))
            # HTTP 入口校验仓库路径；此处用假路径测试流程，不访问真实仓库。
            stack.enter_context(patch.object(api, "checked_root", return_value=Path("C:/demo")))
            stack.enter_context(patch.object(graph, "OUTPUT_DIR", root / "outputs"))
            stack.enter_context(patch.object(
                graph, "decide_next", side_effect=lambda _state, allowed: SupervisorDecision(
                    next_agent=allowed[0], reason="API 测试派工"
                ),
            ))
            stack.enter_context(patch.object(graph, "researcher_node", lambda state: {
                "sources": [], "research": "证据不足", "research_report": "# 调研报告\n",
                "next_step": "assess_research",
                "events": ["researcher：模拟研究"],
            }))
            stack.enter_context(patch.object(graph, "product_planner_node", lambda state: {
                "draft_plan": "# 产品方案\n", "acceptance_criteria": ["筛选反馈"],
                "open_questions": [], "review_status": "pending", "next_step": "assess_plan",
                "events": ["planner：模拟"],
            }))
            stack.enter_context(patch.object(
                graph, "inspect_repository", return_value=(repo_context, {"app/main.py": "x=1\n"})
            ))
            stack.enter_context(patch.object(
                graph, "verify_repository", return_value={"app/main.py": "x=1\n"}
            ))
            stack.enter_context(patch.object(
                graph, "propose_change_plan", return_value="# 文件级计划\n"
            ))
            for module in (graph, candidate_delivery):
                stack.enter_context(patch.object(module, "planned_paths", return_value={"app/main.py"}))
                stack.enter_context(patch.object(module, "prepare_workspace", return_value=(
                    "data/workspaces/fake/attempt-0", {"app/main.py": "x=1\n"},
                )))
            stack.enter_context(patch.object(candidate_delivery, "generate_code_patch", return_value=CodePatchDraft(
                edits=[{"path": "app/main.py", "content": "x=2\n"}], summary="改接口", risks=[],
            )))
            stack.enter_context(patch.object(candidate_delivery, "apply_edits", return_value=(
                "data/workspaces/fake/attempt-0", "--- a/app/main.py\n+++ b/app/main.py\n",
                [{"path": "app/main.py", "before_sha256": "b" * 64, "after_sha256": "c" * 64}],
            )))
            stack.enter_context(patch.object(artifact_store, "verify_workspace"))

            client = TestClient(api.app)
            created = client.post("/tasks", json={
                "request": "实现反馈筛选", "mode": "code_change", "repo_root": "C:/demo",
            })
            self.assertEqual(created.status_code, 201)
            task_id = created.json()["task_id"]
            plan_approved = client.post(f"/tasks/{task_id}/review", json={"action": "approve"})
            self.assertEqual(plan_approved.status_code, 200)
            self.assertTrue(plan_approved.json()["waiting_for_diff_review"])
            self.assertIn("--- a/app/main.py", plan_approved.json()["diff"])
            self.assertEqual(
                client.post(f"/tasks/{task_id}/review", json={"action": "approve"}).status_code,
                409,
            )
            diff_approved = client.post(f"/tasks/{task_id}/diff-review", json={"action": "approve"})
            self.assertEqual(diff_approved.status_code, 200)
            self.assertEqual(diff_approved.json()["workflow_status"], "diff_approved")


if __name__ == "__main__":
    unittest.main()
