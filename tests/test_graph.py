"""测试 LangGraph 的节点路由、人工暂停/恢复和任务状态变化。

这些是工作流测试，不是对真实 DeepSeek、知识库或业务仓库的端到端测试。
setUp 会替换所有外部依赖；输出文件和测试用 checkpoint 写入临时目录。文件边界本身
另由 test_workspace_tools.py 验证。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.services import artifact_store, candidate_delivery
from app.workflow import graph as agent_graph
from app.agents.coder import CodePatchDraft
from app.agents.intake import IntakeDecision, IntakeDecisionError, ModeDecision
from app.agents.researcher import ResearchCompleted, ResearchNeedsInput
from app.workflow.state import initial_chat_state, initial_state
from app.agents.supervisor import SupervisorDecision
from app.tools.discovery import DiscoveryResult
from app.tools.repository import RepoAccessError


class ResearchNodeBoundaryTests(unittest.TestCase):
    """Researcher 节点只映射调研结果，不在图层做检索或业务判断。"""

    def test_missing_evidence_becomes_user_interrupt_step(self):
        state = initial_state("设计反馈筛选")
        outcome = ResearchNeedsInput(sources=[], question="请补充内部优先级规则")
        with patch.object(agent_graph, "conduct_research", return_value=outcome) as research:
            update = agent_graph.researcher_node(state)
        research.assert_called_once_with(state)
        self.assertEqual(update["next_step"], "await_research_input")
        self.assertEqual(update["research_question"], outcome.question)
        self.assertEqual(update["conversation"][-1], {"role": "assistant", "text": outcome.question})

    def test_completed_research_becomes_plan_step(self):
        state = initial_state("设计反馈筛选")
        outcome = ResearchCompleted(
            sources=[], research="证据不足", report="# 调研报告\n", event_detail="没有可用来源",
        )
        with patch.object(agent_graph, "conduct_research", return_value=outcome) as research:
            update = agent_graph.researcher_node(state)
        research.assert_called_once_with(state)
        self.assertEqual(update["next_step"], "assess_research")
        self.assertEqual(update["research"], outcome.research)
        self.assertEqual(update["research_report"], outcome.report)


class GraphTests(unittest.TestCase):
    """以假节点和假工具驱动真实图，检查每个审核分支会走到哪里。"""

    def setUp(self):
        """为每个测试建立独立临时目录，并替换会联网或访问真实仓库的函数。

        unittest 每运行一个 test_* 方法就重新调用一次 setUp；addCleanup
        会在测试结束后恢复 patch 并删除临时目录，避免测试之间互相污染。
        """

        # 输出目录指向系统临时目录，而不是项目的真实 outputs/。
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name) / "outputs"

        def fake_researcher(state):
            """模拟完成研究，避免调用知识库、Tavily 和大模型。"""

            return {
                "sources": [],
                "research": "证据不足：测试摘要为空",
                "research_report": "# 调研报告\n\n证据不足。\n",
                "next_step": "assess_research",
                "events": ["researcher：完成模拟研究"],
            }

        def fake_planner(state):
            """模拟生成方案；退回后把反馈写进新稿，便于验证回环。"""

            text = state["review_feedback"] if state["review_status"] == "rejected" else "初稿"
            return {
                "draft_plan": f"# 产品方案\n\n{text}\n",
                "acceptance_criteria": ["只返回匹配优先级的反馈"],
                "open_questions": ["外部资料待核验"],
                "review_status": "pending",
                "next_step": "assess_plan",
                "events": ["product_planner：完成模拟方案"],
            }

        # patch.object 临时替换 graph 模块实际调用的对象，而非修改生产代码。
        for target, replacement in [
            ("OUTPUT_DIR", self.output_dir),
            ("researcher_node", fake_researcher),
            ("product_planner_node", fake_planner),
        ]:
            item = patch.object(agent_graph, target, replacement)
            item.start()
            self.addCleanup(item.stop)

        # 工作流测试不调用真实 Supervisor 模型；默认选择程序允许的首选步骤。
        # 需要验证动态补查的测试会临时覆盖这个返回规则。
        routing_patch = patch.object(
            agent_graph, "decide_next",
            side_effect=lambda _state, allowed: SupervisorDecision(
                next_agent=allowed[0], reason="测试选择首个允许节点"
            ),
        )
        self.routing_mock = routing_patch.start()
        self.addCleanup(routing_patch.stop)

        # 假仓库快照只保留流程需要的字段；这里不读取真实演示仓库。
        self.repo_context = {
            "root": "C:/demo",
            "head": "a" * 40,
            "files": [{"path": "app/main.py", "sha256": "b" * 64, "bytes": 10, "lines": 1}],
        }
        self.inspect_patch = patch.object(
            agent_graph, "inspect_repository",
            return_value=(self.repo_context, {"app/main.py": "app = 1"}),
        )
        self.inspect_mock = self.inspect_patch.start()
        self.addCleanup(self.inspect_patch.stop)
        verify_patch = patch.object(
            agent_graph, "verify_repository", return_value={"app/main.py": "app = 1"}
        )
        verify_patch.start()
        self.addCleanup(verify_patch.stop)
        plan_patch = patch.object(
            agent_graph, "propose_change_plan", return_value="# 文件级变更计划\n\n只读规划。\n"
        )
        plan_patch.start()
        self.addCleanup(plan_patch.stop)

        # 阶段 3 图测试只验证路由与门禁：工作区和哈希核验函数被替换。
        # 路径/哈希细节另由 test_workspace_tools 和 test_artifact_store 验证。
        for module in (agent_graph, candidate_delivery):
            # workspace_prepare 节点和候选生成服务各自从基线准备工作区。
            workspace_patch = patch.object(
                module, "prepare_workspace",
                return_value=("data/workspaces/fake/attempt-0", {"app/main.py": "app = 1\n"}),
            )
            workspace_patch.start()
            self.addCleanup(workspace_patch.stop)
            paths_patch = patch.object(module, "planned_paths", return_value={"app/main.py"})
            paths_patch.start()
            self.addCleanup(paths_patch.stop)
        code_patch = patch.object(candidate_delivery, "generate_code_patch", return_value=CodePatchDraft(
            edits=[{"path": "app/main.py", "content": "app = 2\n"}],
            summary="修改演示接口", risks=[],
        ))
        code_patch.start()
        self.addCleanup(code_patch.stop)
        apply_patch = patch.object(
            candidate_delivery, "apply_edits",
            return_value=(
                "data/workspaces/fake/attempt-0",
                "--- a/app/main.py\n+++ b/app/main.py\n@@ -1 +1 @@\n-app = 1\n+app = 2\n",
                [{"path": "app/main.py", "before_sha256": "b" * 64,
                  "after_sha256": "c" * 64}],
            ),
        )
        apply_patch.start()
        self.addCleanup(apply_patch.stop)
        verify_workspace_patch = patch.object(artifact_store, "verify_workspace")
        self.verify_workspace_mock = verify_workspace_patch.start()
        self.addCleanup(verify_workspace_patch.stop)

    @staticmethod
    def config(state):
        """复用同一 thread_id；代码测试显式选择假的仓库路径供 mock 工具检查。"""

        if state["requested_mode"] == "code_change" and not state.get("repo_root"):
            state["repo_root"] = "C:/demo"
        return {"configurable": {"thread_id": state["task_id"]}}

    def test_chat_intake_reopens_and_clarifies_before_research(self):
        """含糊的第一句话停在图内 intake，SQLite 重开后仍可继续澄清。

        模型先回答“还需要具体问题”，第二轮才给出正式 request；
        在此之前 Supervisor 只负责接待派工，不应调用正式任务分类器、
        Researcher 或写入报告。
        """

        db_path = Path(self.temp_dir.name) / "checkpoint-intake.sqlite"
        state = initial_chat_state("我想做一个需求")
        decisions = [
            IntakeDecision(ready=False, reply="你想解决什么具体问题？", request=None),
            IntakeDecision(ready=True, reply="已理解，将开始调研。",
                           request="调研反馈列表优先级筛选，只写方案"),
        ]
        with patch.object(agent_graph, "intake_requirement", side_effect=decisions) as intake:
            with patch.object(agent_graph, "decide_mode", return_value=ModeDecision(
                mode="proposal_only", reason="只要方案"
            )) as supervisor:
                with SqliteSaver.from_conn_string(str(db_path)) as saver:
                    graph = agent_graph.build_graph(saver)
                    graph.invoke(state, config=self.config(state), durability="sync")
                    paused = graph.get_state(self.config(state))
                    self.assertEqual(paused.next, ("await_user",))
                    self.assertEqual(paused.values["request"], "")
                    self.assertEqual(paused.values["intake_reply"], "你想解决什么具体问题？")
                    supervisor.assert_not_called()
                    self.assertFalse(self.output_dir.exists())

                # 新的 saver/graph 模拟 PyCharm 重启后重新打开同一 checkpoint。
                with SqliteSaver.from_conn_string(str(db_path)) as saver:
                    graph = agent_graph.build_graph(saver)
                    graph.invoke(
                        Command(resume={"message": "反馈列表按优先级筛选，只写方案"}),
                        config=self.config(state), durability="sync",
                    )
                    paused = graph.get_state(self.config(state))
                    self.assertEqual(paused.next, ("review",))
                    self.assertEqual(paused.values["intake_status"], "ready")
                    self.assertEqual(paused.values["request"], "调研反馈列表优先级筛选，只写方案")
                    self.assertEqual(len(paused.values["conversation"]), 4)
                self.assertEqual(intake.call_count, 2)
                supervisor.assert_called_once()

    def test_invalid_intake_decision_waits_for_user_without_losing_evidence(self):
        """格式重试仍失败时，图暂停让人补充；会话和旧证据可继续使用。"""

        state = initial_chat_state("我想做一个需求")
        state["sources"] = [{
            "source_id": 1, "source_type": "local_file", "title": "README.md",
            "locator": "C:/demo/README.md", "snippet": "反馈列表尚无筛选",
            "retrieved_at": "2026-10-31T00:00:00+00:00",
        }]
        decisions = [
            IntakeDecisionError("模型输出无效"),
            IntakeDecision(ready=False, reply="你希望先出方案，还是修改代码？"),
        ]
        with patch.object(agent_graph, "intake_requirement", side_effect=decisions) as intake, patch.object(
            agent_graph, "decide_mode"
        ) as supervisor, patch.object(agent_graph, "run_discovery_tool") as tool:
            graph = agent_graph.build_graph(InMemorySaver())
            graph.invoke(state, config=self.config(state))
            paused = graph.get_state(self.config(state))
            self.assertEqual(paused.next, ("await_user",))
            self.assertEqual(paused.values["request"], "")
            self.assertEqual(paused.values["sources"], state["sources"])
            self.assertIn("没能完成这一步判断", paused.values["intake_reply"])
            supervisor.assert_not_called()
            tool.assert_not_called()

            graph.invoke(Command(resume={"message": "先做方案"}), config=self.config(state))
            paused = graph.get_state(self.config(state))
            self.assertEqual(paused.next, ("await_user",))
            self.assertIn("方案", paused.values["intake_reply"])
            self.assertEqual(paused.values["sources"], state["sources"])
            self.assertEqual(intake.call_count, 2)

    def test_intake_can_find_project_and_answer_before_starting_task(self):
        """项目位置由只读工具发现；回答用户问题时不提前启动正式调研。"""

        state = initial_chat_state("你知道演示项目在哪吗？")
        decisions = [
            IntakeDecision(ready=False, tool_name="list_projects"),
            IntakeDecision(ready=False, tool_name="inspect_project", tool_path="demo"),
            IntakeDecision(ready=False, reply="知道，在 C:/projects/demo。你想先做方案吗？"),
        ]
        results = [
            DiscoveryResult('{"projects":[{"name":"demo","path":"C:/projects/demo"}]}'),
            DiscoveryResult(
                '{"name":"demo","path":"C:/projects/demo"}', "C:/projects/demo", [{
                    "source_type": "local_file", "title": "README.md",
                    "locator": "C:/projects/demo/README.md", "snippet": "当前不支持筛选",
                    "retrieved_at": "2026-10-31T00:00:00+00:00",
                }],
            ),
        ]
        with patch.object(agent_graph, "intake_requirement", side_effect=decisions) as intake, patch.object(
            agent_graph, "run_discovery_tool", side_effect=results
        ) as tool, patch.object(agent_graph, "decide_mode") as supervisor:
            graph = agent_graph.build_graph(InMemorySaver())
            graph.invoke(state, config=self.config(state))

        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("await_user",))
        self.assertEqual(snapshot.values["repo_root"], "C:/projects/demo")
        self.assertEqual(snapshot.values["sources"][0]["source_type"], "local_file")
        self.assertIn("C:/projects/demo", snapshot.values["intake_reply"])
        self.assertEqual(snapshot.values["request"], "")
        self.assertEqual(snapshot.values["supervisor_steps"], 0)
        # Intake 没有直接通向工具/用户中断；两次查找和最终暂停都由 Supervisor 派发。
        routes = [event for event in snapshot.values["events"] if event.startswith("supervisor ->")]
        self.assertEqual(routes, [
            "supervisor -> intake：接待阶段",
            "supervisor -> discovery_tool：接待阶段",
            "supervisor -> intake：接待阶段",
            "supervisor -> discovery_tool：接待阶段",
            "supervisor -> intake：接待阶段",
            "supervisor -> await_user：接待阶段",
        ])
        self.assertEqual(tool.call_count, 2)
        self.assertEqual(intake.call_count, 3)
        supervisor.assert_not_called()

    def test_discovery_failure_returns_to_intake_without_claiming_success(self):
        """可选知识库查询失败后保留会话，并在事件中如实标记失败。"""

        state = initial_chat_state("查一下需求背景")
        state["intake_status"] = "lookup"
        state["discovery_tool"] = "search_kb"
        state["discovery_query"] = "需求背景"
        with patch.object(
            agent_graph, "run_discovery_tool",
            side_effect=agent_graph.KnowledgeBaseError("连接失败"),
        ):
            update = agent_graph.discovery_tool_node(state)
        self.assertEqual(update["next_step"], "intake")
        self.assertEqual(update["sources"], [])
        self.assertIn("连接失败", update["discovery_observations"][0])
        self.assertIn("失败", update["events"][0])

    def test_code_request_without_repo_asks_before_research(self):
        """聊天中的代码任务若未选仓库，不应先生成方案再在代码阶段失败。"""

        state = initial_chat_state("请改代码实现筛选")
        with patch.object(agent_graph, "intake_requirement", return_value=IntakeDecision(
            ready=True, reply="理解需求。", request="修改代码实现反馈筛选",
        )), patch.object(agent_graph, "decide_mode", return_value=ModeDecision(
            mode="code_change", reason="明确要求改代码",
        )):
            graph = agent_graph.build_graph(InMemorySaver())
            graph.invoke(state, config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("await_user",))
        self.assertIn("仓库路径", snapshot.values["intake_reply"])
        self.assertEqual(snapshot.values["research"], None)

    def test_chat_code_request_after_plan_approval_reaches_diff_review(self):
        """“先看方案，批准后改代码”仍是同一项代码任务，不在方案保存后结束。"""

        original = "请修改演示项目代码；先看方案，批准后在隔离工作区生成代码和 diff。"
        state = initial_chat_state(original)
        state["repo_root"] = "C:/demo"
        with patch.object(agent_graph, "intake_requirement", return_value=IntakeDecision(
            ready=True, request="实现反馈列表优先级筛选并补充测试",
        )), patch.object(agent_graph, "decide_mode", return_value=ModeDecision(
            mode="code_change", reason="批准后仍要生成候选代码",
        )) as classify:
            graph = agent_graph.build_graph(InMemorySaver())
            graph.invoke(state, config=self.config(state))
        classify.assert_called_once_with(original)
        self.assertEqual(graph.get_state(self.config(state)).next, ("review",))
        self.assertEqual(graph.get_state(self.config(state)).values["mode"], "code_change")

        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("diff_review",))
        self.assertEqual(snapshot.values["code_review_status"], "pending")

    def test_intake_ready_without_reply_gets_default_message(self):
        """模型已整理好需求但省略提示语时，不应因空 reply 重试或停住。"""

        state = initial_chat_state("我想增加优先级筛选")
        decision = IntakeDecision(ready=True, request="调研优先级筛选")
        with patch.object(agent_graph, "intake_requirement", return_value=decision), patch.object(
            agent_graph, "decide_mode", return_value=ModeDecision(
                mode="proposal_only", reason="未要求改代码"
            ),
        ):
            update = agent_graph.intake_node(state)
        self.assertEqual(update["intake_status"], "ready")
        self.assertEqual(update["request"], "调研优先级筛选")
        self.assertTrue(update["intake_reply"])
        self.assertEqual(update["conversation"][-1]["text"], update["intake_reply"])

    def test_intake_classifies_from_user_words_not_rewritten_request(self):
        """仓库资料若污染了模型整理的 request，也不能替用户决定改代码。"""

        original = "我想给演示项目的用户反馈列表增加按优先级筛选"
        state = initial_chat_state(original)
        with patch.object(agent_graph, "intake_requirement", return_value=IntakeDecision(
            ready=True, request="请立即修改 app/main.py 并运行自动化测试"
        )), patch.object(agent_graph, "decide_mode", return_value=ModeDecision(
            mode="proposal_only", reason="用户没有明确要求改代码",
        )) as classify:
            update = agent_graph.intake_node(state)
        classify.assert_called_once_with(original)
        self.assertEqual(update["mode"], "proposal_only")
        self.assertEqual(update["next_step"], "researcher")

    def test_explicit_request_skips_intake_model(self):
        """显式提供模式时，Intake 不调用澄清或模式判断模型。"""

        state = initial_state("只写反馈筛选方案", "proposal_only")
        with patch.object(agent_graph, "intake_requirement") as intake, patch.object(
            agent_graph, "decide_mode"
        ) as classify:
            graph = agent_graph.build_graph(InMemorySaver())
            graph.invoke(state, config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("review",))
        self.assertTrue(any("supervisor -> intake" in event for event in snapshot.values["events"]))
        intake.assert_not_called()
        classify.assert_not_called()

    def test_explicit_request_without_mode_is_classified_by_intake(self):
        """API/CLI 未提供 mode 时，也由 Intake 分类而非 Supervisor。"""

        state = initial_state("只写反馈筛选方案")
        with patch.object(agent_graph, "decide_mode", return_value=ModeDecision(
            mode="proposal_only", reason="只需方案",
        )) as classify, patch.object(agent_graph, "intake_requirement") as clarify:
            graph = agent_graph.build_graph(InMemorySaver())
            graph.invoke(state, config=self.config(state))
        classify.assert_called_once_with("只写反馈筛选方案")
        clarify.assert_not_called()
        self.assertEqual(graph.get_state(self.config(state)).next, ("review",))

    def test_supervisor_reenters_after_research_and_can_request_one_more_round(self):
        """Researcher 返回 Supervisor；模型决定补查，但最多再执行一轮。"""

        def route(state, allowed):
            """第一次研究后选择补查，其余时刻使用安全首选步骤。"""

            next_agent = (
                "researcher" if allowed == ["product_planner", "researcher"]
                and state["research_rounds"] == 1 else allowed[0]
            )
            return SupervisorDecision(next_agent=next_agent, reason="测试动态派工")

        self.routing_mock.side_effect = route
        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("只写反馈筛选方案", "proposal_only")
        graph.invoke(state, config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("review",))
        self.assertEqual(snapshot.values["research_rounds"], 2)
        self.assertEqual(sum("researcher：完成模拟研究" in event for event in snapshot.values["events"]), 2)
        self.assertTrue(any("supervisor -> researcher" in event for event in snapshot.values["events"]))

    def test_supervisor_can_send_draft_back_for_research_before_human_review(self):
        """方案生成后发现证据缺口，可以先补查再重写，不能跳过最终人审。"""

        def route(state, allowed):
            next_agent = (
                "researcher" if allowed == ["review", "researcher"]
                and state["research_rounds"] == 1 else allowed[0]
            )
            return SupervisorDecision(next_agent=next_agent, reason="方案依据不足，先补查")

        self.routing_mock.side_effect = route
        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("只写反馈筛选方案", "proposal_only")
        graph.invoke(state, config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("review",))
        self.assertEqual(snapshot.values["research_rounds"], 2)
        self.assertEqual(sum("product_planner：完成模拟方案" in event for event in snapshot.values["events"]), 2)
        self.assertEqual(snapshot.values["review_status"], "pending")

    def test_supervisor_cannot_route_directly_to_coder_before_review(self):
        """即使模型返回合法节点名，只要当前未获人工批准，也拒绝越级派工。"""

        def unsafe_route(_state, allowed):
            return SupervisorDecision(
                next_agent="coder" if "review" in allowed else allowed[0],
                reason="尝试跳过人审",
            )

        self.routing_mock.side_effect = unsafe_route
        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现反馈筛选", "code_change", "C:/demo")
        with self.assertRaisesRegex(ValueError, "当前不允许的节点"):
            graph.invoke(state, config=self.config(state))
        self.assertFalse(self.output_dir.exists())

    def test_supervisor_step_limit_blocks_before_any_more_dispatch(self):
        """即使未来新增回环，派工预算耗尽后也不能无限调用模型或节点。"""

        state = initial_state("只写方案", "proposal_only")
        state["mode"] = "proposal_only"
        state["next_step"] = "researcher"
        state["supervisor_steps"] = agent_graph.MAX_SUPERVISOR_STEPS
        update = agent_graph.supervisor_node(state)
        self.assertEqual(update["workflow_status"], "blocked")
        self.assertIsNone(update["next_step"])
        self.assertIn("上限", update["error"])
        self.routing_mock.assert_not_called()

    def test_supervisor_only_dispatches_preclassified_task(self):
        """Supervisor 不再调用模式判断模型；缺少模式则明确拒绝派工。"""

        state = initial_state("只写方案")
        state["next_step"] = "researcher"
        with patch.object(agent_graph, "decide_mode") as classify:
            with self.assertRaisesRegex(ValueError, "尚未由 Intake 确定"):
                agent_graph.supervisor_node(state)
            state["mode"] = "proposal_only"
            update = agent_graph.supervisor_node(state)
        self.assertEqual(update["next_step"], "researcher")
        classify.assert_not_called()

    def test_terminated_review_cannot_reenter_supervisor(self):
        """人工终止应直接结束图，不能把终止状态再次当成待派工状态。"""

        state = initial_state("只写方案", "proposal_only")
        state["mode"] = "proposal_only"
        state["review_status"] = "terminated"
        with self.assertRaisesRegex(ValueError, "不允许继续派工"):
            agent_graph._allowed_supervisor_targets(state)

    def test_supervisor_uses_explicit_next_step_not_old_artifacts(self):
        """即使已保存过方案，重做路径也由 next_step 决定，不靠文件字段猜。"""

        state = initial_state("只写方案", "proposal_only")
        state["mode"] = "proposal_only"
        state["review_status"] = "approved"
        state["research_path"] = "outputs/old/research.md"
        state["proposal_path"] = "outputs/old/proposal.md"
        state["next_step"] = "save"
        update = agent_graph.supervisor_node(state)
        self.assertEqual(update["next_step"], "save")
        self.assertEqual(update["supervisor_steps"], 1)

    def test_supervisor_rejects_unknown_next_step(self):
        """损坏的 checkpoint 不能让 Supervisor 猜测或随意跳到某个节点。"""

        state = initial_state("只写方案", "proposal_only")
        state["mode"] = "proposal_only"
        state["next_step"] = "unknown"
        with self.assertRaisesRegex(ValueError, "未知或已结束的下一步"):
            agent_graph.supervisor_node(state)

    def test_finish_requires_approved_saved_plan(self):
        """即使 next_step 被错误设置为 finish，完成节点也会守住人工门禁。"""

        state = initial_state("只写方案", "proposal_only")
        state["mode"] = "proposal_only"
        state["next_step"] = "finish"
        with self.assertRaisesRegex(ValueError, "尚未批准或保存"):
            agent_graph.finish_node(state)

    def test_proposal_only_approve_saves_two_files(self):
        """纯方案任务应先停在 review；批准后只保存报告与方案。

        断言仓库检查没有被调用，证明 proposal_only 不会误入代码分支。
        """

        with patch.object(agent_graph, "decide_mode", return_value=ModeDecision(
            mode="proposal_only", reason="只要方案"
        )):
            graph = agent_graph.build_graph(InMemorySaver())
            state = initial_state("只写方案，不改代码")
            graph.invoke(state, config=self.config(state))

        # 第一次 invoke 遇到 interrupt 后暂停，而不是直接写入批准文件。
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("review",))
        self.assertEqual(snapshot.values["mode"], "proposal_only")
        self.assertFalse(self.output_dir.exists())

        # 用 Command(resume=...) 模拟人输入“批准”，从同一个 checkpoint 继续。
        result = graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        self.assertEqual(result["workflow_status"], "completed")
        self.assertIsNone(result["next_step"])
        directory = self.output_dir / state["task_id"]
        self.assertTrue((directory / "research.md").exists())
        self.assertTrue((directory / "proposal.md").exists())
        self.inspect_mock.assert_not_called()

    def test_code_change_reaches_diff_review_without_touching_demo(self):
        """代码任务批准产品方案后，应生成候选 diff 并停在第二次审核。

        仓库读取和隔离区写入均已 mock；这里只检查图的流转及临时产物。
        """

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("diff_review",))
        self.assertEqual(snapshot.values["workflow_status"], "diff_pending")
        self.assertEqual(snapshot.values["mode"], "code_change")
        self.inspect_mock.assert_called_once()
        self.assertTrue(
            (self.output_dir / state["task_id"] / "change_plan.md").exists()
        )
        self.assertTrue(
            (self.output_dir / state["task_id"] / "diff-attempt-0.patch").exists()
        )

    def test_diff_approve_finishes_without_running_tests(self):
        """第二次批准后状态应是 diff_approved，而非宣称测试已经通过。

        此处还检查审核节点调用了隔离工作区核验；测试执行属于后续阶段。
        """

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        result = graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        self.assertEqual(result["workflow_status"], "diff_approved")
        self.assertEqual(result["code_review_status"], "approved")
        self.verify_workspace_mock.assert_called_once()
        self.assertFalse(any("测试通过" in event for event in result["events"]))

    def test_diff_reject_creates_new_attempt_and_pauses_again(self):
        """退回候选 diff 后保留反馈、增加 attempt 编号，再次等待人工审核。"""

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        graph.invoke(Command(resume={"action": "reject", "feedback": "补测试"}), config=self.config(state))
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("diff_review",))
        self.assertEqual(snapshot.values["code_revision_count"], 1)
        self.assertEqual(snapshot.values["code_review_feedback"], "补测试")
        self.assertTrue((self.output_dir / state["task_id"] / "diff-attempt-1.patch").exists())

    def test_diff_reject_limit_terminates_without_commit(self):
        """连续退回超过上限后图结束，不再派 Coder 继续生成候选版本。"""

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        # MAX_CODE_REVISIONS=2：前两次退回重做，第三次达到上限而终止。
        for number in range(3):
            result = graph.invoke(
                Command(resume={"action": "reject", "feedback": f"修改 {number}"}),
                config=self.config(state),
            )
        self.assertEqual(result["workflow_status"], "terminated")
        self.assertEqual(result["code_review_status"], "terminated")
        self.assertEqual(graph.get_state(self.config(state)).next, ())

    def test_diff_file_tamper_blocks_approval(self):
        """人工审核期间若磁盘上的 patch 被改写，批准应转为 blocked。"""

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        diff_file = self.output_dir / state["task_id"] / "diff-attempt-0.patch"
        # 图节点会核对磁盘 patch 与 checkpoint 中展示给人的 diff 是否相同。
        diff_file.write_text("伪造 diff", encoding="utf-8")
        result = graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        self.assertEqual(result["workflow_status"], "blocked")
        self.assertIn("diff 文件", result["error"])

    def test_workspace_tamper_blocks_approval(self):
        """模拟隔离区哈希核验失败，确认图不会把候选改动标为获批。

        这里通过 side_effect 注入错误；真实改文件后的哈希检测见工作区测试。
        """

        self.verify_workspace_mock.side_effect = agent_graph.WorkspaceError("文件变化")
        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        result = graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        self.assertEqual(result["workflow_status"], "blocked")
        self.assertIn("文件变化", result["error"])

    def test_code_task_cannot_finish_at_proposal_stage(self):
        """代码任务不能在方案保存后冒充完成，必须继续走隔离候选和 diff 审核。"""

        state = initial_state("实现优先级筛选", "code_change")
        state["mode"] = "code_change"
        with self.assertRaisesRegex(ValueError, "不能在方案保存后提前结束"):
            agent_graph.finish_node(state)

    def test_repo_inspection_failure_blocks_without_change_plan(self):
        """仓库只读检查失败时应停止，不能继续生成或保存文件变更计划。"""

        self.inspect_mock.side_effect = RepoAccessError("仓库不干净")
        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("实现优先级筛选", "code_change")
        graph.invoke(state, config=self.config(state))
        result = graph.invoke(Command(resume={"action": "approve"}), config=self.config(state))
        self.assertEqual(result["workflow_status"], "blocked")
        self.assertIn("仓库不干净", result["error"])
        self.assertFalse((self.output_dir / state["task_id"] / "change_plan.md").exists())

    def test_missing_research_evidence_pauses_for_user_then_continues(self):
        """来源缺口先暂停追问；用户补充成为来源后再进方案审核。"""

        def researcher(state):
            """模拟第一轮缺依据、第二轮已收到用户补充的 Researcher。"""

            if not state["research_clarified"]:
                question = "请补充现有的优先级规则。"
                return {
                    "research_question": question,
                    "research": None,
                    "research_report": None,
                    "next_step": "await_research_input",
                    "conversation": state["conversation"] + [{"role": "assistant", "text": question}],
                    "events": ["researcher：等待用户补充"],
                }
            return {
                "sources": state["sources"],
                "research": "用户说明了优先级规则 [1]",
                "research_report": "# 调研报告\n\n用户说明了优先级规则 [1]\n",
                "next_step": "assess_research",
                "events": ["researcher：已整理用户补充"],
            }

        with patch.object(agent_graph, "researcher_node", researcher):
            graph = agent_graph.build_graph(InMemorySaver())
            state = initial_state("调研优先级筛选", "proposal_only")
            graph.invoke(state, config=self.config(state))
            paused = graph.get_state(self.config(state))
            self.assertEqual(paused.next, ("await_research_input",))
            self.assertEqual(paused.values["workflow_status"], "running")
            self.assertIsNone(paused.values["draft_plan"])

            graph.invoke(
                Command(resume={"message": "优先级为高、中、低三档"}),
                config=self.config(state),
            )
            resumed = graph.get_state(self.config(state))
        self.assertEqual(resumed.next, ("review",))
        self.assertTrue(resumed.values["research_clarified"])
        self.assertIsNone(resumed.values["research_question"])
        self.assertEqual(resumed.values["sources"][0]["source_type"], "user")
        self.assertEqual(resumed.values["sources"][0]["snippet"], "优先级为高、中、低三档")
        self.assertEqual(resumed.values["research_rounds"], 2)

    def test_reject_revises_plan_and_pauses_again(self):
        """产品方案被退回后应带着反馈再进 Planner，并再次暂停审核。"""

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("只写方案", "proposal_only")
        graph.invoke(state, config=self.config(state))
        graph.invoke(
            Command(resume={"action": "reject", "feedback": "补充兼容性要求"}),
            config=self.config(state),
        )
        snapshot = graph.get_state(self.config(state))
        self.assertEqual(snapshot.next, ("review",))
        self.assertEqual(snapshot.values["revision_count"], 1)
        self.assertIn("补充兼容性要求", snapshot.values["draft_plan"])
        self.assertFalse(self.output_dir.exists())

    def test_reject_limit_terminates_without_saving(self):
        """产品方案多次被退回达到上限后终止，不写批准文件。"""

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("只写方案", "proposal_only")
        graph.invoke(state, config=self.config(state))
        # MAX_REVISIONS=2：第三次退回后不再给 Planner 修改机会。
        for index in range(3):
            result = graph.invoke(
                Command(resume={"action": "reject", "feedback": f"修改 {index}"}),
                config=self.config(state),
            )
        self.assertEqual(result["review_status"], "terminated")
        self.assertFalse(self.output_dir.exists())

    def test_terminate_does_not_save(self):
        """人工直接终止方案任务时，不保存调研报告或产品方案。"""

        graph = agent_graph.build_graph(InMemorySaver())
        state = initial_state("只写方案", "proposal_only")
        graph.invoke(state, config=self.config(state))
        result = graph.invoke(Command(resume={"action": "terminate"}), config=self.config(state))
        self.assertEqual(result["workflow_status"], "terminated")
        self.assertFalse(self.output_dir.exists())

    def test_sqlite_reopen_then_approve(self):
        """关闭并重新打开 SQLite 后，仍能恢复方案审核并完成任务。"""

        db_path = Path(self.temp_dir.name) / "checkpoint.sqlite"
        state = initial_state("只写方案", "proposal_only")
        # 第一段 with 模拟第一次进程运行：停在 review 后关闭数据库连接。
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = agent_graph.build_graph(saver)
            graph.invoke(state, config=self.config(state), durability="sync")

        # 第二段 with 模拟重启后的新进程：从同一 thread_id 恢复。
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = agent_graph.build_graph(saver)
            self.assertEqual(graph.get_state(self.config(state)).next, ("review",))
            result = graph.invoke(
                Command(resume={"action": "approve"}),
                config=self.config(state), durability="sync",
            )
        self.assertEqual(result["workflow_status"], "completed")

    def test_sqlite_reopen_at_diff_review_then_approve(self):
        """SQLite 重开后，代码任务仍停在 diff_review，能继续第二次审核。"""

        db_path = Path(self.temp_dir.name) / "checkpoint-code.sqlite"
        state = initial_state("实现优先级筛选", "code_change")
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = agent_graph.build_graph(saver)
            graph.invoke(state, config=self.config(state), durability="sync")
            graph.invoke(
                Command(resume={"action": "approve"}),
                config=self.config(state), durability="sync",
            )
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            graph = agent_graph.build_graph(saver)
            self.assertEqual(graph.get_state(self.config(state)).next, ("diff_review",))
            result = graph.invoke(
                Command(resume={"action": "approve"}),
                config=self.config(state), durability="sync",
            )
        self.assertEqual(result["workflow_status"], "diff_approved")


if __name__ == "__main__":
    unittest.main()
