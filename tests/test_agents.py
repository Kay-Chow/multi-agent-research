"""模型与搜索使用假对象，只测试产品 Agent 的业务边界。"""

import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from app.agents import intake as intake_agent
from app.agents import planner as planning_agent
from app.agents import researcher as research_agent
from app.agents import supervisor as supervisor_agent
from app.workflow import graph as agent_graph
from app.workflow.state import initial_chat_state, initial_state


class FakeModel:
    """用预置结构化结果替代真实大模型，测试 Agent 的边界校验。"""

    def __init__(self, results):
        self.results = results

    def with_structured_output(self, schema, method):
        class BoundModel:
            """模拟 `with_structured_output` 绑定 schema 后的模型对象。"""

            def invoke(inner_self, messages):
                return schema.model_validate(self.results[schema.__name__])

        return BoundModel()


class AgentTests(unittest.TestCase):
    """验证任务分类、证据引用、产品方案和文件计划的模型输出约束。"""

    @staticmethod
    def plan_data():
        return {
            "problem": "反馈列表不能筛选",
            "target_users": ["产品经理"],
            "goal": "按优先级查看反馈",
            "in_scope": ["增加可选筛选参数"],
            "out_of_scope": ["不增加数据库"],
            "rules": ["不传参数仍返回全部"],
            "acceptance_criteria": ["priority=high 时只返回 high 反馈"],
            "risks": [],
            "open_questions": ["内部业务背景仍待确认"],
            "source_ids": [1],
        }

    @staticmethod
    def ready_state(with_source=True):
        state = initial_state("增加反馈列表优先级筛选", "proposal_only")
        state["mode"] = "proposal_only"
        state["research"] = "- 研究要点 [1]" if with_source else "证据不足"
        if with_source:
            state["sources"] = [{
                "source_id": 1,
                "source_type": "web",
                "title": "官方文档", "locator": "https://example.com/docs",
                "snippet": "摘要", "retrieved_at": "2026-10-29T00:00:00+00:00",
            }]
        return state

    @staticmethod
    def change_plan_data():
        return {
            "summary": "为反馈列表增加优先级筛选",
            "file_changes": [{
                "path": "app/main.py",
                "change": "给 GET /feedback 增加可选 priority 参数并过滤",
                "reason": "满足产品方案中的筛选规则",
            }, {
                "path": "tests/test_feedback.py",
                "change": "覆盖合法、非法、缺省三类参数",
                "reason": "验证新能力和旧行为兼容",
            }],
            "tests": ["运行反馈 API 单元测试并检查未传参数时返回全量"],
            "risks": [],
            "open_questions": [],
        }

    def ready_change_state(self):
        state = self.ready_state()
        state["mode"] = "code_change"
        state["review_status"] = "approved"
        state["final_plan"] = "# 已批准的产品方案"
        state["acceptance_criteria"] = ["不传参数仍返回全量"]
        state["repo_context"] = {
            "root": "C:/demo", "head": "a" * 40,
            "files": [
                {"path": "app/main.py", "sha256": "b" * 64, "bytes": 1, "lines": 1},
                {"path": "tests/test_feedback.py", "sha256": "c" * 64, "bytes": 1, "lines": 1},
            ],
        }
        return state

    def test_supervisor_classification_is_structured(self):
        model = FakeModel({"ModeDecision": {"mode": "code_change", "reason": "需要修改代码"}})
        with patch.object(intake_agent, "create_model", return_value=model):
            self.assertEqual(intake_agent.decide_mode("实现筛选").mode, "code_change")

    def test_supervisor_can_choose_another_research_round(self):
        """有两条安全路线时，Supervisor 的结构化决定确实影响下一节点。"""

        state = initial_state("调研反馈筛选")
        state["mode"] = "proposal_only"
        state["research"] = "证据不足"
        model = FakeModel({"SupervisorDecision": {
            "next_agent": "researcher", "reason": "先补查业务背景",
        }})
        with patch.object(supervisor_agent, "create_model", return_value=model):
            decision = supervisor_agent.decide_next(state, ["product_planner", "researcher"])
        self.assertEqual(decision.next_agent, "researcher")

    def test_supervisor_invalid_choice_falls_back_to_safe_route(self):
        """模型坚持选当前不允许的 Coder 时，最多纠错一次，再走安全默认步骤。"""

        state = initial_state("只写方案")
        state["mode"] = "proposal_only"
        bound_model = Mock()
        bound_model.invoke.return_value = {"next_agent": "coder", "reason": "越过审核"}
        model = Mock()
        model.with_structured_output.return_value = bound_model
        with patch.object(supervisor_agent, "create_model", return_value=model), self.assertLogs(
            supervisor_agent.logger, level="WARNING"
        ) as logs:
            decision = supervisor_agent.decide_next(state, ["review", "researcher"])
        self.assertEqual(decision.next_agent, "review")
        self.assertEqual(bound_model.invoke.call_count, 2)
        self.assertEqual(len(logs.records), 2)

    def test_supervisor_single_safe_route_does_not_call_model(self):
        """保存等必经步骤由程序强制执行，不为无选择的步骤额外花模型调用。"""

        with patch.object(supervisor_agent, "create_model") as model:
            decision = supervisor_agent.decide_next(initial_state("只写方案"), ["save"])
        self.assertEqual(decision.next_agent, "save")
        model.assert_not_called()

    def test_intake_decision_rejects_conflicting_fields(self):
        """字段类型之外还要检查跨字段关系，不能让矛盾决定进入图路由。"""

        invalid_decisions = [
            {"ready": True, "reply": "开始", "request": "改代码", "tool_name": "list_projects"},
            {"ready": False, "tool_name": "search_kb", "tool_query": "  "},
            {"ready": False, "tool_name": "read_file", "tool_path": ""},
            {"ready": False, "reply": " "},
            {"ready": True, "reply": "开始", "request": " "},
            {"ready": False, "reply": "还要确认", "request": "提前创建的任务"},
            {"ready": False, "tool_name": "unknown_tool"},
        ]
        for data in invalid_decisions:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                intake_agent.IntakeDecision.model_validate(data)

        # 查资料时不需要给用户回复；正常回答时则必须有回复。
        tool_decision = intake_agent.IntakeDecision(ready=False, tool_name="search_kb", tool_query="需求背景")
        answer_decision = intake_agent.IntakeDecision(ready=False, reply="请说说你想解决什么问题？")
        ready_decision = intake_agent.IntakeDecision(ready=True, request="调研反馈筛选")
        self.assertEqual(tool_decision.tool_name, "search_kb")
        self.assertIsNone(answer_decision.tool_name)
        self.assertEqual(ready_decision.reply, "")

    def test_intake_invalid_output_retries_once_then_accepts(self):
        """第一次结构化输出缺少回复，补充纠错指令后再接受有效决定。"""

        bound_model = Mock()
        bound_model.invoke.side_effect = [
            {"ready": False, "reply": ""},
            {"ready": False, "reply": "你想先做方案还是修改代码？"},
        ]
        model = Mock()
        model.with_structured_output.return_value = bound_model
        with patch.object(intake_agent, "create_model", return_value=model), self.assertLogs(
            intake_agent.logger, level="WARNING"
        ) as logs:
            result = intake_agent.intake_requirement(initial_chat_state("我想做个需求"))
        self.assertIn("方案", result.reply)
        self.assertEqual(bound_model.invoke.call_count, 2)
        self.assertIn("上一条决定不符合", bound_model.invoke.call_args.args[0][-1][1])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("decision:value_error", logs.records[0].getMessage())
        self.assertIn("未调用工具且未开始任务", logs.records[0].getMessage())
        self.assertNotIn("我想做个需求", logs.records[0].getMessage())

    def test_intake_two_invalid_outputs_raise_controlled_error(self):
        """最多调用两次模型，随后抛出专用异常供图节点转成人工澄清。"""

        bound_model = Mock()
        bound_model.invoke.return_value = {"ready": False, "reply": ""}
        model = Mock()
        model.with_structured_output.return_value = bound_model
        with patch.object(intake_agent, "create_model", return_value=model), self.assertLogs(
            intake_agent.logger, level="WARNING"
        ) as logs:
            with self.assertRaises(intake_agent.IntakeDecisionError):
                intake_agent.intake_requirement(initial_chat_state("你好"))
        self.assertEqual(bound_model.invoke.call_count, 2)
        self.assertEqual(len(logs.records), 2)

    def test_intake_service_error_is_not_treated_as_bad_user_input(self):
        """网络/模型服务故障应如实暴露，不做两次无意义的格式重试。"""

        bound_model = Mock()
        bound_model.invoke.side_effect = RuntimeError("模型服务暂不可用")
        model = Mock()
        model.with_structured_output.return_value = bound_model
        with patch.object(intake_agent, "create_model", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "模型服务暂不可用"):
                intake_agent.intake_requirement(initial_chat_state("你好"))
        bound_model.invoke.assert_called_once()

    def test_research_without_results_marks_evidence_gap(self):
        """用户已补充过一次后仍无外部依据，就出带缺口的报告而不反复追问。"""

        model = FakeModel({"SearchQueries": {
            "knowledge_queries": ["反馈筛选"], "web_queries": ["反馈筛选"]
        }})
        state = initial_state("反馈筛选")
        state["research_clarified"] = True
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_web", return_value=[]
        ), patch.object(
            research_agent, "search_knowledge_base", return_value=[]
        ):
            result = agent_graph.researcher_node(state)
        self.assertEqual(result["sources"], [])
        self.assertIn("证据不足", result["research"])
        self.assertIn("没有相关命中", result["research_report"])

    def test_research_rejects_fabricated_source_id(self):
        model = FakeModel({
            "SearchQueries": {
                "knowledge_queries": ["反馈筛选"], "web_queries": ["反馈筛选"]
            },
            "ResearchBrief": {"findings": [{"claim": "结论", "source_id": 99}]},
        })
        hits = [{"title": "资料", "url": "https://example.com", "snippet": "摘要"}]
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_web", return_value=hits
        ), patch.object(
            research_agent, "search_knowledge_base", return_value=[]
        ):
            with self.assertRaisesRegex(ValueError, "不存在的来源"):
                agent_graph.researcher_node(initial_state("反馈筛选"))

    def test_irrelevant_sources_prompt_user_instead_of_drafting(self):
        """来源虽非空但无有效要点时先求证，不把无关资料硬凑成方案。"""

        state = initial_state("调研反馈筛选")
        state["sources"] = [{
            "source_id": 1, "source_type": "local_file", "title": "README.md",
            "locator": "C:/demo/README.md", "snippet": "与反馈筛选无关的说明",
            "retrieved_at": "2026-11-01T00:00:00+00:00",
        }]
        model = FakeModel({
            "SearchQueries": {"knowledge_queries": [], "web_queries": []},
            "ResearchBrief": {"findings": []},
        })
        with patch.object(research_agent, "create_model", return_value=model):
            result = agent_graph.researcher_node(state)
        self.assertIsNone(result["research_report"])
        self.assertIn("资料不足以支持明确结论", result["research_question"])

    def test_plan_rejects_fabricated_source_id(self):
        data = self.plan_data()
        data["source_ids"] = [2]
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaisesRegex(ValueError, "不存在的来源"):
                agent_graph.product_planner_node(self.ready_state())

    def test_plan_requires_acceptance_criteria(self):
        data = self.plan_data()
        data["acceptance_criteria"] = []
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaises(ValidationError):
                agent_graph.product_planner_node(self.ready_state())

    def test_no_source_requires_open_question(self):
        data = self.plan_data()
        data["source_ids"] = []
        data["open_questions"] = []
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaisesRegex(ValueError, "待确认问题"):
                agent_graph.product_planner_node(self.ready_state(with_source=False))

    def test_web_only_plan_requires_internal_background_question(self):
        data = self.plan_data()
        data["open_questions"] = []
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaisesRegex(ValueError, "知识库没有相关来源"):
                agent_graph.product_planner_node(self.ready_state())

    def test_research_merges_kb_and_web_with_distinct_citations(self):
        model = FakeModel({
            "SearchQueries": {
                "knowledge_queries": ["反馈列表背景"],
                "web_queries": ["反馈列表筛选设计"],
            },
            "ResearchBrief": {"findings": [
                {"claim": "演示列表尚无筛选", "source_id": 1},
                {"claim": "公开资料可供参考", "source_id": 2},
            ]},
        })
        kb_hits = [{
            "title": "产品反馈列表背景.md",
            "locator": "tech_notes_v1/demo/产品反馈列表背景.md#chunk-0",
            "snippet": "GET /feedback 尚无筛选",
        }]
        web_hits = [{"title": "公开资料", "url": "https://example.com", "snippet": "摘要"}]
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", return_value=kb_hits
        ), patch.object(research_agent, "search_web", return_value=web_hits):
            result = agent_graph.researcher_node(initial_state("增加反馈列表筛选"))
        self.assertEqual([source["source_type"] for source in result["sources"]],
                         ["knowledge_base", "web"])
        self.assertIn("本地知识库片段", result["research_report"])
        self.assertIn("外部网页搜索摘要", result["research_report"])
        self.assertIn("[1]", result["research"])
        self.assertIn("[2]", result["research"])

    def test_research_reuses_discovery_evidence_without_duplicate_search(self):
        """接待阶段已经查过的仓库、知识库和公网证据由正式报告直接复用。"""

        state = initial_state("增加反馈列表筛选")
        state["sources"] = [
            {"source_id": 1, "source_type": "local_file", "title": "README.md",
             "locator": "C:/demo/README.md", "snippet": "当前不支持筛选",
             "retrieved_at": "2026-10-31T00:00:00+00:00"},
            {"source_id": 2, "source_type": "knowledge_base", "title": "背景.md",
             "locator": "demo/背景.md#chunk-0", "snippet": "历史需求背景",
             "retrieved_at": "2026-10-31T00:00:00+00:00"},
            {"source_id": 3, "source_type": "web", "title": "公开参考",
             "locator": "https://example.com", "snippet": "公开设计参考",
             "retrieved_at": "2026-10-31T00:00:00+00:00"},
        ]
        model = FakeModel({
            "SearchQueries": {"knowledge_queries": [], "web_queries": []},
            "ResearchBrief": {"findings": [{"claim": "当前列表没有筛选", "source_id": 1}]},
        })
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", side_effect=AssertionError("不应重复查 KB")
        ), patch.object(research_agent, "search_web", side_effect=AssertionError("不应重复联网")):
            result = agent_graph.researcher_node(state)
        self.assertEqual(result["sources"], state["sources"])
        self.assertIn("已选仓库文件", result["research_report"])
        self.assertIn("[1]", result["research"])

    def test_supervisor_second_research_round_can_add_new_evidence(self):
        """补查不是空转：已有知识库来源仍可按新查询增加不同片段。"""

        state = initial_state("增加反馈列表筛选")
        state["research_rounds"] = 2
        state["research"] = "此前只知道历史背景"
        state["sources"] = [{
            "source_id": 1, "source_type": "knowledge_base", "title": "历史背景",
            "locator": "demo/history#chunk-0", "snippet": "旧版反馈列表",
            "retrieved_at": "2026-10-31T00:00:00+00:00",
        }]
        model = FakeModel({
            "SearchQueries": {"knowledge_queries": ["反馈优先级规则"], "web_queries": []},
            "ResearchBrief": {"findings": [{"claim": "新增优先级规则", "source_id": 2}]},
        })
        new_hit = [{
            "title": "优先级规则", "locator": "demo/priority#chunk-0",
            "snippet": "high、medium、low 三档",
        }]
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", return_value=new_hit
        ) as kb, patch.object(research_agent, "search_web") as web:
            result = agent_graph.researcher_node(state)
        self.assertEqual([item["source_id"] for item in result["sources"]], [1, 2])
        self.assertIn("[2]", result["research"])
        kb.assert_called_once_with("反馈优先级规则")
        web.assert_not_called()

    def test_missing_internal_fact_asks_user_when_kb_is_down(self):
        """内部规则是必需信息，不代表必须来自知识库；应暂停请用户补充。"""

        model = FakeModel({"SearchQueries": {
            "knowledge_queries": ["反馈列表"], "web_queries": ["反馈列表"],
            "internal_facts_required": True,
        }})
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", side_effect=research_agent.KnowledgeBaseError("连接失败")
        ), patch.object(research_agent, "search_web", return_value=[]) as web:
            result = agent_graph.researcher_node(initial_state("反馈列表"))
        self.assertNotIn("workflow_status", result)
        self.assertIn("知识库暂不可用", result["research_question"])
        self.assertIsNone(result["research_report"])
        web.assert_called_once_with("反馈列表")

    def test_missing_internal_fact_asks_user_when_kb_has_no_hits(self):
        """检索成功但没有内部规则时，也应请用户补充而不是编造。"""

        model = FakeModel({"SearchQueries": {
            "knowledge_queries": ["内部审批规则"], "web_queries": ["公开参考"],
            "internal_facts_required": True,
        }})
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", return_value=[]
        ), patch.object(research_agent, "search_web", return_value=[]) as web:
            result = agent_graph.researcher_node(initial_state("依据内部审批规则设计流程"))
        self.assertNotIn("workflow_status", result)
        self.assertIn("请补充相关规则或文档", result["research_question"])
        self.assertIsNone(result["research_report"])
        web.assert_called_once_with("公开参考")

    def test_user_answer_can_replace_missing_kb_evidence(self):
        """知识库仍不可用时，用户补充可供引用，但报告必须保留来源局限。"""

        state = initial_state("依据内部规则设计筛选")
        state["research_clarified"] = True
        state["sources"] = [{
            "source_id": 1, "source_type": "user", "title": "用户补充",
            "locator": f"conversation/{state['task_id']}#research-answer",
            "snippet": "优先级只允许 high 和 low",
            "retrieved_at": "2026-11-01T00:00:00+00:00",
        }]
        model = FakeModel({
            "SearchQueries": {
                "knowledge_queries": ["优先级内部规则"], "web_queries": [],
                "internal_facts_required": True,
            },
            "ResearchBrief": {"findings": [{"claim": "只允许 high 和 low", "source_id": 1}]},
        })
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", side_effect=research_agent.KnowledgeBaseError("连接失败")
        ):
            result = agent_graph.researcher_node(state)
        self.assertIsNotNone(result["research_report"])
        self.assertIn("只允许 high 和 low [1]", result["research"])
        self.assertIn("用户补充（未经独立核验）", result["research_report"])
        self.assertIn("本轮知识库检索失败", result["research_report"])

    def test_optional_kb_failure_keeps_local_evidence(self):
        """知识库并非硬依赖时继续用已读仓库资料，并在报告中标注故障。"""

        state = initial_state("调研反馈列表筛选")
        state["sources"] = [{
            "source_id": 1, "source_type": "local_file", "title": "README.md",
            "locator": "C:/demo/README.md", "snippet": "当前列表没有筛选",
            "retrieved_at": "2026-11-01T00:00:00+00:00",
        }]
        model = FakeModel({
            "SearchQueries": {
                "knowledge_queries": ["反馈筛选背景"], "web_queries": [],
                "internal_facts_required": False,
            },
            "ResearchBrief": {"findings": [{"claim": "当前列表没有筛选", "source_id": 1}]},
        })
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", side_effect=research_agent.KnowledgeBaseError("连接失败")
        ), patch.object(research_agent, "search_web") as web:
            result = agent_graph.researcher_node(state)
        self.assertNotIn("workflow_status", result)
        self.assertEqual(result["sources"], state["sources"])
        self.assertIn("知识库检索失败：连接失败", result["research_report"])
        self.assertIn("内部背景待核验", result["research"])
        self.assertIn("知识库不可用", result["events"][0])
        web.assert_not_called()

    def test_optional_kb_failure_without_any_sources_asks_user(self):
        """全部渠道没有来源时先问用户，不能直接写出无依据的方案。"""

        model = FakeModel({"SearchQueries": {
            "knowledge_queries": ["反馈列表"], "web_queries": ["反馈筛选设计"],
            "internal_facts_required": False,
        }})
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", side_effect=research_agent.KnowledgeBaseError("连接失败")
        ), patch.object(research_agent, "search_web", return_value=[]) as web:
            result = agent_graph.researcher_node(initial_state("调研反馈筛选"))
        self.assertNotIn("workflow_status", result)
        self.assertEqual(result["sources"], [])
        self.assertIsNone(result["research_report"])
        self.assertIn("请补充需求背景", result["research_question"])
        web.assert_called_once_with("反馈筛选设计")

    def test_web_failure_keeps_kb_evidence_and_marks_limitation(self):
        model = FakeModel({
            "SearchQueries": {
                "knowledge_queries": ["反馈列表"], "web_queries": ["反馈列表筛选"],
            },
            "ResearchBrief": {"findings": [
                {"claim": "演示列表尚无筛选", "source_id": 1},
            ]},
        })
        kb_hits = [{
            "title": "产品反馈列表背景.md",
            "locator": "tech_notes_v1/demo/产品反馈列表背景.md#chunk-0",
            "snippet": "GET /feedback 尚无筛选",
        }]
        with patch.object(research_agent, "create_model", return_value=model), patch.object(
            research_agent, "search_knowledge_base", return_value=kb_hits
        ), patch.object(
            research_agent, "search_web", side_effect=research_agent.WebSearchError("TLS 连接失败")
        ):
            result = agent_graph.researcher_node(initial_state("增加反馈列表筛选"))
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source_type"], "knowledge_base")
        self.assertIn("网页检索失败", result["research_report"])
        self.assertIn("公开背景待核验", result["research"])

    def test_valid_plan_contains_testable_criterion(self):
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ProductPlanDraft": self.plan_data()
        })):
            result = agent_graph.product_planner_node(self.ready_state())
        self.assertIn("验收标准", result["draft_plan"])
        self.assertEqual(result["acceptance_criteria"], ["priority=high 时只返回 high 反馈"])

    def test_optional_blank_items_do_not_render_empty_bullets(self):
        """可选列表里的空白文本应被跳过，不能生成只有连字符的假条目。"""

        data = self.plan_data()
        data["risks"] = [" "]
        with patch.object(planning_agent, "create_model", return_value=FakeModel({"ProductPlanDraft": data})):
            result = agent_graph.product_planner_node(self.ready_state())
        self.assertIn("## 风险\n- 暂无", result["draft_plan"])
        self.assertNotIn("\n- \n", result["draft_plan"])

    def test_change_plan_rejects_forbidden_file(self):
        data = self.change_plan_data()
        data["file_changes"][0]["path"] = ".env"
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ChangePlanDraft": data
        })):
            with self.assertRaisesRegex(ValueError, "不在白名单"):
                planning_agent.propose_change_plan(self.ready_change_state(), {"app/main.py": "app = 1"})

    def test_change_plan_lists_files_and_tests_without_writing(self):
        with patch.object(planning_agent, "create_model", return_value=FakeModel({
            "ChangePlanDraft": self.change_plan_data()
        })):
            plan = planning_agent.propose_change_plan(
                self.ready_change_state(), {"app/main.py": "app = 1"}
            )
        self.assertIn("`app/main.py`", plan)
        self.assertIn("tests/test_feedback.py", plan)
        self.assertIn("尚未修改代码", plan)


if __name__ == "__main__":
    unittest.main()
