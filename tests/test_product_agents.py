"""阶段 1 边界测试：真实搜索/模型由假对象替代，不消耗 API 配额。"""

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.product import agents
from app.product.state import initial_product_state


class FakeModel:
    """按 Pydantic schema 名称提供固定结果，验证业务层二次校验。"""

    def __init__(self, results: dict[str, dict]):
        self.results = results

    def with_structured_output(self, schema, method):
        class BoundModel:
            def invoke(inner_self, messages):
                return schema.model_validate(self.results[schema.__name__])

        return BoundModel()


class ProductAgentTests(unittest.TestCase):
    def test_research_rejects_fabricated_source_id(self) -> None:
        model = FakeModel({
            "QueryPlan": {"queries": ["反馈优先级"]},
            "ResearchNotes": {"findings": [{"claim": "结论", "source_id": 99}]},
        })
        hits = [{
            "title": "官方文档", "url": "https://example.com/docs", "snippet": "一段摘要"
        }]
        with patch.object(agents, "_model", return_value=model), patch.object(
            agents, "search_web", return_value=hits
        ):
            with self.assertRaisesRegex(ValueError, "不存在的来源"):
                agents.collect_research("需要调研反馈筛选")

    def test_empty_search_is_explicitly_marked_insufficient(self) -> None:
        model = FakeModel({"QueryPlan": {"queries": ["反馈优先级"]}})
        with patch.object(agents, "_model", return_value=model), patch.object(
            agents, "search_web", return_value=[]
        ):
            sources, findings, report = agents.collect_research("反馈筛选")
        self.assertEqual(sources, [])
        self.assertEqual(findings, [])
        self.assertIn("没有检索到", report)
        self.assertIn("不能据此声称", report)

    def _state_with_source(self):
        state = initial_product_state("给反馈列表增加优先级筛选", "proposal_only")
        state["mode"] = "proposal_only"
        state["sources"] = [{
            "source_id": 1,
            "title": "参考资料",
            "url": "https://example.com/docs",
            "snippet": "一段摘要",
            "retrieved_at": "2026-10-29T00:00:00+00:00",
        }]
        state["research_findings"] = [{"claim": "参考要点", "source_id": 1}]
        return state

    def _valid_plan(self):
        return {
            "problem": "列表无法筛选",
            "target_users": ["产品经理"],
            "goal": "按优先级查看反馈",
            "in_scope": ["增加可选筛选参数"],
            "out_of_scope": ["不增加数据库"],
            "rules": ["不传参数仍返回全部"],
            "acceptance_criteria": ["传 high 时只返回 high 反馈"],
            "risks": [],
            "open_questions": [],
            "source_ids": [1],
        }

    def test_product_plan_rejects_fabricated_source_id(self) -> None:
        data = self._valid_plan()
        data["source_ids"] = [2]
        with patch.object(agents, "_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaisesRegex(ValueError, "不存在的来源"):
                agents.create_product_plan(self._state_with_source())

    def test_product_plan_requires_acceptance_criteria(self) -> None:
        data = self._valid_plan()
        data["acceptance_criteria"] = []
        with patch.object(agents, "_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaises(ValidationError):
                agents.create_product_plan(self._state_with_source())

    def test_product_plan_without_sources_requires_open_question(self) -> None:
        state = initial_product_state("反馈筛选", "proposal_only")
        state["mode"] = "proposal_only"
        data = self._valid_plan()
        data["source_ids"] = []
        with patch.object(agents, "_model", return_value=FakeModel({
            "ProductPlanDraft": data
        })):
            with self.assertRaisesRegex(ValueError, "待确认问题"):
                agents.create_product_plan(state)


if __name__ == "__main__":
    unittest.main()
