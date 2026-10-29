"""Worker 边界测试：模拟模型与搜索结果，不调用外部 API。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents import (
    CodeDraft,
    Finding,
    ReportDraft,
    ResearchBrief,
    SearchQueries,
    coder_node,
    researcher_node,
    writer_node,
)
from app.state import initial_state


class FakeLLM:
    """按结构化输出类型，依次返回预设结果。"""

    def __init__(self, responses):
        self.responses = {
            schema: iter(items)
            for schema, items in responses.items()
        }

    def with_structured_output(self, schema, method):
        return SimpleNamespace(
            invoke=lambda _messages: next(self.responses[schema])
        )


class AgentBoundaryTests(unittest.TestCase):
    def test_researcher_with_no_search_results(self):
        fake_llm = FakeLLM({
            SearchQueries: [SearchQueries(queries=["测试检索词"])],
        })

        with (
            patch("app.agents.ChatDeepSeek", return_value=fake_llm),
            patch("app.agents.search_web", return_value=[]),
        ):
            result = researcher_node(initial_state("测试任务"))

        self.assertEqual(result["sources"], [])
        self.assertTrue(result["research"].startswith("证据不足"))

    def test_researcher_rejects_fabricated_source_id(self):
        fake_llm = FakeLLM({
            SearchQueries: [SearchQueries(queries=["测试检索词"])],
            ResearchBrief: [
                ResearchBrief(findings=[
                    Finding(claim="编造的结论", source_id=99)
                ])
            ],
        })
        search_hits = [{
            "title": "真实搜索结果",
            "url": "https://example.com/test",
            "snippet": "网页摘要",
        }]

        with (
            patch("app.agents.ChatDeepSeek", return_value=fake_llm),
            patch("app.agents.search_web", return_value=search_hits),
        ):
            with self.assertRaisesRegex(ValueError, "不存在的来源"):
                researcher_node(initial_state("测试任务"))

    def test_coder_rejects_two_syntax_errors(self):
        state = initial_state("需要代码示例", True)
        state["research"] = "已有研究资料"

        fake_llm = FakeLLM({
            CodeDraft: [
                CodeDraft(code="def f(", explanation="第一次"),
                CodeDraft(code="x =", explanation="第二次"),
            ],
        })

        with patch("app.agents.ChatDeepSeek", return_value=fake_llm):
            with self.assertRaisesRegex(ValueError, "仍有语法错误"):
                coder_node(state)

    def test_writer_rejects_invalid_citations(self):
        state = initial_state("写一份报告")
        state["research"] = "已有研究结论 [1]"
        state["sources"] = [{
            "source_id": 1,
            "title": "来源一",
            "url": "https://example.com/one",
            "snippet": "摘要",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }]

        bad_report = ReportDraft(
            title="测试报告",
            summary="这个结论引用了不存在的来源 [9]",
            body="正文",
            limitations="局限性",
        )
        fake_llm = FakeLLM({
            ReportDraft: [bad_report, bad_report],
        })

        with patch("app.agents.ChatDeepSeek", return_value=fake_llm):
            with self.assertRaisesRegex(ValueError, "引用校验"):
                writer_node(state)


if __name__ == "__main__":
    unittest.main()
