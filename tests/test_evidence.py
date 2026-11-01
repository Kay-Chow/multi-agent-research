"""来源合并与用户补充的单元测试；不调用模型或外部服务。"""

import unittest

from app.services.evidence import merge_discovery_sources, user_research_source


class EvidenceTests(unittest.TestCase):
    """来源编号由证据层维护，Graph 不自行拼接来源字段。"""

    def test_merge_deduplicates_and_continues_max_source_id(self):
        existing = [{
            "source_id": 4, "source_type": "local_file", "title": "旧文件",
            "locator": "repo/README.md", "snippet": "旧内容", "retrieved_at": "t0",
        }]
        incoming = [
            {"source_type": "local_file", "title": "重复", "locator": "repo/README.md",
             "snippet": "重复内容", "retrieved_at": "t1"},
            {"source_type": "web", "title": "空摘要", "locator": "https://empty",
             "snippet": "  ", "retrieved_at": "t1"},
            {"source_type": "web", "title": "新来源", "locator": "https://example.com",
             "snippet": "可引用摘要", "retrieved_at": "t1", "source_id": 99},
        ]
        merged = merge_discovery_sources(existing, incoming)
        self.assertEqual([item["source_id"] for item in merged], [4, 5])
        self.assertEqual(merged[-1]["locator"], "https://example.com")
        self.assertEqual(len(existing), 1)  # 不原地修改 checkpoint 的旧来源列表。

    def test_user_answer_is_marked_as_user_source(self):
        existing = [{
            "source_id": 3, "source_type": "web", "title": "公开页面",
            "locator": "https://example.com", "snippet": "摘要", "retrieved_at": "t0",
        }]
        source = user_research_source(existing, "a" * 32, "优先级只有 high 和 low")
        self.assertEqual(source["source_id"], 4)
        self.assertEqual(source["source_type"], "user")
        self.assertIn("research-answer", source["locator"])
        self.assertEqual(source["snippet"], "优先级只有 high 和 low")


if __name__ == "__main__":
    unittest.main()
