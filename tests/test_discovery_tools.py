"""探索接待的工具测试：只读定位、知识库/网页证据和公网查询边界。"""

import unittest
from unittest.mock import patch

from app.tools import discovery as discovery_tools


class DiscoveryToolTests(unittest.TestCase):
    """用假工具响应验证调度，不调用真实仓库、Chroma 或 Tavily。"""

    def test_list_and_select_repository(self):
        """查找候选时只列项目；选定后才记录仓库路径和 README 来源。"""

        projects = [{"name": "demo", "path": "C:/projects/demo"}]
        with patch.object(discovery_tools, "list_repositories", return_value=projects):
            listed = discovery_tools.run_discovery_tool(
                "list_projects", query=None, path=None, repo_root=None,
            )
        self.assertIn("demo", listed.observation)
        self.assertIsNone(listed.repo_root)
        self.assertEqual(listed.sources, [])

        with patch.object(discovery_tools, "inspect_project", return_value={
            "name": "demo", "path": "C:/projects/demo", "summary": "当前不支持筛选",
        }):
            selected = discovery_tools.run_discovery_tool(
                "inspect_project", query=None, path="demo", repo_root=None,
            )
        self.assertEqual(selected.repo_root, "C:/projects/demo")
        self.assertEqual(selected.sources[0]["source_type"], "local_file")

    def test_selected_repo_file_is_a_reusable_source(self):
        """读文件的正文作为本地证据返回；工具层不负责凭空生成引用编号。"""

        with patch.object(discovery_tools, "read_repository_text", return_value="GET /feedback 无筛选"):
            result = discovery_tools.run_discovery_tool(
                "read_file", query=None, path="app/main.py", repo_root="C:/projects/demo",
            )
        self.assertEqual(result.sources[0]["source_type"], "local_file")
        self.assertIn("无筛选", result.sources[0]["snippet"])

    def test_kb_and_web_sources_are_kept_distinct(self):
        """本地历史背景和网页摘要有不同 source_type，不能相互冒充。"""

        with patch.object(discovery_tools, "search_knowledge_base", return_value=[{
            "title": "背景.md", "locator": "demo/背景.md#chunk-0", "snippet": "旧规则",
        }]):
            kb = discovery_tools.run_discovery_tool(
                "search_kb", query="优先级筛选", path=None, repo_root=None,
            )
        with patch.object(discovery_tools, "search_web", return_value=[{
            "title": "公开资料", "url": "https://example.com", "snippet": "公开设计参考",
        }]):
            web = discovery_tools.run_discovery_tool(
                "search_web", query="feedback priority filter", path=None, repo_root=None,
            )
        self.assertEqual(kb.sources[0]["source_type"], "knowledge_base")
        self.assertEqual(web.sources[0]["source_type"], "web")

    def test_web_query_rejects_local_paths(self):
        """公网工具至少阻断显式的 Windows 路径和配置文件名。"""

        with patch.object(discovery_tools, "search_web") as web:
            for query in ("C:\\Users\\Kay\\secret", "读取 .env 配置"):
                with self.subTest(query=query), self.assertRaises(ValueError):
                    discovery_tools.run_discovery_tool(
                        "search_web", query=query, path=None, repo_root=None,
                    )
        web.assert_not_called()


if __name__ == "__main__":
    unittest.main()
