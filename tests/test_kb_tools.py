"""本地知识库客户端的安全边界与检索过滤测试。"""

import os
import unittest
from unittest.mock import patch

import httpx

from app import kb_tools


class FakeResponse:
    """模拟本地 RAG 服务的 HTTP 响应，避免单元测试依赖真实服务。"""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    """模拟 httpx.Client 的上下文管理与 POST 调用。"""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, json):
        if self.error:
            raise self.error
        assert url == kb_tools.DEFAULT_KB_URL
        assert json["top_k"] == 3
        return self.response


class KnowledgeBaseToolTests(unittest.TestCase):
    """验证本地知识库地址、来源路径、相关性过滤及错误处理。"""

    def test_filters_irrelevant_hits_and_keeps_document_location(self):
        payload = {"collection": "tech_notes_v1", "hits": [
            {"source": "demo/产品反馈列表背景.md", "chunk_index": 0,
             "content": "GET /feedback 尚无筛选", "rerank_score": 0.82},
            {"source": "Java基础.md", "chunk_index": 3,
             "content": "无关笔记", "rerank_score": 0.002},
        ]}
        with patch.dict(os.environ, {"KB_RETRIEVAL_URL": kb_tools.DEFAULT_KB_URL}), patch.object(
            kb_tools.httpx, "Client", return_value=FakeClient(FakeResponse(payload))
        ):
            hits = kb_tools.search_knowledge_base("反馈列表优先级")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["locator"],
                         "tech_notes_v1/demo/产品反馈列表背景.md#chunk-0")

    def test_rejects_remote_kb_url(self):
        with patch.dict(os.environ, {"KB_RETRIEVAL_URL": "https://example.com/retrieve"}):
            with self.assertRaises(kb_tools.KnowledgeBaseError):
                kb_tools.search_knowledge_base("反馈列表")

    def test_rejects_unsafe_source_path(self):
        payload = {"collection": "tech_notes_v1", "hits": [
            {"source": "../.env", "chunk_index": 0,
             "content": "伪造内容", "rerank_score": 0.9},
        ]}
        with patch.dict(os.environ, {"KB_RETRIEVAL_URL": kb_tools.DEFAULT_KB_URL}), patch.object(
            kb_tools.httpx, "Client", return_value=FakeClient(FakeResponse(payload))
        ):
            with self.assertRaisesRegex(kb_tools.KnowledgeBaseError, "命中字段无效"):
                kb_tools.search_knowledge_base("反馈列表")

    def test_connection_failure_is_not_silent(self):
        error = httpx.ConnectError("无法连接")
        with patch.dict(os.environ, {"KB_RETRIEVAL_URL": kb_tools.DEFAULT_KB_URL}), patch.object(
            kb_tools.httpx, "Client", return_value=FakeClient(error=error)
        ):
            with self.assertRaisesRegex(kb_tools.KnowledgeBaseError, "检索失败"):
                kb_tools.search_knowledge_base("反馈列表")


if __name__ == "__main__":
    unittest.main()
