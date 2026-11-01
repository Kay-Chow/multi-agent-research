"""通过本机只读检索接口查询演示 Chroma 知识库。"""

import math
import os
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from app.config import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_KB_URL = "http://127.0.0.1:8002/retrieve"
# 重排分只是相关性启发式，不是概率；当前阈值用来过滤 Java 笔记等明显跑题的命中。
MIN_RERANK_SCORE = 0.1


class KnowledgeBaseError(RuntimeError):
    """本地知识库不可用或返回数据不可信，任务不得悄悄退化为只靠公网。"""


def _retrieval_url() -> str:
    """演示期只允许本机知识库，避免配置错误把查询发到外部地址。"""

    url = os.getenv("KB_RETRIEVAL_URL", DEFAULT_KB_URL).strip()
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.path != "/retrieve"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise KnowledgeBaseError("KB_RETRIEVAL_URL 只能是本机 http://127.0.0.1:端口/retrieve")
    return url


def _valid_source(source: str) -> bool:
    """拒绝把绝对路径、上级目录或控制字符当成可展示的文档来源。"""

    path = PurePosixPath(source)
    return (
        bool(source)
        and len(source) <= 240
        and not source.startswith("/")
        and "\\" not in source
        and ":" not in source
        and not any(part in {"", ".", ".."} for part in source.split("/"))
        and not any(ord(char) < 32 for char in source)
        and path.suffix.lower() == ".md"
    )


def search_knowledge_base(query: str) -> list[dict[str, str]]:
    """取最多 3 个相关片段；不调用问答模型，也不改知识库。"""

    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("知识库检索词长度必须在 1～1000 字符之间")
    try:
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(_retrieval_url(), json={"query": query, "top_k": 3})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise KnowledgeBaseError(f"本地知识库检索失败：{exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
        raise KnowledgeBaseError("本地知识库返回格式错误：缺少 hits")
    collection = payload.get("collection")
    if not isinstance(collection, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", collection):
        raise KnowledgeBaseError("本地知识库返回格式错误：collection 无效")

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in payload["hits"][:3]:
        if not isinstance(hit, dict):
            raise KnowledgeBaseError("本地知识库返回格式错误：hit 无效")
        source = hit.get("source")
        chunk_index = hit.get("chunk_index")
        content = hit.get("content")
        score = hit.get("rerank_score")
        if (
            not isinstance(source, str) or not _valid_source(source)
            or type(chunk_index) is not int or chunk_index < 0
            or not isinstance(content, str)
            or not isinstance(score, (int, float)) or not math.isfinite(score)
        ):
            raise KnowledgeBaseError("本地知识库返回格式错误：命中字段无效")
        # 排序分数不代表概率；低于阈值的片段仅作为跑题候选丢弃。
        if score < MIN_RERANK_SCORE or not content.strip():
            continue
        locator = f"{collection}/{source}#chunk-{chunk_index}"
        if locator in seen:
            continue
        seen.add(locator)
        results.append({
            "title": source,
            "locator": locator,
            "snippet": content.strip()[:1000],
        })
    return results
