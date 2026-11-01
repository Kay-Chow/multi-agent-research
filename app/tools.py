"""联网搜索工具：只返回有真实 URL 的、长度受限的搜索结果。"""

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from requests import RequestException
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError, ForbiddenError, InvalidAPIKeyError, MissingAPIKeyError,
    TimeoutError as TavilyTimeoutError, UsageLimitExceededError,
)


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class WebSearchError(RuntimeError):
    """统一表示 Tavily 配置、网络或 API 失败，供 Researcher 标注检索局限。"""


def search_web(query: str) -> list[dict[str, str]]:
    """搜索网页并整理结果；不让 Tavily 直接生成最终答案。"""

    query = query.strip()
    if not query:
        raise ValueError("搜索词不能为空")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise WebSearchError("缺少 TAVILY_API_KEY，请检查项目根目录的 .env")

    try:
        response = TavilyClient(api_key=api_key).search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=False,
            include_raw_content=False,
            timeout=20,
        )
    except (
        RequestException, TavilyTimeoutError, UsageLimitExceededError,
        ForbiddenError, InvalidAPIKeyError, BadRequestError, MissingAPIKeyError,
    ) as exc:
        # SDK 会把超时/额度/鉴权问题抛成自己的异常类，并不都继承
        # requests.RequestException；统一转成 WebSearchError 供上层处理。
        raise WebSearchError(f"Tavily 请求失败：{exc}") from exc

    results = []
    seen_urls = set()

    for item in response.get("results", []):
        url = (item.get("url") or "").strip()
        parsed = urlparse(url)

        # 这里只检查 URL 形式，不验证网页确实可访问；重复 URL 只保留一次。
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        results.append({
            "title": (item.get("title") or url).strip(),
            "url": url,
            # 搜索摘要是外部不可信文本，同时限制传给后续模型的长度。
            "snippet": (item.get("content") or "")[:1000],
        })

    return results
