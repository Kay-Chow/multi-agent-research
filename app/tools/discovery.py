"""接待阶段的只读工具：定位项目、读取小段源码、检索本地知识与公开资料。"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.tools.knowledge_base import search_knowledge_base
from app.tools.repository import inspect_project, list_repositories, read_repository_text
from app.tools.web_search import search_web


ToolName = Literal["list_projects", "inspect_project", "read_file", "search_kb", "search_web"]


@dataclass
class DiscoveryResult:
    """一次只读查找的结果；来源编号由图节点合并到 State 时统一分配。"""

    observation: str
    repo_root: str | None = None
    sources: list[dict[str, str]] = field(default_factory=list)


def _source(source_type: str, title: str, locator: str, snippet: str) -> dict[str, str]:
    """截断外部文本，避免单次工具结果无限膨胀或污染后续模型上下文。"""

    return {
        "source_type": source_type,
        "title": title[:240],
        "locator": locator[:500],
        "snippet": snippet.strip()[:2000],
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_discovery_tool(
    name: ToolName, *, query: str | None, path: str | None, repo_root: str | None
) -> DiscoveryResult:
    """按模型提出的工具名执行固定的只读操作；模型不能传任意 Shell 命令。"""

    if name == "list_projects":
        projects = list_repositories()
        return DiscoveryResult(json.dumps({"projects": projects}, ensure_ascii=False))

    if name == "inspect_project":
        if not path:
            raise ValueError("检查项目时需要项目路径或目录名")
        project = inspect_project(path)
        source = _source("local_file", "README.md", f"{project['path']}/README.md", project["summary"])
        return DiscoveryResult(
            json.dumps(project, ensure_ascii=False), project["path"],
            [source] if not project["summary"].startswith("没有可读取") else [],
        )

    if name == "read_file":
        if not repo_root:
            raise ValueError("请先通过 inspect_project 选定仓库")
        if not path:
            raise ValueError("读取文件时需要仓库内相对路径")
        content = read_repository_text(repo_root, path)
        source = _source("local_file", path, f"{repo_root}/{path}", content)
        return DiscoveryResult(
            json.dumps({"path": path, "content": content[:2000]}, ensure_ascii=False),
            repo_root, [source],
        )

    if not query or not query.strip():
        raise ValueError(f"{name} 需要非空检索词")
    query = query.strip()
    if len(query) > 200:
        raise ValueError("检索词超过 200 字符")

    if name == "search_kb":
        hits = search_knowledge_base(query)
        sources = [_source("knowledge_base", hit["title"], hit["locator"], hit["snippet"])
                   for hit in hits]
        return DiscoveryResult(json.dumps({"query": query, "hits": sources}, ensure_ascii=False),
                               sources=sources)

    if name == "search_web":
        # 公网搜索不能携带本地绝对路径或配置文件名；Prompt 中还有更严格的保密约束。
        if re.search(r"[A-Za-z]:[\\/]", query) or ".env" in query.lower():
            raise ValueError("公网检索词不能包含本地路径或配置文件名")
        hits = search_web(query)
        sources = [_source("web", hit["title"], hit["url"], hit["snippet"])
                   for hit in hits if hit["snippet"].strip()]
        return DiscoveryResult(json.dumps({"query": query, "hits": sources}, ensure_ascii=False),
                               sources=sources)

    raise ValueError(f"未知接待工具：{name}")
