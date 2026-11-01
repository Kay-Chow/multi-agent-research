"""把只读工具与用户补充整理为可引用来源；不负责图的路由。"""

from datetime import datetime, timezone
from typing import cast

from app.workflow.state import Source


def merge_discovery_sources(
    existing: list[Source], incoming: list[dict[str, str]], *, limit: int = 12
) -> list[Source]:
    """合并一次接待工具的来源，按 locator 去重并分配稳定编号。

    返回新列表，不修改 checkpoint 中的旧列表。用已有最大编号而非列表
    长度续号，避免旧来源不连续时生成重复引用编号。
    """

    sources = list(existing)
    seen = {source["locator"] for source in sources}
    next_id = max((source["source_id"] for source in sources), default=0) + 1
    for item in incoming:
        if item["locator"] in seen or not item["snippet"].strip():
            continue
        if len(sources) >= limit:
            break
        # 编号由本函数控制；即使工具结果意外携带同名字段也不能覆盖它。
        sources.append(cast(Source, {**item, "source_id": next_id}))
        seen.add(item["locator"])
        next_id += 1
    return sources


def user_research_source(existing: list[Source], task_id: str, message: str) -> Source:
    """把用户对调研追问的回答标成独立来源，而非冒充知识库命中。"""

    source_id = max((source["source_id"] for source in existing), default=0) + 1
    return {
        "source_id": source_id,
        "source_type": "user",
        "title": "用户补充",
        "locator": f"conversation/{task_id}#research-answer",
        "snippet": message,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
