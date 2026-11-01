"""Supervisor 的模型派工决定；候选节点集合由 Graph 的状态机限定。"""

import json
import logging

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError

from app.agents.model_client import create_model
from app.agents.prompts import SUPERVISOR_ROUTE_PROMPT
from app.workflow.state import AgentState, SupervisorTarget


logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Supervisor 的一次派工决定；实际允许集合仍由图根据状态计算。"""

    next_agent: SupervisorTarget
    reason: str = Field(min_length=1, max_length=300)

def decide_next(state: AgentState, allowed: list[SupervisorTarget]) -> SupervisorDecision:
    """在程序给定的安全候选中选择下一位执行者；无效输出回退到安全首选。

    单候选时无需消耗模型调用；多候选只出现在研究后、方案后两个
    补查决策点。模型只能在当前候选中选择，不能越过人工审核。
    """

    if not allowed:
        raise ValueError("Supervisor 没有可执行的下一步")
    if len(allowed) == 1:
        return SupervisorDecision(next_agent=allowed[0], reason="当前阶段只有一项允许的后续操作")

    messages = [
        ("system", SUPERVISOR_ROUTE_PROMPT),
        ("human", json.dumps({
            "request": state["request"],
            "decision_point": state["next_step"],
            "allowed_next_agents": allowed,
            "research_rounds": state.get("research_rounds", 0),
            "source_count": len(state.get("sources", [])),
            "research": (state.get("research") or "")[:1200],
            "draft_plan": (state.get("draft_plan") or "")[:1200],
            "open_questions": state.get("open_questions", [])[:5],
        }, ensure_ascii=False)),
    ]
    model = create_model().with_structured_output(SupervisorDecision, method="function_calling")
    for attempt in range(2):
        try:
            decision = SupervisorDecision.model_validate(model.invoke(messages))
            if decision.next_agent in allowed:
                return decision
        except (ValidationError, OutputParserException):
            pass
        # 模型可能给出合法节点名，却不在当前状态允许的集合里。
        logger.warning("Supervisor 路由无效：task_id=%s，尝试=%s/2", state["task_id"], attempt + 1)
        messages.append(("human", f"上次路由无效。只能从 {allowed} 中选择一个 next_agent。"))
    return SupervisorDecision(next_agent=allowed[0], reason="模型路由无效，采用安全默认步骤")
