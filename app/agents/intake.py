"""接待与任务模式判断：澄清需求、选择只读工具或开始正式任务。"""

import json
import logging
from typing import Literal, Self

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agents.model_client import create_model
from app.agents.prompts import INTAKE_MODE_PROMPT, INTAKE_PROMPT
from app.workflow.state import AgentState, TaskMode


logger = logging.getLogger(__name__)
MAX_INTAKE_DECISION_ATTEMPTS = 2


class IntakeDecisionError(Exception):
    """接待模型连续给出无效结构化决定；由图节点转成可恢复的澄清。"""


class ModeDecision(BaseModel):
    """Intake 的结构化模式判断：是否需要进入代码修改流程。"""

    mode: TaskMode
    reason: str = Field(min_length=1)

class IntakeDecision(BaseModel):
    """探索接待的单步决定：回答/澄清、请求只读工具，或开始正式任务。

    字段类型由 Pydantic 校验；model_validator 再检查字段之间的关系。
    这样模型返回矛盾决定时，错误在入口处集中暴露，而不是留给各节点猜测。
    """

    ready: bool
    reply: str = Field(default="", max_length=600, description="回复用户的文本；开始正式任务时可为空")
    request: str | None = Field(default=None, max_length=1000)
    tool_name: Literal[
        "list_projects", "inspect_project", "read_file", "search_kb", "search_web"
    ] | None = Field(default=None, description="需要查资料时选择一个只读工具；否则为空")
    tool_query: str | None = Field(default=None, max_length=200, description="知识库或网页检索词")
    tool_path: str | None = Field(default=None, max_length=500, description="仓库名/路径或已选仓库内相对文件路径")

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        """一个决定只能选工具、开始任务或回复用户三条路径中的一条。"""

        if self.tool_name is not None:
            # tool_name 的可选值已由 Literal 限定，这里只检查跨字段约束。
            if self.ready or self.request:
                raise ValueError("调用工具时不能同时开始正式任务")
            if self.tool_name in {"search_kb", "search_web"} and not (self.tool_query or "").strip():
                raise ValueError("检索工具缺少查询词")
            if self.tool_name in {"inspect_project", "read_file"} and not (self.tool_path or "").strip():
                raise ValueError("仓库工具缺少路径")
            return self

        if self.ready:
            # 已有可执行需求就能开始；纯提示性的 reply 由图节点补默认文案。
            if not (self.request or "").strip():
                raise ValueError("开始正式任务时必须给出可执行的任务描述")
            return self
        if not self.reply.strip():
            raise ValueError("未调用工具且未开始任务时必须给出用户可见的回复")
        if self.request:
            raise ValueError("仍需澄清时不能提前生成任务")
        return self

def intake_requirement(state: AgentState) -> IntakeDecision:
    """根据对话选择下一步；无效模型输出只重试一次，不执行任何工具。

    只捕获结构化解析/校验错误。网络超时、额度耗尽等调用故障不应伪装成
    “用户没说清楚”；真正的工具调用仍由图内的 discovery_tool 节点负责。
    """

    decision_model = create_model().with_structured_output(
        IntakeDecision, method="function_calling"
    )
    messages = [
        ("system", INTAKE_PROMPT),
        ("human", json.dumps({
            "conversation": state["conversation"][-12:],
            "observations": state.get("discovery_observations", [])[-5:],
            "selected_repo": state.get("repo_root"),
            "remaining_lookups": max(0, 4 - state.get("discovery_lookup_count", 0)),
        }, ensure_ascii=False)),
    ]
    last_error: ValidationError | OutputParserException | None = None
    for attempt in range(1, MAX_INTAKE_DECISION_ATTEMPTS + 1):
        try:
            # with_structured_output 通常已返回 IntakeDecision；再校验一次可防御
            # 返回 dict/None 的模型适配器，且不依赖某个服务商的解析实现。
            return IntakeDecision.model_validate(decision_model.invoke(messages))
        except (ValidationError, OutputParserException) as exc:
            last_error = exc
            # 只记录出错字段与通用错误说明，不记录原始模型输出、
            # 用户对话或文件内容。自定义校验错误的说明均为固定文案。
            if isinstance(exc, ValidationError):
                error_codes = "; ".join(
                    f"{'.'.join(map(str, error['loc'])) or 'decision'}:"
                    f"{error['type']}({error['msg']})"
                    for error in exc.errors(include_input=False)
                )[:300]
            else:
                error_codes = type(exc).__name__
            logger.warning(
                "接待模型决定无效：task_id=%s，尝试=%s/%s，校验=%s",
                state["task_id"], attempt, MAX_INTAKE_DECISION_ATTEMPTS,
                error_codes,
            )
            if attempt < MAX_INTAKE_DECISION_ATTEMPTS:
                # 下次调用时给出纠错指令；不无限重试，也不执行未校验的工具决定。
                messages.append((
                    "human",
                    "上一条决定不符合结构化字段或字段关系。请重新给出一个决定："
                    "调用工具时 ready=false、request=null，并填写该工具所需参数；"
                    "不调用工具且 ready=false 时填写 reply；"
                    "ready=true 时必须填写 request，reply 可为空。",
                ))

    # 循环只会因决定有效而 return，或因达到重试上限而走到这里。
    raise IntakeDecisionError("接待模型连续返回无效决定") from last_error

def decide_mode(request: str) -> ModeDecision:
    """在 Intake 确认需求时判断模式；Supervisor 只读取已确定的结果。"""

    return create_model().with_structured_output(
        ModeDecision, method="function_calling"
    ).invoke([("system", INTAKE_MODE_PROMPT), ("human", request)])
