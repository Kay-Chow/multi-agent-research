"""在隔离工作区生成待审核代码候选；不测试、提交或改动原仓库。"""

from dataclasses import dataclass
from pathlib import Path

from app.agents.coder import generate_code_patch
from app.services.artifact_store import write_diff
from app.tools.workspace import WorkspaceError, apply_edits, planned_paths, prepare_workspace
from app.workflow.state import AgentState, FileHash


@dataclass(frozen=True)
class CandidateResult:
    """一次代码尝试的产物，供 Graph 写入 checkpoint 并交人工审核。"""

    workspace_path: str
    diff_path: str
    diff: str
    file_hashes: list[FileHash]
    summary: str


def generate_candidate(state: AgentState, output_root: Path) -> CandidateResult:
    """复位隔离区、取得模型候选、校验文件并保存统一 diff。

    同一 Coder 节点若因超时重跑，先从已检查的仓库基线重新准备工作区，
    不沿用上次未完成的候选文件。所有写入都限制在隔离区和任务输出目录。
    """

    context = state["repo_context"]
    plan = state["change_plan"]
    if context is None or not plan:
        raise ValueError("缺少仓库基线或文件级变更计划")
    attempt = state.get("code_revision_count", 0)
    paths = planned_paths(plan)
    workspace, baseline = prepare_workspace(context, state["task_id"], attempt)
    draft = generate_code_patch(state, baseline, paths)
    actual_workspace, diff, hashes = apply_edits(
        context, state["task_id"], attempt,
        [edit.model_dump() for edit in draft.edits], paths,
    )
    if workspace != actual_workspace:
        raise WorkspaceError("隔离工作区路径不一致")
    diff_path = write_diff(output_root, state["task_id"], attempt, diff)
    return CandidateResult(
        workspace_path=workspace,
        diff_path=diff_path,
        diff=diff,
        file_hashes=hashes,
        summary=draft.summary.strip(),
    )
