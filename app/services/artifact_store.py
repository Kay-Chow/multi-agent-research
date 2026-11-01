"""任务产物的落盘与核验；节点只调用这里，不实现文件操作细节。"""

import re
from pathlib import Path

from app.tools.workspace import WorkspaceError, verify_workspace
from app.workflow.state import FileHash, RepoContext


def _task_output_dir(output_root: Path, task_id: str) -> Path:
    """定位固定的 outputs/<任务ID>，拒绝链接目录与越界 ID。"""

    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise ValueError("任务 ID 格式不正确")
    directory = output_root / task_id
    if any(path.is_symlink() or path.is_junction() for path in (output_root, directory)):
        raise WorkspaceError("任务输出目录不能是链接或 junction")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_once(path: Path, content: str) -> None:
    """重试时同内容视为成功，不覆盖已有的不同批准版本。"""

    content = content.rstrip() + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    if any(item.is_symlink() or item.is_junction() for item in (path, temp)):
        raise WorkspaceError("任务输出文件不能是链接或 junction")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"已存在不同内容的批准文件：{path}")
        return
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def save_reports(output_root: Path, task_id: str, research: str, proposal: str) -> tuple[str, str]:
    """将获批的调研报告和产品方案各保存一次，返回相对产物路径。"""

    directory = _task_output_dir(output_root, task_id)
    _write_once(directory / "research.md", research)
    _write_once(directory / "proposal.md", proposal)
    return f"outputs/{task_id}/research.md", f"outputs/{task_id}/proposal.md"


def save_change_plan(output_root: Path, task_id: str, plan: str) -> str:
    """保存文件级计划，不接触用户授权的原仓库。"""

    directory = _task_output_dir(output_root, task_id)
    _write_once(directory / "change_plan.md", plan)
    return f"outputs/{task_id}/change_plan.md"


def write_diff(output_root: Path, task_id: str, attempt: int, diff: str) -> str:
    """保存一次隔离尝试的补丁；新 attempt 保留旧补丁供人工对照。"""

    directory = _task_output_dir(output_root, task_id)
    target = directory / f"diff-attempt-{attempt}.patch"
    temp = target.with_suffix(".patch.tmp")
    if any(path.is_symlink() or path.is_junction() for path in (target, temp)):
        raise WorkspaceError("diff 输出文件不能是链接或 junction")
    temp.write_text(diff, encoding="utf-8")
    temp.replace(target)
    return f"outputs/{task_id}/{target.name}"


def verify_diff_candidate(
    output_root: Path, task_id: str, attempt: int, expected_diff: str,
    context: RepoContext | None, file_hashes: list[FileHash],
) -> None:
    """人工批准前重查补丁、原仓库和隔离区，防止审核期间被替换。"""

    if context is None:
        raise WorkspaceError("缺少仓库基线，不能批准 diff")
    diff_file = output_root / task_id / f"diff-attempt-{attempt}.patch"
    if (
        output_root.is_symlink() or output_root.is_junction()
        or diff_file.parent.is_symlink() or diff_file.parent.is_junction()
        or not diff_file.is_file() or diff_file.is_symlink() or diff_file.is_junction()
        or diff_file.read_text(encoding="utf-8") != expected_diff
    ):
        raise WorkspaceError("待审核 diff 文件在审核期间变化")
    verify_workspace(context, task_id, attempt, file_hashes)
