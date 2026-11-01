"""阶段 3：固定文件快照的隔离工作区、补丁校验与完整性检查。"""

import ast
import difflib
import hashlib
import re
from pathlib import Path

from app.config import PROJECT_ROOT
from app.tools.repository import READ_ALLOWLIST, RepoAccessError, validate_change_path, verify_repository
from app.workflow.state import FileHash, RepoContext


WORKSPACE_ROOT = PROJECT_ROOT / "data" / "workspaces"
MAX_EDIT_FILES = 3
MAX_FILE_BYTES = 16_000
MAX_DIFF_BYTES = 24_000
MAX_DIFF_LINES = 400


class WorkspaceError(ValueError):
    """候选代码、目标路径或隔离工作区不符合阶段 3 的安全约束。"""


def planned_paths(change_plan: str) -> set[str]:
    """从文件级计划中提取允许修改的文件集合。

    只读取 `## 目标文件` 下的反引号路径，计划其他章节出现的文件名
    不会自动获得修改权限；最终仍要经过静态变更白名单校验。
    """

    match = re.search(r"(?ms)^## 目标文件\s*\n(.*?)(?=^## |\Z)", change_plan)
    if not match:
        raise WorkspaceError("文件级变更计划缺少目标文件段")
    paths = re.findall(r"(?m)^- `([^`]+)`：", match.group(1))
    if not paths or len(paths) > MAX_EDIT_FILES or len(paths) != len(set(paths)):
        raise WorkspaceError("文件级变更计划的目标文件数量或格式无效")
    for path in paths:
        validate_change_path(path)
    return set(paths)


def _workspace(task_id: str, attempt: int) -> tuple[Path, str]:
    """计算并建立一次代码尝试专用的隔离目录。

    task_id 必须是系统生成的十六进制 ID，attempt 只能在有限次数内递增。
    同时拒绝路径中的链接和 junction，防止目录看似在工作区内、实际指向别处。
    """

    if not re.fullmatch(r"[0-9a-f]{32}", task_id) or type(attempt) is not int or not 0 <= attempt <= 2:
        raise WorkspaceError("任务 ID 或代码修改次数无效")
    relative = f"data/workspaces/{task_id}/attempt-{attempt}"
    for parent in (WORKSPACE_ROOT.parent, WORKSPACE_ROOT, WORKSPACE_ROOT / task_id):
        if parent.exists() and (parent.is_symlink() or parent.is_junction()):
            raise WorkspaceError("工作区路径经过链接或 junction")
    path = PROJECT_ROOT / relative
    if path.exists() and (path.is_symlink() or path.is_junction()):
        raise WorkspaceError("工作区不能是链接或 junction")
    path.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(WORKSPACE_ROOT.resolve()):
        raise WorkspaceError("工作区越出固定根目录")
    return path, relative


def _check_workspace_entries(workspace: Path) -> None:
    """检查隔离目录只含固定快照文件，不含额外文件或链接。

    这一步在准备、写入和审核时都会调用；如果有人在暂停审核期间
    往工作区塞进新文件，后续节点不会把它误认为 Coder 的合法产物。
    """

    allowed_files = set(READ_ALLOWLIST)
    allowed_dirs = {"app", "tests"}
    for entry in workspace.rglob("*"):
        relative = entry.relative_to(workspace).as_posix()
        if entry.is_symlink() or entry.is_junction():
            raise WorkspaceError(f"工作区包含链接：{relative}")
        if entry.is_dir():
            if relative not in allowed_dirs:
                raise WorkspaceError(f"工作区包含额外目录：{relative}")
        elif not entry.is_file() or relative not in allowed_files:
            raise WorkspaceError(f"工作区包含额外文件：{relative}")


def prepare_workspace(context: RepoContext, task_id: str, attempt: int) -> tuple[str, dict[str, str]]:
    """从已选原仓库的只读快照重建本次隔离工作区。

    先确认 Git HEAD、仓库干净状态和白名单文件哈希都未变化，再复制文件。
    节点重试时也从同一基线重建，避免在半写入结果上继续叠加修改。
    返回的相对路径用于 State，源码字典只传给当前 Coder 节点。
    """

    baseline = verify_repository(context)
    workspace, relative = _workspace(task_id, attempt)
    _check_workspace_entries(workspace)
    for file_path in READ_ALLOWLIST:
        target = workspace / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # 所有路径都来自静态白名单，但仍防止本地并发把目录替换成链接。
        if target.is_symlink() or target.is_junction() or target.parent.is_symlink() or target.parent.is_junction():
            raise WorkspaceError(f"工作区文件路径经过链接：{file_path}")
        target.write_bytes(baseline[file_path].encode("utf-8"))
    return relative, baseline


def apply_edits(
    context: RepoContext,
    task_id: str,
    attempt: int,
    edits: list[dict[str, str]],
    expected_paths: set[str],
) -> tuple[str, str, list[FileHash]]:
    """校验 Coder 的文件全文，并只写进隔离工作区。

    候选文件集合必须与已批准的计划完全一致；逐文件检查大小、Python
    语法和实际变化，最后检查整个 unified diff 的大小。准备工作区时
    可能创建目录；候选文件正文要等全部校验通过才写入，并返回 diff
    与前后哈希供人工审核。
    """

    if not edits or len(edits) > MAX_EDIT_FILES:
        raise WorkspaceError("代码修改文件数量超限")
    workspace, relative = _workspace(task_id, attempt)
    _check_workspace_entries(workspace)
    baseline = verify_repository(context)
    paths = [edit["path"] for edit in edits]
    for path in paths:
        validate_change_path(path)
    if len(paths) != len(set(paths)) or set(paths) != expected_paths:
        raise WorkspaceError("代码修改文件必须与已批准的文件级计划完全一致")

    # 写候选内容前，四份隔离区文件必须仍与原仓库基线逐字一致。
    for file_path in READ_ALLOWLIST:
        target = workspace / file_path
        if not target.is_file() or target.is_symlink() or target.read_bytes().decode("utf-8") != baseline[file_path]:
            raise WorkspaceError(f"工作区基线被改动：{file_path}")

    # 先在内存中完成所有校验和 diff 计算，避免前一个文件已写、后一个失败。
    generated: dict[str, str] = {}
    hashes: list[FileHash] = []
    diff_parts: list[str] = []
    for edit in sorted(edits, key=lambda item: item["path"]):
        path = edit["path"]
        before = baseline[path]
        # 保留基线文件的换行风格，减少与业务改动无关的整文件 diff。
        newline = "\r\n" if "\r\n" in before else "\n"
        normalized = edit["content"].replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
        content = normalized.replace("\n", newline)
        raw = content.encode("utf-8")
        if len(raw) > MAX_FILE_BYTES:
            raise WorkspaceError(f"生成文件超过大小限制：{path}")
        # AST 只解析语法，不 import，也不会运行生成的 Python 代码。
        if path.endswith(".py"):
            try:
                ast.parse(content, filename=path)
            except SyntaxError as exc:
                raise WorkspaceError(f"生成的 Python 文件语法错误：{path}:{exc.lineno}") from exc
        if content == before:
            raise WorkspaceError(f"计划文件没有实际改动：{path}")
        diff_parts.extend(difflib.unified_diff(
            before.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
        hashes.append({
            "path": path,
            "before_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
            "after_sha256": hashlib.sha256(raw).hexdigest(),
        })
        generated[path] = content

    diff_text = "".join(diff_parts)
    if not diff_text or len(diff_text.encode("utf-8")) > MAX_DIFF_BYTES or len(diff_text.splitlines()) > MAX_DIFF_LINES:
        raise WorkspaceError("diff 为空或超过大小/行数限制")

    # 所有内容校验完成后才写；若进程在写入中断，重试会先从仓库基线重建。
    for path, content in generated.items():
        (workspace / path).write_bytes(content.encode("utf-8"))
    return relative, diff_text, hashes


def verify_workspace(
    context: RepoContext, task_id: str, attempt: int, expected_hashes: list[FileHash]
) -> None:
    """批准 diff 前再次核对原仓库与隔离区所有文件的内容哈希。

    计划修改的文件应匹配 Coder 产出时记录的 after_sha256；未修改文件
    应继续匹配原仓库基线。这样人工看到的 diff 才对应实际待处理文件。
    """

    baseline = verify_repository(context)
    workspace, _ = _workspace(task_id, attempt)
    _check_workspace_entries(workspace)
    after_by_path = {item["path"]: item["after_sha256"] for item in expected_hashes}
    if len(after_by_path) != len(expected_hashes):
        raise WorkspaceError("文件哈希记录重复")
    # 不只检查已修改文件，也检查工作区里的其他基线文件未被偷偷改动。
    for file_path in READ_ALLOWLIST:
        target = workspace / file_path
        if not target.is_file() or target.is_symlink():
            raise WorkspaceError(f"工作区文件丢失或被替换：{file_path}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = after_by_path.get(file_path)
        if expected is None:
            expected = hashlib.sha256(baseline[file_path].encode("utf-8")).hexdigest()
        if actual != expected:
            raise WorkspaceError(f"工作区文件在审核期间变化：{file_path}")
