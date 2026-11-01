"""受限的仓库发现与只读检查；代码阶段仍只接收小范围文件快照。"""

import hashlib
import os
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

from dotenv import load_dotenv

from app.state import RepoContext, RepoFile


# 可以在启动前通过环境变量改为另一个项目目录；模型不能自行扩大此范围。
# 当前只允许选择该目录的直接子目录作为仓库，不扫描整块磁盘。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
PROJECTS_ROOT = Path(os.getenv("AGENT_PROJECTS_ROOT") or Path(__file__).resolve().parents[2])
# 代码生成阶段暂时仍限于这个演示工程布局；接待阶段可安全读取其他已选仓库的文本。
READ_ALLOWLIST = ("README.md", "requirements.txt", "app/main.py", "tests/test_feedback.py")
CHANGE_ALLOWLIST = frozenset({"app/main.py", "tests/test_feedback.py", "README.md"})
MAX_FILE_BYTES = 16_000
MAX_TOTAL_BYTES = 40_000
DISCOVERY_FILE_BYTES = 4_000
DISCOVERY_EXTENSIONS = frozenset({".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"})


class RepoAccessError(ValueError):
    """仓库路径、文件或 Git 基线不符合受限只读访问要求。"""


def _git(root: Path, *args: str) -> str:
    """只允许程序写死的 Git 只读参数；绝不拼接 Shell 命令。"""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepoAccessError(f"Git 只读检查失败：{exc}") from exc
    if result.returncode != 0:
        raise RepoAccessError(f"Git 只读检查失败：{result.stderr.strip()[:300]}")
    return result.stdout


def _projects_root() -> Path:
    """取得开发者允许的项目父目录；拒绝用链接偷偷扩大读取范围。"""

    if PROJECTS_ROOT.is_symlink() or PROJECTS_ROOT.is_junction():
        raise RepoAccessError("项目目录不能是符号链接或 junction")
    try:
        root = PROJECTS_ROOT.resolve(strict=True)
    except OSError as exc:
        raise RepoAccessError(f"找不到项目目录：{PROJECTS_ROOT}") from exc
    if not root.is_dir():
        raise RepoAccessError("项目目录不是文件夹")
    return root


def checked_root(path: str | Path) -> Path:
    """只允许项目目录下的一个真实 Git 仓库；用户给路径也必须过同一边界。"""

    projects = _projects_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = projects / candidate
    if candidate.is_symlink() or candidate.is_junction():
        raise RepoAccessError("仓库根目录不能是符号链接或 junction")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise RepoAccessError(f"找不到仓库：{candidate}") from exc
    if root.parent != projects or not root.is_dir():
        raise RepoAccessError(f"仓库必须是项目目录 {projects} 下的直接子目录")

    top_level = _git(root, "rev-parse", "--show-toplevel").strip()
    if Path(top_level).resolve(strict=True) != root:
        raise RepoAccessError("所选路径不是 Git 仓库根目录")
    return root


def list_repositories() -> list[dict[str, str]]:
    """只列一层仓库名称与路径，不读源码；最多展示 30 个候选。"""

    projects = _projects_root()
    result: list[dict[str, str]] = []
    for item in sorted(projects.iterdir(), key=lambda path: path.name.lower()):
        if item.is_symlink() or item.is_junction() or not item.is_dir() or not (item / ".git").exists():
            continue
        result.append({"name": item.name, "path": str(item)})
        if len(result) >= 30:
            break
    return result


def read_repository_text(repo_root: str | Path, relative_path: str) -> str:
    """只读已跟踪的普通文本，限制大小并拒绝隐藏/密钥路径与链接。"""

    root = checked_root(repo_root)
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise RepoAccessError("文件路径无效")
    if PureWindowsPath(relative_path).drive or relative_path.startswith("/"):
        raise RepoAccessError("不能读取绝对文件路径")
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        raise RepoAccessError("文件路径包含隐藏目录或越界片段")
    if any(word in relative_path.lower() for word in ("secret", "token", "credential", "password", "private")):
        raise RepoAccessError("不能读取可能包含凭据的文件")
    if PurePosixPath(relative_path).suffix.lower() not in DISCOVERY_EXTENSIONS:
        raise RepoAccessError("接待阶段只允许读取常见文本文件")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise RepoAccessError("文件路径经过链接或 junction")
    if not current.is_file() or not current.resolve(strict=True).is_relative_to(root):
        raise RepoAccessError("文件不存在或越出仓库")
    # 只读已跟踪文件，避免模型看到本地临时文件、未提交草稿或凭据。
    _git(root, "ls-files", "--error-unmatch", "--", relative_path)
    raw = current.read_bytes()
    if len(raw) > DISCOVERY_FILE_BYTES:
        raise RepoAccessError("文件超过接待阶段读取大小限制")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepoAccessError("文件不是 UTF-8 文本") from exc


def inspect_project(repo_root: str | Path) -> dict[str, str]:
    """选定仓库并读取最多一份 README 摘要，供接待 Agent 回答项目问题。"""

    root = checked_root(repo_root)
    try:
        summary = read_repository_text(root, "README.md")[:2000]
    except RepoAccessError:
        summary = "没有可读取的 README.md；不能据此推断项目功能。"
    return {"name": root.name, "path": str(root), "summary": summary}


def checked_file(root: Path, relative_path: str) -> Path:
    """同时校验词法路径和实际解析路径，防止 `..`、链接、junction 越界。"""

    if (
        not isinstance(relative_path, str)
        or "\\" in relative_path
        or PureWindowsPath(relative_path).drive
        or relative_path.startswith("/")
    ):
        raise RepoAccessError(f"不允许的文件路径：{relative_path}")
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise RepoAccessError(f"不允许的文件路径：{relative_path}")
    if relative_path not in READ_ALLOWLIST:
        raise RepoAccessError(f"文件不在只读白名单：{relative_path}")

    current = root
    for part in parts:
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise RepoAccessError(f"文件路径经过链接或 junction：{relative_path}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise RepoAccessError(f"白名单文件不存在：{relative_path}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RepoAccessError(f"文件越出仓库或不是普通文件：{relative_path}")
    return resolved


def inspect_repository(repo_root: str | Path) -> tuple[RepoContext, dict[str, str]]:
    """只读检查仓库并建立可复核的基线快照。

    仓库必须干净；仅读取固定白名单中的少量 UTF-8 文件。
    `RepoContext` 存 HEAD 与哈希，可进入 checkpoint；完整源码只作为
    本次节点的临时输入，避免把大量代码持久化到任务状态里。
    """

    root = checked_root(repo_root)
    # --porcelain=v1 输出稳定，含未暂存、已暂存及未跟踪文件；脏仓库先停止。
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.strip():
        raise RepoAccessError("所选仓库存在未提交变更，请人工检查后再开始新任务")
    head = _git(root, "rev-parse", "HEAD").strip()

    files: list[RepoFile] = []
    contents: dict[str, str] = {}
    total = 0
    for relative_path in READ_ALLOWLIST:
        path = checked_file(root, relative_path)
        raw = path.read_bytes()
        if len(raw) > MAX_FILE_BYTES:
            raise RepoAccessError(f"文件超过单文件大小限制：{relative_path}")
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise RepoAccessError("仓库上下文超过总大小限制")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepoAccessError(f"文件不是 UTF-8 文本：{relative_path}") from exc
        files.append({
            "path": relative_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines": len(text.splitlines()),
        })
        contents[relative_path] = text
    return {"root": str(root), "head": head, "files": files}, contents


def verify_repository(expected: RepoContext) -> dict[str, str]:
    """重新读取仓库并与旧快照逐项比较，避免使用过期基线。

    阶段 2 规划、阶段 3 建立工作区和审核时都会复查；HEAD、工作树
    或白名单文件只要有变化，就停止而不是默默覆盖人的修改。
    """

    actual, contents = inspect_repository(expected["root"])
    if actual != expected:
        raise RepoAccessError("所选仓库在检查后发生变化，请重新启动任务")
    return contents


def validate_change_path(relative_path: str) -> None:
    """Change Planner 只能提议修改已知业务文件，不能改密钥或 CI 配置。"""

    if relative_path not in CHANGE_ALLOWLIST:
        raise RepoAccessError(f"变更文件不在白名单：{relative_path}")
