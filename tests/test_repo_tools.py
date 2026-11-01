"""只读仓库工具的白名单、脏仓库和快照一致性测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.tools import repository as repo_tools


class RepoToolTests(unittest.TestCase):
    """验证用户选定仓库的只读白名单、发现范围与基线变更检测。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.projects = Path(self.temp_dir.name).resolve() / "projects"
        self.projects.mkdir()
        self.root = self.projects / "demo"
        (self.root / ".git").mkdir(parents=True)
        (self.root / "app").mkdir()
        (self.root / "tests").mkdir()
        for name in repo_tools.READ_ALLOWLIST:
            (self.root / name).write_text(f"content of {name}\n", encoding="utf-8")

        root_patch = patch.object(repo_tools, "PROJECTS_ROOT", self.projects)
        root_patch.start()
        self.addCleanup(root_patch.stop)

        def fake_git(root, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(self.root)
            if args == ("rev-parse", "HEAD"):
                return "a" * 40
            if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                return ""
            if args[:3] == ("ls-files", "--error-unmatch", "--"):
                return args[3] + "\n"
            raise AssertionError(f"意外的 Git 调用：{args}")

        git_patch = patch.object(repo_tools, "_git", side_effect=fake_git)
        self.git_mock = git_patch.start()
        self.addCleanup(git_patch.stop)

    def test_inspection_reads_only_allowlist_and_stores_hashes(self):
        context, contents = repo_tools.inspect_repository(self.root)
        self.assertEqual(set(contents), set(repo_tools.READ_ALLOWLIST))
        self.assertEqual(len(context["files"]), len(repo_tools.READ_ALLOWLIST))
        self.assertEqual(len(context["files"][0]["sha256"]), 64)

    def test_path_traversal_and_secret_path_are_rejected(self):
        for path in ("../outside.txt", "C:/outside.txt", ".env", "app\\main.py"):
            with self.subTest(path=path), self.assertRaises(repo_tools.RepoAccessError):
                repo_tools.checked_file(self.root, path)
        with self.assertRaises(repo_tools.RepoAccessError):
            repo_tools.validate_change_path(".github/workflows/ci.yml")

    def test_dirty_repo_is_rejected(self):
        def dirty_git(root, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(self.root)
            if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                return " M app/main.py\n"
            return "a" * 40

        self.git_mock.side_effect = dirty_git
        with self.assertRaisesRegex(repo_tools.RepoAccessError, "未提交变更"):
            repo_tools.inspect_repository(self.root)

    def test_snapshot_detects_file_change(self):
        context, _ = repo_tools.inspect_repository(self.root)
        (self.root / "app/main.py").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(repo_tools.RepoAccessError, "发生变化"):
            repo_tools.verify_repository(context)

    def test_discovery_lists_and_reads_only_selected_repo(self):
        """可发现仓库并读取已跟踪 README；不能越界到相邻仓库或秘密文件。"""

        projects = repo_tools.list_repositories()
        self.assertEqual([item["name"] for item in projects], ["demo"])
        self.assertEqual(repo_tools.inspect_project("demo")["path"], str(self.root))
        self.assertIn("content of README.md", repo_tools.read_repository_text(self.root, "README.md"))
        for path in (".env", "../secret.md", "app/private_key.py", "C:/outside.txt"):
            with self.subTest(path=path), self.assertRaises(repo_tools.RepoAccessError):
                repo_tools.read_repository_text(self.root, path)
        with self.assertRaises(repo_tools.RepoAccessError):
            repo_tools.checked_root(self.projects.parent)


if __name__ == "__main__":
    unittest.main()
