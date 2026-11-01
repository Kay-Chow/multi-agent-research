"""直接测试阶段 3 的隔离文件逻辑，而不调用图或大模型。

与 test_graph.py 中 mock 掉工作区函数不同，本文件真正调用
prepare_workspace / apply_edits / verify_workspace；但仓库基线是假的，
写入位置是临时目录，不会修改真实演示仓库。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import workspace_tools as workspace


class WorkspaceTests(unittest.TestCase):
    """验证模型候选文件从“计划路径”到“可审核 diff”的安全边界。"""

    def setUp(self):
        """为每个测试准备独立的假仓库快照和临时工作区。

        verify_repository 被替换为固定文本，所以测试不依赖 Git 状态；
        工作区函数本身没有被替换，仍会真实创建、读取、检查临时文件。
        """

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        # 模拟允许读取的四个文件；本测试的所有“原文件”都来自这个字典。
        self.baseline = {
            "README.md": "# 演示\n",
            "requirements.txt": "fastapi\n",
            "app/main.py": "value = 1\n",
            "tests/test_feedback.py": "def test_old():\n    assert True\n",
        }
        self.context = {"root": "C:/demo", "head": "a" * 40, "files": []}
        # 把工作区根目录改到临时目录，避免碰项目下的真实 data/workspaces。
        for target, value in [
            ("PROJECT_ROOT", root), ("WORKSPACE_ROOT", root / "data" / "workspaces"),
        ]:
            item = patch.object(workspace, target, value)
            item.start()
            self.addCleanup(item.stop)
        # 仅替换“读取真实仓库”这一步；其余工作区逻辑保持原样。
        verify = patch.object(workspace, "verify_repository", return_value=self.baseline)
        verify.start()
        self.addCleanup(verify.stop)
        self.task_id = "a" * 32

    def test_planned_paths_only_come_from_target_section(self):
        """只有计划的“目标文件”小节能授予修改范围。

        风险说明里虽出现 README.md，但它不应被当作本次可修改文件。
        """

        plan = "# 计划\n\n## 目标文件\n- `app/main.py`：修改接口\n\n## 风险\n- `README.md` 不在计划中\n"
        self.assertEqual(workspace.planned_paths(plan), {"app/main.py"})

    def test_rejects_forbidden_plan_path(self):
        """即使路径写在“目标文件”中，也不能越过静态白名单修改 .env。"""

        plan = "## 目标文件\n- `.env`：写入密钥\n"
        with self.assertRaises(ValueError):
            workspace.planned_paths(plan)

    def test_apply_diff_and_verify_hashes(self):
        """走通正常链路：复制基线、应用候选文件、生成 diff、核对哈希。"""

        # prepare_workspace 复制假仓库文件，此时原值仍为 value = 1。
        relative, _ = workspace.prepare_workspace(self.context, self.task_id, 0)
        self.assertTrue(relative.endswith("attempt-0"))
        # apply_edits 只接受已规划的 app/main.py，返回人工可看的统一 diff。
        _, diff, hashes = workspace.apply_edits(
            self.context, self.task_id, 0,
            [{"path": "app/main.py", "content": "value = 2\n"}], {"app/main.py"},
        )
        self.assertIn("-value = 1", diff)
        self.assertIn("+value = 2", diff)
        # 审核前再次读取临时文件，确认内容仍与生成时记录的哈希一致。
        workspace.verify_workspace(self.context, self.task_id, 0, hashes)

    def test_rejects_unplanned_and_invalid_python_without_writing(self):
        """未规划的文件或 Python 语法错误都应被拒绝，且不留下部分修改。"""

        workspace.prepare_workspace(self.context, self.task_id, 0)
        # 即使 README.md 属于只读文件清单，也不在本次 expected_paths 中。
        with self.assertRaises(workspace.WorkspaceError):
            workspace.apply_edits(
                self.context, self.task_id, 0,
                [{"path": "README.md", "content": "# 新文档\n"}], {"app/main.py"},
            )
        # AST 只检查语法，不执行候选 Python 代码；语法错误必须写入前发现。
        with self.assertRaisesRegex(workspace.WorkspaceError, "语法错误"):
            workspace.apply_edits(
                self.context, self.task_id, 0,
                [{"path": "app/main.py", "content": "def broken(:\n"}], {"app/main.py"},
            )
        # 两种失败之后，工作区里的 app/main.py 仍应是原始基线内容。
        self.assertEqual(
            (workspace.PROJECT_ROOT / "data/workspaces" / self.task_id / "attempt-0/app/main.py").read_text(),
            self.baseline["app/main.py"],
        )

    def test_rejects_empty_or_oversized_diff(self):
        """没有实际变化或 diff 超过体积上限时，都不产生可审核的改动。"""

        workspace.prepare_workspace(self.context, self.task_id, 0)
        # 新旧内容完全相同，不能把“没有改动”伪装成一次代码交付。
        with self.assertRaisesRegex(workspace.WorkspaceError, "没有实际改动"):
            workspace.apply_edits(
                self.context, self.task_id, 0,
                [{"path": "app/main.py", "content": "value = 1\n"}], {"app/main.py"},
            )
        # 临时把上限压到 10 字节，用极小输入稳定触发“diff 过大”分支。
        with patch.object(workspace, "MAX_DIFF_BYTES", 10):
            with self.assertRaisesRegex(workspace.WorkspaceError, "diff"):
                workspace.apply_edits(
                    self.context, self.task_id, 0,
                    [{"path": "app/main.py", "content": "value = 2\n"}], {"app/main.py"},
                )

    def test_tampered_workspace_cannot_be_approved(self):
        """生成 diff 后若隔离文件被人改写，审核前哈希校验必须失败。"""

        workspace.prepare_workspace(self.context, self.task_id, 0)
        _, _, hashes = workspace.apply_edits(
            self.context, self.task_id, 0,
            [{"path": "app/main.py", "content": "value = 2\n"}], {"app/main.py"},
        )
        target = workspace.PROJECT_ROOT / "data/workspaces" / self.task_id / "attempt-0/app/main.py"
        # 模拟人工审核等待期间发生“文件调包”；保存的 after_sha256 不变。
        target.write_text("value = 999\n", encoding="utf-8")
        with self.assertRaisesRegex(workspace.WorkspaceError, "审核期间变化"):
            workspace.verify_workspace(self.context, self.task_id, 0, hashes)


if __name__ == "__main__":
    unittest.main()
