"""任务产物保存与审核前核验的单元测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import artifact_store
from app.tools.workspace import WorkspaceError


class ArtifactStoreTests(unittest.TestCase):
    """验证文件写入幂等、任务目录约束及 diff 防篡改。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_root = Path(self.temp_dir.name) / "outputs"
        self.task_id = "a" * 32

    def test_reports_are_saved_once_without_overwriting_approved_content(self):
        paths = artifact_store.save_reports(self.output_root, self.task_id, "# 调研", "# 方案")
        self.assertEqual(paths, (
            f"outputs/{self.task_id}/research.md", f"outputs/{self.task_id}/proposal.md",
        ))
        self.assertEqual((self.output_root / self.task_id / "research.md").read_text(encoding="utf-8"),
                         "# 调研\n")
        self.assertEqual(artifact_store.save_reports(
            self.output_root, self.task_id, "# 调研", "# 方案",
        ), paths)
        with self.assertRaises(FileExistsError):
            artifact_store.save_reports(self.output_root, self.task_id, "# 被改写的调研", "# 方案")

    def test_invalid_task_id_cannot_escape_output_root(self):
        with self.assertRaisesRegex(ValueError, "任务 ID"):
            artifact_store.save_change_plan(self.output_root, "../other", "计划")
        self.assertFalse(self.output_root.exists())

    def test_diff_must_match_reviewed_content_before_workspace_verification(self):
        diff = "--- a/app/main.py\n+++ b/app/main.py\n"
        path = artifact_store.write_diff(self.output_root, self.task_id, 0, diff)
        self.assertEqual(path, f"outputs/{self.task_id}/diff-attempt-0.patch")
        context = {"root": "C:/demo", "head": "a" * 40, "files": []}
        with patch.object(artifact_store, "verify_workspace") as verify:
            artifact_store.verify_diff_candidate(
                self.output_root, self.task_id, 0, diff, context, [],
            )
            verify.assert_called_once_with(context, self.task_id, 0, [])
            (self.output_root / self.task_id / "diff-attempt-0.patch").write_text(
                "伪造补丁", encoding="utf-8",
            )
            with self.assertRaises(WorkspaceError):
                artifact_store.verify_diff_candidate(
                    self.output_root, self.task_id, 0, diff, context, [],
                )
            verify.assert_called_once()  # 补丁不一致时不能继续核验并批准工作区。


if __name__ == "__main__":
    unittest.main()
