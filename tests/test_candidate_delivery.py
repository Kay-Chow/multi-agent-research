"""隔离候选生成服务的边界测试；不访问真实仓库或模型。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import candidate_delivery
from app.agents.coder import CodePatchDraft
from app.workflow.state import initial_state
from app.tools.workspace import WorkspaceError


class CandidateDeliveryTests(unittest.TestCase):
    """候选代码只写隔离工作区，路径不一致时不保存待审 diff。"""

    def setUp(self):
        self.state = initial_state("修改反馈接口", "code_change", "C:/demo")
        self.state["repo_context"] = {"root": "C:/demo", "head": "a" * 40, "files": []}
        self.state["change_plan"] = "# 文件级变更计划"
        self.output_root = Path(tempfile.gettempdir()) / "candidate-test-outputs"

    def test_candidate_collects_validated_files_and_diff(self):
        hashes = [{"path": "app/main.py", "before_sha256": "b" * 64,
                   "after_sha256": "c" * 64}]
        draft = CodePatchDraft(
            edits=[{"path": "app/main.py", "content": "value = 2\n"}],
            summary="  修改接口  ", risks=[],
        )
        with patch.object(candidate_delivery, "planned_paths", return_value={"app/main.py"}), patch.object(
            candidate_delivery, "prepare_workspace",
            return_value=("data/workspaces/task/attempt-0", {"app/main.py": "value = 1\n"}),
        ), patch.object(candidate_delivery, "generate_code_patch", return_value=draft), patch.object(
            candidate_delivery, "apply_edits",
            return_value=("data/workspaces/task/attempt-0", "--- a/app/main.py\n", hashes),
        ) as apply, patch.object(
            candidate_delivery, "write_diff", return_value="outputs/task/diff-attempt-0.patch",
        ) as write:
            result = candidate_delivery.generate_candidate(self.state, self.output_root)
        self.assertEqual(result.summary, "修改接口")
        self.assertEqual(result.file_hashes, hashes)
        self.assertEqual(result.diff_path, "outputs/task/diff-attempt-0.patch")
        self.assertEqual(apply.call_args.args[3], [{"path": "app/main.py", "content": "value = 2\n"}])
        write.assert_called_once_with(self.output_root, self.state["task_id"], 0, result.diff)

    def test_workspace_mismatch_does_not_save_diff(self):
        draft = CodePatchDraft(
            edits=[{"path": "app/main.py", "content": "value = 2\n"}],
            summary="修改接口", risks=[],
        )
        with patch.object(candidate_delivery, "planned_paths", return_value={"app/main.py"}), patch.object(
            candidate_delivery, "prepare_workspace", return_value=("workspace-a", {}),
        ), patch.object(candidate_delivery, "generate_code_patch", return_value=draft), patch.object(
            candidate_delivery, "apply_edits", return_value=("workspace-b", "diff", []),
        ), patch.object(candidate_delivery, "write_diff") as write:
            with self.assertRaisesRegex(WorkspaceError, "路径不一致"):
                candidate_delivery.generate_candidate(self.state, self.output_root)
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
