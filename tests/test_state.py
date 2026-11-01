"""检查聊天入口和显式任务入口是否建立了不同的初始状态。"""

import unittest

from app.state import initial_chat_state, initial_state


class StateTests(unittest.TestCase):
    """防止一句含糊聊天被误当成正式需求送去检索。"""

    def test_chat_starts_without_formal_request(self):
        """聊天先保存原话，待 Intake 判断 ready 后才填写 request。"""

        state = initial_chat_state("  我想做一个需求  ")
        self.assertEqual(state["intake_status"], "collecting")
        self.assertEqual(state["next_step"], "intake")
        self.assertEqual(state["request"], "")
        self.assertEqual(state["conversation"], [
            {"role": "user", "text": "我想做一个需求"}
        ])

    def test_explicit_task_skips_chat_intake(self):
        """CLI start 和 HTTP 创建接口已提供完整需求，只需 Intake 确认模式。"""

        state = initial_state("  调研反馈筛选，只写方案  ", "proposal_only")
        self.assertEqual(state["intake_status"], "ready")
        self.assertEqual(state["next_step"], "intake")
        self.assertEqual(state["request"], "调研反馈筛选，只写方案")
        self.assertEqual(state["requested_mode"], "proposal_only")
        self.assertEqual(state["conversation"], [])


if __name__ == "__main__":
    unittest.main()
