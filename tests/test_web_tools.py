"""公网搜索适配层测试：不联网，只验证 SDK 异常与结果过滤。"""

import os
import unittest
from unittest.mock import patch

from requests import ConnectionError as RequestsConnectionError
from tavily.errors import TimeoutError as TavilyTimeoutError, UsageLimitExceededError

from app.tools import web_search as web_tools


class WebToolTests(unittest.TestCase):
    """确保 Tavily 的多种失败都变成 Researcher 可识别的 WebSearchError。"""

    def test_sdk_and_network_errors_are_wrapped(self):
        """SDK 自定义异常不继承 requests.RequestException，也必须被捕获。"""

        errors = [
            TavilyTimeoutError(20),
            UsageLimitExceededError("额度不足"),
            RequestsConnectionError("连接失败"),
        ]
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            for error in errors:
                with self.subTest(error=type(error).__name__), patch.object(
                    web_tools, "TavilyClient"
                ) as client:
                    client.return_value.search.side_effect = error
                    with self.assertRaises(web_tools.WebSearchError):
                        web_tools.search_web("反馈列表筛选")


if __name__ == "__main__":
    unittest.main()
