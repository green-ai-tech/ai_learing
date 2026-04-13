"""屏蔽 transformers 库的 __path__ 警告（必须在所有 import 之前调用）"""

import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.CRITICAL)
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"


class _QuietStderr:
    """只过滤 __path__ 相关警告的 stderr 包装器"""

    def __init__(self, original):
        self._original = original

    def write(self, text):
        if "__path__" not in text and "future versions" not in text:
            self._original.write(text)

    def flush(self, *args, **kwargs):
        self._original.flush()

    def isatty(self):
        return self._original.isatty()


def apply():
    """启用警告屏蔽（在文件最开头调用）"""
    sys.stderr = _QuietStderr(sys.stderr)


def restore():
    """恢复原始 stderr（在关键 import 完成后调用）"""
    if isinstance(sys.stderr, _QuietStderr):
        sys.stderr = sys.stderr._original
