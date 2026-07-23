"""全局日志测试：验证 import 不劫持 stdout，且 install/restore 行为正确。"""

import sys

from mobo.common.logging import GlobalLogger, logger


def test_logger_is_singleton():
    assert GlobalLogger() is logger


def test_import_does_not_hijack_stdout():
    """仅 import / 使用 logger 不应改变 sys.stdout。"""
    original = sys.stdout
    logger.info("some log without redirect")
    assert sys.stdout is original


def test_install_and_restore_stdout_redirect():
    original = sys.stdout
    try:
        logger.install_stdout_redirect()
        assert sys.stdout is logger
        assert sys.stderr is logger
    finally:
        logger.restore_stdout()
    assert sys.stdout is original


def test_restore_is_idempotent():
    original = sys.stdout
    logger.restore_stdout()  # 未安装时调用应安全
    assert sys.stdout is original


def test_write_filters_blank(monkeypatch):
    """write 对空白消息不产生日志记录。"""
    recorded = []
    monkeypatch.setattr(logger.logger, "info", lambda msg: recorded.append(msg))
    logger.write("   \n")
    logger.write("hello")
    assert recorded == ["hello"]
