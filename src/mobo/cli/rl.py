"""强化学习（PPO）优化命令行入口。"""

from mobo.common.logging import logger
from mobo.optimization.rl.run import train_and_optimize


def main() -> int:
    logger.install_stdout_redirect()
    train_and_optimize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
