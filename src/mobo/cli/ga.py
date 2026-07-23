"""遗传算法（NSGA-II）优化命令行入口。"""

from mobo.common.logging import logger
from mobo.optimization.ga.run import NSGA2_run


def main() -> int:
    logger.install_stdout_redirect()
    NSGA2_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
