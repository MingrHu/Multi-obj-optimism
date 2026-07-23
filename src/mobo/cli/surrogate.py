"""代理模型训练/评估命令行入口。

替代原 ``SurrogateModel/Interface.py`` 与 ``evaluate_model.py`` 中带 macOS 硬编码
路径的 ``__main__`` 逻辑，改用集中式 :data:`mobo.common.paths.TEST_DIR` 下的示例
数据集 ``simulated.txt``。
"""

import os

from mobo.common.logging import logger
from mobo.common.paths import TEST_DIR
from mobo.surrogate.evaluate import SurrogateModelEvaluator
from mobo.surrogate.interface import Doe_surrogateModel


def train_demo() -> None:
    """使用示例数据集训练一个代理模型（默认 DNN）。"""
    vars_out = ["1", "2", "3", "grain", "load"]
    data_file = str(TEST_DIR / "simulated.txt")
    doe_s = Doe_surrogateModel(data_file, vars_out, n_vars=3)
    doe_s.train_save_model(1)


def evaluate_demo() -> None:
    """使用示例数据集对多种代理模型做交叉验证评估。"""
    vars_out = ["1", "2", "3", "grain", "load"]
    data_file = str(TEST_DIR / "simulated.txt")

    evaluator = SurrogateModelEvaluator(
        data_file=data_file,
        vars_out=vars_out,
        n_vars=3,
        n_splits=5,
        model_params={
            "PRG": {"degree": 2},
            "SVR": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            "RF": {"n_estimators": 300, "n_jobs": -1},
            "KM": {"alpha": 0.1, "n_restarts_optimizer": 10},
            "DNN": {"epochs": 300, "batch_size": 16, "verbose": 0, "patience": 30},
        },
    )

    summaries = evaluator.evaluate(
        models=["PRG", "SVR", "RF", "KM", "DNN"],
        target_indices=[0, 1],
        score_weights=(0.9, 0.1),
    )

    evaluator.save_report(
        summaries,
        text_path=os.path.join(str(TEST_DIR), "evaluate_history_result.txt"),
        json_path=os.path.join(str(TEST_DIR), "evaluate_history_result.json"),
    )


def main() -> int:
    logger.install_stdout_redirect()
    evaluate_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
