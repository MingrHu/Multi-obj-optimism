from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from common import (
    build_single_output_dnn,
    evaluate_model as accuracy_time_score,
    load_and_preprocess_data,
    normal_max_absolute_error,
)

output_dir = "../../data/TEST"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "evaluate_history_result.txt")


# @brief  目标值交叉验证摘要类
# @return None
# @author MingrHu
# @date   2026/01/20
# @param  model_name         模型名称
# @param  target_index       目标值索引
# @param  target_name        目标值名称
# @param  n_splits           折数
# @param  n_samples          样本数
# @param  r2_mean            R2 系数均值
# @param  r2_std             R2 系数标准差
# @param  nmae_mean          归一化平均绝对误差均值
# @param  nmae_std           归一化平均绝对误差标准差
# @param  mae_mean           平均绝对误差均值
# @param  mae_std            平均绝对误差标准差
# @param  rmse_mean          均方根误差均值
# @param  rmse_std           均方根误差标准差
# @param  max_error_mean     最大误差均值
@dataclass(frozen=True)
class TargetCVSummary:
    model_name: str
    target_index: int
    target_name: str
    n_splits: int
    n_samples: int

    r2_mean: float
    r2_std: float

    nmae_mean: float
    nmae_std: float

    mae_mean: float
    mae_std: float

    rmse_mean: float
    rmse_std: float

    max_error_mean: float
    max_error_std: float

    train_time_mean_s: float
    train_time_min_s: float
    train_time_max_s: float

    predict_time_mean_s: float
    predict_time_min_s: float
    predict_time_max_s: float

    score: Optional[float] = None


# @brief  代理模型评估类
# @return None
# @author MingrHu
# @date   2026/01/20
# @param  data_file          数据集文件
# @param  vars_out           数据集的输入参数和输出目标值名
# @param  n_vars             自变量X 输入参数个数
# @param  n_splits           折数（可选）
# @param  shuffle            是否打乱数据（可选）
# @param  random_state       随机种子（可选）
# @param  model_params       模型参数字典（可选）
class SurrogateModelEvaluator:
    DEFAULT_MODELS: Tuple[str, ...] = ("PRG", "SVR", "RF", "KM", "DNN")

    def __init__(
        self,
        data_file: str,
        vars_out: List[str],
        n_vars: int,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.data_file = data_file
        self.vars_out = vars_out
        self.n_vars = n_vars

        if n_splits < 2:
            raise ValueError("n_splits 必须 >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        self.model_params: Dict[str, Dict[str, Any]] = model_params or {}

    # @brief  评估代理模型
    # @return 目标值交叉验证摘要列表
    # @author MingrHu
    # @date   2026/01/20
    # @param  models             模型名称序列（可选）
    # @param  target_indices     目标值索引序列（可选）
    # @param  target_names       目标值名称序列（可选）
    # @param  score_weights      分数权重元组 (w1, w2)
    def evaluate(
        self,
        models: Optional[Sequence[str]] = None,
        *,
        target_indices: Optional[Sequence[int]] = None,
        target_names: Optional[Sequence[str]] = None,
        score_weights: Tuple[float, float] = (0.9, 0.1),
    ) -> List[TargetCVSummary]:
        X, Y = load_and_preprocess_data(self.data_file, self.vars_out, self.n_vars)
        n_targets = Y.shape[1]

        if target_indices is None:
            target_indices = list(range(n_targets))
        else:
            target_indices = list(target_indices)

        if any((idx < 0 or idx >= n_targets) for idx in target_indices):
            raise ValueError(f"target_indices 越界：Y 共有 {n_targets} 列")

        if models is None:
            models = list(self.DEFAULT_MODELS)
        else:
            models = list(models)

        resolved_target_names = self._resolve_target_names(
            n_targets=n_targets,
            target_indices=target_indices,
            target_names=target_names,
        )

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        summaries: List[TargetCVSummary] = []
        for model_name in models:
            for local_i, target_idx in enumerate(target_indices):
                summary = self._evaluate_one(
                    X=X,
                    y=Y[:, target_idx],
                    kf=kf,
                    model_name=model_name,
                    target_index=target_idx,
                    target_name=resolved_target_names[local_i],
                )
                summaries.append(summary)

        self._attach_scores_in_place(
            summaries,
            score_weights=score_weights,
        )

        return summaries

    # @brief  保存评估报告
    # @return None
    # @author MingrHu
    # @date   2026/01/20
    # @param  summaries         目标值交叉验证摘要序列
    # @param  text_path         文本报告路径（可选）
    # @param  json_path         JSON 报告路径（可选）
    def save_report(
        self,
        summaries: Sequence[TargetCVSummary],
        *,
        text_path: str = output_path,
        json_path: Optional[str] = None,
    ) -> None:
        lines = self.format_report_lines(summaries)
        with open(text_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + os.linesep)
            f.write(os.linesep)

        if json_path is not None:
            payload = [asdict(s) for s in summaries]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    # @brief  格式化报告行
    # @return 报告行列表
    # @author MingrHu
    # @date   2026/01/20
    # @param  summaries         目标值交叉验证摘要序列
    def format_report_lines(self, summaries: Sequence[TargetCVSummary]) -> List[str]:
        lines: List[str] = []
        lines.append("=" * 100)
        # 生成当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"MingrHu-{current_time}-进行模型评价")
        lines.append(f"数据集Data: {self.data_file}")
        lines.append(f"K交叉验证KFold: n_splits={self.n_splits}, shuffle={self.shuffle}, random_state={self.random_state}")
        lines.append("=" * 100)

        for s in summaries:
            score_str = "-" if s.score is None else f"{s.score:.4f}"
            lines.append(
                f"[{s.model_name}] target={s.target_name} (Y[{s.target_index}]) | "
                f"R2={s.r2_mean:.4f}±{s.r2_std:.4f} | "
                f"NMAE={s.nmae_mean:.4f}±{s.nmae_std:.4f} | "
                f"MAE={s.mae_mean:.4f}±{s.mae_std:.4f} | "
                f"RMSE={s.rmse_mean:.4f}±{s.rmse_std:.4f} | "
                f"MAXERR={s.max_error_mean:.4f}±{s.max_error_std:.4f} | "
                f"Train={s.train_time_mean_s:.4f}s (min={s.train_time_min_s:.4f}, max={s.train_time_max_s:.4f}) | "
                f"Pred={s.predict_time_mean_s:.6f}s (min={s.predict_time_min_s:.6f}, max={s.predict_time_max_s:.6f}) | "
                f"Score={score_str}"
            )

        return lines

    # @brief  解析目标值名称
    # @return 目标值名称列表
    # @author MingrHu
    # @date   2026/01/20
    # @param  n_targets         目标值列数
    # @param  target_indices    目标值索引序列
    # @param  target_names      目标值名称序列（可选）
    def _resolve_target_names(
        self,
        *,
        n_targets: int,
        target_indices: Sequence[int],
        target_names: Optional[Sequence[str]],
    ) -> List[str]:
        if target_names is not None:
            target_names = list(target_names)
            if len(target_names) == len(target_indices):
                return list(target_names)
            if len(target_names) == n_targets:
                return [target_names[i] for i in target_indices]
            raise ValueError("target_names 长度必须等于 len(target_indices) 或等于 Y 的列数")

        if len(self.vars_out) >= self.n_vars + n_targets:
            default_names = self.vars_out[self.n_vars : self.n_vars + n_targets]
            return [default_names[i] for i in target_indices]

        return [f"target_{i}" for i in target_indices]

    # @brief  评估模型
    # @return None
    # @author MingrHu
    # @date   2026/01/20
    # @param  X                输入特征矩阵
    # @param  y                目标值向量
    # @param  kf               KFold 交叉验证对象
    # @param  model_name       模型名称
    # @param  target_index     目标值索引
    # @param  target_name      目标值名称
    def _evaluate_one(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        kf: KFold,
        model_name: str,
        target_index: int,
        target_name: str,
    ) -> TargetCVSummary:
        fold_r2: List[float] = []
        fold_nmae: List[float] = []
        fold_mae: List[float] = []
        fold_rmse: List[float] = []
        fold_maxerr: List[float] = []
        fold_train_s: List[float] = []
        fold_pred_s: List[float] = []

        for train_idx, test_idx in kf.split(X):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

            model = self._build_model(model_name=model_name, input_dim=X_train_scaled.shape[1])

            train_start = time.perf_counter()
            # 差异化训练：DNN 单独处理
            if model_name == "DNN":
                self._fit_dnn(model, X_train_scaled, y_train_scaled)
            else:
                model.fit(X_train_scaled, y_train_scaled)
            train_end = time.perf_counter()

            pred_start = time.perf_counter()
            y_pred_scaled = model.predict(X_test_scaled)
            pred_end = time.perf_counter()

            y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).ravel()
            y_true = y_test.reshape(-1)

            fold_r2.append(float(r2_score(y_true, y_pred)))
            fold_nmae.append(float(normal_max_absolute_error(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))))
            fold_mae.append(float(mean_absolute_error(y_true, y_pred)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
            fold_maxerr.append(float(max_error(y_true, y_pred)))
            fold_train_s.append(float(train_end - train_start))
            fold_pred_s.append(float(pred_end - pred_start))

        return TargetCVSummary(
            model_name=model_name,
            target_index=target_index,
            target_name=target_name,
            n_splits=self.n_splits,
            n_samples=int(X.shape[0]),
            r2_mean=float(np.mean(fold_r2)),
            r2_std=float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
            nmae_mean=float(np.mean(fold_nmae)),
            nmae_std=float(np.std(fold_nmae, ddof=1)) if len(fold_nmae) > 1 else 0.0,
            mae_mean=float(np.mean(fold_mae)),
            mae_std=float(np.std(fold_mae, ddof=1)) if len(fold_mae) > 1 else 0.0,
            rmse_mean=float(np.mean(fold_rmse)),
            rmse_std=float(np.std(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0,
            max_error_mean=float(np.mean(fold_maxerr)),
            max_error_std=float(np.std(fold_maxerr, ddof=1)) if len(fold_maxerr) > 1 else 0.0,
            train_time_mean_s=float(np.mean(fold_train_s)),
            train_time_min_s=float(np.min(fold_train_s)),
            train_time_max_s=float(np.max(fold_train_s)),
            predict_time_mean_s=float(np.mean(fold_pred_s)),
            predict_time_min_s=float(np.min(fold_pred_s)),
            predict_time_max_s=float(np.max(fold_pred_s)),
            score=None,
        )

    # @brief  构建模型
    # @return None
    # @author MingrHu
    # @date   2026/01/20
    # @param  model_name       模型名称
    # @param  input_dim        输入维度
    def _build_model(self, *, model_name: str, input_dim: int):
        params = self.model_params.get(model_name, {})

        if model_name == "SVR":
            return SVR(
                kernel=params.get("kernel", "rbf"),
                C=float(params.get("C", 1.0)),
                epsilon=float(params.get("epsilon", 0.1)),
            )

        if model_name == "RF":
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 300)),
                random_state=int(params.get("random_state", self.random_state)),
                n_jobs=int(params.get("n_jobs", -1)),
            )

        if model_name == "PRG":
            degree = int(params.get("degree", 2))
            return make_pipeline(
                PolynomialFeatures(degree, include_bias=False),
                LinearRegression(),
            )

        if model_name == "KM":
            kernel = C(1.0, (1e-3, 1e6)) * RBF(
                length_scale=float(params.get("length_scale", 1.0)),
                length_scale_bounds=params.get("length_scale_bounds", (1e-1, 1e4)),
            )
            return GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=int(params.get("n_restarts_optimizer", 10)),
                alpha=float(params.get("alpha", 0.1)),
                random_state=int(params.get("random_state", self.random_state)),
            )

        if model_name == "DNN":
            return build_single_output_dnn(input_dim)

        raise ValueError(f"Unsupported model_name: {model_name}")

    # @brief  训练 DNN 模型
    # @return None
    # @author MingrHu
    # @date   2026/01/20
    # @param  model             DNN 模型实例
    # @param  X_train_scaled    缩放后的训练特征矩阵
    # @param  y_train_scaled    缩放后的训练目标值向量
    def _fit_dnn(self, model, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray) -> None:
        params = self.model_params.get("DNN", {})
        epochs = int(params.get("epochs", 300))
        batch_size = int(params.get("batch_size", 16))
        verbose = int(params.get("verbose", 0))
        patience = int(params.get("patience", 30))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
        ]

        model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    # @brief  计算并更新模型评估分数
    # @return None
    # @author MingrHu
    # @date   2026/01/20
    # @param  summaries         目标值交叉验证摘要列表
    # @param  score_weights     分数权重元组 (w1, w2)
    def _attach_scores_in_place(
        self,
        summaries: List[TargetCVSummary],
        *,
        score_weights: Tuple[float, float],
    ) -> None:
        w1, w2 = score_weights

        by_target: Dict[Tuple[int, str], List[TargetCVSummary]] = {}
        for s in summaries:
            by_target.setdefault((s.target_index, s.target_name), []).append(s)

        updated: List[TargetCVSummary] = []
        for (_, _), group in by_target.items():
            times = [g.train_time_mean_s for g in group]
            t_min = float(np.min(times))
            t_max = float(np.max(times))

            for g in group:
                score = float(accuracy_time_score(g.r2_mean, g.train_time_mean_s, t_min, t_max, w1, w2))
                updated.append(TargetCVSummary(**{**asdict(g), "score": score}))

        summaries[:] = sorted(updated, key=lambda x: (x.target_index, x.model_name))


def main():
    vars_out = ["1", "2", "3", "4", "5", "6", "7","8","9","10",
                "11","12","13","14","15","16","17","18","19","20", "res1", "res2", "res3"]
    data_file = '/Users/bytedance/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'

    evaluator = SurrogateModelEvaluator(
        data_file=data_file,
        vars_out=vars_out,
        n_vars=7,
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
        target_indices=[0, 1, 2],
        score_weights=(0.9, 0.1),
    )

    evaluator.save_report(
        summaries,
        text_path=output_path,
        json_path=os.path.join(output_dir, "evaluate_history_result.json"),
    )


if __name__ == "__main__":
    main()