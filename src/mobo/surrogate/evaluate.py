from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from datetime import datetime
from keras import backend as keras_backend
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

from mobo.common.paths import TEST_DIR
from .common import (
    build_single_output_dnn,
    evaluate_model as accuracy_time_score,
    load_and_preprocess_data,
    normal_max_absolute_error,
)
from .hyperparameters import normalize_model_params

output_dir = str(TEST_DIR)
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
        max_workers: Optional[int] = None,
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
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers 必须 >= 1 或为 None")
        self.max_workers = max_workers

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
            model_summaries = self._evaluate_targets(
                X=X,
                Y=Y,
                kf=kf,
                model_name=model_name,
                target_indices=target_indices,
                target_names=resolved_target_names,
            )
            summaries.extend(model_summaries)

        self._attach_scores_in_place(
            summaries,
            score_weights=score_weights,
        )

        return summaries

    def _evaluate_targets(
        self,
        *,
        X: np.ndarray,
        Y: np.ndarray,
        kf: KFold,
        model_name: str,
        target_indices: Sequence[int],
        target_names: Sequence[str],
    ) -> List[TargetCVSummary]:
        """并行评价不同输出目标；单个目标内部的 K 折保持串行。"""
        jobs = list(zip(target_indices, target_names, strict=True))
        workers = self._resolve_target_workers(len(jobs))

        def evaluate_target(job: tuple[int, str]) -> TargetCVSummary:
            target_idx, target_name = job
            # 每个目标使用独立的 KFold 实例，避免在线程间共享可迭代器状态。
            target_kf = KFold(
                n_splits=kf.n_splits,
                shuffle=kf.shuffle,
                random_state=kf.random_state,
            )
            return self._evaluate_one(
                X=X,
                y=Y[:, target_idx],
                kf=target_kf,
                model_name=model_name,
                target_index=target_idx,
                target_name=target_name,
            )

        if workers == 1:
            results = []
            for job in jobs:
                results.append(evaluate_target(job))
                if model_name == "DNN":
                    keras_backend.clear_session()
            return results

        ordered: List[Optional[TargetCVSummary]] = [None] * len(jobs)
        try:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"cv-{model_name.lower()}",
            ) as executor:
                futures = {
                    executor.submit(evaluate_target, job): index
                    for index, job in enumerate(jobs)
                }
                for future in as_completed(futures):
                    ordered[futures[future]] = future.result()
        finally:
            if model_name == "DNN":
                # clear_session 是 Keras 全局操作，不能在其他目标仍训练时调用。
                keras_backend.clear_session()
        return [summary for summary in ordered if summary is not None]

    def _resolve_target_workers(self, target_count: int) -> int:
        if target_count <= 1:
            return 1
        if self.max_workers is not None:
            return min(target_count, self.max_workers)
        logical_cpus = os.cpu_count() or 1
        return min(target_count, max(1, logical_cpus // 2), 8)

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
                f"Pred={s.predict_time_mean_s:.6f}s "
                f"(min={s.predict_time_min_s:.6f}, max={s.predict_time_max_s:.6f}) | "
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
            if model_name == "DNN":
                y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            else:
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
        params = normalize_model_params(model_name, self.model_params.get(model_name, {}))

        if model_name == "SVR":
            return SVR(**params)

        if model_name == "RF":
            return RandomForestRegressor(**params)

        if model_name == "PRG":
            return make_pipeline(
                PolynomialFeatures(
                    params["degree"], include_bias=params["include_bias"]
                ),
                LinearRegression(fit_intercept=params["fit_intercept"]),
            )

        if model_name == "KM":
            kernel = C(
                params["constant_value"],
                (params["constant_lower"], params["constant_upper"]),
            ) * RBF(
                length_scale=params["length_scale"],
                length_scale_bounds=(
                    params["length_scale_lower"], params["length_scale_upper"]
                ),
            )
            return GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=params["n_restarts_optimizer"],
                alpha=params["alpha"],
                normalize_y=params["normalize_y"],
                random_state=params["random_state"],
            )

        if model_name == "DNN":
            return build_single_output_dnn(
                input_dim,
                hidden_layer_1=params["hidden_layer_1"],
                hidden_layer_2=params["hidden_layer_2"],
                hidden_layer_3=params["hidden_layer_3"],
                activation=params["activation"],
                dropout_1=params["dropout_1"],
                dropout_2=params["dropout_2"],
                batch_normalization=params["batch_normalization"],
                learning_rate=params["learning_rate"],
            )

        raise ValueError(f"Unsupported model_name: {model_name}")

    # @brief  训练 DNN 模型
    # @return None
    # @author MingrHu
    # @date   2026/01/20
    # @param  model             DNN 模型实例
    # @param  X_train_scaled    缩放后的训练特征矩阵
    # @param  y_train_scaled    缩放后的训练目标值向量
    def _fit_dnn(self, model, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray) -> None:
        params = normalize_model_params("DNN", self.model_params.get("DNN", {}))

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=params["patience"],
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=params["reduce_lr_factor"],
                patience=params["reduce_lr_patience"],
                min_lr=params["min_lr"],
            ),
        ]

        model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=params["verbose"],
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
