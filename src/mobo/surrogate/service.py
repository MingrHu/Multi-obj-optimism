"""代理模型服务层。

按 ``interface_protocol.md`` 的协议，对外提供以 ``model_id`` 为主键的训练接口：
训练前落盘请求参数，训练后把响应（含模型路径、超参、耗时）写入
``TASKS_DIR/<model_id>/state.json``，之后优化阶段可仅凭 ``model_id`` 反查模型目录。

协议 ``model_index`` 与底层 :class:`~mobo.surrogate.interface.Doe_surrogateModel`
的 ``which_model`` 顺序不同，本层负责映射，不改动底层训练函数体。
"""

from __future__ import annotations

import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mobo.common import task_store
from mobo.common.logging import logger
from mobo.common.paths import model_family_dir
from .interface import Doe_surrogateModel

_KIND = "surrogate"

# 协议 model_index -> (底层 which_model 下标, 模型族名)
# 协议: 0-PRG 1-SVR 2-RF 3-KM 4-DNN
# 底层 which_model: 0-KM 1-DNN 2-PRG 3-SVR 4-RF
_INDEX_MAP = {
    0: (2, "PRG"),
    1: (3, "SVR"),
    2: (4, "RF"),
    3: (0, "KM"),
    4: (1, "DNN"),
}

# 各模型 biz_params 到底层 model_par 列表的顺序约定
_PARAM_ORDER = {
    "PRG": ["degree"],
    "SVR": ["kernel", "C", "epsilon"],
    "RF": ["n_estimators", "n_jobs"],
    "KM": ["alpha", "n_restarts_optimizer"],
    "DNN": ["epochs", "batch_size", "verbose", "patience"],
}

# 训练续跑所需的参数键（三路解析：记录 > 传入 > 报错）
_REQUIRED_TRAIN_KEYS = ("data_file", "vars_out", "n_vars", "model_index", "biz_params")


def _new_model_id() -> str:
    return "tr_" + datetime.now().strftime("%Y%m%d_%H%M_%S%f")[:-3]


def _to_model_par(family: str, biz_params: Dict[str, Any]) -> List[str]:
    """按各模型约定顺序把 biz_params 转成底层 model_par 字符串列表。"""
    return [str(biz_params[k]) for k in _PARAM_ORDER.get(family, []) if k in biz_params]


def _snapshot_model_artifacts(
    model_id: str,
    family: str,
    target_names: List[str],
) -> tuple[str, Dict[str, str]]:
    """把共享模型族目录中的本次训练产物复制到 model_id 专属目录。"""
    source_dir = model_family_dir(family)
    destination_dir = Path(task_store.state_path(model_id)).parent / "models"
    destination_dir.mkdir(parents=True, exist_ok=True)
    extension = "keras" if family == "DNN" else "pkl"
    model_paths: Dict[str, str] = {}

    for target_name in target_names:
        filenames = [
            f"{target_name}_model.{extension}",
            f"{target_name}_scalers.pkl",
        ]
        for filename in filenames:
            source = source_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"训练完成但未找到模型产物：{source}")
            shutil.copy2(source, destination_dir / filename)
        model_paths[target_name] = str(destination_dir / filenames[0])

    return str(destination_dir), model_paths


def train_surrogate(
    data_file: Optional[str] = None,
    vars_out: Optional[List[str]] = None,
    n_vars: Optional[int] = None,
    model_index: Optional[int] = None,
    biz_params: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """训练一个代理模型并把请求/响应落盘到 state.json。

    输入参数走三路解析：优先用 ``model_id`` 记录里的 req，缺失时用本次传入值并
    回填记录，两者都没有则报错。

    :param data_file: 数据集文件（绝对路径）
    :param vars_out: 变量名列表（前 n_vars 为输入，其余为目标）
    :param n_vars: 输入变量个数
    :param model_index: 协议模型索引 0-PRG/1-SVR/2-RF/3-KM/4-DNN
    :param biz_params: 模型超参
    :param model_id: 复用已有 model_id（None 则新建）
    :return: 协议 resp 字典（code/msg/model_id/data）
    """
    model_id = model_id or _new_model_id()
    try:
        req = task_store.resolve_req(model_id, _KIND, {
            "data_file": data_file, "vars_out": vars_out, "n_vars": n_vars,
            "model_index": model_index, "biz_params": biz_params or {},
        }, _REQUIRED_TRAIN_KEYS)
    except ValueError as exc:
        return {"code": 1, "msg": f"续跑参数缺失：{exc}", "model_id": model_id, "data": {}}

    model_index = req["model_index"]
    if model_index not in _INDEX_MAP:
        return {"code": 1, "msg": f"不支持的 model_index：{model_index}", "model_id": model_id, "data": {}}

    which_model, family = _INDEX_MAP[model_index]
    data_file, vars_out, n_vars = req["data_file"], req["vars_out"], req["n_vars"]
    biz_params = req["biz_params"] or {}
    task_store.update(model_id, stage="train", status="running")

    try:
        model_par = _to_model_par(family, biz_params)
        started = time.time()
        Doe_surrogateModel(data_file, vars_out, n_vars).train_save_model(which_model, model_par)
        cost = round(time.time() - started, 2)

        target_names = vars_out[n_vars:]
        model_dir, model_paths = _snapshot_model_artifacts(
            model_id, family, target_names
        )

        data = {
            "model_index": model_index, "model_family": family,
            "train_status": "finished", "train_cost_sec": cost,
            "hyper_params": biz_params, "model_dir": model_dir,
            "model_save_paths": model_paths, "target_names": target_names,
        }
        task_store.update(model_id, stage="train", status="finished", data=data)
        return {"code": 0, "msg": "训练完成", "model_id": model_id, "data": data}
    except Exception as exc:
        logger.error(f"代理模型训练失败：{exc}")
        task_store.update(model_id, stage="train", status="failed",
                          data={"train_status": "failed"})
        return {"code": 1, "msg": f"训练失败：{exc}", "model_id": model_id, "data": {}}


def query_model_status(model_id: str) -> Dict[str, Any]:
    """查询代理模型训练任务状态（从 state.json 读取）。"""
    state = task_store.load(model_id)
    if state is None:
        return {"code": 1, "msg": "模型任务不存在", "model_id": model_id, "data": {}}
    return {"code": 0, "msg": "ok", "model_id": model_id,
            "data": {"status": state.get("status"), "stage": state.get("stage"),
                     **(state.get("data") or {})}}


__all__ = ["train_surrogate", "query_model_status"]
