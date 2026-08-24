"""DEFORM 自动化服务层。

面向任务的服务接口：抽样、初始化执行任务、逐阶段推进、查询状态、提取数据。
任务的必要输入（路径、参数表、目标表等）在 :func:`init_execution_task` 时落盘到
``TASKS_DIR/<task_id>/state.json``；之后的 :func:`run_execution_step` /
:func:`run_extract_data` / :func:`query_execution_status` 仅凭 ``task_id`` 即可继续，
:class:`~mobo.automation.pipeline.ForgingTask` 会按保存的目录约定从磁盘重建，
无需重新传入参数（支持跨进程续跑）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mobo.common import task_store
from mobo.common.logging import logger
from mobo.common.paths import task_dir
from .pipeline import ForgingTask, generate_sample_file

_KIND = "automation"

# 初始化执行任务所需的路径键
_REQUIRED_PATH_KEYS = (
    "smp_file",
    "std_key_file",
    "temp_key_path",
    "res_db_path",
    "res_key_path",
    "res_txt_path",
)

# 执行任务续跑所需的参数键（三路解析：记录 > 传入 > 报错）
_REQUIRED_EXEC_KEYS = (
    "paths_config",
    "param_table",
    "target_table",
    "in_progress",
)


def _result(task_id: str, ok: bool, message: str) -> Dict[str, str]:
    """构造统一的服务返回结构。"""
    return {"task_id": task_id, "status": "success" if ok else "failed", "message": message}


def _key_sample_index(key_path: str, template_stem: str) -> int:
    """从输入 KEY 文件名解析样本序号（generate_key_files 用 <模板名><序号>.KEY 命名）。

    剥去模板主名前缀后取剩余部分的整数序号；无法解析时返回一个极大值，使其排到
    末尾而不打乱正常样本的相对顺序。
    """
    stem = Path(key_path).stem
    suffix = stem[len(template_stem):] if stem.startswith(template_stem) else stem
    return int(suffix) if suffix.isdigit() else 2 ** 63 - 1


def _rebuild_task(task_id: str, provided: Optional[Dict[str, Any]] = None) -> ForgingTask:
    """按参数从磁盘目录重建 ForgingTask（中间产物由确定性文件名推导）。

    参数走三路解析：优先用任务记录里的 req，缺失时用本次传入值并回填记录，
    两者都没有则报错。
    """
    req = task_store.resolve_req(task_id, _KIND, provided or {}, _REQUIRED_EXEC_KEYS)
    paths = req["paths_config"]
    incremental = bool(req.get("incremental", False))
    task = ForgingTask(
        sample_file=paths["smp_file"],
        template_key=paths["std_key_file"],
        temp_key_dir=paths["temp_key_path"],
        result_db_dir=paths["res_db_path"],
        result_key_dir=paths["res_key_path"],
        result_txt_dir=paths["res_txt_path"],
        param_table=[list(row) for row in req["param_table"]],
        target_table=[list(row) for row in req["target_table"]],
        in_progress=list(req["in_progress"]),
        process_info_file=paths.get("process_info_file") or (
            str(task_dir(task_id) / "process_info.json") if incremental else ""
        ),
        incremental=incremental,
        incremental_state_file=paths.get(
            "incremental_state_file",
            str(task_dir(task_id) / "incremental_dataset.json"),
        ),
        incremental_output_file=paths.get(
            "incremental_output_file",
            os.path.join(paths["res_txt_path"], f"{task_id}_incremental_result.txt"),
        ),
    )
    # 从临时 KEY 目录恢复已生成的输入 KEY（文件名形如 <模板名><样本序号>.KEY）。
    temp_dir = paths["temp_key_path"]
    if os.path.isdir(temp_dir):
        template_stem = Path(paths["std_key_file"]).stem
        task.key_files = sorted(
            (os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".KEY")),
            key=lambda p: _key_sample_index(p, template_stem),
        )
    return task


def align_result_db_dirs(task_id: str, apply: bool = True) -> Dict[str, str]:
    """校正结果 DB 目录序号，使其与真实样本号（即 KEY 顺序）对齐
    :param task_id: 任务 ID（据 state.json 重建任务信息）
    :param apply: True 真正重命名；False 仅返回将要改动的目录数（预演）
    """
    try:
        task = _rebuild_task(task_id)
        res_db = task.result_db_dir
        if not os.path.isdir(res_db):
            return _result(task_id, False, f"结果 DB 目录不存在：{res_db}")
        template_stem = Path(task.template_key).stem

        # 逐目录读实际 DB 名解析真实样本号，构造 {旧目录号: 真实样本号}
        plan: Dict[str, int] = {}
        seen: Dict[int, str] = {}
        for name in os.listdir(res_db):
            sub_dir = os.path.join(res_db, name)
            if not os.path.isdir(sub_dir):
                continue
            dbs = [f for f in os.listdir(sub_dir) if f.endswith(".DB")]
            if len(dbs) != 1:
                continue  # 无 DB 或多个 DB，无法判定，跳过
            sample_idx = _key_sample_index(dbs[0], template_stem)
            if sample_idx == 2 ** 63 - 1:
                continue  # 文件名解析不出样本号，跳过
            if sample_idx in seen:
                return _result(task_id, False,
                               f"样本号 {sample_idx} 被目录 {seen[sample_idx]}/ 与 {name}/ 同时占用，已中止")
            seen[sample_idx] = name
            plan[name] = sample_idx

        changes = {old: new for old, new in plan.items() if old != str(new)}
        if not changes:
            return _result(task_id, True, "结果 DB 目录已对齐，无需改动")
        if not apply:
            return _result(task_id, True, f"预演：{len(changes)} 个目录需重命名（未改动磁盘）")

        # 两阶段重命名，规避 4→2 与 2→10 这类环状占用冲突
        tmp_map: Dict[str, int] = {}
        for i, (old_name, new_i) in enumerate(changes.items()):
            tmp_name = f".__align_tmp_{i}__hmr"
            os.rename(os.path.join(res_db, old_name), os.path.join(res_db, tmp_name))
            tmp_map[tmp_name] = new_i
        for tmp_name, new_i in tmp_map.items():
            os.rename(os.path.join(res_db, tmp_name), os.path.join(res_db, str(new_i)))

        return _result(task_id, True, f"已对齐 {len(changes)} 个结果 DB 目录到真实样本号")
    except Exception as exc:
        logger.error(f"结果 DB 目录对齐失败：{exc}")
        return _result(task_id, False, f"结果 DB 目录对齐失败：{exc}")


def create_sampling_task(
    task_id: str,
    save_dir: str,
    method: str,
    param_ranges: Dict[str, tuple[float, float]],
    n_samples: int = 0,
    level_nums: Optional[List[int]] = None,
) -> Dict[str, str]:
    """创建并执行抽样任务，结果落盘到 state.json。"""
    if n_samples == 0:
        return {}
    level_nums = level_nums or []
    try:
        out_path = generate_sample_file(task_id, method, param_ranges, save_dir, n_samples, level_nums)
        task_store.init_state(task_id, _KIND, {
            "sampling": {"method": method, "save_dir": save_dir,
                         "n_samples": n_samples, "level_nums": level_nums},
        })
        task_store.update(task_id, stage="sampling", status="finished",
                          data={"sample_file": out_path})
        return _result(task_id, True, f"成功使用 {method} 方法生成样本")
    except Exception as exc:
        logger.error(f"抽样任务创建失败：{exc}")
        return _result(task_id, False, f"使用 {method} 方法生成样本失败")


def init_execution_task(
    task_id: str,
    paths_config: Dict[str, str],
    param_table: List[List[str]],
    target_table: List[List[Any]],
    in_progress: List[bool],
    incremental: bool = False,
) -> Dict[str, str]:
    """初始化执行任务：校验路径、落盘输入参数、构建任务并生成 KEY 文件。"""
    try:
        if any(not paths_config.get(k) for k in _REQUIRED_PATH_KEYS):
            return _result(task_id, False, "未指定样本、模板 KEY、临时/结果路径等必填项")

        for path in paths_config.values():
            target_dir = path if not os.path.splitext(path)[1] else os.path.dirname(path)
            os.makedirs(target_dir, exist_ok=True)

        # 三路解析并落盘续跑所需的全部输入参数（记录已有则沿用，缺失则回填）
        task = _rebuild_task(task_id, {
            "paths_config": paths_config,
            "param_table": param_table,
            "target_table": target_table,
            "in_progress": in_progress,
            "incremental": incremental,
        })
        task.generate_keys()
        task_store.update(task_id, stage="generate_keys", status="finished",
                          data={"key_file_count": len(task.key_files)})
        return _result(task_id, True, "执行任务初始化成功，KEY 文件生成完成")
    except Exception as exc:
        logger.error(f"执行任务初始化失败：{exc}")
        if task_store.exists(task_id):
            task_store.update(task_id, stage="generate_keys", status="failed")
        return _result(task_id, False, f"执行任务初始化失败：{exc}")


def run_execution_step(task_id: str, **overrides: Any) -> Dict[str, str]:
    """推进求解阶段。

    优先从任务记录续跑；记录缺失的参数可用 ``overrides`` 补齐（会回填记录），
    记录与传入都没有则报错。求解前会按样本补生成缺失或空的 KEY，
    已存在的非空 KEY 保持不变。
    """
    try:
        task = _rebuild_task(task_id, overrides)
        task.generate_keys()
        task.run_solver()
        task_store.update(task_id, stage="run_solver", status="finished",
                          data={
                              "db_file_count": len(task.db_files),
                              "incremental_state_file": getattr(
                                  task, "incremental_state_file", ""
                              ) if getattr(task, "incremental", False) else "",
                              "incremental_output_file": getattr(
                                  task, "incremental_output_file", ""
                              ) if getattr(task, "incremental", False) else "",
                          })
        return _result(task_id, True, "计算任务运行完成")
    except Exception as exc:
        logger.error(f"求解运行失败：{exc}")
        if task_store.exists(task_id):
            task_store.update(task_id, stage="run_solver", status="failed")
        return _result(task_id, False, f"求解运行失败：{exc}")


def query_execution_status(task_id: str) -> Dict[str, str]:
    """查询执行任务状态（从 state.json 读取阶段与状态）。"""
    state = task_store.load(task_id)
    if state is None:
        return _result(task_id, False, "执行任务不存在")
    return {
        "task_id": task_id,
        "status": state.get("status", "unknown"),
        "message": f"当前阶段：{state.get('stage')}",
    }


def run_extract_data(task_id: str, **overrides: Any) -> Dict[str, str]:
    """推进数据提取阶段。

    优先从任务记录续跑；记录缺失的参数可用 ``overrides`` 补齐（会回填记录），
    记录与传入都没有则报错。
    """
    try:
        task = _rebuild_task(task_id, overrides)
        # 续跑重建时 param_table 仅含 2 行表头（样本行未落盘 req），从样本文件补回，
        # 顺序与 generate_keys 一致：文件第 i 行 -> <模板名>i.KEY -> db_files[i] -> param_table[i+2]
        task.load_samples_into_table()
        task.prepare_db_files()  # 按结果 DB 目录约定重建 db_files
        task.extract()
        task_store.update(task_id, stage="extract", status="finished",
                          data={"result_dir": task.result_txt_dir})
        return _result(task_id, True, "数据提取完成")
    except Exception as exc:
        logger.error(f"数据提取失败：{exc}")
        if task_store.exists(task_id):
            task_store.update(task_id, stage="extract", status="failed")
        return _result(task_id, False, f"数据提取失败：{exc}")


__all__ = [
    "create_sampling_task",
    "init_execution_task",
    "run_execution_step",
    "query_execution_status",
    "run_extract_data",
    "align_result_db_dirs",
]
