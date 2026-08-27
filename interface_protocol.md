# Python 内部服务协议

本文描述 `mobo` 包内部可复用的任务级 Python 服务。端上或其他外部系统不应直接依赖
这些函数，应使用 [DOE_HTTP_API.md](DOE_HTTP_API.md) 中的 Flask HTTP API。

## 服务边界

| 流程 | 模块 | 主键 | 主要函数 |
|---|---|---|---|
| 代理模型 | `mobo.surrogate.service` | `model_id`（`tr_` 前缀） | `train_surrogate`、`query_model_status` |
| 优化 | `mobo.optimization.service` | `task_id`（`opt_` 前缀） | `run_optimization`、`query_optimization_status` |
| 单工步 DEFORM | `mobo.automation.service` | `task_id` | `init_execution_task`、`run_execution_step`、`run_extract_data` |
| 多工步 DEFORM | `mobo.automation.multi_operation_service` | `task_id` | `init_multi_operation_task`、`run_multi_operation_task`、`run_multi_operation_extract` |

任务状态统一存放在 `data/tasks/<task_id>/state.json`。状态包含 `kind`、`status`、
`stage`、`req`、`data` 和只追加的 `history`。服务可通过任务 ID 读取已保存的请求参数，
用于查询和断点恢复。

## 代理模型服务

```python
from mobo.surrogate.service import train_surrogate

response = train_surrogate(
    data_file="/absolute/path/training.tsv",
    vars_out=["temperature", "speed", "grain", "load"],
    n_vars=2,
    model_index=2,
    biz_params={"n_estimators": 300, "n_jobs": -1},
)
```

服务层模型编号固定为：

| `model_index` | 模型族 |
|---:|---|
| 0 | PRG，多项式回归 |
| 1 | SVR |
| 2 | RF，随机森林 |
| 3 | KM，Kriging/GPR |
| 4 | DNN |

成功响应包含 `model_id`、`model_dir`、目标名称、模型路径、超参数和训练耗时。模型快照
位于任务专属目录，不要求调用方拼接共享模型路径。

## 优化服务

```python
from mobo.optimization.service import run_optimization

response = run_optimization(
    {
        "model_id": "tr_example",
        "objective_names": ["grain", "load"],
        "all_var_list": ["temperature", "speed", "grain", "load"],
        "input_var_count": 2,
        "decision_var_indices": [0, 1],
        "decision_var_names": ["temperature", "speed"],
        "decision_bounds": [
            {"lower": 900, "upper": 1100},
            {"lower": 10, "upper": 50},
        ],
        "constraints": [],
        "objective_config": [
            {"name": "grain", "minimize": True},
            {"name": "load", "minimize": True},
        ],
        "optimizer_config": {
            "pop_size": 100,
            "n_offsprings": 100,
            "eliminate_duplicates": True,
            "n_gen": 200,
            "seed": 42,
        },
        "output_config": {},
    },
    optimizer="nsga2",
)
```

参数化 NSGA-II 的解集是无表头 TSV。列顺序通过响应
`data.task_info.result_columns` 单独记录，顺序为决策变量、目标变量、`feasible`。

## HTTP 聚合层

`mobo.api` 将上述内部服务聚合到一个 DOE ID 下，并把样本、训练数据、模型、推理结果和
优化结果隔离到 `data/doe_tasks/<doe_id>/`。HTTP 层负责参数校验、统一响应、后台线程
控制、模型选择以及按字段读取结果；具体调用以 [DOE_HTTP_API.md](DOE_HTTP_API.md) 为准。

## 路径与部署约定

- 代码通过 `mobo.common.paths` 获取项目和数据目录。
- 外置运行数据使用 `MOBO_PROJECT_DIR`、`MOBO_DATA_DIR`，不在代码中写平台绝对路径。
- Flask 内置服务仅用于本地测试；生产容器应由 WSGI 服务加载
  `mobo.api.app:create_app`。
- DEFORM 子进程仅能在装有 DEFORM 的 Windows 环境真实执行；HTTP Demo 不依赖 DEFORM。
