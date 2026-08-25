# 优化与代理模型对外调用接口

本文描述仓库当前可直接使用的同步 Python 接口。范围仅包括代理模型训练与
NSGA-II 多目标优化，不包含 DEFORM 自动化和批处理流程。

## 设计边界

- 对外入口统一放在 `mobo.api`，接受 Python 字典或 JSON 字符串，返回可直接 JSON
  序列化的字典。
- `mobo.api` 只做参数校验和编排；算法仍位于 `mobo.surrogate` 与
  `mobo.optimization`，历史算法函数体不变。
- 调用是同步的。外部系统若需要 HTTP、消息队列或异步任务，只需把这里的三个函数
  包在传输层中，不应让 Web 路由直接依赖底层算法类。
- 训练与优化状态统一保存在 `data/tasks/<id>/state.json`。每次训练的模型产物保存在
  `data/tasks/<model_id>/models/`，避免同模型族的后续训练覆盖已有 `model_id`。

## 公开函数

```python
from mobo.api import query_task, run_optimization, train_surrogate
```

| 函数 | 输入 | 用途 |
|---|---|---|
| `train_surrogate(request)` | 字典或 JSON 字符串 | 训练代理模型，返回 `model_id` |
| `run_optimization(request)` | 字典或 JSON 字符串 | 使用 `model_id` 执行参数化 NSGA-II |
| `query_task(task_id)` | `tr_...` 或 `opt_...` | 查询训练或优化任务状态 |

所有响应都包含 `code` 和 `msg`。`code == 0` 表示成功；参数错误和运行错误均以
`code != 0` 返回，不要求传输层捕获业务异常。

## 1. 训练代理模型

完整请求示例：

```python
train_response = train_surrogate({
    "data_file": "C:/project/data/samples.tsv",
    "all_var_list": ["temperature", "die_temperature", "speed", "grain", "load"],
    "input_var_count": 3,
    "model_index": 0,
    "params": {"degree": 2},
})
```

必要字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `data_file` | string | 制表符分隔数据文件；路径必须存在 |
| `all_var_list` | string[] | 按数据列顺序排列；前面是输入，后面是目标 |
| `input_var_count` | int | 输入列数量，必须至少保留一个目标列 |
| `model_index` | int | `0=PRG, 1=SVR, 2=RF, 3=KM, 4=DNN` |
| `params` | object | 可省略；当前仅接受下表中的实际固定值 |
| `model_id` | string | 可选；仅凭已有 ID 续跑时使用 |

当前历史训练函数虽然保留了超参数形参，但函数体尚未使用这些参数。为防止调用方误以为
调参已生效，对外接口会拒绝不同于真实配置的值：

| 模型 | 当前真实固定配置 |
|---|---|
| PRG | `degree=2` |
| SVR | `kernel="rbf", C=1.0, epsilon=0.1` |
| RF | `n_estimators=300, n_jobs=-1`（使用全部可用 CPU 核心） |
| KM | `alpha=0.1, n_restarts_optimizer=20` |
| DNN | `epochs=1000, batch_size=16, verbose=1, patience=50` |

只有在确认允许更改算法行为后，才应让这些参数真正进入历史训练函数。

成功响应中的关键字段：

```json
{
  "code": 0,
  "msg": "训练完成",
  "model_id": "tr_...",
  "data": {
    "model_family": "PRG",
    "train_status": "finished",
    "model_dir": ".../data/tasks/tr_.../models",
    "model_save_paths": {"grain": ".../grain_model.pkl"},
    "target_names": ["grain", "load"]
  }
}
```

## 2. 参数化 NSGA-II 优化

优化必须引用一个已完成的 `model_id`，变量顺序也必须与训练时一致。

```python
optimization_response = run_optimization({
    "model_id": train_response["model_id"],
    "objective_names": ["grain", "load"],
    "input_var_count": 3,
    "all_var_list": ["temperature", "die_temperature", "speed", "grain", "load"],
    "decision_var_indices": [0, 1, 2],
    "decision_var_names": ["temperature", "die_temperature", "speed"],
    "decision_bounds": [
        {"lower": 875, "upper": 965, "desc": "工件温度[°C]"},
        {"lower": 300, "upper": 700, "desc": "模具温度[°C]"},
        {"lower": 10, "upper": 50, "desc": "上模速度[mm/s]"}
    ],
    "constraints": [
        {"target_obj": "grain", "constraint_kind": "upper", "limit_value": 30},
        {"target_obj": "load", "constraint_kind": "upper", "limit_value": 330000}
    ],
    "objective_config": [
        {"name": "grain", "minimize": True},
        {"name": "load", "minimize": True}
    ],
    "optimizer_config": {
        "pop_size": 100,
        "n_offsprings": 100,
        "eliminate_duplicates": True,
        "n_gen": 200,
        "seed": 42
    },
    "output_config": {}
})
```

补充规则：

- `decision_var_indices` 是输入列中的下标；`decision_var_names` 必须与这些下标对应。
- `decision_bounds` 与决策变量一一对应，且必须满足 `lower < upper`。
- `objective_names` 必须来自 `all_var_list` 的输出部分。
- `objective_config` 可省略，默认所有目标最小化；最大化目标设为 `minimize=false`。
- `constraints` 可省略。`constraint_kind` 支持 `upper` 和 `lower`。
- `optimizer_config` 可省略，默认值就是示例中的值。
- `output_config.pareto_txt_path` 可选；默认输出到当前优化任务目录。
- 自定义输出路径只能位于 `MOBO_DATA_DIR` 中；任务 ID 只允许字母、数字、下划线和连字符。

解集使用 UTF-8 TSV 格式，首行为列名，最后一列 `feasible` 表示约束是否满足。
最大化目标在文件中会恢复为真实数值，不暴露优化器内部使用的负号。

## 3. 状态查询与传输层适配

```python
status = query_task(train_response["model_id"])
```

如果后续需要 REST，可以保持非常薄的路由：请求体原样交给 `mobo.api`，响应字典原样
返回。HTTP 状态码、鉴权、限流与任务队列属于部署层，不应加入算法包。CPU 密集训练和
优化若部署为多用户服务，建议由进程队列执行，并对同一模型族的训练任务串行调度。

## 4. 完整演示

演示明确分为调用方和服务方。调用方生成数据并写入 JSON 请求：

```bash
python -m mobo.api.client_demo
python -m mobo.api.server_demo
```

`client_demo.py` 只负责样本和请求序列化；`server_demo.py` 只负责读取请求、训练、评价、
选优和优化。端上提供的 `request_id` 是整条流程的幂等键。相同 ID 和相同请求再次提交时，
服务直接返回已有结果；相同 ID 的请求内容发生变化则报冲突。文件按 ID 隔离：

```text
temp/api_demo/requests/<request_id>/
├── samples.tsv
├── request.json
├── response.json
└── server/
    ├── evaluation.json
    └── pareto_solutions.tsv
```
