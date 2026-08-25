# DOE HTTP API（v1）

服务启动：`mobo-api`，默认监听 `0.0.0.0:5000`。可通过 `MOBO_API_HOST` 和
`MOBO_API_PORT` 修改。客户端示例：`python -m mobo.api.demo`。

从环境安装到完整优化 Demo 的逐步操作见
[API 后端启动文档](src/mobo/api/BACKEND_STARTUP.md)。

## 通用响应

```json
{
  "code": 0,
  "message": "ok",
  "data": {}
}
```

`code=0` 表示成功。参数错误、资源不存在和状态冲突分别使用 HTTP 400、404、409；
后台训练和优化提交成功使用 HTTP 202。

## 接口

| 方法与路径 | 关键请求参数 | 说明 |
|---|---|---|
| `POST /api/v1/doe/add` | `id?`, `name?`, `description?`, `metadata?` | 创建 DOE；未传 ID 时自动生成 |
| `GET /api/v1/doe/list` | - | 查询 DOE 列表 |
| `POST /api/v1/doe/delete` | `id` | 删除 DOE 及其样本、模型、训练和优化文件 |
| `POST /api/v1/hust/doe/sample/generate` | `id`, `method`, `param_ranges`, `n_samples?`, `level_nums?` | LHS/全因子采样 |
| `POST /api/v1/hust/doe/dataset/generate` | `id`, `param_ranges`, `target_names`, `input_names?`, `n_samples?`, `seed?`, `noise_ratio?` | 在 DOE 训练目录生成完整流程演示数据集 |
| `GET /api/hust/v1/doe/train/progress?id=...` | `id` | 查询代理模型训练状态、阶段、进度及已训练模型 |
| `POST /api/v1/hust/doe/train/delete` | `id` | 删除训练记录和代理模型 |
| `POST /api/v1/hust/doe/train/stop` | `id` | 发出训练中止请求 |
| `POST /api/v1/hust/doe/train/startTrain` | 见下文 | 后台训练并交叉验证 |
| `POST /api/v1/hust/doe/inference/startInference` | `id`, `inputs`, `model_id?` | 代理模型批量推理 |
| `POST /api/v1/hust/doe/optimize/start` | 见下文 | 后台提交 NSGA-II、单目标或 RL 优化 |
| `POST /api/v1/hust/doe/optimize/stop` | `id` | 发出优化中止请求 |
| `GET /api/v1/hust/doe/optimize/getById?id=...` | `id` | 查询优化状态、参数与结果文件 |

## 训练示例

完整流程 demo 可以先调用训练数据生成接口，服务会根据输入范围构造可复现的非线性目标，
并把无表头制表符数据写入当前 DOE 的 `training` 目录。该数据只用于接口联调和流程验证，
不代表真实仿真结果。

```json
{
  "id": "doe_ring_001",
  "input_names": ["temperature", "speed"],
  "target_names": ["grain", "load"],
  "param_ranges": {
    "temperature": [900, 1100],
    "speed": [10, 50]
  },
  "n_samples": 80,
  "seed": 42,
  "noise_ratio": 0
}
```

```json
{
  "id": "doe_ring_001",
  "data_file": "<数据集生成响应中的data.data_file>",
  "all_var_list": ["temperature", "speed", "grain", "load"],
  "input_var_count": 2,
  "models": [
    {"model_index": 0, "params": {"degree": 2}},
    {"model_index": 2, "params": {"n_estimators": 300, "n_jobs": -1}}
  ],
  "evaluation": {"enabled": true, "n_splits": 5, "random_state": 42}
}
```

模型编号为 `0=PRG, 1=SVR, 2=RF, 3=KM, 4=DNN`。未传 `models` 时默认训练前四种模型。

## 推理示例

```json
{
  "id": "doe_ring_001",
  "inputs": [[950, 20], [1000, 30]],
  "model_id": null
}
```

不指定 `model_id` 时自动选择平均评价分最高的模型。响应中的 `targets` 给出列顺序，
`predictions` 为相同顺序的二维数组。

## 优化提交

`algorithm` 支持 `nsga2` 和 `rl`；单目标问题仍使用 `nsga2`，只传一个目标即可。
NSGA-II 的其余字段沿用参数化优化服务：`objective_names`、`all_var_list`、
`input_var_count`、`decision_var_indices`、`decision_var_names`、`decision_bounds`、
`objective_config`、`constraints`、`optimizer_config`、`output_config`。不指定 `model_id`
时自动使用评分最高的代理模型。

完整 NSGA-II 请求示例：

```json
{
  "id": "doe_ring_001",
  "algorithm": "nsga2",
  "objective_names": ["grain", "load"],
  "all_var_list": ["temperature", "speed", "grain", "load"],
  "input_var_count": 2,
  "decision_var_indices": [0, 1],
  "decision_var_names": ["temperature", "speed"],
  "decision_bounds": [
    {"lower": 900, "upper": 1100},
    {"lower": 10, "upper": 50}
  ],
  "constraints": [],
  "optimizer_config": {
    "pop_size": 20,
    "n_offsprings": 10,
    "eliminate_duplicates": true,
    "n_gen": 10,
    "seed": 42
  }
}
```

## 落盘结构

```text
data/doe_tasks/<id>/
├── doe.json
├── samples/
├── models/<model_id>/
├── training/
└── optimization/
```

`doe.json` 是任务元数据、进度和产物索引的唯一入口；算法生成的最终结果会复制到该
DOE 的对应目录中。
