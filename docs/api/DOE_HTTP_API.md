# DOE HTTP API（v1）

服务启动：`mobo-api`，默认监听 `0.0.0.0:5000`。可通过 `MOBO_API_HOST` 和
`MOBO_API_PORT` 修改。客户端示例：`python -m mobo.api.demo`。

Docker 环境可执行 `docker compose up --build -d`，容器会自动使用 Gunicorn 启动服务；
详见 [Docker 部署文档](../deployment/DOCKER_DEPLOYMENT.md)。

从环境安装到完整优化 Demo 的逐步操作见
[API 后端启动文档](../deployment/BACKEND_STARTUP.md)。

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
| `GET /health` | - | 健康检查 |
| `POST /api/v1/doe/add` | `id?`, `name?`, `description?`, `metadata?` | 创建 DOE；未传 ID 时自动生成 |
| `GET /api/v1/doe/list` | - | 查询 DOE 列表 |
| `POST /api/v1/doe/delete` | `id` | 删除 DOE 及其样本、模型、训练和优化文件 |
| `POST /api/v1/hust/doe/sample/generate` | `id`, `method`, `param_ranges`, `n_samples?`, `level_nums?` | LHS/全因子采样 |
| `POST /api/v1/hust/doe/dataset/generate` | `id`, `param_ranges`, `target_names`, `input_names?`, `n_samples?`, `seed?`, `noise_ratio?` | 在 DOE 训练目录生成完整流程演示数据集 |
| `GET /api/v1/hust/doe/data/get` | `id`, `data_type`, `fields` | 按字段获取样本、数据集、优化或最近一次推理结果 |
| `GET /api/hust/v1/doe/train/progress?id=...` | `id` | 查询代理模型训练状态、阶段、进度及已训练模型 |
| `POST /api/v1/hust/doe/train/delete` | `id` | 删除训练记录和代理模型 |
| `POST /api/v1/hust/doe/train/stop` | `id` | 发出训练中止请求 |
| `POST /api/v1/hust/doe/train/startTrain` | 见下文 | 后台训练并交叉验证 |
| `POST /api/v1/hust/doe/inference/startInference` | `id`, `inputs`, `fields?`, `model_id?` | 代理模型批量推理并按目标字段返回 |
| `POST /api/v1/hust/doe/optimize/start` | 见下文 | 后台提交 NSGA-II、单目标或 RL 优化 |
| `POST /api/v1/hust/doe/optimize/stop` | `id` | 发出优化中止请求 |
| `GET /api/v1/hust/doe/optimize/getById?id=...` | `id` | 查询优化状态、参数与结果文件 |

除健康检查外，成功响应统一使用 `code/message/data`。GET 参数通过 query string 传递，
POST 参数使用 JSON 对象；GET 接口不读取请求体。

## 样本生成

LHS 示例：

```json
{
  "id": "doe_ring_001",
  "method": "lhs",
  "param_ranges": {
    "temperature": [900, 1100],
    "speed": [10, 50]
  },
  "n_samples": 20
}
```

响应 `data` 包含 `method`、`param_ranges`、`sample_file` 和 `columns`。样本文件为无表头
TSV，`columns` 保存文件列顺序。LHS 实现会在请求的随机样本之外追加上下界组合，因此
实际行数可能大于 `n_samples`。

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

数据集生成响应中的 `data` 格式为：

```json
{
  "data_file": "<DOE目录>/training/demo_training_dataset.tsv",
  "all_var_list": ["temperature", "speed", "grain", "load"],
  "input_var_count": 2,
  "sample_count": 80,
  "input_names": ["temperature", "speed"],
  "target_names": ["grain", "load"]
}
```

`data_file` 是服务内部产物索引；端上读取数据应使用“按字段获取数据”接口。

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
  "inputs": {
    "temperature": [950, 1000],
    "speed": [20, 30]
  },
  "fields": ["grain"],
  "model_id": null
}
```

`inputs` 既可使用上述按字段组织的数组，也兼容原有二维数组。不指定 `model_id` 时自动
选择平均评价分最高的模型；不传 `fields` 时返回模型的全部目标。响应示例：

```json
{
  "code": 0,
  "message": "推理完成",
  "data": {"grain": [123.45, 120.67]}
}
```

服务会在 DOE 状态的 `inference` 区块保存本次所用 `model_id` 和所有目标的预测结果，
因此之后可用字段取数接口读取最近一次推理的任意目标。新的推理会覆盖该区块。

## 按字段获取数据

样本、训练数据集、优化结果以及最近一次推理结果统一通过以下接口读取：

```http
GET /api/v1/hust/doe/data/get
```

请求中的 `data_type` 支持：

| `data_type` | 数据来源 | 可用字段记录位置 |
|---|---|---|
| `sample` | 最近生成的 LHS/全因子样本 | `doe.json` 的 `sample.columns` |
| `dataset` | Demo 合成训练数据集 | `training.dataset.all_var_list` |
| `optimization` | 最近完成的参数化 NSGA-II 解集 | `optimization.result.task_info.result_columns` |
| `inference` | 最近一次推理 | `inference.columns` |

例如只获取 LHS 样本中的温度：

```http
GET /api/v1/hust/doe/data/get?id=doe_ring_001&data_type=sample&fields=temperature
```

多个字段重复传递 `fields`，例如
`fields=temperature&fields=speed`。GET 接口不接收 JSON 请求体。

响应只包含所请求的字段：

```json
{
  "code": 0,
  "message": "数据获取完成",
  "data": {"temperature": [900.0, 925.5, 980.2, 1100.0]}
}
```

优化结果文件为无表头 TSV。服务端在 DOE 状态中单独记录列顺序，典型顺序为决策变量、
目标变量和 `feasible`；端上无需解析或依赖文件表头。

请求不存在的字段返回 HTTP 400；对应数据尚未生成或文件不存在返回 HTTP 409。该接口
只读取当前协议生成的 DOE，不兼容开发阶段的旧状态格式。

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
