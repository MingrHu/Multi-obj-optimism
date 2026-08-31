# DOE HTTP API（v1）

服务启动：`mobo-api`，默认监听 `0.0.0.0:5000`。可通过 `MOBO_API_HOST` 和
`MOBO_API_PORT` 修改。客户端示例：`python -m mobo.api.demo`

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
后台训练和优化提交成功使用 HTTP 202

## 接口

| 方法与路径 | 关键请求参数 | 说明 |
|---|---|---|
| `GET /health` | - | 健康检查 |
| `POST /api/v1/doe/add` | `id?`, `name?`, `description?`, `metadata?` | 创建 DOE；未传 ID 时自动生成 |
| `GET /api/v1/doe/list` | - | 查询 DOE 列表 |
| `POST /api/v1/doe/delete` | `id` | 删除 DOE 及其样本、模型、训练和优化文件 |
| **`POST /api/v1/hust/doe/sample/generate`** | **`id`, `method`, `param_ranges`, `n_samples?`, `level_nums?`** | **LHS/全因子采样** |
| `POST /api/v1/hust/doe/dataset/generate` | `id`, `param_ranges`, `target_names`, `input_names?`, `n_samples?`, `seed?`, `noise_ratio?` | 在 DOE 训练目录生成完整流程演示数据集 |
| `GET /api/v1/hust/doe/data/get` | `id`, `resource_id`, `fields` | 按资源索引和字段获取样本、数据集、优化或推理结果 |
| **`GET /api/v1/hust/doe/train/progress`** | **`id`** | **查询代理模型训练状态、阶段、进度及已训练模型** |
| **`POST /api/v1/hust/doe/train/delete`** | **`id`** | **删除训练记录和代理模型** |
| **`POST /api/v1/hust/doe/train/stop`** | **`id`** | **发出训练中止请求** |
| **`POST /api/v1/hust/doe/train/startTrain`** | **见下文** | **后台训练并交叉验证** |
| **`POST /api/v1/hust/doe/inference/startInference`** | **`id`, `inputs`, `fields?`, `model_id?`** | **代理模型批量推理并按目标字段返回** |
| **`POST /api/v1/hust/doe/optimize/start`** | **见下文** | **后台提交 NSGA-II、单目标或 RL 优化** |
| **`POST /api/v1/hust/doe/optimize/stop`** | **`id`** | **发出优化中止请求** |
| **`GET /api/v1/hust/doe/optimize/getById`** | **`id`** | **查询优化状态、参数与结果文件** |

除健康检查外，成功响应统一使用 `code/message/data`。GET 参数通过 query string 传递，
POST 参数使用 JSON 对象；GET 接口不读取请求体



## 1 样本数据生成

**POST /api/v1/hust/doe/sample/generate**

**请求字段说明**：

LHS 拉丁超立方：

```json
{
  "id": "doe_sample_001",
  "method": "lhs",
  "param_ranges": {
    "X1": [0, 1],
    "X2": [10, 15],
    "X3": [1000, 2000]
  },
  "n_samples": 4
}
```

Full 全因子：

```json
{
  "id": "doe_sample_001",
  "method": "full",
  "param_ranges": {
    "X1": [0, 1],
    "X2": [10, 15],
    "X3": [1000, 2000]
  },
  "level_nums": [3, 3, 3]
}
```

LHS 使用 `n_samples` 指定基础随机样本数，后端还会追加所有变量上下界的笛卡尔
组合并去重。Full 使用 `level_nums` 指定各变量水平数，其顺序必须与
`param_ranges` 一致，不需要另外输入样本总数。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已创建的 DOE 唯一标识 |
| `method` | string | 否 | `lhs` 或 `full`，默认 `lhs` |
| `param_ranges` | object | 是 | 变量名称到 `[lower, upper]` 的映射，必须满足 `lower < upper` |
| `n_samples` | integer | LHS 是 | LHS 基础随机样本数，必须为正整数 |
| `level_nums` | integer array | Full 是 | 各变量水平数，必须与变量数量相同且全部为正整数 |

**成功响应字段说明**：

LHS 成功返回 HTTP 200：

```json
{
  "code": 0,
  "message": "样本生成完成",
  "data": {
    "id": "doe_sample_001",
    "method": "lhs",
    "param_ranges": {
      "X1": [0.0, 1.0],
      "X2": [10.0, 15.0],
      "X3": [1000.0, 2000.0]
    },
    "resource_id": "tos-a1b2c3d4e5f60718293a",
    "resource_type": "sample",
    "columns": ["X1", "X2", "X3"],
    "sample_count": 12,
    "n_samples": 4
  }
}
```

上述3变量请求会生成4个基础 LHS 样本并追加最多 `2³ = 8` 个边界组合，去重前
合计12行。`sample_count` 始终以实际落盘行数为准。

Full 成功返回 HTTP 200：

```json
{
  "code": 0,
  "message": "样本生成完成",
  "data": {
    "id": "doe_sample_001",
    "method": "full",
    "param_ranges": {
      "X1": [0.0, 1.0],
      "X2": [10.0, 15.0],
      "X3": [1000.0, 2000.0]
    },
    "resource_id": "tos-b2c3d4e5f60718293a4b",
    "resource_type": "sample",
    "columns": ["X1", "X2", "X3"],
    "sample_count": 27,
    "level_nums": [3, 3, 3]
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功为 `样本生成完成` |
| `data.id` | string | DOE 唯一标识 |
| `data.method` | string | 实际采样方法 `lhs` 或 `full` |
| `data.param_ranges` | object | 变量名称及上下界 |
| `data.resource_id` | string | 样本数据的不透明资源索引，格式为 `tos-` 加20位十六进制字符 |
| `data.resource_type` | string | 资源类型，样本数据固定为 `sample` |
| `data.columns` | string array | 无表头 TSV 的字段顺序 |
| `data.sample_count` | integer | 文件中实际生成的样本总数 |
| `data.n_samples` | integer | LHS 请求的基础随机样本数，仅 LHS 返回 |
| `data.level_nums` | integer array | 各变量水平数，仅 Full 返回 |

样本文件的实际服务器路径只在服务端 DOE 状态中维护，不会返回给端上。端上使用
`data.resource_id` 调用下一节的数据获取接口读取样本内容。

**失败响应字段说明**：

请求参数不合法返回 HTTP 400：

```json
{
  "code": 1,
  "message": "full 的 level_nums 必须是与 param_ranges 等长的正整数数组",
  "data": {}
}
```

DOE 不存在返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_sample_001",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |



## 2 按资源索引获取DOE数据

**GET /api/v1/hust/doe/data/get**

**请求字段说明**：

以下示例读取样本资源中的 `X1` 和 `X3` 两列。`fields` 是可重复的 query string
参数，GET 请求不使用 JSON 请求体。

```http
GET /api/v1/hust/doe/data/get?id=doe_sample_001&resource_id=tos-a1b2c3d4e5f60718293a&fields=X1&fields=X3
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 资源所属的 DOE 唯一标识 |
| `resource_id` | string | 是 | 生成、训练、推理或优化接口返回的 `tos-xxxxx` 不透明资源索引 |
| `fields` | string | 是 | 需要返回的字段名称，可重复传递，至少一个 |

`resource_id` 不是服务器路径或下载 URL，只能与其所属 DOE 的 `id` 配合使用。服务端
当前支持 `sample`、`dataset`、`inference` 和 `optimization` 四类资源。相同 DOE
重新生成同类资源后，旧索引失效，端上应保存最新业务接口响应中的索引。

**成功响应字段说明**：

成功返回 HTTP 200：

```json
{
  "code": 0,
  "message": "数据获取完成",
  "data": {
    "id": "doe_sample_001",
    "resource_id": "tos-a1b2c3d4e5f60718293a",
    "resource_type": "sample",
    "row_count": 12,
    "values": {
      "X1": [0.25, 0.42, 0.18, 0.35],
      "X3": [1120.0, 1185.0, 1090.0, 1210.0]
    }
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功为 `数据获取完成` |
| `data.id` | string | DOE 唯一标识 |
| `data.resource_id` | string | 本次读取的资源索引 |
| `data.resource_type` | string | `sample`、`dataset`、`inference` 或 `optimization` |
| `data.row_count` | integer | 该资源的总数据行数，不受本次字段数量影响 |
| `data.values` | object | 字段名称到数据数组的映射，键顺序与请求的 `fields` 顺序一致 |

**失败响应字段说明**：

字段不存在返回 HTTP 400：

```json
{
  "code": 1,
  "message": "请求字段不存在：X9，可用字段：X1, X2, X3",
  "data": {}
}
```

资源索引无效、已失效或不属于当前 DOE 时返回 HTTP 404：

```json
{
  "code": 404,
  "message": "数据资源不存在：tos-00000000000000000000",
  "data": {}
}
```

资源记录存在但服务端文件已缺失时返回 HTTP 409：

```json
{
  "code": 409,
  "message": "对应数据尚未生成或结果文件不存在",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，资源不存在为404，资源文件状态冲突为409 |
| `message` | string | 具体错误原因，不包含服务器文件路径 |
| `data` | object | 失败时为空对象 |



## 3 代理模型训练提交/开始

**POST /api/v1/hust/doe/train/startTrain**

**请求字段说明**：

```json
{
  "id": "doe_20260622_001",
  "data_source": {
    "input_data": {
      "labels": ["X1", "X2", "X3", "X4", "X5", "X6", "X7"],
      "samples": [
        [0.25, 18.6, 1120, 220, 240, 22.5, 18.3],
        [0.42, 25.3, 1185, 310, 280, 35.2, 42.6],
        [0.18, 12.4, 1090, 180, 200, 18.6, 15.2],
        [0.35, 21.7, 1210, 290, 310, 38.1, 35.4]
      ]
    },
    "output_data": {
      "labels": ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"],
      "samples": [
        [1050, 0.42, 0.083, 18.5, 2.4, 0.92, 0.92, 0.98],
        [980, 0.55, 0.091, 20.1, 2.8, 0.86, 0.92, 0.95],
        [1125, 0.38, 0.075, 17.8, 2.1, 0.95, 0.92, 1.00],
        [1018, 0.47, 0.088, 19.3, 2.6, 0.89, 0.92, 0.97]
      ]
    }
  },
  "models": [
    {
      "name": "RF",
      "params": {
        "n_estimators": 300,
        "n_jobs": -1
      }
    }
  ],
  "evaluation": {
    "enabled": true,
    "method": "k_fold",
    "n_splits": 2,
    "random_state": 42
  }
}
```

后端校验输入和输出样本后，按照输入字段、输出字段的顺序合并每一行，并在当前
DOE 的 `training` 目录落盘为无表头 TSV。`all_var_list`、`input_var_count` 和
样本总数均由后端计算，端上不需要传递服务端文件路径。

输入和输出样本行数必须相同，每行数据宽度必须与对应 `labels` 数量相同。
`models` 支持 `PRG`、`SVR`、`RF`、`KM` 和 `DNN`，同一模型不能重复提交。
四行数据仅用于展示协议结构，实际训练应根据输入维度和模型复杂度提供足够样本。

各模型参数示例：

```json
[
  {"name": "PRG", "params": {"degree": 2}},
  {"name": "SVR", "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}},
  {"name": "RF", "params": {"n_estimators": 300, "n_jobs": -1}},
  {"name": "KM", "params": {"alpha": 0.1, "n_restarts_optimizer": 10}},
  {
    "name": "DNN",
    "params": {"epochs": 300, "batch_size": 16, "verbose": 0, "patience": 30}
  }
]
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已创建的 DOE 唯一标识 |
| `data_source` | object | 是 | 本次训练使用的内嵌数据 |
| `data_source.input_data` | object | 是 | 输入变量和输入样本 |
| `input_data.labels` | string array | 是 | 输入变量名称，不能为空或重复 |
| `input_data.samples` | `number[][]` | 是 | 输入样本二维数组 |
| `data_source.output_data` | object | 是 | 输出目标和输出样本 |
| `output_data.labels` | string array | 是 | 输出目标名称，不能为空或重复 |
| `output_data.samples` | `number[][]` | 是 | 输出样本二维数组 |
| `models` | object array | 否 | 模型配置列表，默认训练 PRG、SVR、RF 和 KM |
| `models[].name` | string | 是 | 模型名称 `PRG`、`SVR`、`RF`、`KM` 或 `DNN` |
| `models[].params` | object | 否 | 模型超参数，默认使用模型自身默认值 |
| `evaluation` | object | 否 | 模型交叉验证配置 |
| `evaluation.enabled` | boolean | 否 | 是否执行评价，默认 `true` |
| `evaluation.method` | string | 否 | 当前仅支持 `k_fold`，默认 `k_fold` |
| `evaluation.n_splits` | integer | 否 | 折数，范围为2到样本总数，默认5 |
| `evaluation.random_state` | integer | 否 | 评价随机种子，默认42 |

**成功响应字段说明**：

训练在后台异步执行，任务成功提交返回 HTTP 202。HTTP 202 只表示请求已经接受，
不表示所有代理模型已经训练完成。

```json
{
  "code": 0,
  "message": "训练任务已提交",
  "data": {
    "id": "doe_20260622_001",
    "status": "queued",
    "stage": "queued",
    "progress": 0,
    "sample_count": 4,
    "input_names": ["X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    "target_names": ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"],
    "models": ["RF"]
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功为 `训练任务已提交` |
| `data.id` | string | DOE 唯一标识 |
| `data.status` | string | 初始训练状态，固定为 `queued` |
| `data.stage` | string | 初始训练阶段，固定为 `queued` |
| `data.progress` | integer | 初始训练进度，固定为0，完整范围为0到100 |
| `data.sample_count` | integer | 本次训练样本总数 |
| `data.input_names` | string array | 输入变量名称 |
| `data.target_names` | string array | 输出目标名称 |
| `data.models` | string array | 本次提交的模型名称 |

**失败响应字段说明**：

请求数据不合法返回 HTTP 400：

```json
{
  "code": 1,
  "message": "输入样本数量与输出样本数量必须一致",
  "data": {}
}
```

DOE 不存在返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

当前 DOE 已有训练正在运行返回 HTTP 409：

```json
{
  "code": 409,
  "message": "training 已在运行",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，资源不存在为404，状态冲突为409 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |

训练线程启动后发生的模型训练或评价错误不会再由本接口返回。调用方应通过训练进度
接口查询 `status`、`stage`、`progress` 和 `error`。



## 4 代理模型训练中止

**POST /api/v1/hust/doe/train/stop**

当前协议直接使用 DOE 的 `id`，不使用独立 `TrainId`。同一 DOE 同时只允许一个
代理模型训练任务。

**请求字段说明**：

```json
{
  "id": "doe_20260622_001"
}
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 正在训练或曾经训练过的 DOE 唯一标识 |

**成功响应字段说明**：

运行中的训练接受中止请求时返回 HTTP 200：

```json
{
  "code": 0,
  "message": "已发送中止请求",
  "data": {
    "id": "doe_20260622_001",
    "accepted": true,
    "status": "stopping",
    "stage": "stopping",
    "progress": 40
  }
}
```

没有运行中的训练时仍返回 HTTP 200，`accepted` 为 `false`，并返回当前已落盘状态：

```json
{
  "code": 0,
  "message": "没有运行中的训练",
  "data": {
    "id": "doe_20260622_001",
    "accepted": false,
    "status": "finished",
    "stage": "finished",
    "progress": 100
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 是否成功发送中止请求 |
| `data.id` | string | DOE 唯一标识 |
| `data.accepted` | boolean | 运行线程是否接受了本次中止请求 |
| `data.status` | string | 当前训练状态 |
| `data.stage` | string | 当前训练阶段 |
| `data.progress` | integer | 当前训练进度，范围为0到100 |

中止采用协作式取消。`accepted=true` 只表示已设置中止信号，不表示线程已经退出；
调用方应继续查询训练进度，直到状态变为 `stopped`、`finished` 或 `failed`。

**失败响应字段说明**：

缺少或使用非法 `id` 返回 HTTP 400：

```json
{
  "code": 1,
  "message": "id 只能包含字母、数字、下划线、短横线，且长度为 1-128",
  "data": {}
}
```

DOE 不存在返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |



## 5 删除训练代理模型

**POST /api/v1/hust/doe/train/delete**

该接口删除指定 DOE 下的代理模型文件、训练数据文件和训练记录，不删除 DOE 本身、
采样记录或优化记录。训练仍在运行时必须先调用中止接口，并等待训练进入终止状态。

**请求字段说明**：

```json
{
  "id": "doe_20260622_001"
}
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 需要清理训练内容的 DOE 唯一标识 |

**成功响应字段说明**：

删除完成返回 HTTP 200：

```json
{
  "code": 0,
  "message": "训练记录和模型文件已删除",
  "data": {
    "id": "doe_20260622_001",
    "status": "not_started",
    "stage": "not_started",
    "progress": 0
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功为 `训练记录和模型文件已删除` |
| `data.id` | string | DOE 唯一标识 |
| `data.status` | string | 清理后的训练状态，固定为 `not_started` |
| `data.stage` | string | 清理后的训练阶段，固定为 `not_started` |
| `data.progress` | integer | 清理后的训练进度，固定为0 |

**失败响应字段说明**：

训练仍在运行时返回 HTTP 409：

```json
{
  "code": 409,
  "message": "训练正在运行，请先中止",
  "data": {}
}
```

DOE 不存在返回 HTTP 404；缺少或使用非法 `id` 返回 HTTP 400，响应结构与训练中止
接口相同。

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404，训练冲突为409 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |



## 6 查询代理模型训练进度

**GET /api/v1/hust/doe/train/progress**

该接口是 GET 请求，`id` 通过 query string 传递，不接收 JSON 请求体，也不使用
独立 `TrainId`。

**请求字段说明**：

```http
GET /api/v1/hust/doe/train/progress?id=doe_20260622_001
```

| 请求字段 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---:|---|
| `id` | query | string | 是 | 需要查询训练进度的 DOE 唯一标识 |

**成功响应字段说明**：

查询成功返回 HTTP 200：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "id": "doe_20260622_001",
    "status": "running",
    "stage": "training",
    "progress": 40,
    "models": [],
    "error": null,
    "updated_at": "2026-08-28T16:30:00+08:00"
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功固定为 `ok` |
| `data.id` | string | DOE 唯一标识 |
| `data.status` | string | `not_started`、`queued`、`running`、`stopping`、`stopped`、`finished` 或 `failed` |
| `data.stage` | string | 当前训练阶段 |
| `data.progress` | integer | 当前训练进度，范围为0到100 |
| `data.models` | object array | 已完成训练或正在累计的模型记录 |
| `data.error` | string or null | 训练失败提示，无错误时为 `null`，内部异常详情仅由服务端维护 |
| `data.updated_at` | string | DOE 状态最后更新时间，带时区的 ISO 8601 格式 |

训练线程启动后发生错误时，本接口返回 HTTP 200，并通过 `status=failed` 和通用
`error` 提示报告后台任务结果，不向端上暴露内部路径或异常细节。

**失败响应字段说明**：

缺少或使用非法 `id` 返回 HTTP 400：

```json
{
  "code": 1,
  "message": "id 只能包含字母、数字、下划线、短横线，且长度为 1-128",
  "data": {}
}
```

DOE 不存在返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |



## 7 代理模型推理

**POST /api/v1/hust/doe/inference/startInference**

端上协议中的 `ModelId` 对应 `model_id`，`InferencePara` 对应单样本 `inputs`。
仓库协议还需要 DOE 的 `id`，用于定位该 DOE 下的模型和训练字段记录。

**请求字段说明**：

单个样本使用一维数值数组：

```json
{
  "id": "doe_20260622_001",
  "model_id": "tr_doe_20260622_001_2_a1b2c3",
  "inputs": [0.25, 18.6, 1120, 220, 240, 22.5, 18.3],
  "fields": ["Y1", "Y2"]
}
```

批量推理也可以按训练时的输入字段组织，每个字段数组长度必须一致：

```json
{
  "id": "doe_20260622_001",
  "inputs": {
    "X1": [0.25, 0.42],
    "X2": [18.6, 25.3],
    "X3": [1120, 1185],
    "X4": [220, 310],
    "X5": [240, 280],
    "X6": [22.5, 35.2],
    "X7": [18.3, 42.6]
  }
}
```

不指定 `model_id` 时，自动选择当前 DOE 下平均评价分最高的模型；不传 `fields`
时返回该模型的全部输出目标。`inputs` 还支持二维数组形式，用于一次提交多个样本。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已完成代理模型训练的 DOE 唯一标识 |
| `model_id` | string | 否 | 指定当前 DOE 下的代理模型，不传时自动选择评分最高模型 |
| `inputs` | `number[]`、`number[][]` 或 object | 是 | 单样本、批量样本或按输入字段组织的批量数据 |
| `fields` | string array | 否 | 需要返回的输出目标，默认返回全部目标 |

**成功响应字段说明**：

推理成功返回 HTTP 200。即使只推理一个样本，每个输出目标仍使用数组返回：

```json
{
  "code": 0,
  "message": "推理完成",
  "data": {
    "id": "doe_20260622_001",
    "model_id": "tr_doe_20260622_001_2_a1b2c3",
    "resource_id": "tos-c3d4e5f60718293a4b5c",
    "resource_type": "inference",
    "columns": ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"],
    "predictions": {
      "Y1": [1050.25],
      "Y2": [0.42]
    }
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功为 `推理完成` |
| `data.id` | string | DOE 唯一标识 |
| `data.model_id` | string | 本次实际加载的代理模型标识 |
| `data.resource_id` | string | 本次完整推理结果的资源索引 |
| `data.resource_type` | string | 推理结果固定为 `inference` |
| `data.columns` | string array | 该模型全部输出目标的字段顺序 |
| `data.predictions` | object | 本次请求字段到预测结果数组的映射 |
| `data.predictions.<field>` | number array | 对应目标的批量预测结果，顺序与输入样本一致 |

服务会在 DOE 状态的 `inference` 区块保存本次使用的 `model_id` 和全部输出目标结果。
本接口可以通过 `fields` 只返回部分目标，但之后仍可将 `data.resource_id` 传给按资源
索引获取数据接口，读取本次推理的其他目标。新的推理结果会使上一次索引失效。

**失败响应字段说明**：

输入参数数量与训练字段不一致时返回 HTTP 400：

```json
{
  "code": 1,
  "message": "inputs[0] 的参数数量必须为 7",
  "data": {}
}
```

DOE 不存在返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

DOE 尚无可用模型或指定的 `model_id` 不属于当前 DOE 时返回 HTTP 409：

```json
{
  "code": 409,
  "message": "没有可用的已训练代理模型",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404，模型不可用为409 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |



## 8 优化任务提交与开始

**POST /api/v1/hust/doe/optimize/start**

接口支持标准化加权单目标、Pareto 多目标和 PPO 强化学习三种模式。调用方只提交
目标、约束和设计变量，后端从 DOE 训练记录推导完整字段顺序和变量下标。
不指定 `model_id` 时自动选择平均评价分最高的代理模型。

**请求字段说明**：

标准化加权单目标使用 NSGA-II：

```json
{
  "id": "doe_20260622_001",
  "mode": "single",
  "objectives": [
    {"name": "Y1", "direction": "min", "weight": 0.7},
    {"name": "Y2", "direction": "max", "weight": 0.3}
  ],
  "objective_normalization": "standard",
  "constraints": [
    {"name": "Y3", "upper": 0.3},
    {"name": "Y4", "lower": 10, "upper": 50}
  ],
  "decision_variables": [
    {"name": "X1", "lower": 0, "upper": 1},
    {"name": "X2", "lower": 10, "upper": 15}
  ],
  "algorithm": {
    "name": "nsga2",
    "params": {
      "pop_size": 100,
      "n_offsprings": 100,
      "n_gen": 200,
      "seed": 42,
      "eliminate_duplicates": true
    }
  }
}
```

`single` 模式先使用训练时保存的目标标准化器把各目标转换到可比较尺度，再根据
`direction` 调整符号并计算加权和。所有 `weight` 必须非负且总和为1。

Pareto 多目标使用 NSGA-II，各目标不传 `weight`：

```json
{
  "id": "doe_20260622_001",
  "mode": "multi",
  "objectives": [
    {"name": "Y1", "direction": "min"},
    {"name": "Y2", "direction": "min"},
    {"name": "Y6", "direction": "max"}
  ],
  "constraints": [
    {"name": "Y3", "upper": 0.3}
  ],
  "decision_variables": [
    {"name": "X1", "lower": 0, "upper": 1},
    {"name": "X2", "lower": 10, "upper": 15}
  ],
  "algorithm": {
    "name": "nsga2",
    "params": {
      "pop_size": 100,
      "n_offsprings": 100,
      "n_gen": 200,
      "seed": 42,
      "eliminate_duplicates": true
    }
  }
}
```

强化学习使用动态代理模型 PPO 环境：

```json
{
  "id": "doe_20260622_001",
  "mode": "reinforcement_learning",
  "objectives": [
    {"name": "Y1", "direction": "min", "weight": 0.6},
    {"name": "Y6", "direction": "max", "weight": 0.4}
  ],
  "objective_normalization": "standard",
  "constraints": [
    {"name": "Y3", "upper": 0.3},
    {"name": "Y4", "lower": 10, "upper": 50}
  ],
  "decision_variables": [
    {"name": "X1", "lower": 0, "upper": 1},
    {"name": "X2", "lower": 10, "upper": 15}
  ],
  "algorithm": {
    "name": "ppo",
    "params": {
      "total_timesteps": 20000,
      "episode_steps": 100,
      "learning_rate": 0.001,
      "constraint_penalty": 5.0,
      "evaluation_episodes": 10,
      "seed": 42
    }
  }
}
```

PPO 的观测是当前设计变量，动作是相对于变量范围的增量。奖励为标准化加权目标的
相反数，违反上下界约束时扣除 `constraint_penalty` 对应的惩罚。历史硬编码的3变量、
`grain/load` 和公共 PRG 模型不再用于本接口。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已完成代理模型训练的 DOE 唯一标识 |
| `model_id` | string | 否 | 指定当前 DOE 下的模型，不传时自动选择评分最高模型 |
| `mode` | string | 是 | `single`、`multi` 或 `reinforcement_learning` |
| `objectives` | object array | 是 | 优化目标配置，名称必须是所选模型的输出字段 |
| `objectives[].name` | string | 是 | 输出目标名称 |
| `objectives[].direction` | string | 是 | `min` 或 `max` |
| `objectives[].weight` | number | Single和RL是 | 非负权重，总和必须为1 |
| `objective_normalization` | string | Single和RL否 | 当前仅支持 `standard`，默认 `standard` |
| `constraints` | object array | 否 | 目标约束，默认空数组 |
| `constraints[].name` | string | 是 | 被约束的代理模型输出字段 |
| `constraints[].lower` | number | 条件必填 | 下限，与 `upper` 至少提供一个 |
| `constraints[].upper` | number | 条件必填 | 上限，与 `lower` 至少提供一个 |
| `decision_variables` | object array | 是 | 参与优化的输入变量及范围 |
| `decision_variables[].name` | string | 是 | 代理模型输入字段名称 |
| `decision_variables[].lower` | number | 是 | 变量下界 |
| `decision_variables[].upper` | number | 是 | 变量上界，必须大于下界 |
| `algorithm` | object | 是 | 算法名称和参数 |
| `algorithm.name` | string | 是 | Single和Multi使用 `nsga2`，RL使用 `ppo` |
| `algorithm.params` | object | 否 | 算法参数，未传字段使用默认值 |

当前不支持 GA、PSO 和 DE。传入这些名称会返回 HTTP 400，不会自动替换成其他算法。

**成功响应字段说明**：

优化在后台执行，任务成功提交返回 HTTP 202：

```json
{
  "code": 0,
  "message": "优化任务已提交",
  "data": {
    "id": "doe_20260622_001",
    "status": "queued",
    "stage": "queued",
    "progress": 0,
    "mode": "reinforcement_learning",
    "algorithm": "ppo",
    "model_id": "tr_doe_20260622_001_2_a1b2c3",
    "objectives": ["Y1", "Y6"]
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功为 `优化任务已提交` |
| `data.id` | string | DOE 唯一标识 |
| `data.status` | string | 初始状态，固定为 `queued` |
| `data.stage` | string | 初始阶段，固定为 `queued` |
| `data.progress` | integer | 初始进度，固定为0 |
| `data.mode` | string | 本次优化模式 |
| `data.algorithm` | string | 实际使用的 `nsga2` 或 `ppo` |
| `data.model_id` | string | 实际使用的代理模型标识 |
| `data.objectives` | string array | 本次优化目标名称 |

HTTP 202 只表示任务已接受。算法完成、失败或中止状态通过优化查询接口获取。
NSGA-II 和 PPO 的结果均保存为无表头 TSV，字段顺序为设计变量、目标变量、`feasible`。

**失败响应字段说明**：

权重总和不为1或参数不合法返回 HTTP 400：

```json
{
  "code": 1,
  "message": "objectives.weight 总和必须为1",
  "data": {}
}
```

DOE 不存在返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

DOE 没有可用代理模型或优化已在运行时返回 HTTP 409：

```json
{
  "code": 409,
  "message": "没有可用的已训练代理模型",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404，状态冲突为409 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |


## 9 中止优化任务

**POST /api/v1/hust/doe/optimize/stop**

接口根据 DOE 唯一标识请求中止该 DOE 下正在执行的优化线程。中止采用协作式取消，接口
返回成功表示中止信号已发送，不表示算法线程已经在响应返回前退出。最终状态应通过优化
查询接口确认。

**请求字段说明**：

```json
{
  "id": "doe_20260622_001"
}
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | DOE 唯一标识，同时用于定位其正在运行的优化任务 |

**成功响应字段说明**：

运行中的优化接受中止信号时返回 HTTP 200：

```json
{
  "code": 0,
  "message": "已发送中止请求",
  "data": {
    "id": "doe_20260622_001",
    "accepted": true,
    "status": "stopping",
    "stage": "stopping",
    "progress": 10
  }
}
```

DOE 存在但当前没有运行中的优化时仍返回 HTTP 200：

```json
{
  "code": 0,
  "message": "没有运行中的优化",
  "data": {
    "id": "doe_20260622_001",
    "accepted": false,
    "status": "not_started",
    "stage": "not_started",
    "progress": 0
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 是否已发送中止请求 |
| `data.id` | string | DOE 唯一标识 |
| `data.accepted` | boolean | `true` 表示运行线程接受了中止信号，`false` 表示没有运行中的优化 |
| `data.status` | string | 当前优化状态，接受中止后为 `stopping` |
| `data.stage` | string | 当前优化阶段，接受中止后为 `stopping` |
| `data.progress` | integer | 当前记录的优化进度，范围0到100 |

线程处理完中止信号后，`status` 和 `stage` 会更新为 `stopped`。由于底层算法可能正在
执行一个不可立即打断的计算步骤，从 `stopping` 变为 `stopped` 可能存在短暂延迟。

**失败响应字段说明**：

未传 `id` 或 `id` 格式不合法时返回 HTTP 400：

```json
{
  "code": 1,
  "message": "id 只能包含字母、数字、下划线、短横线，且长度为 1-128",
  "data": {}
}
```

DOE 不存在时返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |


## 10 查询优化任务

**GET /api/v1/hust/doe/optimize/getById**

接口根据 DOE 唯一标识查询该 DOE 下最近一次优化的状态、提交参数和执行结果。GET 请求
不使用 JSON 请求体，`id` 通过查询参数传递。

**请求字段说明**：

```text
GET /api/v1/hust/doe/optimize/getById?id=doe_20260622_001
```

| 请求字段 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---:|---|
| `id` | query | string | 是 | DOE 唯一标识 |

**成功响应字段说明**：

尚未提交优化时返回 HTTP 200：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "id": "doe_20260622_001",
    "status": "not_started",
    "stage": "not_started",
    "progress": 0,
    "request": null,
    "result": null,
    "error": null,
    "updated_at": "2026-08-28T10:30:00+00:00"
  }
}
```

优化完成后，`data.result` 包含如下结构：

```json
{
  "optimization_id": "opt_doe_20260622_001_a1b2c3",
  "task_info": {
    "model_id": "tr_doe_20260622_001_2_a1b2c3",
    "optimizer": "nsga2",
    "mode": "multi",
    "decision_var_names": ["X1", "X2"],
    "objective_names": ["Y1", "Y2"],
    "result_columns": ["X1", "X2", "Y1", "Y2", "feasible"],
    "total_generation": 200,
    "pop_size": 100,
    "run_time_sec": 12.5
  },
  "resource_id": "tos-d4e5f60718293a4b5c6d",
  "resource_type": "optimization",
  "columns": ["X1", "X2", "Y1", "Y2", "feasible"],
  "constraint_check": {
    "all_solution_feasible": true,
    "solution_count": 20
  }
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功固定为 `ok` |
| `data.id` | string | DOE 唯一标识 |
| `data.status` | string | `not_started`、`queued`、`running`、`stopping`、`stopped`、`finished` 或 `failed` |
| `data.stage` | string | 当前优化阶段，与状态变化同步 |
| `data.progress` | integer | 当前优化进度，范围0到100 |
| `data.request` | object或null | 后端归一化后的本次优化请求，尚未提交时为 `null` |
| `data.result` | object或null | 优化完成后的结果索引，未完成或失败时为 `null` |
| `data.result.optimization_id` | string | 后端生成的底层优化执行标识 |
| `data.result.task_info` | object | 模型、算法、模式、字段顺序、规模和耗时信息 |
| `data.result.task_info.result_columns` | string array | 无表头结果 TSV 的列顺序 |
| `data.result.resource_id` | string | 优化结果的不透明资源索引 |
| `data.result.resource_type` | string | 优化结果固定为 `optimization` |
| `data.result.columns` | string array | 优化结果的字段顺序，与 `result_columns` 一致 |
| `data.result.constraint_check` | object | 可行解数量与约束检查摘要 |
| `data.error` | string或null | 优化失败提示，其他状态为 `null`，内部异常详情仅由服务端维护 |
| `data.updated_at` | string | DOE 状态最近更新时间，ISO 8601 格式 |

客户端获取优化结果数据时，将 `data.result.resource_id` 作为 `resource_id` 调用第2节
的数据获取接口。优化结果文件路径仅由服务端维护，不出现在 HTTP 响应中。查询接口
返回 HTTP 200 且 `status=failed` 表示后台优化执行失败，`error` 只返回通用失败提示。

**失败响应字段说明**：

未传 `id` 或 `id` 格式不合法时返回 HTTP 400：

```json
{
  "code": 1,
  "message": "id 只能包含字母、数字、下划线、短横线，且长度为 1-128",
  "data": {}
}
```

DOE 不存在时返回 HTTP 404：

```json
{
  "code": 404,
  "message": "DOE 任务不存在：doe_20260622_001",
  "data": {}
}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE 不存在为404 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |









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
