# DOE HTTP API（v1）

服务启动：`mobo-api`，默认监听 `0.0.0.0:5050`。可通过 `MOBO_API_HOST` 和
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
| `GET /health` | - | 健康检查并返回采样、代理模型、推理、遗传优化和强化学习依赖的就绪状态 |
| `POST /api/v1/doe/add` | `id?`, `name?`, `description?`, `metadata?` | 创建 DOE；未传 ID 时自动生成；名称允许重复，ID 唯一 |
| `GET /api/v1/doe/list` | - | 查询 DOE 列表 |
| `POST /api/v1/doe/delete` | `id` | 删除 DOE 及其样本、模型、训练和优化文件 |
| **`POST /api/v1/hust/doe/sample/generate`** | **`id`, `method`, `param_ranges`, `n_samples?`, `include_boundaries?`, `level_nums?`** | **LHS/全因子采样** |
| `POST /api/v1/hust/doe/dataset/generate` | `id`, `param_ranges`, `target_names`, `input_names?`, `n_samples?`, `seed?`, `noise_ratio?` | 在 DOE 训练目录生成完整流程演示数据集 |
| `POST /api/v1/hust/doe/dataset/save` | `id`, `data_source` | 保存训练数据及输入/输出字段定义，不启动训练 |
| `GET /api/v1/hust/doe/data/get` | `id`, `resource_id`, `fields` | 按资源索引和字段获取样本、数据集、优化或推理结果 |
| **`GET /api/v1/hust/doe/train/hyperparameters`** | - | **查询五类代理模型支持的超参数、默认值和约束** |
| **`GET /api/v1/hust/doe/train/progress`** | **`id`, `run_id?`** | **查询当前或指定训练轮次的进度及模型评价** |
| **`POST /api/v1/hust/doe/train/delete`** | **`id`** | **删除训练记录和代理模型** |
| **`POST /api/v1/hust/doe/train/stop`** | **`id`** | **发出训练中止请求** |
| **`POST /api/v1/hust/doe/train/startTrain`** | **见下文** | **后台训练并交叉验证** |
| **`POST /api/v1/hust/doe/inference/startInference`** | **`id`, `inputs`, `fields?`, `model_id?`** | **代理模型批量推理并按目标字段返回** |
| **`POST /api/v1/hust/doe/optimize/start`** | **`id`, `training_run_id?`, `model_id?` 等，见第9节** | **提交优化并返回新的优化 `run_id`** |
| **`POST /api/v1/hust/doe/optimize/stop`** | **`id`** | **发出优化中止请求** |
| **`GET /api/v1/hust/doe/optimize/getById`** | **`id`** | **查询当前优化状态及全部历史轮次的参数与结果索引** |
| **`GET /api/v1/hust/doe/optimize/process`** | **`id`、可选 `run_id`** | **查询单目标、多目标或强化学习的优化过程曲线** |

除健康检查外，成功响应统一使用 `code/message/data`。GET 参数通过 query string 传递，
POST 参数使用 JSON 对象；GET 接口不读取请求体

## 标识与轮次规则

字段名区分大小写，使用下划线形式，不使用 `runid`、`TrainId` 或 `ModelId`。
同一个 DOE 可以多次训练和优化，每次提交产生新的轮次，不使用任务名称定位历史结果。

| 标识 | 来源 | 用途与范围 |
|---|---|---|
| `id` | 创建 DOE 的 `data.id` | DOE 唯一标识，所有任务操作使用它，名称允许重复 |
| 训练 `run_id` | 训练提交的 `data.run_id`，例如 `train_1a2b3c4d5e6f` | 传给训练进度查询的 `run_id`，或优化提交的 `training_run_id` |
| 优化 `run_id` | 优化提交的 `data.run_id`，例如 `run_1a2b3c4d5e6f` | 传给优化过程查询的 `run_id`，也用于在优化查询的 `history` 中定位一轮结果 |
| `current_run_id` | 训练进度或优化查询响应 | 对应模块最近提交的轮次，不一定仍在运行；优化重启恢复时可为 `null` |
| `selected_run_id` | 训练进度查询响应 | 本次实际展示的训练轮次，查询历史时可以不同于 `current_run_id` |
| `training_run_id` | 优化请求字段，以及优化查询的 `request.training_run_id` | 优化所使用的训练轮次，不能传优化 `run_id` |
| `model_id` | 训练进度的 `models[].model_id` | 某轮训练中生成的模型标识，模型族名称 `RF` 等不能替代它 |
| `optimization_id` | 优化结果的 `result.optimization_id` | 底层算法执行标识，不是轮次查询参数，不用于查询收敛曲线 |
| `resource_id` | 业务响应或历史结果中的 `tos-...` | 传给数据获取接口，不是路径，必须与所属 DOE 的 `id` 一起使用 |

| 接口 | 是否支持轮次选择 | 不传轮次时的行为 |
|---|---|---|
| 训练提交 | 不接收客户端指定的 `run_id` | 每次成功提交生成新的训练轮次 |
| 训练进度 | query `run_id`，必须是训练轮次 | 返回当前训练聚合状态，并返回全部轮次摘要 |
| 训练中止 / 删除 | 仅使用 `id`，不支持按轮次操作 | 中止当前训练；删除该 DOE 全部训练记录与模型 |
| 推理 | 仅 `model_id`，不支持 `training_run_id` | 在当前训练聚合记录的模型中选择；指定历史模型不会自动查找历史轮次 |
| 优化提交 | JSON `training_run_id`，必须是训练轮次 | 使用当前训练聚合记录；在所选范围内按 `model_id` 或最高评分选模型 |
| 优化中止 | 仅使用 `id`，不支持 `run_id` | 请求中止当前优化 |
| 优化查询 | 仅 query `id`，不支持 query `run_id` | 返回当前状态及全部 `history`，客户端按 `history[].run_id` 选历史结果 |
| 优化过程 | query `run_id`，必须是优化轮次 | 使用 `current_run_id`；为空时回退到最后一条历史轮次 |

建议客户端保存两个独立字段 `training_run_id` 和 `optimization_run_id`，分别保存两类
提交响应的 `data.run_id`。不要用一个变量覆盖两种轮次，也不要把 `resource_id` 当作轮次。
查询或提交时，轮次与模型必须属于请求的 DOE；历史训练轮次用于优化时必须已经 `finished`。
未声明支持的参数不会因为出现在 JSON 或查询字符串中就生效。

服务会在 Flask/Gunicorn worker 启动阶段预加载 NumPy、Pandas、PyDOE、Joblib、
采样、代理模型训练与评价、推理、遗传优化和强化学习模块。任一运行依赖加载失败时
worker 启动失败，不接受业务请求。依赖全部就绪时健康检查返回 HTTP 200：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "service": "mobo-doe",
    "dependencies": {
      "ready": true,
      "components": {
        "numpy": "2.4.0",
        "pandas": "2.3.3",
        "pyDOE": "0.3.8",
        "sampling": "ready",
        "surrogate_training": "ready",
        "surrogate_evaluation": "ready",
        "inference": "ready",
        "optimization_ga": "ready",
        "optimization_rl": "ready"
      },
      "error": null
    }
  }
}
```

上面的版本号仅为示例，以实际部署的响应为准。健康接口不需要参数。

| 健康响应字段 | 类型 | 说明 |
|---|---|---|
| `code/message` | integer / string | 就绪时为 `0/ok`，未就绪时为 `1/service not ready` |
| `data.service` | string | 固定为 `mobo-doe` |
| `data.dependencies.ready` | boolean | 全部预加载依赖是否就绪，不代表某个 DOE 训练完成 |
| `data.dependencies.components` | object | NumPy、Pandas、PyDOE 实际版本及各运行组件就绪状态 |
| `data.dependencies.error` | string或null | 就绪时为null；状态异常时记录预加载错误 |

未就绪返回 HTTP 503，结构示例：

```json
{"code": 1, "message": "service not ready", "data": {"service": "mobo-doe", "dependencies": {"ready": false, "components": {}, "error": null}}}
```

正常启动阶段导入失败会直接阻止 worker 启动，可能无法访问健康接口，而不是必然收到503。

DOE 的 `id` 是任务唯一标识，展示名称 `name` 允许重复。显式名称会去除首尾空白，未传
`name` 时使用唯一 `id` 作为名称。未传 `id` 时服务端自动生成并通过创建响应的 `data.id`
返回；同 ID 创建返回 HTTP 409。客户端必须保存并使用 `id` 调用训练、推理、优化、查询和
删除接口，不应使用 `name` 定位任务。

创建两个同名任务时会得到两个不同的任务 ID：

```json
{
  "name": "7050环件优化",
  "description": "第一组参数"
}
```

成功返回 HTTP 201：

```json
{
  "code": 0,
  "message": "DOE 任务已创建",
  "data": {
    "id": "doe_a1b2c3d4e5f67890",
    "name": "7050环件优化",
    "description": "第一组参数",
    "metadata": {},
    "status": "created",
    "stage": "created",
    "progress": 0,
    "created_at": "2026-09-15T10:00:00+08:00",
    "updated_at": "2026-09-15T10:00:00+08:00",
    "optimization_run_count": 0,
    "has_optimization_result": false
  }
}
```

再次使用相同 `name` 创建任务仍返回 HTTP 201，但 `data.id` 不同。



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
  "n_samples": 4,
  "include_boundaries": false
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

LHS 使用 `n_samples` 指定基础随机样本数。`include_boundaries=false` 时严格生成
`n_samples` 行；显式设为 `true` 时才追加所有变量上下界的笛卡尔组合并去重。
Full 使用 `level_nums` 指定各变量水平数，其顺序必须与
`param_ranges` 一致，不需要另外输入样本总数。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已创建的 DOE 唯一标识 |
| `method` | string | 否 | `lhs` 或 `full`，默认 `lhs` |
| `param_ranges` | object | 是 | 变量名称到 `[lower, upper]` 的映射，必须满足 `lower < upper` |
| `n_samples` | integer | LHS 是 | LHS 基础随机样本数，必须为正整数 |
| `include_boundaries` | boolean | 否 | LHS 是否追加所有上下界组合，默认 `false`；默认实际行数等于 `n_samples` |
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
    "n_samples": 4,
    "include_boundaries": true
  }
}
```

上述响应对应 `include_boundaries=true`：3变量会生成4个基础 LHS 样本并追加最多
`2³ = 8` 个边界组合，去重前合计12行。设为 `false` 时只落盘4行。
`sample_count` 始终以实际落盘行数为准。

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
| `data.include_boundaries` | boolean | LHS 是否追加边界组合，仅 LHS 返回 |
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
资源有效期按业务区分：重新生成样本或推理会替换同类旧索引；每次训练的独立数据集快照与
每次优化结果使用独立索引，新轮次不会覆盖历史结果。保存或生成新的 DOE 数据集会替换已有
`dataset` 类型资源索引，端上应刷新该索引；历史训练快照属于 `training_dataset` 类型，不受这次替换影响。
删除 DOE 后所有索引失效，删除训练后训练数据集和模型记录被清理；不存在或失效的索引返回404。

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
    "row_count": 4,
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
| `data.resource_type` | string | `sample`、`dataset`、`training_dataset`、`inference` 或 `optimization`，training_dataset表示某轮训练的独立数据快照 |
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



## 3 保存 DOE 训练配置

**POST /api/v1/hust/doe/dataset/save**

该接口用于在不启动训练的情况下，显式保存当前 DOE 的训练数据、字段名称及输入/输出角色。
`data_source` 的结构和校验规则与训练提交接口相同；可选的 `source_name` 仅保存客户端文件名，
不保存或暴露客户端绝对路径。后端将数据按“输入字段在前、输出字段在后”写入 DOE 自身的
`dataset` 目录，并保存字段顺序、输入列数、输入边界、样本数和数据资源索引。

```json
{
  "id": "doe_20260622_001",
  "data_source": {
    "source_name": "training-data.txt",
    "input_data": {
      "labels": ["工件温度", "压下速度"],
      "samples": [[900, 10], [1100, 50]]
    },
    "output_data": {
      "labels": ["载荷", "晶粒尺寸"],
      "samples": [[8.2, 42], [7.9, 37]]
    }
  }
}
```

成功响应返回 `id`、`resource_id`、`columns`、`input_names`、`target_names`、
`input_bounds` 和 `sample_count`。训练或优化正在运行时修改配置返回 HTTP 409。
后续训练仍会校验并保存本次训练实际使用的数据；仅保存配置不会启动后台任务。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已创建的DOE唯一标识 |
| `data_source` | object | 是 | 数据对象，字段与第4节训练请求一致 |
| `data_source.source_name` | string | 否 | 来源文件名，不用于读取客户端文件 |
| `data_source.input_data.labels` | string array | 是 | 非空、不重复的输入字段名称 |
| `data_source.input_data.samples` | number二维数组 | 是 | 输入数值矩阵，每行宽度与输入标签数一致 |
| `data_source.output_data.labels` | string array | 是 | 非空、不重复且不与输入重名的输出字段名称 |
| `data_source.output_data.samples` | number二维数组 | 是 | 输出数值矩阵，行数须与输入一致 |

成功返回 HTTP 200：

```json
{
  "code": 0,
  "message": "DOE 配置与训练数据已保存",
  "data": {
    "id": "doe_20260622_001",
    "resource_id": "tos-a1b2c3d4e5f60718293",
    "columns": ["工件温度", "压下速度", "载荷", "晶粒尺寸"],
    "input_names": ["工件温度", "压下速度"],
    "target_names": ["载荷", "晶粒尺寸"],
    "input_bounds": [
      {"name": "工件温度", "lower": 900, "upper": 1100},
      {"name": "压下速度", "lower": 10, "upper": 50}
    ],
    "sample_count": 2
  }
}
```

| 响应字段 | 类型 | 说明 |
|---|---|---|
| `code/message` | integer / string | 成功码和保存提示 |
| `data.id` | string | DOE唯一标识 |
| `data.resource_id` | string | 数据集索引，按第2节读取；本响应不包含resource_type |
| `data.columns` | string array | 无表头TSV列顺序，先输入后输出 |
| `data.input_names/target_names` | string array | 输入和输出名称 |
| `data.input_bounds` | object array | 输入范围摘要，每项含name、lower、upper，取数据最小值与最大值 |
| `data.sample_count` | integer | 样本行数，不是输入或输出字段数 |

输入输出行数不一致返回 HTTP 400：

```json
{"code": 1, "message": "输入样本数量与输出样本数量必须一致", "data": {}}
```

后台任务运行时返回 HTTP 409：

```json
{"code": 409, "message": "任务正在运行，无法修改 DOE 配置", "data": {}}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE不存在为404，运行冲突为409 |
| `message/data` | string / object | 错误原因和空对象 |

本接口不创建训练轮次，不返回run_id。随后训练仍需按第4节提交data_source或服务端兼容文件参数，
不能只传本接口的resource_id或id就默认启动已保存数据的训练。


## 4 代理模型训练提交/开始

**POST /api/v1/hust/doe/train/startTrain**

**请求字段说明**：

```json
{
  "id": "doe_20260622_001",
  "data_source": {
    "source_name": "training-data.txt",
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
    "random_state": 42,
    "max_workers": "auto"
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
  {"name": "KM", "params": {"alpha": 0.1, "n_restarts_optimizer": 20}},
  {
    "name": "DNN",
    "params": {"epochs": 1000, "batch_size": 16, "verbose": 0, "patience": 50}
  }
]
```

`params` 可以省略或传空对象。后端只接受下表列出的字段，并把调用方覆盖值与默认值合并；未知字段、
错误类型、越界值或不一致的上下界在启动后台任务前返回 HTTP 400。训练模型快照与 K 折交叉验证使用
同一组完整有效参数，完整参数写入训练请求与模型记录，调用方显式覆盖项另存为
`param_overrides`，便于界面恢复。

**PRG 参数**

| 字段 | 默认值 | 约束 |
|---|---:|---|
| `degree` | `2` | 1～10 的整数 |
| `include_bias` | `false` | 布尔值 |
| `fit_intercept` | `true` | 布尔值 |

**SVR 参数**

| 字段 | 默认值 | 约束 |
|---|---:|---|
| `kernel` | `rbf` | `linear` / `poly` / `rbf` / `sigmoid` |
| `C` | `1.0` | 正数 |
| `epsilon` | `0.1` | 非负数 |
| `gamma` | `scale` | 正数或 `scale` / `auto` |
| `degree` | `3` | 1～10 的整数，仅 `poly` 核使用 |
| `coef0` | `0.0` | 数值 |
| `shrinking` | `true` | 布尔值 |
| `tol` | `0.001` | 正数 |
| `max_iter` | `-1` | `-1` 或正整数 |

**RF 参数**

| 字段 | 默认值 | 约束 |
|---|---:|---|
| `n_estimators` | `300` | 1～5000 的整数 |
| `criterion` | `squared_error` | `squared_error` / `absolute_error` / `friedman_mse` / `poisson` |
| `max_depth` | `null` | `null` 或正整数 |
| `min_samples_split` | `2` | ≥2 的整数，或 (0,1] 小数 |
| `min_samples_leaf` | `1` | ≥1 的整数，或 (0,1] 小数 |
| `max_features` | `1.0` | `null`、正整数、(0,1] 小数、`sqrt` 或 `log2` |
| `bootstrap` | `true` | 布尔值 |
| `max_samples` | `null` | `bootstrap=true` 时可用：正整数或 (0,1] 小数 |
| `random_state` | `42` | `null` 或整数 |
| `n_jobs` | `-1` | 非零整数，`-1` 表示使用全部可用核心 |

**KM 参数**

| 字段 | 默认值 | 约束 |
|---|---:|---|
| `alpha` | `0.1` | 非负数 |
| `n_restarts_optimizer` | `20` | 非负整数 |
| `normalize_y` | `false` | 布尔值 |
| `random_state` | `42` | `null` 或整数 |
| `constant_value` | `1.0` | 正数 |
| `constant_lower` / `constant_upper` | `0.001` / `1000000` | 正数且下界小于上界 |
| `length_scale` | `1.0` | 正数 |
| `length_scale_lower` / `length_scale_upper` | `0.1` / `10000` | 正数且下界小于上界 |

**DNN 参数**

| 字段 | 默认值 | 约束 |
|---|---:|---|
| `hidden_layer_1` / `hidden_layer_2` / `hidden_layer_3` | `64` / `32` / `16` | 1～4096 的整数 |
| `activation` | `relu` | `relu` / `tanh` / `sigmoid` / `elu` / `selu` |
| `dropout_1` / `dropout_2` | `0.2` / `0.1` | [0,1) 数值 |
| `batch_normalization` | `true` | 布尔值 |
| `learning_rate` | `0.001` | 正数 |
| `epochs` | `1000` | 1～100000 的整数 |
| `batch_size` | `16` | 正整数 |
| `verbose` | `0` | `0` / `1` / `2`；默认关闭逐轮控制台输出，避免容器日志被大量训练轮次淹没 |
| `patience` | `50` | 正整数 |
| `reduce_lr_factor` | `0.2` | (0,1) 数值 |
| `reduce_lr_patience` | `5` | 正整数 |
| `min_lr` | `0.000001` | 非负数 |

客户端可通过 **GET `/api/v1/hust/doe/train/hyperparameters`** 动态读取以上元数据。响应中
`data.models.<模型>.<参数>` 包含 `type`、`default`、`label`，以及适用的 `choices`、
`minimum`、`maximum`、`exclusive_minimum`、`exclusive_maximum` 和 `nullable`。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已创建的 DOE 唯一标识 |
| `data_source` | object | 是 | 本次训练使用的内嵌数据 |
| `data_source.source_name` | string | 否 | 原始文件名，仅用于界面恢复；不要传客户端绝对路径 |
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
| `evaluation.max_workers` | integer or string | 否 | 不同输出目标的并发线程数，可取1～64或`auto`，默认`auto` |

交叉验证按模型族依次执行，避免多个模型同时争抢 CPU 和内存；同一模型的不同输出目标由线程池并行，
但单个输出目标内部的 K 折仍保持串行。`max_workers=auto` 时取输出目标数、逻辑 CPU 数一半和8三者
的最小值（至少1）。如模型自身还启用了内部并行（例如 RF 的 `n_jobs=-1`），可显式传
`max_workers=1` 避免嵌套并行。

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
    "run_id": "train_1a2b3c4d5e6f",
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
| `data.run_id` | string | 本次独立训练轮次标识，前缀为 `train_` |
| `data.sample_count` | integer | 本次训练样本总数 |
| `data.input_names` | string array | 输入变量名称 |
| `data.target_names` | string array | 输出目标名称 |
| `data.models` | string array | 本次提交的模型名称 |

保存 `data.run_id` 后，可使用第7节的 `run_id` 查询本轮进度和评价结果；本轮完成后，
把同一值作为第9节优化提交的 `training_run_id`，固定优化使用的模型与训练字段来源。
重新训练会生成新的轮次，不覆盖历史模型；提交响应不包含最终模型 `model_id`，需查询训练进度获取。

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



## 5 代理模型训练中止

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
DNN 正式训练的调度线程收到中止信号后会终止并回收该任务的所有训练子进程。

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



## 6 删除训练代理模型

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



## 7 查询代理模型训练进度

**GET /api/v1/hust/doe/train/progress**

该接口是 GET 请求，`id` 通过 query string 传递，不接收 JSON 请求体，也不使用
独立 `TrainId`。

**请求字段说明**：

```http
GET /api/v1/hust/doe/train/progress?id=doe_20260622_001&run_id=train_1a2b3c4d5e6f
```

| 请求字段 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---:|---|
| `id` | query | string | 是 | 需要查询训练进度的 DOE 唯一标识 |
| `run_id` | query | string | 否 | 指定训练历史轮次；不传时返回当前或最新训练状态 |

**成功响应字段说明**：

训练进行中查询成功返回 HTTP 200。以下两个训练响应示例省略了较长的 `history` 数组，
该字段实际始终返回；历史条目结构详见下方字段表。完成示例仅展示一个目标的评价条目：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "id": "doe_20260622_001",
    "status": "running",
    "stage": "training",
    "progress": 40,
    "current_run_id": "train_1a2b3c4d5e6f",
    "selected_run_id": "train_1a2b3c4d5e6f",
    "current_model": "RF",
    "current_model_index": 1,
    "total_models": 1,
    "input_names": ["workpiece_temperature", "die_temperature"],
    "target_names": ["load", "grain_size", "roundness"],
    "input_bounds": [
      {"name": "workpiece_temperature", "lower": 900.0, "upper": 1100.0},
      {"name": "die_temperature", "lower": 150.0, "upper": 250.0}
    ],
    "dataset": {
      "resource_id": "tos-a1b2c3d4e5f6",
      "columns": ["workpiece_temperature", "die_temperature", "load", "grain_size", "roundness"],
      "sample_count": 100,
      "source_name": "training-data.txt"
    },
    "model_configs": [
      {
        "name": "RF",
        "params": {
          "n_estimators": 500,
          "criterion": "squared_error",
          "max_depth": null,
          "min_samples_split": 2,
          "min_samples_leaf": 1,
          "max_features": 1.0,
          "bootstrap": true,
          "max_samples": null,
          "random_state": 42,
          "n_jobs": -1
        },
        "param_overrides": {"n_estimators": 500}
      }
    ],
    "models": [],
    "error": null,
    "updated_at": "2026-08-28T16:30:00+08:00"
  }
}
```

训练和交叉验证完成后，`models` 返回代理模型及各输出目标的评价结果：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "id": "doe_20260622_001",
    "status": "finished",
    "stage": "finished",
    "progress": 100,
    "current_run_id": "train_1a2b3c4d5e6f",
    "selected_run_id": "train_1a2b3c4d5e6f",
    "current_model": null,
    "current_model_index": null,
    "total_models": 1,
    "input_names": ["workpiece_temperature", "die_temperature"],
    "target_names": ["load", "grain_size", "roundness"],
    "input_bounds": [
      {"name": "workpiece_temperature", "lower": 900.0, "upper": 1100.0},
      {"name": "die_temperature", "lower": 150.0, "upper": 250.0}
    ],
    "dataset": {
      "resource_id": "tos-a1b2c3d4e5f6",
      "columns": ["workpiece_temperature", "die_temperature", "load", "grain_size", "roundness"],
      "sample_count": 100,
      "source_name": "training-data.txt"
    },
    "model_configs": [
      {
        "name": "RF",
        "params": {
          "n_estimators": 500,
          "criterion": "squared_error",
          "max_depth": null,
          "min_samples_split": 2,
          "min_samples_leaf": 1,
          "max_features": 1.0,
          "bootstrap": true,
          "max_samples": null,
          "random_state": 42,
          "n_jobs": -1
        },
        "param_overrides": {"n_estimators": 500}
      }
    ],
    "models": [
      {
        "model_id": "tr_doe_20260622_001_2_a1b2c3",
        "model_index": 2,
        "model_family": "RF",
        "hyper_params": {
          "n_estimators": 500, "criterion": "squared_error", "max_depth": null,
          "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0,
          "bootstrap": true, "max_samples": null, "random_state": 42, "n_jobs": -1
        },
        "target_names": ["load", "grain_size", "roundness"],
        "train_cost_sec": 0.52,
        "score": 0.91,
        "evaluation": [
          {
            "model_name": "RF",
            "target_index": 0,
            "target_name": "load",
            "n_splits": 5,
            "n_samples": 100,
            "r2_mean": 0.956,
            "r2_std": 0.018,
            "nmae_mean": 0.042,
            "nmae_std": 0.006,
            "mae_mean": 1.25,
            "mae_std": 0.13,
            "rmse_mean": 1.62,
            "rmse_std": 0.17,
            "max_error_mean": 3.84,
            "max_error_std": 0.41,
            "train_time_mean_s": 0.11,
            "train_time_min_s": 0.09,
            "train_time_max_s": 0.14,
            "predict_time_mean_s": 0.002,
            "predict_time_min_s": 0.001,
            "predict_time_max_s": 0.003,
            "score": 0.93
          }
        ]
      }
    ],
    "error": null,
    "updated_at": "2026-08-28T16:35:00+08:00"
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
| `data.current_run_id` | string or null | 当前 DOE 最近提交的训练轮次 |
| `data.selected_run_id` | string or null | 本次响应展示的训练轮次 |
| `data.history` | object array | 按提交顺序排列的全部训练轮次摘要，不是当前轮次的详细评价结果 |
| `data.history[].run_id` | string | 训练轮次，可作为本接口的 query `run_id` 或优化提交的 `training_run_id` |
| `data.history[].status/stage/progress` | string / string / number | 该历史轮次的状态、阶段和进度 |
| `data.history[].dataset` | object | 该轮数据集的资源索引、列顺序、样本数和来源名称，字段可能按实际记录省略 |
| `data.history[].models` | object array | 模型摘要，包含 `model_id/model_index/model_family/target_names/train_cost_sec/score`，不包含完整评价指标 |
| `data.history[].model_configs` | object array | 该轮模型参数，结构同顶层 `model_configs` |
| `data.history[].best_model` | string或null | 最佳模型的 `model_id`；尚未评选时为null |
| `data.history[].error` | string或null | 历史记录的错误信息；可能保留历史内部异常，不等同于顶层通用失败提示 |
| `data.history[].created_at/updated_at` | string | 该轮提交和更新时间，ISO 8601 格式 |
| `data.current_model` | string or null | 当前正在训练或交叉验证的模型族；终态为空 |
| `data.current_model_index` | integer or null | 当前模型在本次所选模型中的从1开始序号；终态为空 |
| `data.total_models` | integer or null | 本次所选模型总数；尚未提交训练时为空 |
| `data.input_names` | string array | 本次训练定义的输入参数名称；未提交训练时为空数组 |
| `data.target_names` | string array | 本次训练定义的输出目标名称；未提交训练时为空数组 |
| `data.input_bounds` | object array | 各输入参数在训练数据中的最小值和最大值；优化界面可据此恢复默认设计边界 |
| `data.dataset` | object | 服务端保存的训练数据集摘要；未保存数据集时为空对象 |
| `data.dataset.resource_id` | string | 训练数据集的不透明资源索引，可通过数据获取接口读取，端上不使用服务端路径 |
| `data.dataset.columns` | string array | 训练数据集的无表头 TSV 列顺序 |
| `data.dataset.sample_count` | integer | 训练数据集样本行数 |
| `data.dataset.source_name` | string | 客户端提交的数据源名称或服务端生成的数据集名称 |
| `data.model_configs` | object array | 本次训练选择的模型配置；未提交训练时为空数组 |
| `data.model_configs[].name` | string | 模型族名称 |
| `data.model_configs[].params` | object | 与默认值合并并通过校验后的完整有效参数 |
| `data.model_configs[].param_overrides` | object | 调用方显式提交的覆盖项；界面据此恢复用户设置，空对象表示全部使用默认值 |
| `data.models` | object array | 已完成训练或正在累计的模型记录 |
| `data.models[].hyper_params` | object | 生成模型快照和交叉验证时共同使用的完整有效参数 |
| `data.models[].model_id` | string | 当前 DOE 下唯一的代理模型标识，可用于推理和优化接口 |
| `data.models[].model_index` | integer | 后端代理模型类型编号 |
| `data.models[].model_family` | string | 代理模型类型，例如 `PRG`、`SVR`、`RF`、`KM` 或 `DNN` |
| `data.models[].target_names` | string array | 当前模型包含的输出目标及其顺序 |
| `data.models[].train_cost_sec` | number | 生成可推理模型快照的训练耗时，单位为秒 |
| `data.models[].score` | number or null | 当前模型所有输出目标评价分的平均值，评价完成前为 `null` |
| `data.models[].evaluation` | object array | 各输出目标的 K 折交叉验证结果，评价完成前为空数组 |
| `data.models[].evaluation[].model_name` | string | 本项评价对应的代理模型类型 |
| `data.models[].evaluation[].target_index` | integer | 输出目标从0开始的索引 |
| `data.models[].evaluation[].target_name` | string | 输出目标名称 |
| `data.models[].evaluation[].n_splits` | integer | K 折交叉验证折数 |
| `data.models[].evaluation[].n_samples` | integer | 参与交叉验证的数据集样本数 |
| `data.models[].evaluation[].r2_mean` | number | 各折 R² 决定系数的平均值，通常越大越好 |
| `data.models[].evaluation[].r2_std` | number | 各折 R² 决定系数的标准差 |
| `data.models[].evaluation[].nmae_mean` | number | 各折归一化最大绝对误差的平均值，不是 MSE |
| `data.models[].evaluation[].nmae_std` | number | 各折归一化最大绝对误差的标准差 |
| `data.models[].evaluation[].mae_mean` | number | 各折平均绝对误差 MAE 的平均值，通常越小越好 |
| `data.models[].evaluation[].mae_std` | number | 各折 MAE 的标准差 |
| `data.models[].evaluation[].rmse_mean` | number | 各折均方根误差 RMSE 的平均值，通常越小越好 |
| `data.models[].evaluation[].rmse_std` | number | 各折 RMSE 的标准差 |
| `data.models[].evaluation[].max_error_mean` | number | 各折最大绝对误差的平均值 |
| `data.models[].evaluation[].max_error_std` | number | 各折最大绝对误差的标准差 |
| `data.models[].evaluation[].train_time_mean_s` | number | 各折训练耗时平均值，单位为秒 |
| `data.models[].evaluation[].train_time_min_s` | number | 各折训练耗时最小值，单位为秒 |
| `data.models[].evaluation[].train_time_max_s` | number | 各折训练耗时最大值，单位为秒 |
| `data.models[].evaluation[].predict_time_mean_s` | number | 各折预测耗时平均值，单位为秒 |
| `data.models[].evaluation[].predict_time_min_s` | number | 各折预测耗时最小值，单位为秒 |
| `data.models[].evaluation[].predict_time_max_s` | number | 各折预测耗时最大值，单位为秒 |
| `data.models[].evaluation[].score` | number or null | 单个输出目标的综合评价分，用于计算模型平均分 |
| `data.error` | string or null | 训练失败提示，无错误时为 `null`，内部异常详情仅由服务端维护 |
| `data.updated_at` | string | DOE 状态最后更新时间，带时区的 ISO 8601 格式 |

当前后端使用 MSE 计算 RMSE，但没有独立返回 `mse_mean` 或 `mse_std`。端上如确实需要
MSE，可根据 `rmse_mean` 平方得到近似展示值，但该结果不等于各折 MSE 的严格平均值；
若协议需要准确的 MSE，应另行确认后再扩展后端返回字段。

训练线程启动后发生错误时，本接口返回 HTTP 200，并通过 `status=failed` 和通用
`error` 提示报告后台任务结果，不向端上暴露内部路径或异常细节。
如果服务进程重启或后台训练线程异常退出，但落盘状态仍为 `queued`、`running` 或
`stopping`，查询接口会将该记录收敛为 `stopped`，避免客户端无限显示运行中。

DNN 正式训练对每个输出目标使用独立进程，默认并发上限为输出数、可用逻辑 CPU 数一半和8的
最小值（至少1）；每个子进程使用1个计算线程，模型超参数保持不变。服务端可通过正整数
环境变量 `MOBO_DNN_MAX_WORKERS` 调整并发上限，端上无需新增请求字段。
进度仍按模型族和训练/评价阶段汇总，不表示各输出的逐轮进度。
启用 K 折评价后，还会针对每个目标额外训练 K 次。
例如8个输出、3折评价会额外训练24个 DNN。调用方可根据 `current_model` 和序号展示
当前阶段；DNN 默认 `verbose=0`。每个正式输出模型结束后会释放 Keras 计算图；并行交叉验证
会在该模型的全部输出目标结束后统一释放，避免训练中的线程被全局会话清理干扰。

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



指定历史训练轮次时，顶层 `models/model_configs/dataset/input_names/target_names/status`
对应所选轮次，`current_run_id` 仍表示 DOE 最近提交轮次，`history` 仍返回全部轮次。
要读取历史模型的完整 `r2/mae/mse/rmse`，请用该训练轮次的 `run_id` 再查询本接口，
不要从仅含模型摘要的 `history[].models` 中寻找完整指标。

训练轮次不存在或不属于当前 DOE 时返回 HTTP 404：

```json
{"code": 404, "message": "训练轮次不存在：train_missing", "data": {}}
```

## 8 代理模型推理

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

不指定 `model_id` 时，自动选择当前训练聚合记录中平均评价分最高的模型；不传 `fields`
时返回该模型的全部输出目标。`inputs` 还支持二维数组形式，用于一次提交多个样本。

本接口当前不读取 `training_run_id` 或 `run_id`，无法指定历史训练轮次。`model_id` 也只在
当前训练记录中匹配；若重新训练清空了当前模型列表，旧模型不能仅凭历史 `model_id` 在本接口中加载。
输入字段顺序取当前训练请求的 `input_names`，获取它们及模型标识使用第7节训练进度接口。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已完成代理模型训练的 DOE 唯一标识 |
| `model_id` | string | 否 | 指定当前训练记录中的代理模型，不传时在该范围内选择评分最高模型 |
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



## 9 优化任务提交与开始

**POST /api/v1/hust/doe/optimize/start**

接口支持标准化加权单目标、Pareto 多目标和 PPO 强化学习三种模式。调用方只提交
目标、约束和设计变量，后端从 DOE 训练记录推导完整字段顺序和变量下标。
不指定 `model_id` 时，在所选训练轮次的模型中自动选择平均评价分最高的模型。
下列示例显式指定 `training_run_id`，其值来自第4节训练提交响应的 `data.run_id`；
省略时使用当前训练聚合记录，不会自动在全部历史轮次中搜索模型。

**请求字段说明**：

标准化加权单目标使用 NSGA-II：

```json
{
  "id": "doe_20260622_001",
  "mode": "single",
  "training_run_id": "train_1a2b3c4d5e6f",
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
  "training_run_id": "train_1a2b3c4d5e6f",
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
  "training_run_id": "train_1a2b3c4d5e6f",
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
| `training_run_id` | string | 否 | 指定本 DOE 下已经完成的训练轮次；不传或空字符串使用当前训练聚合记录；建议显式固定来源 |
| `model_id` | string | 否 | 指定所选训练轮次内的模型；省略时在该范围内选择评分最高模型，不在全部历史轮次中搜索 |
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

| `algorithm.params` 字段 | 适用算法 | 默认值 | 约束与含义 |
|---|---|---|---|
| `pop_size` | nsga2 | 100 | 正整数，种群规模 |
| `n_offsprings` | nsga2 | 100 | 正整数，每代子代数量 |
| `n_gen` | nsga2 | 200 | 正整数，迭代代数 |
| `eliminate_duplicates` | nsga2 | true | boolean，是否去除重复个体 |
| `seed` | 两种算法 | 42 | 非负整数，随机种子 |
| `total_timesteps` | ppo | 20000 | 正整数，请求的训练步数，不是回合数 |
| `episode_steps` | ppo | 100 | 正整数，每个训练回合最大步数 |
| `learning_rate` | ppo | 0.001 | 正数，学习率 |
| `constraint_penalty` | ppo | 5.0 | 正数，约束惩罚系数 |
| `evaluation_episodes` | ppo | 10 | 正整数，训练后评价回合数，不写入训练收敛曲线 |
| `action_step_ratio` | ppo | 0.05 | 大于0且不超过1，动作相对变量范围的步长比例 |
| `max_solutions` | ppo | 100 | 正整数，最多保留的结果数量 |

无约束时省略 `constraints` 或传 `[]`；也可只传需要约束的部分目标。
`lower=0, upper=0` 表示要求该目标等于0，不表示未设置约束。
设计变量可只包含部分输入，其余输入由算法按所选训练数据确定基准值。

**成功响应字段说明**：

优化在后台执行，任务成功提交返回 HTTP 202：

```json
{
  "code": 0,
  "message": "优化任务已提交",
  "data": {
    "id": "doe_20260622_001",
    "run_id": "run_1a2b3c4d5e6f",
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
| `data.run_id` | string | 本次新建的优化轮次，前缀为 `run_`，保存后用于第12节收敛曲线查询与第11节历史结果定位 |
| `data.status` | string | 初始状态，固定为 `queued` |
| `data.stage` | string | 初始阶段，固定为 `queued` |
| `data.progress` | integer | 初始进度，固定为0 |
| `data.mode` | string | 本次优化模式 |
| `data.algorithm` | string | 实际使用的 `nsga2` 或 `ppo` |
| `data.model_id` | string | 实际使用的代理模型标识 |
| `data.objectives` | string array | 本次优化目标名称 |

HTTP 202 只表示任务已接受。算法完成、失败或中止状态通过优化查询接口获取。
每次重新优化返回新的 `data.run_id`。客户端应在提交成功后更新所展示的优化轮次，
轮询第11节状态并使用第12节 `run_id` 查询同一轮曲线，不能继续展示上一轮的曲线。
NSGA-II 和 PPO 的结果均保存为无表头 TSV，字段顺序为设计变量、目标变量、`feasible`。
NSGA-II 会在单次运行内缓存已经评估的设计变量；同批候选解先去重，未缓存候选解按代批量送入
代理模型。该处理只减少重复推理和并行调度开销，不改变目标方向、约束或选择、交叉、变异语义。

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
| `code` | integer | 参数错误为1，DOE或指定训练轮次不存在为404，状态冲突为409 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |

指定不存在或不属于本DOE的训练轮次返回 HTTP 404：

```json
{"code": 404, "message": "训练轮次不存在：train_missing", "data": {}}
```

指定尚未完成的历史训练轮次返回 HTTP 409：

```json
{"code": 409, "message": "所选训练轮次尚未完成", "data": {}}
```


## 10 中止优化任务

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
如果服务进程在 `queued`、`running` 或 `stopping` 阶段退出，后台线程不会跨进程恢复；
服务再次启动时会把没有对应运行线程的优化任务及其当前运行版本收敛为 `stopped`，保留请求、
进度和历史记录，并清空 `current_run_id`，避免客户端无限显示正在运行或正在停止。

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


## 11 查询优化任务

**GET /api/v1/hust/doe/optimize/getById**

接口根据 DOE 唯一标识查询该 DOE 下最近一次优化的状态、提交参数和执行结果。GET 请求
不使用 JSON 请求体，`id` 通过查询参数传递。
本接口不读取 query `run_id`，不能通过它改变顶层 `request/result` 的轮次。
历史查询方式是在 `data.history` 中按 `run_id` 选择记录，再读取该记录的 `result`。

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
    "current_run_id": null,
    "history": [],
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
| `data.current_run_id` | string或null | 最近一次提交的优化轮次，来自提交响应的 `data.run_id`；服务重启收敛中断状态后可为空，不表示历史被清空 |
| `data.history` | object array | 按提交顺序保存的全部优化轮次，尚未提交时为空数组 |
| `data.history[].run_id` | string | 优化轮次，用于定位该条记录及查询第12节曲线 |
| `data.history[].status/stage/progress` | string / string / number | 该轮状态、阶段、进度，不一定与顶层当前状态一致 |
| `data.history[].request` | object | 该轮归一化请求，结构与顶层 `request` 一致，不包含模型文件路径 |
| `data.history[].result` | object或null | 该轮结果，结构与顶层 `result` 一致，未产出结果时为null |
| `data.history[].error` | string或null | 该轮记录的错误信息，历史记录可能保留内部异常，不等同于顶层通用失败提示 |
| `data.history[].created_at/updated_at` | string | 该轮提交及更新时间，ISO 8601 格式 |
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
的数据获取接口。历史版本使用 `data.history[].result.resource_id` 读取。每次优化运行使用独立
结果目录和资源索引，新运行不会覆盖旧版本。优化结果文件路径仅由服务端维护，不出现在 HTTP 响应中。查询接口
返回 HTTP 200 且 `status=failed` 表示后台优化执行失败，`error` 只返回通用失败提示。

`request` 是归一化执行配置，不是请求原文，字段如下。历史记录的 `request` 使用同一结构。

| `request` 字段 | 类型 | 含义 |
|---|---|---|
| `training_run_id` | string或null | 实际绑定的训练轮次，历史兼容记录可能为空 |
| `model_id` | string | 实际选用的模型标识 |
| `requested_mode` | string | 客户端请求的 `single/multi/reinforcement_learning`，界面应使用它恢复模式 |
| `mode` | string | 内部目标组合方式，仅为 `single/multi`；强化学习此字段为 `single` |
| `optimizer` | string | 内部算法名 `nsga2/rl`，其中 `rl` 对应提交接口的 `ppo` |
| `objective_names` | string array | 参与优化的输出目标顺序 |
| `objective_config` | object array | 每项包含 `name/minimize/weight`；`minimize=true` 对应请求的 `direction=min`，多目标的weight可为空 |
| `objective_normalization` | string | 目标标准化设置 |
| `constraints` | object array | 每项含 `target_obj/constraint_kind/limit_value`，其中 `constraint_kind` 为 `lower/upper`；双边约束拆为两项 |
| `all_var_list` | string array | 所选训练轮次的全部字段，先输入后输出 |
| `input_var_count` | integer | 输入字段数量 |
| `decision_var_names` | string array | 实际参与优化的输入变量顺序 |
| `decision_var_indices` | integer array | 各设计变量在输入字段中的从0开始下标 |
| `decision_bounds` | object array | 与设计变量对应的 `lower/upper` |
| `optimizer_config` | object | 合并默认值后的算法参数 |
| `output_config` | object | 内部输出配置，当前提交接口默认空对象 |

例如，历史轮次摘要可能如下（仅展示 `data.history` 的一条记录）：

```json
{
  "run_id": "run_1a2b3c4d5e6f",
  "status": "finished",
  "stage": "finished",
  "progress": 100,
  "request": {
    "training_run_id": "train_1a2b3c4d5e6f",
    "model_id": "tr_doe_20260622_001_2_a1b2c3",
    "requested_mode": "multi",
    "mode": "multi",
    "objective_names": ["Y1", "Y2"],
    "objective_config": [
      {"name": "Y1", "minimize": true, "weight": null},
      {"name": "Y2", "minimize": true, "weight": null}
    ],
    "objective_normalization": "standard",
    "constraints": [],
    "all_var_list": ["X1", "X2", "Y1", "Y2"],
    "input_var_count": 2,
    "decision_var_names": ["X1", "X2"],
    "decision_var_indices": [0, 1],
    "decision_bounds": [{"lower": 0, "upper": 1}, {"lower": 10, "upper": 15}],
    "optimizer_config": {"pop_size": 100, "n_offsprings": 100, "eliminate_duplicates": true, "n_gen": 200, "seed": 42},
    "output_config": {},
    "optimizer": "nsga2"
  },
  "result": {
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
    "constraint_check": {"all_solution_feasible": true, "solution_count": 20}
  },
  "error": null,
  "created_at": "2026-10-10T10:00:00+08:00",
  "updated_at": "2026-10-10T10:00:13+08:00"
}
```

取本轮工艺参数和目标值：使用同一条记录的 `result.resource_id` 调用第2节数据获取接口。
取本轮收敛曲线：使用同一条记录的 `run_id` 调用第12节优化过程接口。
两者是不同标识；结果尚未产出时，可能已有过程数据，但 `result` 仍为null。

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









## 12 获取优化过程与收敛曲线

**GET /api/v1/hust/doe/optimize/process**

按 DOE 和优化轮次查询已经落盘的过程数据，支持运行中轮询、完成后查询及历史轮次查询。
本接口只读，不启动优化，不暴露服务器路径，也不需要客户端读取任务文件。

**请求示例与字段说明**：

```text
GET /api/v1/hust/doe/optimize/process?id=doe_20260622_001
GET /api/v1/hust/doe/optimize/process?id=doe_20260622_001&run_id=run_a1b2c3d4e5f6
```

GET 参数放在 URL query 中，不使用 JSON 请求体。`run_id` 取自优化提交响应的
`data.run_id` 或第11节 `data.history[].run_id`，不是代理模型标识或底层 `optimization_id`。

| 请求字段 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---:|---|
| `id` | query | string | 是 | DOE 唯一标识 |
| `run_id` | query | string | 否 | 优化轮次标识，不传时查询最近一次提交的轮次，传入时必须属于当前 DOE |

**成功响应示例与字段说明**：

多目标优化返回 HTTP 200：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "id": "doe_20260622_001",
    "run_id": "run_a1b2c3d4e5f6",
    "status": "running",
    "available": true,
    "point_count": 3,
    "mode": "multi",
    "metric": "hypervolume",
    "x_label": "generation",
    "x": [1, 2, 3],
    "y": [0.0, 12.5, 15.2],
    "direction": "max",
    "hv_config": {
      "objective_names": ["grain", "strength"],
      "offset": [15.0, -650.0],
      "scale": [5.0, 100.0],
      "reference_point": [4.0, 4.0]
    }
  }
}
```

三种模式均只返回一条曲线，`x` 和 `y` 都是一维数组，长度一致。
端上直接将 `x[i]` 和 `y[i]` 配对绘图；单目标的 `null` 绘制为空缺而不是0。

单目标模式的曲线字段示例，其余外层结构相同：

```json
{
  "mode": "single",
  "metric": "best_feasible_weighted_objective",
  "x_label": "generation",
  "x": [1, 2, 3],
  "y": [0.8, 0.5, 0.4],
  "direction": "min",
  "hv_config": null
}
```

强化学习模式的曲线字段示例，其余外层结构相同：

```json
{
  "mode": "reinforcement_learning",
  "metric": "episode_reward_mean_100",
  "x_label": "episode",
  "x": [1, 2, 3],
  "y": [-35.0, -30.0, -28.0],
  "direction": "max",
  "hv_config": null
}
```

| 成功响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为0 |
| `message` | string | 成功固定为 `ok` |
| `data.id` | string | DOE 唯一标识 |
| `data.run_id` | string或null | 查询的优化轮次，尚未提交优化时为 `null` |
| `data.status` | string | 该轮次状态，支持 `not_started/queued/running/stopping/stopped/finished/failed` |
| `data.available` | boolean | 是否已记录至少一个横轴点，不代表已经找到可行解 |
| `data.point_count` | integer | 横轴点数，与 `x` 长度一致 |
| `data.mode` | string或null | `single/multi/reinforcement_learning`，未提交时为 `null` |
| `data.metric` | string或null | 指标定义，见下表，尚未提交优化时为null |
| `data.x_label` | string或null | `generation` 为遗传算法代数，`episode` 为 PPO 已完成训练回合数，尚未提交时为null |
| `data.x` | integer array | 横轴点，按记录顺序递增 |
| `data.y` | number或null array | 单条曲线的一维纵轴数组，长度与x一致 |
| `data.direction` | string或null | 曲线优化方向，`min` 越小越好，`max` 越大越好，尚未提交时为null |
| `data.hv_config` | object或null | 多目标HV的固定计算尺度和参考点，第一代记录前及其他模式为null |
| `data.hv_config.objective_names` | string array | HV所用目标名称与坐标顺序 |
| `data.hv_config.offset` | number array | 固定中心值，最小化目标使用训练均值，最大化目标使用训练均值的负值 |
| `data.hv_config.scale` | number array | 固定尺度，使用模型目标标准化器的scale值，常量目标按标准化器规则使用1 |
| `data.hv_config.reference_point` | number array | 标准化、统一最小化方向后的参考点，第一代确定后不再改变 |

### 三种曲线的坐标轴定义

| `mode` | 横轴 `x[i]` 的含义 | 建议横轴标题 | 纵轴 `y[i]` 的含义 | 建议纵轴标题 | 单位与优劣方向 |
|---|---|---|---|---|---|
| `single` | NSGA-II 当前记录的代数，从第1代开始，第1代包含初始种群评价 | 优化代数（代） | 截至该代，所有已观察可行解中的最小加权标准化目标函数值 | 最佳可行加权目标值 | 无物理单位，越小越好，`direction=min` |
| `multi` | NSGA-II 当前记录的代数，从第1代开始，第1代包含初始种群评价 | 优化代数（代） | 当前这一代种群的可行解前沿相对固定参考点的HV | 可行前沿超体积（HV） | 标准化空间中的无量纲体积，越大越好，`direction=max` |
| `reinforcement_learning` | PPO 已完成的训练回合编号，从第1回合开始，每结束一个回合增加1 | 已完成训练回合数（回合） | 最近最多100个已完成训练回合的累计奖励的算术平均 | 平均回合累计奖励（最近100回合） | 无物理单位，越大越好，`direction=max` |

这三条曲线均属于优化阶段，不是代理模型训练的epoch、loss或R²曲线。
横轴也不是耗时、样本数或接口轮询次数。三种模式的纵轴含义不同，不能使用一个统一的
“目标值”标题，也不能将它们放在同一尺度下比较。

### 单目标：代数与最佳可行加权目标值

响应通过 `metric=best_feasible_weighted_objective`、`x_label=generation` 标识此曲线。
每完成一代种群评价记录一个点。`x=1` 表示初始种群评价完成，不表示代理模型训练了一轮；
正常完成时横轴通常为1到请求的 `algorithm.params.n_gen`，中止或失败时可能只返回其中一段。

纵轴先对模型预测的原始目标值做训练尺度标准化，再按方向和权重组成一个标量。
以 `f_j(p)` 表示工艺参数p对应的第j个预测目标，`mu_j/scale_j` 为训练时保存的
目标标准化器中心值和尺度，公式为：

```text
z_j(p) = (f_j(p) - mu_j) / scale_j
s_j = +1（direction=min），-1（direction=max）
J(p) = sum_j(weight_j * s_j * z_j(p))
y[k] = 截至x[k]这一代已观察到的所有可行解中，J(p)的最小值
```

例如材料利用率最大、成型载荷最小，权重分别为0.3和0.7：
`J=0.7*标准化载荷-0.3*标准化材料利用率`。即使包含最大化目标，组合后的曲线仍然是
越小越好。它不是原始载荷、不是原始材料利用率，也不是把两个原始物理量直接相加。

示例 `x=[1,2,3], y=[0.8,0.5,0.4]` 表示第1、2、3代结束时的历史最佳可行J分别为
0.8、0.5、0.4。第3代没有更优解时，纵轴会保持0.5，不会回升。
尚未发现可行解时为 `null`，例如 `y=[null,null,0.4]`；应画为前两代无有效点，不能补成0。
J可以为负数，0也不代表达到最优；没有原始物理单位，不能在纵轴标注N、MPa或百分比。

### 多目标：代数与可行前沿超体积HV

响应通过 `metric=hypervolume`、`x_label=generation` 标识此曲线。
横轴含义与单目标相同，每完成一代记录一个点；纵轴不是Y1或Y2的原始值，也不是
多个目标的加权和，而是把这一代可行前沿的整体质量汇总成一个HV值。

多目标没有唯一的标量目标函数，因此使用 HV 表示单条收敛曲线。计算时先将最大化目标取负，
统一为最小化方向，再使用固定训练尺度：`z=(F-offset)/scale`。
第一代用整个种群的有限目标值确定每个坐标的参考点：`r=max(3, 第一代z最大值)+1`，
没有有限目标行时使用4，后续不重新归一化、不移动参考点。只使用满足全部约束的解计算HV，
在任何坐标不优于参考点的解不贡献体积。HV定义参考
[pymoo官方指标文档](https://pymoo.org/misc/indicators.html)。
不同轮次可能具有不同模型尺度、目标数或参考点，不能直接比较HV绝对值。
过程观测不改变算法的选择、目标、权重、约束或随机数；获取解对应的工艺参数和目标值仍使用第2节。

直观上，二维目标的HV是可行前沿与固定参考点围成的覆盖面积，多维则是对应的体积，
重叠部分不重复计算。同一轮次使用同一参考点，前沿越接近理想方向或覆盖越好，HV通常越大。
例如 `x=[1,2,3], y=[0,12.5,15.2]` 表示第1代HV为0，第2代为12.5，第3代为15.2。
0可能表示没有可行解，也可能是可行解均未对参考点贡献体积，不能单凭HV=0认定算法失败。

端上直接绘制返回的HV，不需要根据 `hv_config` 再计算或进行归一化；这些元数据用于解释
计算尺度和参考点。HV不是百分比、不限定在0到1、没有工艺物理单位。记录的是当前代前沿，
不是所有历史解的累计前沿，因此曲线可能下降，不应由端上强制改成历史最大值。

### 强化学习：训练回合数与平均回合累计奖励

响应通过 `metric=episode_reward_mean_100`、`x_label=episode` 标识此曲线。
一个回合是从环境reset到回合结束的完整交互过程，内部包含多次PPO动作和代理模型预测。
当前环境每回合按 `algorithm.params.episode_steps` 步结束，默认100步。
因此横轴 `x=3` 是已经结束3个训练回合，不是训练3步，也不是更新网络3次。

每一步奖励为加权标准化目标的相反数减去约束惩罚；回合内先求和，再对最近的回合求均值：

```text
reward_t = -J(p_t) - constraint_penalty * violation_t
R_e = sum_t(reward_t)                      # 第e回合内的累计奖励，不是每步平均
m = min(e, 100)
y(e) = (R_(e-m+1) + ... + R_e) / m         # 最近最多100回合的累计奖励均值
x(e) = e                                  # 已完成训练回合编号
```

其中J按单目标小节的权重、标准化和方向计算。`violation_t` 为各约束越界程度的总和：
上界约束取 `max(0,预测值-上界)/max(abs(上界),1)`，下界约束取
`max(0,下界-预测值)/max(abs(下界),1)`。不满足约束的训练回合仍参与奖励曲线，
只是奖励会被扣分，所以有曲线或奖励较高不等于最终解一定可行。

第1个点只用R1，第2个点是 `(R1+R2)/2`，第101个点只用第2到101回合的累计奖励。
例如前3个回合累计奖励为-35、-25、-24，则接口返回
`x=[1,2,3], y=[-35,-30,-28]`，并不是三个回合各自的累计奖励。
从-35升到-28是奖励改善；负奖励不表示请求失败，曲线也不保证单调上升。

改变回合步数、权重、模型尺度或惩罚系数都会影响奖励量级，不同配置下不能直接比较奖励绝对值。
它不是代理模型训练误差、原始目标值、折扣回报、PPO loss，也不是单步奖励。
训练结束后的 `evaluation_episodes` 不计入这条曲线。

### 端上读取与绘图约定

1. 从 `data.mode/metric/x_label/direction` 选择上表的轴标题与“越大/越小越好”提示，字段值是协议枚举，不是中文展示文案
2. 按 `data.x[i]` 与 `data.y[i]` 配对绘制一条曲线，使用真实横轴值，不使用数组下标、请求次数或百分比替代
3. 纵轴保持服务端返回值，不取绝对值、不再次取负号、不二次归一化、不自行累计最优值，也不把单目标null替换为0
4. 单目标null显示为缺失点；多目标0和PPO负值属于有效数值，不能当作空值过滤
5. 每次轮询返回截至当前的完整数组，应替换当前图表序列，不直接追加整个响应，否则会重复绘制旧点
6. 轮询期间使用提交时保存的优化run_id，切换轮次时先清空旧曲线；空数组显示“暂无过程数据”，终态保留已经记录的曲线

例如多目标响应中的 `x[1]=2, y[1]=12.5` 应形成坐标点 `(2,12.5)`，
图表标题可用“多目标优化收敛曲线”，横轴“优化代数（代）”，纵轴“可行前沿超体积（HV）”。
单目标和强化学习按上表换轴标题，不沿用多目标标题。若要展示载荷、利用率等带物理单位的
原始目标值或最优工艺参数，应通过第11节结果索引与第2节数据接口获取，不从本收敛曲线反推。

单目标尚未发现可行解时纵轴为 `null`；发现后保留历史最佳值。
多目标没有可行解或没有对参考点贡献体积的解时HV为0；当前前沿HV不保证单调。
PPO 回合奖励均值也不保证单调，横轴不是epoch或训练步数；其实际训练步数可能因完整 rollout
超过请求的 `total_timesteps`，因此记录的回合数量由实际训练决定。

尚未优化、排队中尚无记录、旧轮次没有过程文件，或 PPO 尚未完成一个回合时，仍返回 HTTP 200，
`available=false`、`point_count=0`、`x=[]`、`y=[]`。
未提交优化时 `mode=null`、`metric=null`、`x_label=null`、`direction=null`、`hv_config=null`。
中止或失败后保留已经写出的点，新轮次不会覆盖历史轮次。
旧版单目标记录可转换成一维数组；旧版PPO单环境记录可将已完成回合顺序转换成横轴。
旧版多目标逐目标最优值不能重建前沿和HV，返回空曲线，不伪造数据，需要重新优化后获取。

**失败响应示例与字段说明**：

未传 `id` 或标识格式非法返回 HTTP 400：

```json
{"code": 1, "message": "id 只能包含字母、数字、下划线、短横线，且长度为 1-128", "data": {}}
```

优化轮次不存在或不属于当前 DOE 返回 HTTP 404：

```json
{"code": 404, "message": "优化轮次不存在：run_missing", "data": {}}
```

DOE 不存在返回 HTTP 404：

```json
{"code": 404, "message": "DOE 任务不存在：doe_missing", "data": {}}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE或轮次不存在为404，内部错误为500 |
| `message` | string | 错误说明，内部错误仅返回 `服务内部错误` |
| `data` | object | 失败时为空对象 |

## 13 创建 DOE 任务

**POST /api/v1/doe/add**

请求和成功响应示例见本文开头的同名任务示例。请求也可使用空对象 `{}`，由服务端生成
`id`，并将它用作默认名称。成功返回 HTTP 201，`data` 是任务摘要，不包含后台轮次标识。

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 否 | 唯一标识，未传时自动生成；仅字母、数字、下划线和短横线，长度1～128 |
| `name` | string | 否 | 展示名称，允许重复，去首尾空白后须非空且不超过128字符；默认id |
| `description` | string | 否 | 任务描述，默认空字符串 |
| `metadata` | object | 否 | 扩展信息，默认空对象 |

| 成功响应 `data` 字段 | 类型 | 说明 |
|---|---|---|
| `id/name/description/metadata` | string / string / string / object | 任务标识、名称、描述和扩展信息 |
| `status/stage/progress` | string / string / number | 新建时分别为 `created/created/0`，列表查询时为任务聚合状态 |
| `created_at/updated_at` | string | 创建和最近更新时间，ISO 8601 格式 |
| `optimization_run_count` | integer | 优化历史轮次数量，新建时为0 |
| `has_optimization_result` | boolean | 当前或任意历史轮次是否有优化结果，新建时为false |

同id重复创建返回 HTTP 409：

```json
{"code": 409, "message": "DOE 任务已存在：doe_20260622_001", "data": {}}
```

名称为空返回 HTTP 400：

```json
{"code": 1, "message": "DOE 任务名称不能为空", "data": {}}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，id冲突为409 |
| `message` | string | 具体错误原因 |
| `data` | object | 失败时为空对象 |

## 14 查询 DOE 列表

**GET /api/v1/doe/list**

```http
GET /api/v1/doe/list
```

本接口无请求字段，不读取 JSON 请求体，也没有分页、排序或名称过滤参数。
成功返回 HTTP 200，示例展示一个刚创建的任务：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "items": [{
      "id": "doe_20260622_001",
      "name": "环件优化",
      "description": "",
      "metadata": {},
      "status": "created",
      "stage": "created",
      "progress": 0,
      "created_at": "2026-10-10T10:00:00+08:00",
      "updated_at": "2026-10-10T10:00:00+08:00",
      "optimization_run_count": 0,
      "has_optimization_result": false
    }],
    "total": 1
  }
}
```

| 响应字段 | 类型 | 说明 |
|---|---|---|
| `code/message` | integer / string | 成功为 `0/ok` |
| `data.items` | object array | 任务摘要列表，每项字段同第13节创建响应；无任务时为空数组 |
| `data.total` | integer | 任务数量，与items长度一致 |

内部错误返回 HTTP 500：

```json
{"code": 500, "message": "服务内部错误", "data": {}}
```

失败的 `code/message/data` 分别为错误码、错误说明和空对象。

## 15 删除 DOE 任务

**POST /api/v1/doe/delete**

```json
{"id": "doe_20260622_001"}
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | DOE唯一标识，不接受name或run_id替代 |

成功返回 HTTP 200。注意本接口的 `data` 是被删除id的字符串，不是 `{ "id": ... }` 对象：

```json
{"code": 0, "message": "DOE 任务及相关文件已删除", "data": "doe_20260622_001"}
```

| 响应字段 | 类型 | 说明 |
|---|---|---|
| `code/message` | integer / string | 成功码及删除提示 |
| `data` | string | 被删除的DOE唯一标识 |

删除包含全部样本、数据集、训练轮次、模型和优化轮次，相关资源索引失效。
训练或优化仍在运行时返回 HTTP 409，需先中止并等待后台退出：

```json
{"code": 409, "message": "任务正在运行，请先中止", "data": {}}
```

DOE不存在返回 HTTP 404：

```json
{"code": 404, "message": "DOE 任务不存在：doe_missing", "data": {}}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，不存在为404，运行冲突为409 |
| `message/data` | string / object | 错误原因和空对象 |

## 16 生成演示训练数据集

**POST /api/v1/hust/doe/dataset/generate**

仅用于联调或 Demo，不执行仿真，不应作为真实工艺评价数据。

```json
{
  "id": "doe_20260622_001",
  "param_ranges": {"X1": [0, 1], "X2": [10, 15]},
  "input_names": ["X1", "X2"],
  "target_names": ["Y1", "Y2"],
  "n_samples": 100,
  "seed": 42,
  "noise_ratio": 0
}
```

| 请求字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `id` | string | 是 | 已创建的DOE唯一标识 |
| `param_ranges` | object | 是 | 输入名称到 `[lower, upper]` 的映射，须lower小于upper |
| `input_names` | string array | 否 | 输入列顺序，与param_ranges包含相同名称且不重复，默认其键顺序 |
| `target_names` | string array | 是 | 非空目标名称数组，名称不能与输入重复 |
| `n_samples` | integer | 否 | 至少10，默认100 |
| `seed` | integer | 否 | 随机种子，默认42 |
| `noise_ratio` | number | 否 | 非负噪声比例，默认0 |

成功返回 HTTP 200。当前响应不包含 `id`，端上保留请求中的DOE标识：

```json
{
  "code": 0,
  "message": "训练数据集生成完成",
  "data": {
    "all_var_list": ["X1", "X2", "Y1", "Y2"],
    "input_var_count": 2,
    "sample_count": 100,
    "input_names": ["X1", "X2"],
    "target_names": ["Y1", "Y2"],
    "source_name": "demo_training_dataset.tsv",
    "resource_id": "tos-a1b2c3d4e5f60718293",
    "resource_type": "dataset",
    "columns": ["X1", "X2", "Y1", "Y2"]
  }
}
```

| 响应字段 | 类型 | 说明 |
|---|---|---|
| `code/message` | integer / string | 成功码和生成提示 |
| `data.all_var_list/columns` | string array | 无表头TSV列顺序，先输入后输出 |
| `data.input_var_count/sample_count` | integer | 输入列数和样本行数 |
| `data.input_names/target_names` | string array | 输入和输出名称 |
| `data.source_name` | string | 数据来源名称，不是服务器路径 |
| `data.resource_id/resource_type` | string | 数据资源索引，类型为dataset |

使用第2节获取数据，将输入与输出按 `data_source` 结构组织后提交第4节训练接口。
生成数据集不会自动创建训练轮次，也不会返回训练 `run_id`。

样本数量不足返回 HTTP 400：

```json
{"code": 1, "message": "n_samples 必须是大于等于10的整数", "data": {}}
```

| 失败响应字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 参数错误为1，DOE不存在为404 |
| `message/data` | string / object | 错误原因和空对象 |

## 17 查询模型超参数元数据

**GET /api/v1/hust/doe/train/hyperparameters**

```http
GET /api/v1/hust/doe/train/hyperparameters
```

无请求参数，不需要DOE标识。成功返回 HTTP 200，下例仅展示 `data.models` 的PRG部分，
实际响应同时包含PRG、SVR、RF、KM、DNN全部模型字段，完整参数见第4节：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "models": {
      "PRG": {
        "degree": {"type": "integer", "default": 2, "minimum": 1, "maximum": 10, "label": "多项式阶数"},
        "include_bias": {"type": "boolean", "default": false, "label": "包含偏置特征"},
        "fit_intercept": {"type": "boolean", "default": true, "label": "拟合截距"}
      }
    }
  }
}
```

| 响应字段 | 类型 | 说明 |
|---|---|---|
| `code/message` | integer / string | 成功为0/ok |
| `data.models` | object | 模型族名称到参数定义的映射，不是数组 |
| `data.models.<模型>.<参数>.type` | string | 参数类型，例如integer、number、string、boolean或number_or_string |
| `data.models.<模型>.<参数>.default` | 任意JSON值 | 默认值，具体类型由参数决定 |
| `data.models.<模型>.<参数>.label` | string | 参数展示名称 |
| `choices` | array | 支持的枚举值，适用时才返回 |
| `minimum/maximum` | number | 含边界的下限/上限，适用时才返回 |
| `exclusive_minimum/exclusive_maximum` | number | 不含边界的下限/上限，适用时才返回 |
| `nullable` | boolean | 是否允许null，适用时才返回 |

内部错误返回 HTTP 500：

```json
{"code": 500, "message": "服务内部错误", "data": {}}
```

失败的 `code/message/data` 分别为错误码、错误说明和空对象。

## 18 客户端轮次调用顺序

1. 创建DOE，保存 `data.id`
2. 保存真实数据集或生成演示数据集，保存 `resource_id` 和字段顺序
3. 提交训练，保存 `data.run_id` 为客户端的 `training_run_id`
4. 使用 `GET /api/v1/hust/doe/train/progress?id=<id>&run_id=<training_run_id>` 轮询，完成后读取模型及评价
5. 提交优化时传 `training_run_id`，可选 `model_id`，保存响应的 `data.run_id` 为客户端的 `optimization_run_id`
6. 使用 `GET /api/v1/hust/doe/optimize/getById?id=<id>` 查询状态，在history中匹配 `optimization_run_id`
7. 使用 `GET /api/v1/hust/doe/optimize/process?id=<id>&run_id=<optimization_run_id>` 查询该轮单条x/y曲线
8. 完成后取该轮 `result.resource_id`，按第2节获取工艺参数、目标值和feasible，不能用run_id替代resource_id

重新训练或优化时更新相应的客户端轮次变量，历史页面则保留选中的历史轮次。
中止接口只有DOE的 `id`，作用于当前后台任务，不作用于历史页面选中的轮次。

## 落盘结构

```text
data/doe_tasks/<id>/
├── doe.json
├── samples/
├── dataset/
│   └── training_dataset.tsv
├── training/runs/<training_run_id>/
│   ├── dataset.tsv
│   ├── models/<model_id>/
│   │   ├── <target>_model.pkl|keras
│   │   └── <target>_scalers.pkl
│   ├── internal/
│   ├── best_model.json
│   └── training_result.json
└── optimization/runs/<optimization_run_id>/
    ├── pareto_solutions.tsv|rl_solutions.tsv
    ├── convergence.json
    ├── internal/
    └── optimization_result.json
```

`doe.json` 是任务元数据、进度和产物索引入口。每个训练和优化轮次都拥有独立目录；
HTTP 聚合流程不会再把模型或运行状态写入顶层 `data/models`、`data/tasks`。旧目录不参与
新协议读取，既有旧任务需要重新训练或重新优化。
