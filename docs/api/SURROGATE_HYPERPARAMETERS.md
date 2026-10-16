# 代理模型超参数中文说明与请求参考

## 1. 文档范围

本文整理当前 DOE HTTP 训练接口实际支持的五类代理模型及其超参数：多项式回归（PRG）、
支持向量回归（SVR）、随机森林回归（RF）、Kriging/高斯过程回归（KM）和深度神经网络
（DNN）。字段名、默认值和约束以服务端实现及
`GET /api/v1/hust/doe/train/hyperparameters` 的返回为准。

本文是 [DOE_HTTP_API.md](DOE_HTTP_API.md) 的超参数专项说明，不替代主协议。研究文档中出现的
ExtraTrees 和 GBR 尚未接入当前 HTTP 训练接口，不能作为模型 `name` 提交。
三个碾环数据集仅使用接口现有模型的定向达标参数见
[三个碾环数据集代理模型结果](../research/RING_SURROGATE_MODEL_RESULTS.md)。

## 2. 请求控制字段

训练接口：

```http
POST /api/v1/hust/doe/train/startTrain
Content-Type: application/json
```

推荐把模型配置放在请求体的 `target_models` 对象中，以便每个输出独立选择模型和参数：

```json
{
  "id": "doe_example",
  "data_source": {
    "input_data": {
      "labels": ["x1", "x2"],
      "samples": [[0.1, 1.0], [0.2, 1.2], [0.3, 1.4]]
    },
    "output_data": {
      "labels": ["y1", "y2"],
      "samples": [[10.0, 5.0], [11.0, 4.0], [12.0, 3.0]]
    }
  },
  "target_models": {
    "y1": [
      {"name": "SVR", "params": {"kernel": "rbf", "C": 10.0}}
    ],
    "y2": [
      {"name": "RF", "params": {"n_estimators": 500, "max_depth": 16}}
    ]
  }
}
```

| 控制字段 | 中文名称 | 类型 | 必填 | 说明 |
|---|---|---|---:|---|
| `target_models` | 逐输出模型配置 | 对象 | 否 | 推荐；键必须完整覆盖输出名称，每个输出至少选择一个候选模型 |
| `target_models.<输出名>` | 单个输出的候选模型 | 对象数组 | 条件必填 | 数组元素字段与 `models[]` 相同；不同输出可对同一模型设置不同参数 |
| `models` | 全输出共用模型列表 | 对象数组 | 否 | 兼容格式；省略两种模型字段时默认训练 PRG、SVR、RF、KM |
| `models[].name` | 模型族名称 | 字符串 | 推荐必填 | 可取 `PRG`、`SVR`、`RF`、`KM`、`DNN`，区分大小写 |
| `models[].model_index` | 模型族编号 | 整数 | 兼容字段 | 未传 `name` 时使用；映射见下表，推荐新请求使用 `name` |
| `models[].params` | 超参数覆盖项 | 对象 | 否 | 只传需要覆盖的字段；省略或 `{}` 时使用该模型全部默认值 |

模型名称与兼容编号：

| `name` | `model_index` | 中文名称 |
|---|---:|---|
| `PRG` | `0` | 多项式回归 |
| `SVR` | `1` | 支持向量回归 |
| `RF` | `2` | 随机森林回归 |
| `KM` | `3` | Kriging / 高斯过程回归 |
| `DNN` | `4` | 深度神经网络 |

`models` 与 `target_models` 不能同时提交。同一输出的候选数组中不能重复模型族，但不同输出
可以重复并使用不同参数。若一个模型对象同时提供 `name` 和 `model_index`，服务端以
`name` 为准；为避免歧义，不应同时提供两者。未知参数、类型错误、越界值或参数关系不合法会在
启动训练前返回 HTTP 400。

## 3. 参数类型约定

| 元数据类型 | 中文含义 | JSON 示例 |
|---|---|---|
| `integer` | 整数 | `300` |
| `number` | 数值，可为整数或小数 | `0.001` |
| `boolean` | 布尔值 | `true` |
| `string` | 字符串 | `"rbf"` |
| `integer_or_number` | 整数计数或小数比例 | `2` 或 `0.5` |
| `number_or_string` | 数值或约定字符串 | `0.1` 或 `"scale"` |
| `null` | 不限制或使用底层默认行为 | `null` |

下文“建议起点”是便于首次请求的经验参考，不是新的接口默认值，也不保证对任意数据集达到指定
R²。服务端会先对输入和每个输出做标准化；调参仍应以留出的测试集结果和业务误差要求为准。

## 4. PRG：多项式回归

适合低维、较平滑且可用低阶多项式表达的关系。阶数升高会快速增加多项式特征数量，也更容易
过拟合。

| 请求字段 | 中文翻译 | 类型与约束 | 默认值 | 建议起点 | 控制作用 |
|---|---|---|---:|---:|---|
| `degree` | 多项式阶数 | 1～10 的整数 | `2` | `2` | 控制非线性阶数；欠拟合时可尝试 `3` |
| `include_bias` | 包含偏置特征 | 布尔值 | `false` | `false` | 是否生成常数为 1 的多项式特征 |
| `fit_intercept` | 拟合截距 | 布尔值 | `true` | `true` | 是否由线性回归器单独拟合截距 |

推荐基线请求：

```json
{
  "name": "PRG",
  "params": {
    "degree": 2,
    "include_bias": false,
    "fit_intercept": true
  }
}
```

通常保持 `include_bias=false`、`fit_intercept=true`，避免重复引入常数项。输入维度较多时应谨慎
提高 `degree`。

## 5. SVR：支持向量回归

适合中小样本和非线性关系。当前实现对每个输出单独训练一个 SVR，并使用标准化后的输入与目标。

| 请求字段 | 中文翻译 | 类型与约束 | 默认值 | 建议起点 | 控制作用 |
|---|---|---|---:|---:|---|
| `kernel` | 核函数 | `linear` / `poly` / `rbf` / `sigmoid` | `"rbf"` | `"rbf"` | 决定特征空间映射形式 |
| `C` | 惩罚系数 | 正数 | `1.0` | `10.0` | 越大越重视训练误差，也可能更易过拟合 |
| `epsilon` | ε 不敏感区间 | 非负数 | `0.1` | `0.01` | 越小越严格拟合细小误差 |
| `gamma` | 核宽度系数 | 正数或 `"scale"` / `"auto"` | `"scale"` | `0.1` | RBF、poly、sigmoid 核的影响范围 |
| `degree` | 多项式核阶数 | 1～10 的整数 | `3` | `3` | 仅在 `kernel="poly"` 时生效 |
| `coef0` | 核函数常数项 | 数值 | `0.0` | `0.0` | 主要影响 poly 和 sigmoid 核 |
| `shrinking` | 启用收缩启发式 | 布尔值 | `true` | `true` | 是否使用收缩策略加速求解 |
| `tol` | 收敛容差 | 正数 | `0.001` | `0.001` | 越小收敛判定越严格、训练可能更慢 |
| `max_iter` | 最大迭代次数 | `-1` 或正整数 | `-1` | `-1` | `-1` 表示不限制迭代次数 |

调优起点请求：

```json
{
  "name": "SVR",
  "params": {
    "kernel": "rbf",
    "C": 10.0,
    "epsilon": 0.01,
    "gamma": 0.1,
    "degree": 3,
    "coef0": 0.0,
    "shrinking": true,
    "tol": 0.001,
    "max_iter": -1
  }
}
```

对标准化数据，可优先比较 `C` 为 `1`、`10`、`100`、`1000`，`gamma` 为 `"scale"`、
`0.01`、`0.1`、`1.0`，`epsilon` 为 `0.001`、`0.01`、`0.1` 的组合。训练分数高而测试分数
下降时，通常应减小 `C`、增大 `epsilon` 或减小 `gamma`。

## 6. RF：随机森林回归

适合存在阈值效应、变量交互或非光滑关系的数据，对特征缩放不敏感。树数量增加通常提高稳定性，
但会增加训练时间和模型体积。

| 请求字段 | 中文翻译 | 类型与约束 | 默认值 | 建议起点 | 控制作用 |
|---|---|---|---:|---:|---|
| `n_estimators` | 决策树数量 | 1～5000 的整数 | `300` | `500` | 森林中的树数量 |
| `criterion` | 节点划分准则 | `squared_error` / `absolute_error` / `friedman_mse` / `poisson` | `"squared_error"` | `"squared_error"` | 衡量候选划分质量 |
| `max_depth` | 单棵树最大深度 | `null` 或正整数 | `null` | `null` | 限制树复杂度；`null` 表示不预设深度上限 |
| `min_samples_split` | 节点最少拆分样本 | ≥2 的整数，或 (0,1] 小数 | `2` | `2` | 控制内部节点继续拆分的门槛 |
| `min_samples_leaf` | 叶节点最少样本 | ≥1 的整数，或 (0,1] 小数 | `1` | `1` | 增大可使预测更平滑、抑制过拟合 |
| `max_features` | 每次划分最大特征数 | `null`、正整数、(0,1] 小数、`"sqrt"` 或 `"log2"` | `1.0` | `1.0` | 控制每次节点划分可检查的特征范围 |
| `bootstrap` | Bootstrap 抽样 | 布尔值 | `true` | `true` | 是否为每棵树进行有放回抽样 |
| `max_samples` | Bootstrap 最大样本量 | `null`、正整数或 (0,1] 小数 | `null` | `null` | 仅当 `bootstrap=true` 时可设置 |
| `random_state` | 随机种子 | `null` 或整数 | `42` | `42` | 固定抽样和建树随机性以便复现 |
| `n_jobs` | 并行任务数 | 非零整数 | `-1` | `-1` | `-1` 表示使用全部可用 CPU 核心 |

推荐基线请求：

```json
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
  }
}
```

若测试集过拟合，可依次尝试增大 `min_samples_leaf`、限制 `max_depth`、减小 `max_features`。
当评价配置按输出并行时，RF 的 `n_jobs=-1` 可能形成嵌套并行；资源紧张时可将
`evaluation.max_workers` 或 `n_jobs` 设为 `1`。

## 7. KM：Kriging / 高斯过程回归

当前 KM 使用“常数核 × RBF 核”。适合样本量不大、响应较平滑且需要核函数插值能力的场景；
样本数增大时训练成本会明显上升。

| 请求字段 | 中文翻译 | 类型与约束 | 默认值 | 建议起点 | 控制作用 |
|---|---|---|---:|---:|---|
| `alpha` | 观测噪声/对角稳定项 | 非负数 | `0.1` | `0.1` | 越大越平滑，并增强数值稳定性 |
| `n_restarts_optimizer` | 核参数优化器重启次数 | 非负整数 | `20` | `20` | 多次从不同初值优化核参数，增加找到较优解的机会 |
| `normalize_y` | 在模型内部标准化目标 | 布尔值 | `false` | `false` | 是否让高斯过程再次标准化目标；外层已标准化时通常保持关闭 |
| `random_state` | 随机种子 | `null` 或整数 | `42` | `42` | 固定优化器重启时的随机性 |
| `constant_value` | 常数核初值 | 正数 | `1.0` | `1.0` | 常数核的初始幅值 |
| `constant_lower` | 常数核下界 | 正数且小于上界 | `0.001` | `0.001` | 常数核优化范围下界 |
| `constant_upper` | 常数核上界 | 正数且大于下界 | `1000000.0` | `1000000.0` | 常数核优化范围上界 |
| `length_scale` | RBF 长度尺度初值 | 正数 | `1.0` | `1.0` | 越大通常表示函数变化更平缓 |
| `length_scale_lower` | 长度尺度下界 | 正数且小于上界 | `0.1` | `0.1` | RBF 长度尺度优化范围下界 |
| `length_scale_upper` | 长度尺度上界 | 正数且大于下界 | `10000.0` | `10000.0` | RBF 长度尺度优化范围上界 |

推荐基线请求：

```json
{
  "name": "KM",
  "params": {
    "alpha": 0.1,
    "n_restarts_optimizer": 20,
    "normalize_y": false,
    "random_state": 42,
    "constant_value": 1.0,
    "constant_lower": 0.001,
    "constant_upper": 1000000.0,
    "length_scale": 1.0,
    "length_scale_lower": 0.1,
    "length_scale_upper": 10000.0
  }
}
```

拟合过度振荡或矩阵数值不稳定时可适当增大 `alpha`；明显欠拟合时可尝试减小 `alpha`。
`constant_lower < constant_upper` 且 `length_scale_lower < length_scale_upper` 是硬性关系约束。

## 8. DNN：深度神经网络

当前 DNN 固定为三层隐藏层结构，通过字段控制每层宽度、激活、Dropout、批归一化和训练过程。
DNN 不在默认模型列表中，必须显式提交 `name="DNN"`。

| 请求字段 | 中文翻译 | 类型与约束 | 默认值 | 建议起点 | 控制作用 |
|---|---|---|---:|---:|---|
| `hidden_layer_1` | 隐藏层 1 神经元数 | 1～4096 的整数 | `64` | `64` | 第一隐藏层宽度 |
| `hidden_layer_2` | 隐藏层 2 神经元数 | 1～4096 的整数 | `32` | `32` | 第二隐藏层宽度 |
| `hidden_layer_3` | 隐藏层 3 神经元数 | 1～4096 的整数 | `16` | `16` | 第三隐藏层宽度 |
| `activation` | 激活函数 | `relu` / `tanh` / `sigmoid` / `elu` / `selu` | `"relu"` | `"relu"` | 三个隐藏层使用的非线性激活 |
| `dropout_1` | 隐藏层 1 丢弃率 | [0,1) 数值 | `0.2` | `0.2` | 随机丢弃部分神经元以抑制过拟合 |
| `dropout_2` | 隐藏层 2 丢弃率 | [0,1) 数值 | `0.1` | `0.1` | 第二处 Dropout 的丢弃比例 |
| `batch_normalization` | 批归一化 | 布尔值 | `true` | `true` | 是否在网络中启用批归一化 |
| `learning_rate` | 学习率 | 正数 | `0.001` | `0.001` | 控制每次参数更新步长 |
| `epochs` | 最大训练轮数 | 1～100000 的整数 | `1000` | `1000` | 早停前允许的最大训练轮数 |
| `batch_size` | 批大小 | 正整数 | `16` | `16` | 每次梯度更新使用的样本数 |
| `verbose` | 训练日志级别 | `0` / `1` / `2` | `0` | `0` | `0` 静默，`1` 进度条，`2` 每轮一行 |
| `patience` | 早停耐心轮数 | 正整数 | `50` | `50` | 验证指标连续未改善多少轮后停止 |
| `reduce_lr_factor` | 学习率衰减系数 | (0,1) 数值 | `0.2` | `0.2` | 触发降学习率时乘以该系数 |
| `reduce_lr_patience` | 学习率衰减耐心轮数 | 正整数 | `5` | `5` | 验证指标未改善多少轮后降低学习率 |
| `min_lr` | 最小学习率 | 非负数 | `0.000001` | `0.000001` | 学习率衰减的下限 |

推荐基线请求：

```json
{
  "name": "DNN",
  "params": {
    "hidden_layer_1": 64,
    "hidden_layer_2": 32,
    "hidden_layer_3": 16,
    "activation": "relu",
    "dropout_1": 0.2,
    "dropout_2": 0.1,
    "batch_normalization": true,
    "learning_rate": 0.001,
    "epochs": 1000,
    "batch_size": 16,
    "verbose": 0,
    "patience": 50,
    "reduce_lr_factor": 0.2,
    "reduce_lr_patience": 5,
    "min_lr": 0.000001
  }
}
```

小样本数据应先控制网络宽度和 Dropout，再调整学习率。`epochs` 是上限；启用的早停机制可能
在达到该轮数前终止训练。为避免服务日志膨胀，接口请求通常保持 `verbose=0`。

## 9. 逐输出多模型请求示例

一次请求可为每个输出训练不同的候选模型，甚至对相同模型族设置不同超参数。服务端会独立评价
每个候选，并按各输出的测试 R² 组装最终 `MIXED` 模型。

```json
{
  "target_models": {
    "load": [
      {"name": "PRG", "params": {"degree": 2}},
      {"name": "RF", "params": {"n_estimators": 500, "max_depth": 12}}
    ],
    "grain_size": [
      {"name": "SVR", "params": {"kernel": "rbf", "C": 20.0}},
      {"name": "RF", "params": {"n_estimators": 1000, "max_depth": 24}}
    ]
  }
}
```

以上片段应与完整训练请求中的 `id`、`data_source` 等字段合并后提交。省略的模型参数会自动补入
服务端默认值。

## 10. 动态读取权威元数据

前端或其他调用方不应长期硬编码参数选项，应在构建表单时读取：

```http
GET /api/v1/hust/doe/train/hyperparameters
```

响应结构：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "models": {
      "PRG": {
        "degree": {
          "type": "integer",
          "default": 2,
          "minimum": 1,
          "maximum": 10,
          "label": "多项式阶数"
        }
      }
    }
  }
}
```

每个参数至少包含 `type`、`default` 和中文 `label`，并按需包含 `choices`、`minimum`、
`maximum`、`exclusive_minimum`、`exclusive_maximum`、`nullable`。请求端应据此生成控件和
执行基础校验，服务端仍会进行最终校验。

## 11. 常见错误

| 错误写法 | 原因 | 正确做法 |
|---|---|---|
| `{"name": "svr"}` | 模型名称区分大小写 | 使用 `{"name": "SVR"}` |
| `{"name": "RF", "params": {"n_jobs": 0}}` | `n_jobs` 不能为 0 | 使用 `-1` 或其他非零整数 |
| `{"name": "RF", "params": {"bootstrap": false, "max_samples": 0.8}}` | `max_samples` 仅在 Bootstrap 开启时可用 | 设置 `bootstrap=true` 或 `max_samples=null` |
| `{"name": "SVR", "params": {"max_iter": 0}}` | 只允许 `-1` 或正整数 | 使用 `-1` 或大于 0 的整数 |
| `{"name": "KM", "params": {"length_scale_lower": 10, "length_scale_upper": 1}}` | 下界必须小于上界 | 调整为严格递增的正数边界 |
| `{"name": "DNN", "params": {"dropout_1": 1.0}}` | Dropout 必须小于 1 | 使用 [0,1) 范围内的数值 |
| `{"name": "ExtraTrees", "params": {}}` | 当前 HTTP 接口不支持该模型 | 使用五类已支持模型，或另行扩展接口 |
