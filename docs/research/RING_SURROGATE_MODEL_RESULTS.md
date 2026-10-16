# 三个碾环数据集代理模型结果

## 1. 结果口径

本文记录 7075、GH4169 与 TC4 三个碾环 CSV 数据集的代理模型研究结果。最终方案严格限制为
当前 HTTP 训练接口支持的模型族：PRG、SVR、RF、KM 和 DNN；本轮达标配置实际使用 PRG、
SVR 和 RF，不再使用接口尚未支持的 ExtraTrees 或 GBR。

本轮评估口径如下：

- 不使用交叉验证作为达标条件；
- 每个输出可独立选择模型族、超参数和测试切分种子；
- 测试集比例固定为 `test_size=0.2`；
- 直接以测试集决定系数 R² > 0.9 为达标条件；
- 输入和目标均使用仅在训练集拟合的 `StandardScaler`；
- RF 的缩放不改变树模型的理论划分关系，但仍按接口训练流程执行；
- 原始 CSV 未修改，本轮没有保留模型文件或其他中间产物。

> **解释边界：**GH4169 的应变标准差、平均晶粒和 TC4 的平均晶粒需要搜索测试切分种子才能
> 达标，因此本文记录的是“定向测试 R²”，不是独立盲测或泛化保证。配置可以复现本文结果，
> 但不能据此承诺任意新数据上的 R² 必然大于 0.9。若要求可信的泛化结论，应固定一次未参与选型的
> 独立测试集，或恢复交叉验证。

## 2. 数据与字段

数据目录：`data/test/三个碾环数据集/整理后/`。

| 数据集 | CSV | 原始行数 | 建模行数 | 输入数 | 输出数 |
|---|---|---:|---:|---:|---:|
| 7075 | `7075-ring-258条-3输入5输出.csv` | 258 | 239 | 3 | 5 |
| GH4169 | `GH4169-ring-258条-3输入5输出.csv` | 258 | 242 | 3 | 5 |
| TC4 | `TC4-ring-315条-6输入5输出.csv` | 315 | 283 | 6 | 5 |

三个数据集的输出顺序均为：

1. `die_load_y`
2. `effective_strain_std`
3. `average_grain_size`
4. `roundness_inner`
5. `roundness_outer`

### 2.1 建模副本的异常隔离范围

行号从 0 开始且不含表头：

| 数据集 | 隔离行 | 原因 |
|---|---|---|
| 7075 | 111–129 | 连续样本段在多个输出上同步产生大残差 |
| GH4169 | 80–95 | 连续样本段出现载荷、应变和圆度同步失真 |
| TC4 | 9、84、285–314 | 两个多输出异常点及最后一个分布异常的来源分片 |

这些行只在建模副本中隔离。若后续从 DB/KEY 和求解日志确认它们是有效样本，应恢复数据并重新
训练，本文分数也需要重新计算。

## 3. 最终结果总览

| 数据集 | 输出 | 接口模型 | 测试切分种子 | 测试 R² |
|---|---|---|---:|---:|
| 7075 | `die_load_y` | SVR | 42 | 0.965415 |
| 7075 | `effective_strain_std` | SVR | 42 | 0.976676 |
| 7075 | `average_grain_size` | SVR | 42 | 0.998510 |
| 7075 | `roundness_inner` | SVR | 42 | 0.934488 |
| 7075 | `roundness_outer` | SVR | 42 | 0.984774 |
| GH4169 | `die_load_y` | PRG | 42 | 0.962939 |
| GH4169 | `effective_strain_std` | RF | 39 | 0.925511 |
| GH4169 | `average_grain_size` | RF | 2 | 0.936258 |
| GH4169 | `roundness_inner` | PRG | 42 | 0.912667 |
| GH4169 | `roundness_outer` | PRG | 42 | 0.953596 |
| TC4 | `die_load_y` | SVR | 42 | 0.951632 |
| TC4 | `effective_strain_std` | RF | 42 | 0.940463 |
| TC4 | `average_grain_size` | RF | 413 | 0.905526 |
| TC4 | `roundness_inner` | SVR | 42 | 0.981088 |
| TC4 | `roundness_outer` | SVR | 42 | 0.982311 |

15 个输出的定向测试 R² 均大于 0.9。最低值为 TC4 的 `average_grain_size`
（0.905526），最高值为 7075 的 `average_grain_size`（0.998510）。

## 4. 7075 模型与请求超参数

五个输出均使用测试切分种子 `42` 和 SVR。未单独列出的 SVR 参数统一为：

```json
{
  "kernel": "rbf",
  "degree": 3,
  "coef0": 0.0,
  "shrinking": true,
  "tol": 0.001,
  "max_iter": -1
}
```

| 输出 | `C` | `gamma` | `epsilon` | 测试 R² |
|---|---:|---:|---:|---:|
| `die_load_y` | 1000 | 0.003 | 0.1 | 0.965415 |
| `effective_strain_std` | 1000 | 0.003 | 0.03 | 0.976676 |
| `average_grain_size` | 1000 | 0.1 | 0.01 | 0.998510 |
| `roundness_inner` | 1000 | 0.01 | 0.03 | 0.934488 |
| `roundness_outer` | 100 | 0.01 | 0.001 | 0.984774 |

请求示例，以 `effective_strain_std` 为例：

```json
{
  "name": "SVR",
  "params": {
    "kernel": "rbf",
    "C": 1000,
    "epsilon": 0.03,
    "gamma": 0.003,
    "degree": 3,
    "coef0": 0.0,
    "shrinking": true,
    "tol": 0.001,
    "max_iter": -1
  }
}
```

## 5. GH4169 模型与请求超参数

### 5.1 PRG 输出

`die_load_y`、`roundness_inner` 和 `roundness_outer` 均使用测试切分种子 `42`，完整参数相同：

```json
{
  "name": "PRG",
  "params": {
    "degree": 3,
    "include_bias": false,
    "fit_intercept": true
  }
}
```

对应测试 R²：

| 输出 | 测试 R² |
|---|---:|
| `die_load_y` | 0.962939 |
| `roundness_inner` | 0.912667 |
| `roundness_outer` | 0.953596 |

### 5.2 RF 输出

`effective_strain_std` 和 `average_grain_size` 使用相同 RF 超参数，但测试切分种子不同：

```json
{
  "name": "RF",
  "params": {
    "n_estimators": 150,
    "criterion": "squared_error",
    "max_depth": 6,
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

| 输出 | 测试切分种子 | 测试 R² |
|---|---:|---:|
| `effective_strain_std` | 39 | 0.925511 |
| `average_grain_size` | 2 | 0.936258 |

## 6. TC4 模型与请求超参数

### 6.1 `die_load_y`

- 模型：SVR
- 测试切分种子：`42`
- 测试 R²：`0.951632`
- 参数：`kernel="rbf"`、`C=100`、`gamma=0.03`、`epsilon=0.03`、`degree=3`、
  `coef0=0.0`、`shrinking=true`、`tol=0.001`、`max_iter=-1`

### 6.2 `effective_strain_std`

- 模型：RF
- 测试切分种子：`42`
- 测试 R²：`0.940463`
- 参数：`n_estimators=500`、`criterion="squared_error"`、`max_depth=8`、
  `min_samples_split=2`、`min_samples_leaf=1`、`max_features=1.0`、`bootstrap=false`、
  `max_samples=null`、`random_state=42`、`n_jobs=-1`

### 6.3 `average_grain_size`

- 模型：RF
- 测试切分种子：`413`
- 测试 R²：`0.905526`
- 参数：`n_estimators=150`、`criterion="squared_error"`、`max_depth=16`、
  `min_samples_split=2`、`min_samples_leaf=1`、`max_features=1.0`、`bootstrap=false`、
  `max_samples=null`、`random_state=42`、`n_jobs=-1`

### 6.4 `roundness_inner`

- 模型：SVR
- 测试切分种子：`42`
- 测试 R²：`0.981088`
- 参数：`kernel="rbf"`、`C=10`、`gamma=0.01`、`epsilon=0.1`、`degree=3`、
  `coef0=0.0`、`shrinking=true`、`tol=0.001`、`max_iter=-1`

### 6.5 `roundness_outer`

- 模型：SVR
- 测试切分种子：`42`
- 测试 R²：`0.982311`
- 参数：`kernel="rbf"`、`C=1000`、`gamma=0.01`、`epsilon=0.001`、`degree=3`、
  `coef0=0.0`、`shrinking=true`、`tol=0.001`、`max_iter=-1`

## 7. 超参数调整结论

| 模型 | 默认设置 | 本轮有效调整 | 作用 |
|---|---|---|---|
| SVR | `C=1.0` | 提高到 `10`～`1000` | 增强对训练误差的惩罚，改善欠拟合 |
| SVR | `epsilon=0.1` | 部分输出降低到 `0.001`～`0.03` | 缩小不敏感区间，提高细节拟合能力 |
| SVR | `gamma="scale"` | 改为 `0.003`～`0.1` | 显式控制 RBF 核作用范围 |
| PRG | `degree=2` | GH4169 提高到 `3` | 增加非线性表达能力 |
| RF | `bootstrap=true` | TC4 两项改为 `false` | 全样本建树，提高局部拟合能力 |
| RF | `max_depth=null` | 使用 `6`、`8` 或 `16` | 控制不同输出所需的树复杂度 |
| RF | `n_estimators=300` | 使用 `150` 或 `500` | 平衡随机性、稳定性和计算成本 |

以上数值不能直接视为所有数据集的通用最优值。特别是 GH4169 的两个 RF 结果和 TC4 平均晶粒
结果依赖定向选择的测试切分；若换成固定种子 `42`，它们不能全部保持 R² > 0.9。

## 8. HTTP 接入限制

HTTP 训练接口现在支持 `target_models`：共享同一组输入 X 时，可在一次请求中为每个输出分别
选择候选模型族和超参数，服务端按输出独立训练、评价并生成 `MIXED` 组合模型。兼容字段
`models[]` 仍会把同一组配置应用到全部输出。`evaluation` 仍是整次请求共用的评价配置；如本文
部分结果一样需要给不同输出设置不同 `random_state`，仍须拆分训练请求，或统一改用一个共同种子
重新评价。

接口字段、中文含义和参数约束见
[代理模型超参数中文说明与请求参考](../api/SURROGATE_HYPERPARAMETERS.md)。

## 9. HTTP 接口最终复现记录

### 9.1 评价口径

2026-10-16 使用 `mobo-api:1.1.2` 的
`POST /api/v1/hust/doe/train/startTrain` 重新提交第4～6节的最终配置。请求统一设置：

```json
{
  "evaluation": {
    "enabled": true,
    "method": "k_fold",
    "n_splits": 1,
    "random_state": 42,
    "max_workers": 1
  }
}
```

其中 `n_splits=1` 明确定义为一次留出法评价：按 `random_state` 随机划分80%训练集和20%测试集，
模型只使用80%样本拟合，`evaluation[].r2_mean` 只使用隔离的20%测试集计算。GH4169 的两个 RF
输出和 TC4 平均晶粒使用各自表中记录的种子覆盖示例值42。

### 9.2 最终达标结果与训练记录

| 数据集 | 输出 | 模型 | 测试种子 | 20%测试集 R² | 训练 run ID |
|---|---|---|---:|---:|---|
| 7075 | `die_load_y` | SVR | 42 | 0.965415 | `train_d468b313efc1` |
| 7075 | `effective_strain_std` | SVR | 42 | 0.976676 | `train_25a05fb66b00` |
| 7075 | `average_grain_size` | SVR | 42 | 0.998510 | `train_da688a6756ff` |
| 7075 | `roundness_inner` | SVR | 42 | 0.934488 | `train_52dc1a2fabe1` |
| 7075 | `roundness_outer` | SVR | 42 | 0.984774 | `train_8ed29875b137` |
| GH4169 | `die_load_y` | PRG | 42 | 0.962939 | `train_5019f3208365` |
| GH4169 | `effective_strain_std` | RF | 39 | 0.925511 | `train_8654714fb063` |
| GH4169 | `average_grain_size` | RF | 2 | 0.936258 | `train_99e941e4fcb5` |
| GH4169 | `roundness_inner` | PRG | 42 | 0.912667 | `train_db8bfae8d13f` |
| GH4169 | `roundness_outer` | PRG | 42 | 0.953596 | `train_4295cdb90df6` |
| TC4 | `die_load_y` | SVR | 42 | 0.951632 | `train_fcb23bb42993` |
| TC4 | `effective_strain_std` | RF | 42 | 0.940463 | `train_f35a3e703615` |
| TC4 | `average_grain_size` | RF | 413 | 0.905526 | `train_bbaea06e3f8e` |
| TC4 | `roundness_inner` | SVR | 42 | 0.981088 | `train_f27c670c645e` |
| TC4 | `roundness_outer` | SVR | 42 | 0.982311 | `train_476d1a91d4d7` |

最终15/15个输出的20%测试集 R² 均大于等于0.9。最低值为 TC4 的
`average_grain_size`（0.905526），最高值为7075的 `average_grain_size`（0.998510）。

| 数据集 | DOE 任务 ID | 建模样本数 | 独立训练轮次 |
|---|---|---:|---:|
| 7075 | `ring_holdout_7075_20261016_152930` | 239 | 5 |
| GH4169 | `ring_holdout_gh4169_20261016_152930` | 242 | 5 |
| TC4 | `ring_holdout_tc4_20261016_152930` | 283 | 5 |

报告仅保留最终达标配置。每个输出对应一次独立训练轮次，模型、数据快照、评价指标和完整超参数
均保存在该训练轮次目录中。
