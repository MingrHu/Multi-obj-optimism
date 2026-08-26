# 架构说明（ARCHITECTURE）

本文档描述 `mobo` 包的分层结构、数据流、原子能力层机制、路径与日志策略，以及平台边界。

## 分层总览

```
                         ┌───────────────────────────┐
                         │        mobo.cli           │  命令行入口
                         │  ga / rl / surrogate /    │  （install_stdout_redirect）
                         │  ring_roundness / demo    │
                         └─────────────┬─────────────┘
                                       │ 调用
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
┌──────────────┐ ┌───────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────┐
│  surrogate   │ │optimization│ │extraction /   │ │  automation  │ │  （复用）  │
│              │ │           │ │ replacement   │ │              │ │            │
│  代理模型    │ │  ga / rl   │ │  原子能力层   │ │ DEFORM 流水线│ │            │
└──────┬───────┘ └─────┬─────┘ └───────┬───────┘ └──────┬───────┘ │            │
       │               │               │                │         │            │
       └───────────────┴───────────────┴────────────────┴─────────┘            │
                                       │                                         │
                                       ▼                                         │
                         ┌───────────────────────────┐                          │
                         │        mobo.common        │◄─────────────────────────┘
                         │   paths（集中式路径）      │
                         │   logging（全局日志）      │
                         └───────────────────────────┘
```

对外系统不直接依赖 CLI 或历史算法类，而是通过 `mobo.api` 的 Flask HTTP 服务调用
DOE 聚合服务。路由层位于 `api.handler`，实际处理层位于 `api.service`，每个 DOE 的
样本、模型、训练与优化信息统一落盘到 `data/doe_tasks/<id>`。详细协议见
`DOE_HTTP_API.md`。

- **common**：最底层基础设施，被其余所有子包依赖。
- **surrogate / optimization / extraction / automation**：业务子包。
  - `automation.config.DeformConfig` 复用 `extraction.deform_targets` 的提取函数。
  - `optimization` 运行期从 `data/models` 加载 `surrogate` 产出的模型。
- **cli**：仅编排各子包的入口，是唯一会调用 `logger.install_stdout_redirect()` 的层。

## 目录与模块

| 子包 | 模块 | 职责 |
|---|---|---|
| `common` | `paths.py` | 集中式路径解析（包括 `DOE_TASKS_DIR/AUTO_SINGLE_DIR/AUTO_MULTI_DIR`）+ 环境变量覆盖 |
| `common` | `logging.py` | `GlobalLogger` 单例；显式 stdout 重定向 |
| `common` | `task_store.py` | 任务状态持久化（`data/tasks/<id>/state.json`），三流程共用；`history` 完整记录阶段转移（只追加不覆盖），`resolve_req` 三路解析续跑参数（记录 > 传入 > 报错） |
| `surrogate` | `common.py` | 数据加载/划分/标准化、指标、`save_model`、`Time`、DNN 构建 |
| `surrogate` | `dnn/polynomial/svr/random_forest/kriging.py` | 五种代理模型训练入口 |
| `surrogate` | `interface.py` | `Doe_surrogateModel` 统一训练接口 |
| `surrogate` | `evaluate.py` | `SurrogateModelEvaluator` K 折交叉验证与报告 |
| `surrogate` | `service.py` | `train_surrogate`/`query_model_status`：`model_id` 主键，req/resp 落盘 |
| `api` | `app.py` / `handler.py` / `service.py` | Flask 应用、HTTP 路由及 DOE 聚合处理层 |
| `api` | `store.py` / `runtime.py` | DOE 独立目录持久化与后台任务中止控制 |
| `optimization/ga` | `problem.py` | `SurrogateOptimizationProblem`（pymoo 问题）|
| `optimization/ga` | `operators.py` | `AdaptiveSBX` 自适应交叉、Pareto 结果读写 |
| `optimization/ga` | `run.py` | `NSGA2_run` 运行入口 |
| `optimization/rl` | `env.py` | `ForgingEnv`（Gymnasium 环境）|
| `optimization/rl` | `run.py` | `train_and_optimize`（PPO）|
| `optimization` | `service.py` | `run_optimization`/`query_optimization_status`：`opt_` 主键，参数化 NSGA-II 与结果落盘 |
| `optimization/ga` | `parameterized.py` | 由协议装配 NSGA-II，输出任务级 TSV 解集；历史 `NSGA2_run` 保持不变 |
| `extraction` | `base.py` / `registry.py` | 原子能力层类型与注册/分派 |
| `extraction` | `deform_targets.py` | DEFORM 目标提取原子函数（`_extract*`）|
| `extraction` | `ring_roundness.py` | 碾环截面圆度纯函数 + `extract_ring_roundness` 适配器 |
| `replacement` | `base.py` / `registry.py` | 工艺参数替换原子能力类型与按参数名注册/分派 |
| `replacement` | `deform_parameters.py` | DEFORM 普通关键字与 MOVCTL 控制点块替换原子函数 |
| `automation` | `config.py` | `DeformConfig` 关键字/对象/目标函数映射 |
| `automation` | `sampling.py` | LHS / 全因子采样（纯逻辑）|
| `automation` | `keyfile.py` | KEY 文件文本处理：格式化、路径派生、`generate_key_files`（纯逻辑）|
| `automation` | `solver.py` | DEFORM 子进程驱动（KEY↔DB）与 `DeformSolver` 求解调度；求解进度落盘到 `process_info_file`（记录各 DB 是否完成），支持中断后仅凭进度文件续跑 |
| `automation` | `extract.py` | 结果 DB→KEY 逐步导出与数据集提取编排 |
| `automation` | `incremental.py` | 可选的边求解边提取检查点；按样本序号幂等保存数据行并原子重建数据集，支持并发乱序完成和宕机续跑 |
| `automation` | `pipeline.py` | `TaskStatus`（枚举）/ `ForgingTask` 三阶段状态机 / `generate_sample_file` |
| `automation` | `service.py` | 任务级服务函数：state.json 落盘 + 仅凭 task_id 从磁盘重建续跑 |
| `automation` | `multi_operation.py` / `task_collection.py` | 多工步换模、恢复、参数化 KEY 预生成与可导入任务集合 |

## 数据流

```
参数范围 ──(采样 automation.sampling: LHS/Full)──► 样本 smp.txt
                                            │
                              (automation.ForgingTask: 生成 KEY → 转 DB → 求解)
                                            ▼
                                     DEFORM 结果 DB/KEY
                                            │
                    (extraction: 按 workpiece_type/target 选择提取器)
                                            ▼
                                   数据集（X 工艺参数 + Y 目标值）
                                            │
                              (surrogate: 训练并保存代理模型 + scalers)
                                            ▼
                                   data/models/<family>/*
                                            │
                    (optimization: GA / RL 加载模型做多目标寻优)
                                            ▼
                          Pareto 前沿 / RL 解集（data/*.txt, *.png）
```

## 原子能力层机制（extraction）

上游按 `(workpiece_type, target_name)` 请求提取能力，注册表返回一个
`ExtractorSpec`，调用方据 `spec.kind` 选择调用约定：

- **`key_lines`**：`fn(all_lines: list[list[str]], obj: str, in_progress: bool) -> str`
  —— 对应 DEFORM 目标提取（`_extractMaxStress/_extractMaxLoad/_extractGrainStdv`）。
- **`key_file`**：`fn(key_path, **kwargs) -> float`
  —— 对应碾环圆度 `extract_ring_roundness`。

关键设计：

- 键为 `(workpiece_type, target_name)`；`resolve()` 在工件专属提取器缺失时**回退到
  通用工件**（默认 `generic`），实现「根据上游传入的工件类型选择抽取函数」。
- 内置注册在 `extraction/__init__.py` 完成；新增能力用
  `registry.register_fn(...)` 或 `@registry.register(...)`，**无需改动底层提取函数体**。
- `DeformConfig.TAR_FUNC` 仍直接引用同一批提取函数，保证 `automation` 流水线零改动；
  registry 作为统一对外原子层与之并行存在。

## 工艺参数替换原子能力（replacement）

`automation.keyfile` 只负责 KEY 文件读写、样本参数组装和能力路由。具体文本修改位于
`replacement.deform_parameters`，注册表按参数名返回两类能力：

- **`line`**：匹配 KEY 关键字与对象 ID，替换普通参数行；
- **`document`**：处理跨多行或多个参数共同决定的结构，例如根据上下界缩放完整
  `MOVCTL` 控制点块。

新增参数类型时实现独立函数并在 `replacement/__init__.py` 注册，无需修改 KEY 生成流程。

TC4 碾环多工步任务由 `task_collection.TC4_RING_MULTI_TASK_1` 完整定义。
工步1不含参数绑定，预生成时字节复制；工步2、3将一个模具温度样本值
展开到所有模具对象。`prepare_keys()` 只验证参数到 KEY 的映射，不调用 DEFORM。

7050 和 GH4169 碾环单工步任务分别由 `task_collection.RING_7050_SINGLE_TASK_1` 与
`task_collection.GH4169_RING_SINGLE_TASK_1` 定义，复用
`ForgingTask` 的 KEY 生成、求解与提取流水线。一个 `ring_die_temperature` 样本值通过
文档级替换原子能力同步作用于对象 2～5；GH4169 模板保留其自带的 GRNDAT 晶粒模型，
初始和平均晶粒尺寸均为 50 μm；`material_fill` 是尚未确定计算方法的目标占位符。
结果提取不再要求调用方提供最大步数：终态目标用空步号导出 DB 最新结果集；全过程目标
先从 DEFORM 的 `Step Numbers` 查询 DB 实际保存步号，再逐帧导出。多工步全过程目标从
各工步 checkpoint DB 提取，因此启用这类目标时必须保留工步检查点。

## 路径集中化

所有路径统一由 `mobo.common.paths` 提供，取代旧代码中的 `../../data/...` 相对路径与
`os.chdir`、以及 macOS/Windows 硬编码绝对路径：

- `PROJECT_DIR` 通过向上查找含 `pyproject.toml` 或 `data/` 的祖先目录推导（比硬数层数稳健）。
- 支持环境变量覆盖：`MOBO_PROJECT_DIR`、`MOBO_DATA_DIR`。
- 函数改造仅限「默认参数/路径来源」，算法体保持 byte-for-byte 不变。
- 运行时工作区分为 `data/AUTO/single/<task>` 和 `data/AUTO/mult/<task>`。
- 多工步样本按工步归档产物：`runs/<sample>/op<n>/` 内集中保存
  `<模板名>_parameterized.KEY`、`result.DB`、`terminal.KEY`、`checkpoint.DB`、
  DEFORM 日志及换模 KEY；下一工步从前一工步 DB 副本继续计算。
- 任务状态集中在 `data/tasks/<task_id>/`：`state.json` 记录任务阶段，多工步的
  `multi_operation_state.json` 记录逐样本/逐工步恢复状态，单工步的
  `process_info.json` 记录逐 DB 求解进度；启用增量数据集后，
  `incremental_dataset.json` 记录逐样本提取状态与数据行。`AUTO` 只保存样本、DB、KEY
  和结果数据。
- HTTP 层以 `data/doe_tasks/<doe_id>/doe.json` 作为聚合状态入口，并在同一 DOE 目录下
  隔离 `samples/models/training/optimization` 产物；底层代理训练和优化仍复用
  `data/tasks/<model_id或optimization_id>` 记录，删除 DOE 时一并清理关联记录。

## 增量数据集与断点恢复

- 单工步调用 `init_execution_task(..., incremental=True)` 启用；每个 DB 求解完成后立即
  导出各步 KEY 并提取该样本。未显式指定时，结果固定写入
  `<res_txt_path>/<task_id>_incremental_result.txt`。
- 多工步调用 `init_multi_operation_task(..., incremental=True)` 启用；每个样本的最终
  工步完成后立即使用各工步终态 KEY 提取该样本，结果固定保存到任务工作区的
  `results` 目录。
- 检查点以样本序号为主键，重复恢复只覆盖同一行；数据集始终按样本序号排序，并通过
  临时文件加 `os.replace` 原子更新。求解已完成而提取未完成的样本会在续跑时补提取。

## 日志设计取舍

旧 `GlobalLogger` 在构造（import）时即执行 `sys.stdout = self`，产生全局副作用并破坏
pytest 的输出捕获。现调整为：

- 构造时**只配置** file/console handler，不劫持 stdout，也不在 import 时创建 `logs/`
  目录（`FileHandler(delay=True)` + 惰性建目录）。
- 提供显式的 `install_stdout_redirect()` / `restore_stdout()`。
- 仅 **CLI 入口**在运行时调用 `install_stdout_redirect()`，库导入与测试默认不劫持。

## 平台边界

`mobo.automation` 通过子进程驱动 Windows 平台的 `DEF_PRE_64.exe` /
`DEF_ARM_CTL.COM`，**仅能在装有 DEFORM 的 Windows 环境真实运行**。在 Linux/macOS 上：

- 采样/格式化/路径等纯逻辑函数可正常使用与测试；
- `ForgingTask` 提供 `dry_run=True` 分支（只推进状态、不真正调用 DEFORM），可用于状态机测试；
- 真实子进程调用相关用例以 `@pytest.mark.deform` 标记并在非 Windows 跳过；
  `solver` 相关用例通过打桩 `subprocess.Popen` 验证命令串与调度逻辑。

路径拼接已从 Windows 反斜杠改为跨平台的 `os.path.join`，其余逻辑不变。
