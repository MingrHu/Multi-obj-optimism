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
│  surrogate   │ │optimization│ │  extraction   │ │  automation  │ │  （复用）  │
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

- **common**：最底层基础设施，被其余所有子包依赖。
- **surrogate / optimization / extraction / automation**：业务子包。
  - `automation.config.DeformConfig` 复用 `extraction.deform_targets` 的提取函数。
  - `optimization` 运行期从 `data/models` 加载 `surrogate` 产出的模型。
- **cli**：仅编排各子包的入口，是唯一会调用 `logger.install_stdout_redirect()` 的层。

## 目录与模块

| 子包 | 模块 | 职责 |
|---|---|---|
| `common` | `paths.py` | 集中式路径解析（`PROJECT_DIR/DATA_DIR/MODELS_DIR/TEST_DIR/KEY_FILE_DIR/TASKS_DIR`）+ 环境变量覆盖 |
| `common` | `logging.py` | `GlobalLogger` 单例；显式 stdout 重定向 |
| `common` | `task_store.py` | 任务状态持久化（`data/tasks/<id>/state.json`），三流程共用；`history` 完整记录阶段转移（只追加不覆盖），`resolve_req` 三路解析续跑参数（记录 > 传入 > 报错） |
| `surrogate` | `common.py` | 数据加载/划分/标准化、指标、`save_model`、`Time`、DNN 构建 |
| `surrogate` | `dnn/polynomial/svr/random_forest/kriging.py` | 五种代理模型训练入口 |
| `surrogate` | `interface.py` | `Doe_surrogateModel` 统一训练接口 |
| `surrogate` | `evaluate.py` | `SurrogateModelEvaluator` K 折交叉验证与报告 |
| `surrogate` | `service.py` | `train_surrogate`/`query_model_status`：`model_id` 主键，req/resp 落盘 |
| `optimization/ga` | `problem.py` | `SurrogateOptimizationProblem`（pymoo 问题）|
| `optimization/ga` | `operators.py` | `AdaptiveSBX` 自适应交叉、Pareto 结果读写 |
| `optimization/ga` | `run.py` | `NSGA2_run` 运行入口 |
| `optimization/rl` | `env.py` | `ForgingEnv`（Gymnasium 环境）|
| `optimization/rl` | `run.py` | `train_and_optimize`（PPO）|
| `optimization` | `service.py` | `run_optimization`/`query_optimization_status`：`opt_` 主键，结果落盘 |
| `extraction` | `base.py` / `registry.py` | 原子能力层类型与注册/分派 |
| `extraction` | `deform_targets.py` | DEFORM 目标提取原子函数（`_extract*`）|
| `extraction` | `ring_roundness.py` | 碾环截面圆度纯函数 + `extract_ring_roundness` 适配器 |
| `automation` | `config.py` | `DeformConfig` 关键字/对象/目标函数映射 |
| `automation` | `sampling.py` | LHS / 全因子采样（纯逻辑）|
| `automation` | `keyfile.py` | KEY 文件文本处理：格式化、路径派生、`generate_key_files`（纯逻辑）|
| `automation` | `solver.py` | DEFORM 子进程驱动（KEY↔DB）与 `DeformSolver` 求解调度 |
| `automation` | `extract.py` | 结果 DB→KEY 逐步导出与数据集提取编排 |
| `automation` | `pipeline.py` | `TaskStatus`（枚举）/ `ForgingTask` 三阶段状态机 / `generate_sample_file` |
| `automation` | `service.py` | 任务级服务函数：state.json 落盘 + 仅凭 task_id 从磁盘重建续跑 |

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

## 路径集中化

所有路径统一由 `mobo.common.paths` 提供，取代旧代码中的 `../../data/...` 相对路径与
`os.chdir`、以及 macOS/Windows 硬编码绝对路径：

- `PROJECT_DIR` 通过向上查找含 `pyproject.toml` 或 `data/` 的祖先目录推导（比硬数层数稳健）。
- 支持环境变量覆盖：`MOBO_PROJECT_DIR`、`MOBO_DATA_DIR`。
- 函数改造仅限「默认参数/路径来源」，算法体保持 byte-for-byte 不变。

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
