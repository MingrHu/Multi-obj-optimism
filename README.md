# mobo · 多目标锻造工艺优化工具包

`mobo`（Multi-Objective Bayesian/Black-box Optimization）是一个面向锻造工艺参数优化的
Python 工具包，覆盖从数据生成到多目标寻优的完整链路：

- **代理模型**（`mobo.surrogate`）：DNN / 多项式回归 / SVR / 随机森林 / Kriging(GPR)，
  统一训练接口与 K 折交叉验证评估。
- **多目标优化**（`mobo.optimization`）：基于 pymoo 的 NSGA-II 遗传算法，以及基于
  stable-baselines3 的强化学习（PPO）优化器。
- **原子能力层**（`mobo.extraction`）：按「工件类型 + 目标名称」分派的 KEY 文件提取能力
  （DEFORM 应力/载荷/晶粒、碾环内外圈圆度），可注册、可回退。
- **DEFORM 自动化**（`mobo.automation`）：采样 → KEY/DB 处理 → 求解调度 → 结果提取的
  流水线（依赖 Windows 平台的 DEFORM）。
- **命令行入口**（`mobo.cli`）：圆度提取、GA/RL 优化、代理模型评估。

> 详细分层与数据流见 [ARCHITECTURE.md](ARCHITECTURE.md)；开发约定见 [AGENTS.md](AGENTS.md)。

---

## 环境要求

- Python **3.11 或 3.12**（推荐 3.12）
- CPU-only 即可运行（无需 GPU/CUDA）
- DEFORM 自动化求解仅支持已安装 DEFORM 的 Windows

## 快速安装

Windows PowerShell（DEFORM 自动化推荐）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

Linux / macOS（不能真实调用 DEFORM，可运行其余功能）：

```bash
bash setup_env.sh            # 核心 + 开发依赖
bash setup_env.sh --with-gui # 额外安装 GUI (PySide6)
```

或手动安装：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# torch 从 CPU 源安装，避免拉取 GPU 版与大量 CUDA 包
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

安装完成后自检：

```bash
python -c "import torch, mobo; print(torch.__version__, mobo.__version__)"
```

> 国内网络可通过 `PIP_INDEX` 环境变量或 PowerShell 的 `-PipIndex` 参数指定镜像。

若 DEFORM 可执行文件没有加入 PATH，可在运行前设置：

```powershell
$env:MOBO_DEF_PRE_64 = "C:\Program Files\SFTC\DEFORM\v11.0\3D\DEF_PRE_64.exe"
$env:MOBO_DEF_ARM_CTL = "C:\Program Files\SFTC\DEFORM\v11.0\3D\DEF_ARM_CTL.COM"
```

安装脚本最后会运行 `pip check` 并导入数值计算、代理模型、优化和自动化所需依赖。

## 快速上手

需要从外部系统按 JSON/字典直接调用代理训练和参数化 NSGA-II 时，使用
[`mobo.api`](PUBLIC_API.md)。该入口会校验变量映射、边界、目标与约束，并以任务 ID
隔离模型和优化产物。

### 1. 训练 / 评估代理模型

```python
from mobo.surrogate.interface import Doe_surrogateModel

vars_out = ["1", "2", "3", "grain", "load"]   # 3 输入 + 2 目标
doe = Doe_surrogateModel("data/TEST/simulated.txt", vars_out, n_vars=3)
doe.train_save_model(2)   # 0:Kriging 1:DNN 2:多项式 3:SVR 4:随机森林
```

命令行做交叉验证评估：

```bash
mobo-surrogate        # 或 python -m mobo.cli.surrogate
```

### 2. NSGA-II 遗传算法优化

```bash
mobo-ga               # 或 python -m mobo.cli.ga
```

加载 `data/models/PRG` 下的代理模型，输出 Pareto 前沿到
`data/pareto_solutions.txt` 与 `data/pareto_front.png`。

### 3. 强化学习（PPO）优化

```bash
mobo-rl               # 或 python -m mobo.cli.rl
```

结果输出到 `data/rl_solutions_sb3.txt`。

### 4. 碾环圆度提取（原子能力层）

```bash
mobo-ring-roundness                         # 默认 data/KEY_FILE/RINGROLL.KEY
mobo-ring-roundness path/to/model.KEY --plane-z 0
```

或在代码中通过原子能力层按工件类型分派：

```python
from mobo.extraction import registry

spec = registry.resolve("ring", "roundness_inner")   # -> ExtractorSpec(kind="key_file")
value = spec.fn("data/KEY_FILE/RINGROLL.KEY", samples=3000)
```

## 项目结构与文件说明

### 顶层目录

```
Multi-obj-optimism/
├── src/mobo/          # 源码（src-layout 单包，pip install -e . 后以 mobo 导入）
├── tests/             # 单元测试（unit/）与集成测试（integration/）
├── data/              # 数据集、训练好的模型、DEFORM KEY 文件（见下方布局）
├── logs/              # 运行日志（DEFORM 操作日志等，不纳入版本管理）
├── pyproject.toml     # 打包与依赖配置、pytest/coverage 配置、CLI 入口点
├── requirements.txt   # 依赖清单（供不走 pyproject 的场景参考）
├── requirements-dev.txt / requirements-gui.txt # 开发测试与可选 GUI 锁定依赖
├── setup_env.ps1      # Windows 一键环境安装脚本
├── setup_env.sh       # 一键环境安装脚本（建 venv + CPU 版 torch + 本包）
├── README.md          # 本文件：总览、安装、上手、结构说明
├── ARCHITECTURE.md    # 分层架构、模块职责表与数据流
└── AGENTS.md          # AI 代理协作的硬约束与开发约定
```

### 源码 `src/mobo/`

```
src/mobo/
├── __init__.py            # 包版本与顶层导出
│
├── api/                   # 对外 JSON/字典门面（代理训练 + 参数化 NSGA-II）
│   ├── validation.py      #   协议解析、字段适配和交叉字段校验
│   └── facade.py          #   train_surrogate/run_optimization/query_task
│
├── common/                # 基础设施（被各子包复用）
│   ├── paths.py           #   集中式路径解析：PROJECT_DIR/DATA_DIR/MODELS_DIR/... 与环境变量覆盖
│   └── logging.py         #   全局 logger（单例 + 统一格式，仅 CLI 入口劫持 stdout）
│
├── surrogate/             # 代理模型：训练、评估、保存/加载
│   ├── common.py          #   公共库：数据加载/预处理、划分、误差指标、模型保存
│   ├── interface.py       #   统一训练入口 Doe_surrogateModel（按编号选模型）
│   ├── dnn.py             #   DNN(keras) 训练：dnn_run
│   ├── polynomial.py      #   多项式回归：prg_fun
│   ├── svr.py             #   支持向量回归
│   ├── random_forest.py   #   随机森林回归
│   ├── kriging.py         #   Kriging / 高斯过程回归(GPR)
│   ├── evaluate.py        #   K 折交叉验证评估与报告
│   └── service.py         #   训练任务状态与 model_id 专属模型产物
│
├── optimization/          # 多目标优化
│   ├── ga/                #   NSGA-II 遗传算法（pymoo）
│   │   ├── operators.py   #     自定义交叉/变异算子
│   │   ├── problem.py     #     多目标问题定义（加载代理模型做评估）
│   │   └── run.py         #     NSGA2_run：装配并运行、输出 Pareto 前沿
│   └── rl/                #   强化学习优化（stable-baselines3 PPO）
│       ├── env.py         #     ForgingEnv 优化环境（gymnasium）
│       └── run.py         #     train_and_optimize：训练 PPO 并导出解集
│
├── extraction/            # 原子能力层：按「工件类型 + 目标」分派的 KEY 文件提取
│   ├── base.py            #   ExtractorSpec/Kind：两种调用约定（key_lines / key_file）
│   ├── registry.py        #   注册表：register_fn / resolve（缺工件专属时回退 generic）
│   ├── deform_targets.py  #   DEFORM 目标提取原子函数：应力/载荷/晶粒（_extract*）
│   └── ring_roundness.py  #   碾环内外圈圆度纯函数 + extract_ring_roundness 适配器
│
├── automation/            # DEFORM 自动化流水线（真实执行依赖 Windows DEFORM）
│   ├── config.py          #   DeformConfig：KEY 关键字/对象 ID/目标函数映射
│   ├── sampling.py        #   采样：LHS / 全因子（纯逻辑，可独立测试）
│   ├── keyfile.py         #   KEY 文件文本处理：数值格式化、路径派生、generate_key_files
│   ├── solver.py          #   DEFORM 子进程驱动（KEY↔DB）与 DeformSolver 求解调度
│   ├── extract.py         #   结果 DB→KEY 逐步导出与数据集提取编排
│   ├── pipeline.py        #   TaskStatus 枚举 + ForgingTask 三阶段状态机
│   └── service.py         #   任务级服务函数（创建/初始化/推进/查询状态）
│
└── cli/                   # 命令行入口（对应 pyproject 的 console_scripts）
    ├── surrogate.py       #   mobo-surrogate：代理模型交叉验证评估
    ├── ga.py              #   mobo-ga：NSGA-II 优化
    ├── rl.py              #   mobo-rl：PPO 强化学习优化
    ├── ring_roundness.py  #   mobo-ring-roundness：碾环圆度提取
    └── automation_demo.py #   DEFORM 自动化流水线调用演示（非 pytest）
```

### 测试 `tests/`

```
tests/
├── conftest.py            # 共享 fixture 与 marker 配置
├── unit/                  # 单元测试（快速、无外部依赖）
│   ├── test_paths.py / test_logging.py    # common 基础设施
│   ├── surrogate/         #   代理模型公共库与评估
│   ├── optimization/      #   GA 算子/问题、RL 环境
│   ├── extraction/        #   原子能力层与提取函数
│   └── automation/        #   采样/KEY 文件/求解/提取/流水线/服务
└── integration/           # 集成测试（依赖 data/ 真实产物，缺失自动跳过）
```

## 数据目录布局

```
data/
├── models/            # 训练好的代理模型（GA/RL 运行时依赖）
│   ├── PRG/           #   多项式模型：<target>_model.pkl + <target>_scalers.pkl
│   └── DNN/           #   DNN 模型：<target>_model.keras + <target>_scalers.pkl
├── TEST/              # 示例数据集 simulated.txt 与评估报告
└── KEY_FILE/          # DEFORM KEY 文件（圆度提取用），如 RINGROLL.KEY
```

自定义路径可用环境变量覆盖：`MOBO_PROJECT_DIR`、`MOBO_DATA_DIR`。

## 测试

```bash
ruff check src tests scripts    # 静态检查（语法、未定义名称、可疑默认参数等）
pytest -m "not slow"          # 默认跳过耗时（DNN 训练等）用例
pytest                        # 全部用例
pytest --cov=mobo             # 覆盖率
```

- `slow`：keras/DNN 训练等耗时用例（默认跳过）。
- `deform`：依赖 Windows DEFORM 环境的用例（非 Windows 跳过）。
- `integration`：依赖 `data/` 真实产物的集成用例（产物缺失自动跳过）。

## 可选 GUI 依赖

PySide6 未被核心逻辑引用，已放入可选依赖组：`pip install -e ".[gui]"`。
