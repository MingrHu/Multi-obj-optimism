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

- Python **3.11+**
- CPU-only 即可运行（无需 GPU/CUDA）

## 快速安装

一键脚本（Linux / macOS，自动建 venv、装 CPU 版 torch、装本包与开发依赖）：

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

> 国内网络可为脚本设置 PyPI 镜像：`PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/ bash setup_env.sh`

## 快速上手

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
pytest -m "not slow"          # 默认跳过耗时（DNN 训练等）用例
pytest                        # 全部用例
pytest --cov=mobo             # 覆盖率
```

- `slow`：keras/DNN 训练等耗时用例（默认跳过）。
- `deform`：依赖 Windows DEFORM 环境的用例（非 Windows 跳过）。
- `integration`：依赖 `data/` 真实产物的集成用例（产物缺失自动跳过）。

## 可选 GUI 依赖

PySide6 未被核心逻辑引用，已放入可选依赖组：`pip install -e ".[gui]"`。
