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
- **DOE HTTP 服务**（`mobo.api`）：通过 Flask 暴露任务、采样、训练、推理和优化接口，
  并按 DOE ID 隔离运行状态与产物。
- **命令行入口**（`mobo.cli`）：圆度提取、GA/RL 优化、代理模型评估。

> 详细分层与数据流见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)；开发约定见 [AGENTS.md](AGENTS.md)；
> HTTP 协议见 [DOE_HTTP_API.md](docs/api/DOE_HTTP_API.md)，其余资料见 [文档导航](docs/README.md)。

---

## 环境要求

- Python **3.11 或 3.12**（推荐 3.12）
- CPU-only 即可运行（无需 GPU/CUDA）
- DEFORM 自动化求解仅支持已安装 DEFORM 的 Windows

## 快速安装

Windows PowerShell（DEFORM 自动化推荐）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

已有虚拟环境失效时使用 `.\scripts\setup_env.ps1 -Recreate`，也可通过 `-PythonPath` 指定
Python 3.11 或 3.12 解释器。

Linux / macOS（不能真实调用 DEFORM，可运行其余功能）：

```bash
bash scripts/setup_env.sh            # 核心 + 开发依赖
bash scripts/setup_env.sh --with-gui # 额外安装 GUI (PySide6)
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
python -c "import flask, requests, torch, mobo; print(torch.__version__, mobo.__version__)"
```

> 国内网络可通过 `PIP_INDEX` 环境变量或 PowerShell 的 `-PipIndex` 参数指定镜像。

若 DEFORM 可执行文件没有加入 PATH，可在运行前设置：

```powershell
$env:MOBO_DEF_PRE_64 = "C:\Program Files\SFTC\DEFORM\v11.0\3D\DEF_PRE_64.exe"
$env:MOBO_DEF_ARM_CTL = "C:\Program Files\SFTC\DEFORM\v11.0\3D\DEF_ARM_CTL.COM"
```

安装脚本最后会运行 `pip check` 并导入数值计算、代理模型、优化和自动化所需依赖。

## Docker 部署与更新

Docker 部署适用于不希望在服务器宿主机安装 Python、TensorFlow、PyTorch 和算法依赖的场景。
容器内自动启动 `mobo-api`，宿主机只需准备 Git、Docker Engine 和 Compose v2。

### 新 Linux 宿主机首次部署

按以下顺序在宿主机执行：

```bash
# 1. CentOS 尚未安装 Git 时先安装 Git
sudo dnf install -y git

# 2. 克隆仓库；仓库提供源码、Dockerfile、Compose、依赖锁定和部署脚本
git clone https://github.com/MingrHu/Multi-obj-optimism.git

# 3. 进入包含 compose.yaml 的仓库根目录
cd Multi-obj-optimism

# 4. 检查系统资源、Docker daemon、Compose v2 和 5000 端口
bash scripts/check_docker_host.sh

# 5. 仅当 CentOS Stream 9/10 尚未安装 Docker 时执行；其他系统使用官方对应安装方式
bash scripts/install_docker_centos.sh --add-current-user

# 6. 首次加入 docker 用户组后应注销并重新登录，再回到仓库根目录重新执行环境检查
bash scripts/check_docker_host.sh

# 7. 构建完整镜像并在后台启动 mobo-api
docker compose up --build -d

# 8. 等待容器状态变为 healthy
docker compose ps

# 9. 从当前宿主机直接请求容器映射出来的 HTTP 服务
curl http://127.0.0.1:5000/health
```

如果第 4 步已经显示 Docker 和 Compose 均可用，则跳过第 5、6 步。CentOS Stream 8 已结束
维护，仓库脚本不会在该系统上自动安装 Docker；若服务器已经具备可工作的 Docker，仍可在
接受宿主机风险的前提下继续构建和启动。

运行仓库自带的完整 HTTP Demo，无需在宿主机安装 Python 依赖：

```bash
# 在已经运行的容器内执行完整采样、训练、推理和优化 Demo
docker exec \
  -e MOBO_API_URL=http://127.0.0.1:5000 \
  mobo-api \
  python -m mobo.api.demo
```

查看服务日志：

```bash
# 在仓库根目录持续查看 Compose 服务日志；Ctrl+C 不会停止容器
docker compose logs -f mobo-api

# 不依赖 compose.yaml，在宿主机任意目录查看同一容器日志
docker logs -f mobo-api
```

### 更新代码或文档并发布到容器

宿主机仓库与运行中的容器是两个独立文件系统。`git pull` 只更新宿主机文件，不会实时改变
当前容器；`docker compose restart` 也只会重启旧容器，不能应用新代码。正确更新流程为：

```bash
# 1. 在宿主机仓库中拉取已经提交的新代码和文档
git pull --ff-only

# 2. 重新构建镜像，并使用新镜像重新创建容器
docker compose up --build -d

# 3. 确认新容器健康
docker compose ps

# 4. 从宿主机验证新容器的 HTTP 服务
curl http://127.0.0.1:5000/health
```

更新原理如下：

1. `compose.yaml` 把仓库根目录作为 Docker 构建上下文，并使用
   `deploy/docker/Dockerfile`。
2. Dockerfile 先安装 `requirements/` 中的锁定依赖；依赖文件没变时直接复用缓存。
3. `COPY . .` 将未被 `.dockerignore` 排除的源码、文档、脚本和配置复制到镜像 `/app`。
4. `pip install . --no-deps` 将当前 `src/mobo` 构建并安装到镜像的 Python 环境。
5. 每次构建产生新的不可变镜像；Compose 发现镜像变化后停止旧容器并创建新容器。
6. `data` 和 `logs` 不进入镜像，而是挂载 Docker 命名卷，因此替换容器不会删除 DOE、模型、
   优化结果和文件日志。

这套 Compose 没有把宿主机源码绑定挂载到 `/app`，所以生产运行不会随宿主机文件即时变化；
任何代码或文档更新都必须重新构建镜像。完整的 CentOS 准备、离线部署、远程访问、日志、
数据备份和回滚说明见 [Docker 部署文档](docs/deployment/DOCKER_DEPLOYMENT.md)。

## 快速上手

需要从外部系统通过 HTTP 调用 DOE 采样、代理训练/评价、推理与优化时，使用
[`mobo.api`](docs/api/DOE_HTTP_API.md)。该入口通过 Flask 暴露 DOE HTTP 服务，并以 DOE ID
隔离模型和优化产物。

启动后端并运行完整训练、推理和 NSGA-II 优化 Demo：

```powershell
# 终端一
mobo-api

# 终端二
python -m mobo.api.demo
```

详细部署、健康检查、端口配置和故障排查见
[API 后端启动文档](docs/deployment/BACKEND_STARTUP.md)。

样本、训练数据集、推理和优化结果可通过同一个 GET 接口按字段读取。`resource_id`
取自对应生成、推理或优化接口的响应，端上无需接触服务器文件路径：

```text
GET /api/v1/hust/doe/data/get?id=<doe_id>&resource_id=<tos-resource-id>&fields=temperature
```

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
mobo-ring-roundness                         # 默认 data/keyfile/RINGROLL.KEY
mobo-ring-roundness path/to/model.KEY --plane-z 0
```

或在代码中通过原子能力层按工件类型分派：

```python
from mobo.extraction import registry

spec = registry.resolve("ring", "roundness_inner")   # -> ExtractorSpec(kind="key_file")
value = spec.fn("data/keyfile/RINGROLL.KEY", samples=3000)
```

## 项目结构与文件说明

### 顶层目录

```
Multi-obj-optimism/
├── src/mobo/          # src-layout 单包源码
├── tests/             # 单元测试与集成测试
├── docs/              # 架构、API、部署和 DEFORM 文档知识库
├── requirements/      # runtime/dev/gui/server 锁定依赖
├── deploy/docker/     # Dockerfile 与 Gunicorn 配置
├── scripts/           # 环境安装及运维/性能脚本
├── tools/             # 仓库维护和文档一致性工具
├── .github/workflows/ # CI 与定期检查
├── data/ / logs/      # 本地运行产物（忽略，不提交）
├── compose.yaml       # 本地和服务器容器编排入口
├── pyproject.toml     # 包、依赖声明、CLI 与测试配置
├── README.md          # 项目入口与快速上手
└── AGENTS.md          # AI 代理开发约定
```

### 源码 `src/mobo/`

```
src/mobo/
├── api/          # Flask DOE 聚合服务、状态持久化、后台任务和 HTTP Demo
├── common/       # 路径、日志及 data/tasks 通用任务状态
├── surrogate/    # 五类代理模型、统一训练接口、评价和任务服务
├── optimization/ # 参数化 NSGA-II、历史 GA 入口、RL/PPO 与优化服务
├── extraction/   # 按工件类型和目标名称注册/分派结果提取能力
├── replacement/  # 按参数名称注册/分派 DEFORM KEY 替换能力
├── automation/   # 单/多工步 DEFORM、增量提取、断点恢复与任务集合
└── cli/          # mobo-surrogate、mobo-ga、mobo-rl、圆度与自动化演示入口
```

具体模块职责以 [ARCHITECTURE.md](docs/ARCHITECTURE.md) 为准，避免在多个文档中维护重复的
逐文件清单。

### 测试 `tests/`

```
tests/
├── conftest.py            # 共享 fixture 与 marker 配置
├── unit/                  # 单元测试（快速、无外部依赖）
│   ├── test_paths.py / test_logging.py    # common 基础设施
│   ├── surrogate/         #   代理模型公共库与评估
│   ├── optimization/      #   GA 算子/问题、RL 环境
│   ├── extraction/        #   原子能力层与提取函数
│   ├── replacement/       #   KEY 参数替换能力
│   ├── api/               #   HTTP 协议、状态和字段取数
│   ├── automation/        #   采样/KEY 文件/求解/提取/流水线/服务
│   └── tools/             #   文档一致性检查器
└── integration/           # 集成测试（依赖 data/ 真实产物，缺失自动跳过）
```

## 数据目录布局

```
data/
├── doe_tasks/         # HTTP 服务 DOE 运行状态及样本、模型和优化产物
├── tasks/             # 底层代理训练、优化和自动化任务状态
├── models/            # 训练好的代理模型（GA/RL 运行时依赖）
│   ├── PRG/           #   多项式模型：<target>_model.pkl + <target>_scalers.pkl
│   └── DNN/           #   DNN 模型：<target>_model.keras + <target>_scalers.pkl
├── AUTO/              # DEFORM 自动化样本、KEY、DB 与结果数据
├── TEST/              # 示例数据集 simulated.txt 与评估报告
└── keyfile/           # DEFORM KEY 文件（圆度提取用）
```

`data/` 是运行期目录，已整体加入 `.gitignore`。容器部署时应挂载持久卷，并通过
`MOBO_PROJECT_DIR`、`MOBO_DATA_DIR` 覆盖默认路径。

## 测试

```bash
ruff check src tests scripts tools  # 静态检查（语法、未定义名称、可疑默认参数等）
pytest -m "not slow"          # 默认跳过耗时（DNN 训练等）用例
pytest                        # 全部用例
pytest --cov=mobo             # 覆盖率
python tools/check_docs.py    # 文档知识库、路由、CLI、包结构和链接一致性
```

- `slow`：keras/DNN 训练等耗时用例（默认跳过）。
- `deform`：依赖 Windows DEFORM 环境的用例（非 Windows 跳过）。
- `integration`：依赖 `data/` 真实产物的集成用例（产物缺失自动跳过）。

## 文档知识库维护

- [README.md](README.md)：项目入口、安装、启动和目录总览。
- [docs/README.md](docs/README.md)：文档知识库导航与职责边界。
- [DOE_HTTP_API.md](docs/api/DOE_HTTP_API.md)：端上调用的唯一 HTTP 协议。
- [interface_protocol.md](docs/api/interface_protocol.md)：Python 内部任务服务协议。
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)：分层、数据流、持久化和平台边界。
- [DOCKER_DEPLOYMENT.md](docs/deployment/DOCKER_DEPLOYMENT.md)：容器构建、数据卷、发布与服务器部署。
- [DEFORM_KEY_KEYWORDS.md](docs/deform/DEFORM_KEY_KEYWORDS.md)：KEY 关键字与能力映射。
- [接口参数文档.md](docs/api/接口参数文档.md)：兼容历史文件名的文档索引。

`python tools/check_docs.py` 会检查知识库链接、HTTP 路由、CLI 入口、包/模块清单和文档
快照。代码公共表面变化后，应先更新文档，再执行
`python tools/check_docs.py --update-snapshot` 确认新的基线。GitHub Actions 会在相关
变更以及每周一自动执行检查。

## 可选 GUI 依赖

PySide6 未被核心逻辑引用，已放入可选依赖组：`pip install -e ".[gui]"`。
