# 依赖清单说明

本目录保存可复现环境使用的锁定依赖。项目元数据和直接依赖声明位于根目录
[`pyproject.toml`](../pyproject.toml)，具体部署或开发环境按用途组合本目录中的清单。

## 文件职责

| 文件 | 内容 | 适用场景 |
|---|---|---|
| `runtime.txt` | Flask API、数据处理、代理模型、优化、强化学习等完整运行依赖及传递依赖 | 后端运行、算法运行、Docker 基础环境 |
| `server.txt` | Gunicorn 生产级 WSGI 服务 | Linux 容器和生产后端，需要与 `runtime.txt` 一起安装 |
| `dev.txt` | `runtime.txt` 加 pytest、Ruff、Radon、Xenon、Vulture 和 pip-audit | 本地开发、测试、质量评分和安全审计 |
| `gui.txt` | PySide6 及其锁定组件 | Windows 桌面界面，纯后端无需安装 |

`server.txt` 和 `gui.txt` 都不是完整运行环境，不能替代 `runtime.txt`。`dev.txt` 第一行通过
`-r runtime.txt` 包含完整运行依赖，因此开发环境不需要重复安装 `runtime.txt`。

## 推荐安装方式

优先使用仓库脚本创建虚拟环境并安装 CPU 版 PyTorch、运行依赖、开发工具和当前包。

Windows 一键安装（推荐，包含桌面 UI）：

```powershell
.\install_windows.cmd
```

也可双击仓库根目录的 `install_windows.cmd`。不需要预先激活 `.venv`、安装 Bash 或修改系统执行策略。
入口启动 Windows PowerShell，自动复用可用的64位 Python 3.11/3.12；找不到时从 Python 官方下载
并验证签名，将 Python 3.12.10 安装到仓库 `.tools/python312`。该版本是 Python 3.12 最后一个
提供官方 Windows 安装器的完整维护版本，已有合适解释器时不会降级它。
缺少 Microsoft Visual C++ x64 运行库时，从微软官方下载并验证签名后安装；Windows 可能弹出管理员授权。
自动化环境可设置 `MOBO_SETUP_NO_PAUSE=1`，跳过命令行入口末尾的按键等待。
下载来源：[Python Windows 安装器](https://www.python.org/downloads/release/python-31210/)、
[微软运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。

脚本安装 CPU PyTorch、完整运行/算法依赖、开发工具、当前包及 GUI；逐步检查退出码，
最后执行 `pip check`、完整算法/API 入口导入、QtCharts 导入及文档检查。
只有全部检查通过才输出 `SUCCESS`，失败立即停止并返回非零退出码。
默认复用现有 `.venv`，重新执行即可按当前锁定清单更新依赖；不要把旧服务器的 `.venv` 直接复制过来。
对迁移来的环境、版本冲突环境或需要干净重装的环境，先关闭使用该环境的程序，再执行：

```powershell
.\install_windows.cmd -Recreate
```

旧环境保留为仓库根目录的 `.venv.backup-<唯一编号>`，不会删除数据、KEY 或计算结果。
Python 安装器、仓库本地 Python 和备份环境均已被 Git 忽略。安装失败可重新执行；
需要重启系统的运行库安装会给出提示。

高级入口仍可单独使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -WithGui
```

可选参数：`-PythonPath /absolute/path/to/python.exe` 指定64位 Python 3.11/3.12；
`-PipIndex` 指定普通依赖镜像（CPU PyTorch 始终使用其专用源）；`-NoAutoPython` 禁止自动下载 Python；
`-SkipVCRuntime` 跳过系统运行库检测/安装；`-DeformDir /absolute/path/to/deform/3D` 指定现有 DEFORM 目录。
PowerShell 5.1 和7均支持，某个 `py` 启动器候选失败后会继续探测其余候选。
不加 `-WithGui` 的高级入口仅安装完整运行和开发依赖。

脚本检测已配置的 DEFORM 路径、PATH 以及 Program Files 下的 SFTC/DEFORM 安装目录，
发现有效前处理器和控制程序后，将 `MOBO_DEF_PRE_64`、`MOBO_DEF_ARM_CTL` 写入当前进程及用户环境变量。
新打开的终端/工作台可直接读取；已打开的其他进程需要重新启动。
DEFORM 商业软件和许可证须另行安装；KEY 模板、样本及结果数据也需自行迁移，不包含在依赖安装中。

Linux 或 macOS（不支持实际调用 Windows DEFORM）：

```bash
bash scripts/setup_env.sh             # 完整运行和开发依赖
bash scripts/setup_env.sh --with-gui  # 另加桌面界面
```

## 按场景手动安装

本地开发和质量检查：

```bash
python -m pip install -r requirements/dev.txt
python -m pip install -e . --no-deps
```

生产后端：

```bash
python -m pip install -r requirements/runtime.txt -r requirements/server.txt
python -m pip install . --no-deps
```

只在已有后端环境上增加桌面界面：

```bash
python -m pip install -r requirements/gui.txt
```

PyTorch 应优先按安装脚本中的方式从 CPU wheel 源安装，再安装其他清单，避免无 GPU 环境
下载 CUDA 运行时。生产 Docker 的实际安装顺序见
[`deploy/docker/Dockerfile`](../deploy/docker/Dockerfile)。

NumPy 锁定为2.4.1，避开因向后兼容问题撤回的2.4.0；修复说明见
[NumPy 2.4.1 发布说明](https://numpy.org/devdocs/release/2.4.1-notes.html)。

## 与 pyproject.toml 的关系

- `pyproject.toml` 的 `project.dependencies` 声明当前包的直接运行依赖，便于构建和安装
- `pyproject.toml` 的 `project.optional-dependencies` 提供 `dev` 和 `gui` 可选依赖入口
- 本目录的 TXT 文件锁定直接依赖和传递依赖版本，用于 Docker、CI 和环境精确复现
- 安装脚本显式处理 CPU 版 PyTorch，因此不要仅依赖普通 `pip install .` 构建运行环境

## 维护规则

- 新增或删除直接依赖时同步检查 `pyproject.toml` 与对应 TXT 清单
- 更新锁定版本后运行 `python -m pip check` 和 `pytest -m "not slow"`
- 更新开发检查工具后运行 `python tools/run_quality_checks.py`
- 更新 Flask、Gunicorn 或部署依赖时同步检查后端启动和 Docker 部署文档
- 更新 PySide6 时保持 `PySide6`、`PySide6_Addons`、`PySide6_Essentials` 和 `shiboken6` 版本一致
- 不在依赖清单中加入本机绝对路径、虚拟环境路径或私有凭据

根目录 [`AGENTS.md`](../AGENTS.md) 的开发约定适用于本目录，无需单独维护另一份
`requirements/AGENTS.md`。
