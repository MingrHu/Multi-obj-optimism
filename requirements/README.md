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

Windows PowerShell：

```powershell
.\scripts\setup_env.ps1
```

Linux 或 macOS：

```bash
bash scripts/setup_env.sh
```

需要桌面界面时增加脚本的 GUI 选项：

```powershell
.\scripts\setup_env.ps1 -WithGui
```

```bash
bash scripts/setup_env.sh --with-gui
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
