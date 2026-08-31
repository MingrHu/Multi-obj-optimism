# AGENTS.md · 面向 AI 代理的开发约定

本文件约定在本仓库中协作开发时的硬性规则与惯例。修改代码前请先阅读
[ARCHITECTURE.md](docs/ARCHITECTURE.md) 了解分层结构。

## 硬约束

- **业务/算法逻辑保持 byte-for-byte 不变**。搬迁或重构既有函数时，**函数体一字不改**。
  允许的改动仅限于：
  - 目录/模块重组；
  - import 修复；
  - 硬编码路径改为集中式路径解析（改默认参数值/路径来源，**不得在函数体内 `os.chdir`**）；
  - 补充函数注释（不改逻辑）；
  - 新增单元测试；
  - 打包与工具配置。
- 涉及算法行为变化的改动，必须先与用户确认。

## 目录与命名约定

- 采用 **src-layout 单包**：所有代码位于 `src/mobo/...`，可 `pip install -e .`。
- 文件名统一 **snake_case**（如 `random_forest.py`、`problem.py`）。
- **函数名 / 类名保持历史命名不变**（如 `dnn_run`、`Doe_surrogateModel`、`NSGA2_run`、
  `DeformConfig`、`ForgingEnv`），即使不符合 PEP 8，也不擅自改名，以免影响引用习惯与 diff。
- 每个子包必须有 `__init__.py`。

## import 规范

- 包内模块之间使用**相对导入**（`from .x import` / `from ..y import`）。
- 跨子包使用**绝对导入** `from mobo.<subpkg> import ...`。
- **禁止** `sys.path.append(...)` 与 `os.chdir(...)`。

## 路径规范

- 一切路径来自 `mobo.common.paths`（包括 `PROJECT_DIR`、`DATA_DIR`、`MODELS_DIR`、
  `TEST_DIR`、`KEY_FILE_DIR`、`TASKS_DIR`、`DOE_TASKS_DIR`、`AUTO_SINGLE_DIR`、
  `AUTO_MULTI_DIR`、`model_family_dir()` 与 `task_dir()`）。
- 禁止新增 `../../data/...` 相对路径或平台绝对路径（macOS `/Users/...`、Windows `C:\...`）。
- 跨平台路径拼接用 `os.path.join` 或 `pathlib.Path`，不要硬编码 `\\` 或 `/`。
- 需要外置数据时用环境变量 `MOBO_PROJECT_DIR` / `MOBO_DATA_DIR`，不改代码。

## 日志规范

- 使用 `from mobo.common.logging import logger`。
- **不要**在库代码/模块导入期劫持 `sys.stdout`；仅 CLI 入口可调用
  `logger.install_stdout_redirect()`。

## HTTP API 规范

- 外部系统只依赖 `mobo.api` 和 [DOE_HTTP_API.md](docs/api/DOE_HTTP_API.md)，不要直接耦合 CLI、
  历史算法入口或 `data/tasks` 内部状态。
- 路由层 `api.handler` 只解析 HTTP 参数和组装统一响应；实际处理放在 `api.service`。
- 只读查询使用 GET，创建、训练、推理、优化、停止和删除操作使用 POST。
- 样本、训练数据集和优化结果文件均为无表头 TSV；字段顺序必须写入 DOE 状态。
- 端上按 `resource_id` 和字段取数统一使用 `GET /api/v1/hust/doe/data/get`，不要要求端上读取服务端路径。
- 新增或修改路由时必须同步 `docs/api/DOE_HTTP_API.md`、HTTP 单元测试和 `mobo.api.demo`。

## 新增「原子提取能力」的方式

原子能力层位于 `mobo.extraction`。新增一个「按工件类型提取目标」的能力时：

1. 把提取函数放到合适模块（DEFORM 文本类 → `deform_targets.py`；几何/文件类 → 独立模块）。
   **保持函数体独立、可单测，不与注册逻辑耦合**。
2. 在 `extraction/__init__.py` 用 `registry.register_fn(workpiece_type, target_name, fn, kind=...)`
   注册，`kind` 取 `"key_lines"` 或 `"key_file"`。
3. 上游通过 `registry.resolve(workpiece_type, target_name)` 获取 `ExtractorSpec`，
   按 `spec.kind` 调用。缺工件专属时自动回退 `generic`。
4. **不要**为了接入 registry 而修改既有提取函数体。

## 测试约定

- 测试位于 `tests/`：`unit/`（快速、无外部依赖）与 `integration/`（依赖 `data/` 产物）。
- 运行：`pytest -m "not slow"`（默认）；完整：`pytest`。
- marker：
  - `slow`：keras/DNN 训练等耗时用例，默认跳过；
  - `deform`：依赖 Windows DEFORM 的用例，非 Windows 跳过；
  - `integration`：依赖仓库 `data/` 真实产物，产物缺失时用 fixture `skip`。
- **禁止污染仓库 `data/`**：写文件的用例一律 `monkeypatch` 路径常量或写入 `tmp_path`。
- DEFORM 子进程用 `monkeypatch` 打桩 `subprocess.Popen`；`ForgingTask(dry_run=True)`
  只推进状态、不真正调用 DEFORM，用于状态机测试。
- 覆盖率：`--cov=mobo`，`fail_under=60`；核心纯逻辑模块（`surrogate/common`、`ga/problem`、
  `extraction/registry`、ring 纯函数）应保持高覆盖。

## 文档维护

- 文档职责以 [README.md](README.md)“文档知识库维护”一节为准；同一协议只保留一个
  维护入口，其他历史文件只做索引，不复制正文。
- 修改 API 路由、CLI 入口、环境变量、包/模块结构或 pytest marker 时，必须同步对应文档。
- 完成文档更新后运行 `python tools/check_docs.py`。公共表面确实发生变化时，在人工确认
  文档已经更新后运行 `python tools/check_docs.py --update-snapshot`，并提交快照。
- 禁止在当前协议文档中加入平台绝对路径或 `../../data`；示例使用环境变量、仓库相对路径
  或明确的 `/absolute/path/...` 占位符。

## 常用命令

```bash
bash scripts/setup_env.sh         # 安装环境（CPU torch + 本包 + dev）
pytest -m "not slow"              # 跑测试
pytest --cov=mobo                 # 覆盖率
python tools/check_docs.py        # 文档知识库与公共表面一致性
python tools/run_quality_checks.py # 本地完整质量检查与汇总报告
python tools/quality_score.py --run-tests  # 本地工程质量评分
mobo-api / mobo-ga / mobo-rl / mobo-surrogate / mobo-ring-roundness   # CLI 入口
```

## 提交规范

- 仅在用户明确要求时提交/推送；在 `main` 分支操作。
- 遵循仓库现有 commit message 风格（`MingrHu: ...`）。
- 不提交 `.venv/`、`logs/`、覆盖率产物（见 `.gitignore`）。
