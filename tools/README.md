# 仓库维护工具

## 文档一致性检查

```bash
python tools/check_docs.py
```

检查内容包括：

- 必需文档及 Markdown 本地链接；
- Flask 实际路由是否全部出现在 `docs/api/DOE_HTTP_API.md`；
- `pyproject.toml` CLI 是否记录在 `README.md` 和 `AGENTS.md`；
- 顶层包和 pytest marker 是否有架构/开发约定；
- 源码模块、路由、CLI、环境变量和 marker 是否偏离人工确认的公共表面快照。

公共表面发生预期变化时，先更新对应文档并检查差异，再执行：

```bash
python tools/check_docs.py --update-snapshot
```

不要只更新快照来绕过文档审阅。定期执行配置位于
`.github/workflows/docs-consistency.yml`。

## 工程质量评分

先安装开发依赖，再运行完整本地检查：

```bash
python -m pip install -r requirements/dev.txt
python tools/run_quality_checks.py
```

总入口依次运行测试与覆盖率、Ruff、Xenon、文档一致性和工程质量评分，并输出
`quality-check-summary.json`、`quality-check-summary.md` 以及每项检查日志。增加
`--with-security` 可联网执行 pip-audit，发现漏洞时脚本返回非零状态。

已有 `quality-reports/coverage.json` 时可跳过测试：

```bash
python tools/quality_score.py --min-score 60
```

评分报告写入忽略提交的 `quality-reports/`。评分由可维护性、圈复杂度、覆盖率、Ruff、
Vulture 和文档一致性组成，只用于本仓库的趋势比较。依赖漏洞、源码安全和供应链安全
分别由 pip-audit、Semgrep CE 与 OpenSSF Scorecard 工作流报告，不混入本地总分。

第三方工具通过 `requirements/dev.txt` 安装，不复制到 `tools/`。`tools/` 只保存本仓库
自己的编排、评分和文档一致性逻辑。
