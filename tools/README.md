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

总入口依次运行测试与覆盖率、Ruff、Xenon、文档一致性和工程质量评分；后端 `src/mobo`
与桌面端 `work_platform/mobo_ui` 均纳入覆盖率、静态检查、复杂度和死代码统计。检查结果输出
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

## 早停变化率检查

`check_early_stop.py` 使用与优化模块相同的公式计算一组数值的相对变化率，并判断变化率
是否小于等于5%。第一行输入数值个数 `n`，随后输入 `n` 个数值；数值可以逐行输入，
也可以使用空格分隔：

```bash
python tools/check_early_stop.py
```

示例输入：

```text
3
100 99 96
```

示例输出：

```text
变化率：4.000000%
满足停止条件
```


## 初末态 KEY 配对复制

从仓库根目录运行以下命令，依次输入三个目录：初态 `input_keys` 目录、终态
`result_keys` 目录、保存配对文件的输出目录。初态目录直接存放 KEY 文件，无子目录；
终态目录直接包含数字命名的工况子目录。输出目录不能与任一源目录相同或互相包含。

```powershell
.\.venv\Scripts\python.exe .\tools\extract_key_pairs.py
```

脚本从每个工况目录的唯一 `*_terminal.KEY` 文件推导初态文件名，不再固定工件名称，
也不再读取 `*_step_1.KEY`。例如终态 `result_keys/100/GH4169100_terminal.KEY`
对应初态 `input_keys/GH4169100.KEY`，复制后为：

```text
输出目录/
└── 100/
    ├── GH4169100.KEY
    └── GH4169100_terminal.KEY
```

已有文件保留原样，只补齐缺失的初态或终态；两份文件都已存在的工况会跳过。
扫描到有效工况时，会先删除其输出子目录中的旧 `*_step_1.KEY` 文件，再补入正确的初态 KEY。
即使初态和终态已经齐全，也会清理旧文件；源目录中的 `*_step_1.KEY` 保持不变。
清理失败会提示并跳过该工况，删除操作不递归进入其他子目录。
待补文件的源文件缺失、终态文件不唯一或编号不匹配时会提示并跳过该工况。
文件先复制至临时目录再发布，防止复制中断留下同名残缺文件。
汇总中的处理数为本次补入文件的工况数，跳过数包含已齐全和异常工况；沿用原脚本约定，
存在跳过时退出码为 1，可根据逐工况提示区分原因。
