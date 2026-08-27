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
