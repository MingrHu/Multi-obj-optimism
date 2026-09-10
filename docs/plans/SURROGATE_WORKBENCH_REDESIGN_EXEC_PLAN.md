# 代理模型工作台重构 ExecPlan

本计划用于把代理模型页面从单块提交表单调整为可追溯的工程工作流。实施期间持续更新
`Progress`、`Surprises & Discoveries`、`Decision Log` 和 `Outcomes & Retrospective`。

## Purpose / Big Picture

代理模型训练应遵循“选择任务 → 准备数据 → 定义输入与响应 → 保存配置 → 训练与评价 → 交给优化”
的阶段边界。后台状态查询不能阻塞数据定义；本地草稿、后端已保存配置和已完成训练模型必须有
不同状态，避免用户把“正在同步”误认为“正在计算”。

参考工作流：

- [Ansys optiSLang Sensitivity Analysis](https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/opti_multi_disc/opti_multi_disc_sensitivity.html)
  先定义设计变量和响应，再生成/评价 DOE，随后构建并评价代理模型。
- [Altair HyperStudy Fit Surface Based Optimisation](https://help.altair.com/feko/topics/feko/user_guide/appendix/fit_optimisation_workflow_feko_c.htm)
  把 Setup、DOE、拟合精度检查和 Optimization 分成连续阶段。

## Progress

- [x] 2026-09-10：确认 DOE 状态读取使用 `training_state=loading`，导致保存配置被错误拦截。
- [x] 2026-09-10：拆分 `status_loading`、训练运行状态和配置保存状态。
- [x] 2026-09-10：DOE 改为只读任务选择；新任务仍由独立创建动作产生。
- [x] 2026-09-10：新增显式配置保存、未保存提示和保存结果摘要。
- [x] 2026-09-10：新建 DOE 后通过待选 ID 在列表刷新完成后可靠选中。
- [ ] 第二阶段：切换 DOE 时检测未保存草稿，支持保存、放弃或取消切换。
- [x] 2026-09-10：选中 DOE 后从数据资源恢复已保存表格、来源文件名和字段角色。
- [x] 2026-09-10：DOE 刷新和状态查询增加请求合并，消除连续点击造成的并发 UI 更新。
- [x] 2026-09-10：切换 DOE 时清理上一个任务的数据上下文，防止跨任务误保存。
- [ ] 第三阶段：把数据、字段定义、训练设置拆成清晰的步骤区，并增加提交前检查摘要。
- [ ] 第三阶段：模型评价增加逐输出指标、训练/验证误差和推荐模型确认动作。

## Surprises & Discoveries

- 页面右侧仍显示“未开始”时，内部状态可能已经被临时改为 `loading`，造成视觉状态与按钮行为冲突。
- 保存配置和训练提交共享同一数据载荷，但生命周期不同：保存是持久化草稿，训练是创建运行任务；
  两者不能共用一个“活动中”布尔条件。
- 可编辑 DOE 下拉框与独立“新建 DOE”动作职责重复，并允许产生不对应后端任务的自由文本。

## Decision Log

- Decision：`status_loading` 只表示只读同步，不写入 `training_state`。
  Rationale：运行状态只描述后端训练生命周期，避免查询行为污染业务状态。
- Decision：状态同步期间允许保存配置，但暂时禁止开始训练。
  Rationale：保存接口由后端做最终并发校验；启动训练必须先取得可信运行状态，防止重复提交。
- Decision：DOE 控件只允许选择已有任务；创建操作由独立按钮完成。
  Rationale：任务名称和 ID 均由后端保证唯一，界面不应接受没有后端实体的自由文本。
- Decision：配置保存按钮保持可点击，前置条件不足时给出明确原因，仅请求提交期间临时禁用。
  Rationale：用户应能发现下一步缺少什么，不能面对无解释的灰色按钮。

## Validation and Acceptance

1. 状态查询尚未返回时，保存配置可以提交，开始训练保持禁用。
2. 训练处于 queued/running/stopping 时，保存操作显示明确提示且不提交请求。
3. 新建 DOE 后列表刷新并选中新任务，任务名称不能通过组合框自由输入。
4. 保存成功后显示样本数、输入数和输出数；字段变化后恢复未保存提示。
5. UI 测试、Ruff、完整测试和文档一致性检查通过。

## Outcomes & Retrospective

第一阶段修复当前状态误判，并建立了任务、配置、状态同步和训练运行四类独立状态。第二阶段将
集中解决跨 DOE 草稿保护和已保存数据恢复；第三阶段再调整页面布局与模型评价深度，避免一次性大改
影响已经稳定的训练与优化接口。
