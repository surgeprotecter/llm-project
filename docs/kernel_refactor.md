# 内核重构与精简真实验收

> 本文保留上一轮重构与验收历史。2026-11-17 已进一步实施动态计划、逐项持久化与语义压缩；
> 当前方案及验收见 [运行机制优化](agent_runtime_optimization_plan.md)。

## 运行结构

真实模式：prepare_turn → model_turn → execute_actions → model_turn。
模型可以直接回答；工具返回结果或结构化错误后，继续由模型决定下一步。
普通追问以本轮答复结束，下一条自然输入继承会话，不要求进入报告图。

会话与检查点使用 LangGraph；外部能力继续由统一 Registry 和 MCP/CLI/API 适配器提供。
原生工具调用负责动作与参数，决策说明不再作为用户回复。
最终交付边界仅做有限检查，避免每个工具之后固定追加评估调用。

## 可扩展能力

- inspect_tools：从名称/描述目录按需加载真实工具 Schema，随后直接调用。
- load_skill：从随包技能和项目 .agents/skills 发现并按需读取 SKILL.md。
- read_state：按当前任务读取相关报告部分或真实证据，支持筛选和分页。
- update_trip、remember：更新明确事实和用户要求记住的偏好。
- report_step：prepare、route、research、transport、schedule、hotels、render 各自返回结果和缺口。

hotels支持hotel_choices指定真实搜索得到的酒店URL，并重新核实报价后作为预算基准，
避免应用默认排序覆盖模型已经做出的选择。版本检查用于区分“建议换酒店”和“文件已更新”。

报告业务不在主图路由中；用户探索候选时不会自动调用报告能力。
渲染仍复用标准模板、预算与质量校验。研究、报价和其他证据保存在本地，按需读取。
多国家城市分别保留国家身份；系统天数分配错误有一次针对性模型修复，不甩给用户。

## 上下文

当前请求、简要旅行事实、能力目录和所选工具 Schema 保留；不直接注入旧报告全部状态。
原始数据与图片不反复发送。当前工具消息保持调用/返回成对，过长历史按引用保留且可再次读取。
预算采用面向混合中文内容的保守估算；实际输入输出 usage 由模型提供并记录。
上下文记录区分实际进入模型的工具、技能和内容，不将“候选已选”冒充“已经发送”。
稳定能力目录置于动态状态之前，减少无谓的前缀变化；缓存费用影响以真实usage明细衡量。
历史版本曾在累计50000 Token后停止追加研究，复杂安静度筛选因此有未通过记录。
当前真实模式已删除该终止条件及配置；工作窗口只触发语义压缩，不停止未完成任务。

## 本轮精简验收

依用户最新要求，不运行50例全量清单。先验证以下高价值场景，出现新问题再针对性重测：

1. 问候与模糊需求：应正常交流并追问，不做无关查询。
2. 印尼组合推荐和继续扩展：真实检索后提供多个方案，不自动生成报告。
3. 高铁单项比价：真实铁路查询，目标日期、站点和价格口径正确。
4. 正确/冲突日期：矛盾时自然澄清，修正后不重复拒绝。
5. 国内短途完整报告：自主组合报告能力，生成真实产物并检查内容质量。
6. 同一报告局部换酒店与随后解释：只更新相应内容，解释不再生成报告。

运行脚本为 scripts/live_kernel_smoke.py；各组独立启动，结果在 output/kernel-smoke。
revision组需使用report组的同一 --root。调用真实模型与工具，不装载Mock夹具。
记录模型输入输出、真实工具及参数、报告前后版本、质量结果和Token。
早期失败、修复后结果及未覆盖范围分别保留。不能以脚本退出成功代替人工内容审阅。

## 验收命令

~~~bash
uv run python scripts/live_kernel_smoke.py basic
uv run python scripts/live_kernel_smoke.py compare_once
uv run python scripts/live_kernel_smoke.py rail
uv run python scripts/live_kernel_smoke.py dates
uv run python scripts/live_kernel_smoke.py report
uv run python scripts/live_kernel_smoke.py local_edit --root <report组结果目录>
~~~

这些命令会产生真实服务请求，按需分组运行；本轮不需要反复执行已经验证且未受后续改动影响的场景。

本轮实际结果见 [精简验收记录](../output/kernel-smoke/delivery/acceptance.md)。
复杂安静度筛选和长组合有失败/中断记录，未计作通过。报告续跑与真实快照分支均保留来源，
不能把它们称为一次冷启动就全部成功。
