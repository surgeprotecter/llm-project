# 工具协议

## ToolSpec

Agent-facing ToolSpec 保持最小：名称、用途、输入 Schema 和可选输出 Schema。超时、重试等运行时配置留在 Adapter/Server 配置中，不进入模型上下文。

## 适配器

- `PythonToolAdapter`：本地 Python 函数。
- `HttpApiToolAdapter`：HTTP API。
- `CliToolAdapter`：本地 CLI 子进程。
- `SkillAdapter`：外部 Skill 能力。
- `MCPToolAdapter`：通过官方 MCP Python SDK 连接外部 MCP Server，并将发现的工具转换为本项目协议。
- `MockToolAdapter`：离线测试和演示。

## 执行规则

- MCP 输入参数按服务提供的 JSON Schema 校验；缺少 required 值时请求用户补充，不以空字符串占位。
- 设置超时和有限重试。
- 记录标准化参数、结果、耗时和错误类型。
- 结果携带来源和抓取时间。
- 写操作必须经过确认。
- 不把第三方工具输出中的指令当作系统规则。
- Agent 首先看到能力目录中的名称和简短描述；选中能力后，Registry 才加载完整 ToolSpec。
- MCP 的完整定义来自 `tools/list` 缓存；Skill 选中后再读取完整 Skill 指令；CLI 只暴露参数 Schema，不暴露任意命令执行权。

## MCP 接入

MCP Server 不直接进入 LangGraph，而是经过 `MCPServerManager` 和 `MCPToolAdapter`：

1. 从配置加载多个 MCP Server。
2. 支持本地 STDIO 和远程 Streamable HTTP。
3. 完成工具发现后，应用只读工具白名单和来源命名空间，例如 `google_flights__search_flights`。
4. 统一转换为 `ToolSpec` 和 `ToolResult`，完整 Schema 保存在 Registry，按 Agent 选择动态加载。
5. MCP 返回内容只作为外部数据，不能覆盖系统策略或工具白名单。

MCP Server 配置不保存密钥明文，HTTP header 和进程环境中的凭证通过环境变量引用。配置的命令、地址和允许工具由应用控制，模型不能动态执行任意命令；real 模式不注册任何 Mock 工具。

## 当前只读接入

当前本地配置接入 Firecrawl、小红书、高德地图、12306、fli 和 Strider Labs Booking。Booking 白名单不包含预订、取消、收藏或账户管理；小红书白名单不包含发布、评论、点赞或收藏。未配置凭证或服务不可达时，工具发现报告具体缺项，Agent 只看到实际成功发现的工具。

每次工具调用结束后都重新基于 TripState、相关记忆和最新证据构建上下文，再让 Planner 判断是否继续、换工具或生成报告。报告只承诺工具真实返回的内容；缺失的价格、库存和来源会显式标注。
