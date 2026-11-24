# Travel Planning Agent

一个基于 LangGraph 的本地旅行规划 Agent，提供中文 CLI、多轮需求澄清、长期记忆与证据召回、动态 MCP 工具选择、执行 trace，以及 Markdown/HTML 旅行报告。

## Agent Loop

```text
用户输入
  → AI 需求抽取并合并旅行状态
  → 按当前任务检查必要信息，缺失时澄清
  → 召回相关记忆/证据，构建带简短工具目录的上下文
  → 模型选择工具
  → 只加载所选 MCP 的完整描述和 JSON Schema，生成并校验参数
  → 调用工具并保存来源、时间、结果和 trace
  → 更新证据，上下文重建后再次规划
  → 基于证据整合报告、标注来源并输出 HTML/Markdown
```

工具目录只包含名称和用途；MCP 完整 Schema 在模型选中工具后才加入参数生成上下文。工具必填参数不全或 Schema 校验失败时会先问用户，不用空字符串或猜测值硬调工具。`--real-only` 会完全禁用 Mock，真实模型或 MCP 调用失败时不会静默回退。

## 环境

- Python 3.12+、`uv`
- DeepSeek API Key（真实模型模式）
- Node.js 18+、`npx`（12306 与 Booking MCP）
- `fli` MCP：使用项目固定上游修复版 `uv run --locked --script scripts/fli_mcp_compat.py`；配置见 `mcp_servers.fli-pinned.example.json`，从项目根目录运行。无需修改全局安装。诊断结果见 `docs/flight_service_debugging.md`。
- 小红书 MCP：按 [官方仓库](https://github.com/xpzouying/xiaohongshu-mcp) 安装对应平台的 MCP 与登录器，先扫码登录再启动 MCP；默认地址 `http://127.0.0.1:18060/mcp`
- 高德地图 MCP：在高德开放平台创建应用并申请“Web 服务”Key（不是 Web 端 JS API Key）
- Firecrawl：可匿名连接，但匿名搜索额度可能耗尽；需要时可配置 Firecrawl API Key

安装 Python 依赖并初始化配置：

```bash
uv sync --extra dev
cp .env.example .env
cp mcp_servers.example.json mcp_servers.json
```

在 `.env` 中配置 `DEEPSEEK_API_KEY`。高德地图 MCP 需要另外设置 `AMAP_MAPS_API_KEY`。项目通过 MCP URL 注入 Key，不会把密钥放进工具上下文或 trace。
提交代码时使用仓库里的示例配置；不要提交实际 Key、Cookie、本地数据库或生成的报告和日志。

如果 Firecrawl 的匿名额度耗尽，可在 `.env` 增加 `FIRECRAWL_API_KEY`；配置会自动加入 Authorization header，不设置时仍尝试 keyless 模式。当前这台开发机已碰到匿名搜索额度限制。

小红书第一次使用需先运行登录器扫码。登录器与 MCP 服务必须共享同一个 cookies 文件路径；当前本机固定使用 `~/.config/xiaohongshu-mcp/cookies.json`：

```bash
mkdir -p ~/.config/xiaohongshu-mcp
chmod 700 ~/.config/xiaohongshu-mcp
export COOKIES_PATH="$HOME/.config/xiaohongshu-mcp/cookies.json"
umask 077
~/.local/bin/xiaohongshu-login-darwin-arm64
chmod 600 "$COOKIES_PATH"
```

登录器会打开扫码登录窗口。启动 MCP 服务时也必须传入相同的 `COOKIES_PATH`：

```bash
COOKIES_PATH="$HOME/.config/xiaohongshu-mcp/cookies.json" ~/.local/bin/xiaohongshu-mcp-darwin-arm64 -port 127.0.0.1:18060
```

当前开发机已安装 LaunchAgent，登录后会自动以相同 cookies 路径启动服务。服务应绑定 `127.0.0.1`，不要把无认证 MCP 端口暴露到局域网或公网。cookies 是登录凭证，不要提交到仓库或发送给他人。

创建高德 Key 时，在“添加 Key”表单中选择“Web 服务”；Android 平台的 SHA1 和 PackageName 不适用。将创建后的 Key 写入 `.env` 的 `AMAP_MAPS_API_KEY`。项目通过官方 Streamable HTTP MCP endpoint `https://mcp.amap.com/mcp?key=...` 接入；Key 的配额和使用范围以高德控制台及服务协议为准。

## MCP 配置

仓库提供 `mcp_servers.example.json`，当前工作目录中的 `mcp_servers.json` 是本地运行配置（已加入以下 MCP；该文件被 `.gitignore` 忽略）：

- Exa：官方免 Key MCP 网页搜索；替换已拒绝匿名请求的 Firecrawl 端点。
- 小红书：Agent 白名单只开放笔记搜索与详情读取；会先由模型生成最多 3 个简短互补关键词，再逐条真实搜索。登录管理工具不暴露给规划器。
- 高德地图：地点/周边搜索、地点详情、天气、地理编码和驾车/步行/骑行/公交路线及距离查询；使用环境变量 `AMAP_MAPS_API_KEY` 注入官方 Streamable HTTP URL。
- 12306：查询国内铁路票务。项目只读取搜索结果，不提供购票操作。
- fli：通过项目内锁定版本的启动脚本查询 Google Flights，不依赖全局旧版；不需要单独 API Key；它通过逆向工程接口取数，非 Google 官方 API，可能随时变化。
- Strider Labs Booking MCP：仅本地 Demo，白名单限制为酒店搜索、详情、可订性、价格和评论读取；不暴露预订、取消、收藏等操作。使用 Playwright，需要 Node.js 与 Chromium。

配置路径由 `.env` 中的 `TRAVEL_AGENT_MCP_CONFIG=mcp_servers.json` 指定。检查真实 MCP 是否连通：

```bash
uv run travel-agent check-tools
```

未安装 Node、缺少高德 Key、小红书服务未启动或未登录时，诊断会逐项显示未连接原因（不会打印密钥）。非严格开发模式可以只使用当前已连接的服务。

## 运行

离线开发（默认 Mock）：

```bash
uv run travel-agent chat
```

真实模型 + 当前成功连接的真实工具（不注册 Mock）：

```bash
uv run travel-agent chat --real-only --trace --user-id alice --trip-id hong-kong-trip
```

`--trace` 展示每轮实际模型阶段、工具名称/来源/耗时、决策、证据和上下文选择数量。查看完整模型输入输出及工具参数/结果：

```bash
uv run travel-agent chat --real-only --trace-full --user-id alice --trip-id hong-kong-trip
```

完整 trace 会遮蔽常见密钥、Cookie、授权头和 Token 字段，但仍可能包含用户个人信息与网页内容；分享前请自行检查。工具输入输出同时会按项目现有机制记录在本地 SQLite，勿将数据库或 trace 提交到公开仓库。

查看 MCP 与模型可用性：

```bash
uv run travel-agent check-tools
```

查看上下文选择清单或最近一次报告：

```bash
uv run travel-agent inspect-context --trip-id hong-kong-trip
uv run travel-agent export-report --trip-id hong-kong-trip
```

## Web 工作台

终端之外还提供了一个单文件网页前端 `index.html`（项目根目录），由极简桥接服务 `src/travel_agent/server.py` 把浏览器请求接到同一个 LangGraph 运行时。界面右上角可选择两种模式：

- **演示沙盒（Demo）**：不启动后端，使用内置示例行程数据，用于查看交互与版式；
- **本地 FastAPI（Live）**：`POST /api/chat` 调用真实模型与已连接的 MCP 服务；页面实际走 `POST /api/chat/stream`（SSE），边跑边推送阶段、模型轮数、工具调用与耗时，等待中可随时中止。

启动：

```bash
uv sync --extra dev          # fastapi / uvicorn 已在项目依赖中
uv run uvicorn travel_agent.server:app --reload --port 8000
```

然后打开 <http://localhost:8000>（前端默认请求 `http://localhost:8000/api/chat`，可在设置面板修改）。接口约定：

```text
POST /api/chat            # 一次性返回，兼容脚本/curl
POST /api/chat/stream     # SSE：start → progress(每 1.5s) → result|error
{"message": "...", "session_id": "TRV-1234", "user_id": "default", "trip_id": "default-trip", "real_only": false}
→ {"reply": "...", "status": "completed|needs_clarification", "session_id": "...",
   "plan": null, "traces": [{"tool", "action", "result", "latency"}],
   "report": {"version", "html_path", "markdown_path", "created_at"}}

GET /api/report?path=output/reports/<trip>/report_v1_xxx.html   # 只读下载报告
GET /api/health                                                 # 实时模式 / 模型 / 工具模式
```

`plan` 为看板渲染用的项目视图（预算分项、航班、酒店、逐日活动），由后端把 `task_plan` 与最新报告投影成前端结构后返回；没有结构化行程时为 `null`，前端会保留上一次看板并在气泡里标注「本轮未生成结构化行程」。

`session_id` 即 LangGraph `thread_id`，同一会话的多轮消息会续跑同一个 checkpoint；`real_only=true` 会让该进程切换到严格真实模式（等价于 CLI 的 `--real-only`）。桥接进程用一把全局锁串行处理回合（共用 SQLite 与单一模型 Key），CORS 对所有来源开放，仅适合本地开发，不要直接暴露到公网。

## 真实数据注意事项

- 未接通的服务不会由 Mock 代替；报告会列明检索来源与时间，无法核实报价时明确标注缺口，并用有说明的规划预留构成预算范围。
- 酒店和交通结果是搜索候选，不代表已预订、锁价或确认库存。
- Booking MCP 是本地演示用的非官方浏览器自动化项目，不应用于生产。
- 高德地图结果、调用配额和数据使用应遵守高德开放平台的服务协议；境外目的地的覆盖情况以实际工具结果为准。

## 测试

```bash
uv run pytest
uv run ruff check src/travel_agent tests
```

## 目录

- `src/travel_agent/models/`：旅行状态、记忆、证据、工具和报告模型。
- `src/travel_agent/persistence/`：SQLite 持久化和事件日志。
- `src/travel_agent/context/`：记忆/证据排序、上下文预算与压缩。
- `src/travel_agent/tools/`：统一工具协议、注册中心、标准化和 Mock 实现。
- `src/travel_agent/mcp/`：MCP 配置、发现、调用和适配器。
- `src/travel_agent/graph/`：LangGraph Agent Loop 与需求/计划节点。
- `src/travel_agent/reports/`：Markdown/HTML 报告。
- `src/travel_agent/server.py`：本地 Web 桥接（FastAPI），`index.html` 为配套单页工作台。
- `docs/project_plan.md`：总体规划；`docs/tool_protocol.md`：工具接入协议。

## 参考与归属

项目参考 Pi 的 Agent session/compaction 设计、LangGraph 的状态编排与 checkpoint、LangMem 的记忆分层思路，以及 OpenHands 的上下文压缩思路。实质性复用代码时保留对应许可证和归属信息。

## 标准旅行报告

生成 HTML 旅行规划默认交付标准报告，校验逐日日期、交通、实际住宿夜晚、分项预算和证据来源。明确要求原样导出聊天时才使用聊天导出。聊天导出可单独打开，不能通过标准报告验收。

修改报告功能时，以已验收的完整报告和当前会话的已确认条件为回归基线。优先复用真实检索记录，交付前至少跑两个使用自然输入的真实测试，并检查实际文件的内容与布局。

完整报告通过真实小红书正文、地点核验、交通与酒店查询、餐饮研究生成。每日一般2–3个主要景点；国内用高德，境外使用 Photon / OpenStreetMap 和 OSRM，按国家与实际城市过滤同名地点。机票分别列出去返程，清楚标注单人往返参考价和按人数推算的全体金额。酒店保留真实图片，核验入住总价与每间每晚口径。

每版均输出全体和每人的预算区间，分项标明检索报价、正文参考或规划预留。查询有缺口时使用明确的支出假设，不将估算称为商家报价。

多轮修改以最新完整报告为基础，仅补查受影响部分；例如增加景点继承已有航班与酒店，某一段换酒店继承其他住宿段和每日行程。详情见 [报告流程](docs/report_pipeline.md)。

真实多轮验收（实际调用模型、MCP、API和只读浏览器）：

```bash
uv run python scripts/live_revision_acceptance.py danang4
uv run python scripts/live_revision_acceptance.py jiangnan7
```

结果在 `output/revision-acceptance/`，含每轮报告、结构化快照、完整模型/工具trace及验收JSON。

## 自然对话、多城市与局部修改

用户无需声明工具或要求“完整报告”。需求足够时自动规划，不足时追问；按统一城市身份与真实抵离时间安排跨城、住宿及每日吃喝体验。只换酒店或航班时保留未受影响的内容。详细策略、真实验收命令与用量说明见 [按需求与缺口规划旅行](docs/adaptive_travel_planning.md)。

机票 MCP 启动脚本及依赖锁位于 `scripts/fli_mcp_compat.py` 和相邻 `.lock`，配置模板已同步。`check-tools` 验证连接和工具发现，具体查询是否成功以真实调用 trace 为准。

## 统一 Agent Loop（真实模式）

真实模式现在由模型原生工具调用驱动：模型选择回答、追问、查询或操作产物，
工具结果和错误返回模型后继续决定。已有报告不会自动把本轮请求变成报告更新。
报告研究、交通、排程、住宿和渲染拆为独立能力，仍使用原有标准模板和确定性校验。
工具目录和技能元数据在各种请求中都可发现；完整定义按需加载。查询、执行计划和写文件的权限在实际调用时检查，
不通过隐藏工具来控制。需要从出发地查机票或按票价择期时，缺出发地先追问；只问当地玩法不需要这个信息。

~~~bash
uv run travel-agent chat --real-only --trace --user-id yuming --trip-id kernel-selftest
~~~

可以直接问“最近有空，帮我看看去哪”“这两个地方有什么区别”，或在已有攻略后说“只调整住宿，其他保留”。
完整输入输出、已加载工具Schema和实际工具调用可用 --trace-full 查看。
项目 .agents/skills/<name>/SKILL.md 可增加领域技能，匹配时按需加载；相关参考文件按需读取。
MCP连接在应用运行期间复用，避免有状态服务的搜索缓存因每次重启而丢失。

复杂任务先展示带版本号的计划，确认后才执行。计划修改还在补条件时，回复“继续”不会误批旧版。
执行中用 `update_plan` 更新 `pending / in_progress / completed`
及结果引用。每个工具结果立即写入 SQLite；城市研究、交通段、住宿和每日安排增量保存。
同一个 `user-id` 和 `trip-id` 下说“继续”可恢复已批准且未完成的任务，已完成任务的新请求会另建执行目标，
保留已有报告和可复用证据。CLI 显示计划变化、当前工具和结果复用；Ctrl+C 取消并保存进度。
交付检查按本轮请求选择要点。`--trace` 会显示 `scope` 和 `delivery`，便于查看比较、进度和报告各自的检查结果。

真实循环没有累计 token 或固定执行轮数的终止门槛。`TRAVEL_AGENT_CONTEXT_TOKEN_BUDGET`
（默认 36000）是上下文整理的工作窗口；接近容量时用 `TRAVEL_AGENT_SUMMARY_MODEL`
（默认 `deepseek-flash`）语义总结旧调用组，保留近期消息、准确状态和结果引用，再继续工作。
原始工具结果完整落盘，模型通过 `read_result` 按字段或分页读取。未变且仍有效的相同查询跨轮复用，
确需更新时用带原因的 `refresh_tool`。用量和缓存命中以服务商返回的 usage 为准。

实现范围、状态结构和精简真实验收见 [运行机制优化方案](docs/agent_runtime_optimization_plan.md)。
历史 [内核重构](docs/kernel_refactor.md) 中的累计用量截断已取消。
当前模块与存储结构、恢复和报告策略见 [实际实现说明](docs/agent_runtime_implementation.md)。
用 `--real-only` 运行真实验收。验收报告、失败轮次和完整用量写入本地 `output/`，不随仓库分发。
