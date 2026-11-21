# 旅行规划 Agent 项目规划

> 2026-11-17 架构更新：真实模式采用统一原生工具循环，当前任务决定行为；
> 报告研究、交通、排程、住宿和渲染为可组合能力，不再按已有报告或请求分类强制进入流水线。
> 本轮经用户同意采用精简真实验收，执行说明见 [内核重构与精简验收](kernel_refactor.md)。
> 下文旧 MVP 分支说明保留为历史背景，冲突处以上述新架构为准。

## 1. 文档状态

- 状态：MVP 核心及多 MCP 接入已实现；小红书、12306、Booking、Flights 已完成本机真实调用验证
- 最后更新：2026-11-16
- 用户交互语言：默认中文，同时允许中英文输入
- 代码标识符：英文
- 初始运行方式：本地 CLI
- 初始业务范围：中国境内旅游优先，保留国际旅行扩展接口

本文档定义项目的 MVP 范围、总体架构、Agent 工作流、上下文构建策略、记忆策略、工具协议、实施阶段、参考代码库及统一命名规范。后续如有重大范围变更，应先更新本文档，再修改实现。

### 当前实现状态

- 已完成：中文 CLI、多轮澄清、AI 结构化需求抽取、LangGraph checkpoint、SQLite 旅行状态/事件、分层记忆、Context Builder、动态工具 Spec、JSON Schema 参数校验、有界规划循环、真实/Mock 模式隔离、多 MCP 客户端、来源型 Markdown/HTML 报告与 trace。
- MCP 配置：Firecrawl、小红书、高德地图、12306、fli 和本地 Demo 用 Strider Booking MCP。真实可用状态由 `travel-agent check-tools` 探测；未安装依赖或缺凭证时不做 Mock 兜底。
- 已完成的外部验证：DeepSeek + `fli-mcp` 完成真实航班搜索和报告生成；小红书完成真实搜索、详情读取、报告正文综合和四日候选行程生成；12306 完成真实余票查询；Booking 完成真实酒店搜索。Node.js 24.22.3 与 `npx` 已安装到用户目录。
- 小红书 MCP 由 LaunchAgent 自动启动并绑定 `127.0.0.1`；登录助手与服务共享 `~/.config/xiaohongshu-mcp/cookies.json`，该文件仅当前用户可读。
- 当前外部阻塞：高德地图尚缺 Web 服务 Key；Firecrawl 匿名搜索额度耗尽。尚未验证高德实时路线/距离以及跨所有供应商的完整行程整合。

## 2. 项目目标

构建一个可在本地独立运行的旅行规划 Agent，主要服务以下用户场景：

1. 用户没有明确目的地：Agent 通过对话了解出发地、时间、天数、人数、预算、气候偏好、拥挤度偏好和其他个性化要求，推荐目的地并生成初步方案。
2. 用户已有明确目的地：Agent 完成景点搜索、风评总结、景点推荐、酒店和机票候选查询、路线规划、预算预估以及行程整合。
3. 用户提出不属于预设场景的旅行需求：Agent 使用有边界的自由 Planner 拆解任务、选择工具并产出结果。

最终旅行计划支持导出为 Markdown 和 HTML。

## 3. MVP 范围

### 3.1 第一版包含

- 中文 CLI 多轮对话。
- 用户、旅行和会话管理。
- 结构化需求收集。
- 缺失信息和冲突信息澄清。
- 无目的地推荐工作流。
- 已知目的地规划工作流。
- 有步数限制的自由 Planner。
- Python、HTTP API、CLI 和 Skill 的统一工具适配层。
- 可替换的真实工具和 Mock 工具。
- 用户、旅行、对话、外部证据和程序性知识的分层记忆。
- 动态构建模型输入的 Context Builder。
- 长对话上下文压缩。
- 外部信息的来源、新鲜度和可信度记录。
- Markdown 和 HTML 旅行计划生成。
- Agent 决策、工具调用和上下文选择轨迹。

### 3.2 第一版暂不包含

- 真实酒店或机票下单。
- 支付操作。
- 生产级长期价格监控。
- Web UI。
- 同时接入所有第三方 CLI 或 Skill。
- 完整课程评测体系。

在不增加 MVP 复杂度的前提下，应为上述能力保留扩展接口。

## 4. 默认技术决策

- Python 3.12。
- 使用 `uv` 管理环境和依赖。
- LangGraph 作为唯一工作流和 Agent loop 运行时。
- 在需要时使用 LangChain Core 和模型集成，不再引入第二套主控框架。
- DeepSeek API 作为初始模型服务。
- 默认模型为 `deepseek-v4-pro`，开发调试可切换为 `deepseek-flash`。
- 使用 Pydantic 定义状态、记忆、证据、工具参数和旅行计划。
- 使用 SQLite 保存本地应用数据。
- 使用 LangGraph SQLite checkpointer 保存线程级执行状态。
- 使用 Typer 和 Rich 实现 CLI。
- 使用 Jinja2 渲染 HTML。
- 使用 HTTPX 调用外部 HTTP 服务。
- 使用 Pytest 编写测试。
- LangMem 作为可选模块预留接口，不作为第一版运行的强制依赖。

## 5. 总体架构

```text
CLI
 `-- Session Service
      |-- User Profile Store
      |-- Trip State Store
      `-- Append-only Event Log
             |
             v
      LangGraph Orchestrator
      |-- Requirement Extraction
      |-- Missing-field Check
      |-- Scenario Router
      |-- Deterministic Workflows
      `-- Bounded General Planner
             |
             v
      Context Builder
      |-- Memory Retrieval
      |-- Evidence Selection
      |-- Dynamic Tool/Skill Selection
      |-- Token Budgeting
      `-- Context Compaction
             |
             v
      DeepSeek Model + Tool Registry
             |
             v
      Validation + Travel Plan Builder
             |
             v
      Markdown / HTML Artifacts
```

## 6. Agent 编排策略

采用“固定骨架 + 动态规划”的混合模式。

### 6.1 固定阶段

- 加载用户和旅行状态。
- 使用预设提示词让模型抽取结构化需求和意图。
- 检查必要字段。
- 澄清缺失或互相冲突的信息。
- 召回相关记忆和已有证据。
- 构建本轮上下文和能力目录。
- 限制规划循环、参数边界和恢复路径。
- 核验工具证据和最终输出。
- 渲染旅行报告。
- 无法继续时安全停止并说明原因。

### 6.2 允许模型动态决策的部分

- 生成搜索关键词。
- 在当前允许的工具中选择工具。
- 判断现有证据是否足够。
- 工具失败后调整计划。
- 对目的地、景点、酒店和交通候选进行排序。
- 解释推荐理由和取舍。
- 对非标准请求进行有边界的任务规划。

### 6.3 Planner 边界

- 最大规划步数。
- 单个工具和单个任务的最大重试次数。
- 按阶段配置的工具白名单。
- Token 和成本预算。
- 总执行时间限制。
- 外部写操作必须人工确认。
- 无法完成目标时必须解释并安全停止。

## 7. 状态策略

聊天记录不是业务状态的唯一事实来源。Agent 在对话过程中持续维护结构化 `TripState`。

计划字段包括：

```text
user_id
trip_id
thread_id
origin
destinations
travel_dates
duration_days
traveler_count
budget
preferences
hard_constraints
confirmed_decisions
rejected_options
open_questions
current_phase
selected_places
selected_hotel
transport_options
budget_breakdown
evidence_ids
report_version
```

所有状态更新都经过 Pydantic 校验。用户纠正信息时，新值替代旧值，但完整变化过程保留在事件日志中。

## 8. 上下文构建策略

记忆存储和模型上下文是两个独立问题。数据库可以保留完整历史，但每次模型调用只接收根据当前任务编译出的上下文视图。

### 8.1 Context Builder 输入

```python
ContextRequest(
    user_id=...,
    trip_id=...,
    thread_id=...,
    phase=...,
    current_query=...,
    token_budget=...,
)
```

### 8.2 始终进入上下文

- 系统规则和安全规则。
- 当前用户请求。
- 当前结构化旅行状态。
- 尚未解决的问题和当前承诺。
- 有长度限制的最近完整对话。
- 当前工作阶段的目标。

### 8.3 按需进入上下文

- 用户档案中与当前任务有关的字段。
- 与当前请求相关的历史旅行摘要。
- 当前阶段需要且尚未过期的外部证据。
- 对当前修改仍有影响的历史决定及理由。
- 当前阶段允许使用的工具定义。
- 当前任务需要加载的 Skill 指令。

### 8.4 默认不进入上下文

- 完整聊天历史。
- 完整原始网页。
- 所有历史工具输出。
- 全部工具和 Skill 定义。
- 过期的价格、天气、班次或库存。
- 无关的历史旅行。
- 密钥和凭证。

### 8.5 上下文选择流程

1. 加入强制上下文。
2. 根据用户、旅行、阶段、类型、新鲜度和权限过滤候选项。
3. 根据相关性、阶段匹配度、新鲜度、重要性和可信度排序。
4. 惩罚过期、冲突、重复和低可信内容。
5. 对证据做去重和多样化，避免加入大量相似搜索结果。
6. 压缩或截断过长的工具观察结果。
7. 在 Token 预算内装入上下文。
8. 为模型推理、工具调用和最终输出保留 Token。
9. 生成 `ContextManifest`，记录本次上下文构建结果。

初始评分策略可配置为：

```text
score =
    0.40 * relevance
  + 0.20 * phase_match
  + 0.15 * recency
  + 0.15 * importance
  + 0.10 * confidence
  - stale_penalty
  - contradiction_penalty
```

这些权重只是初始默认值，不应写死为不可修改的业务规则。

### 8.6 Context Manifest

每次模型调用记录：

- 选中的记忆和证据 ID。
- 选择原因。
- 被排除候选项的数量及原因分类。
- 各上下文分区的 Token 数量。
- 向模型暴露的工具定义。
- 是否发生上下文压缩。
- 数据来源、新鲜度和抓取时间。

## 9. 记忆策略

### 9.1 记忆层级

- Thread Memory：当前对话和图执行 checkpoint。
- Trip Memory：某次旅行的需求、选择、否决项、证据和进度。
- User Memory：可跨旅行使用的稳定偏好。
- Episodic Memory：历史旅行和历史交互摘要。
- Procedural Memory：可复用工作流和 Skill 指令。

### 9.2 写入规则

- 用户明确表达的稳定偏好可以进入 User Memory。
- 只针对当前旅行的偏好保留在 Trip Memory。
- Agent 推断出的偏好标记为 `inferred`，置信度低于用户明确陈述。
- 重要推断在成为长期事实前应由用户确认。
- 用户纠正信息时，新记忆通过 `supersedes` 关联并替代旧记忆。
- 酒店、机票、天气等动态信息必须记录 `fetched_at` 和 `valid_until`。
- 密钥、支付信息和不必要的个人信息不能作为普通记忆保存。

### 9.3 检索规则

- 优先使用结构化精确过滤，再考虑语义检索。
- 按 `user_id`、`trip_id`、阶段、实体、证据类型和新鲜度筛选。
- 语义检索只用于非结构化历史摘要和笔记。
- Top-k 和 Token 预算必须可配置。
- 基础旅行状态查询不能依赖向量数据库。

### 9.4 上下文压缩规则

参考 Pi 的 compaction 设计：

- 保存完整原始历史。
- 只压缩发送给模型的上下文视图。
- 最近完整对话保持原文。
- 较早且已经结束的对话压缩为结构化旅行摘要。
- 工具调用和对应结果不能被拆开。
- 摘要必须保留目标、限制、决定、否决项、进度、待确认问题和关键证据 ID。
- 在达到输入预算前提前触发压缩，并为输出保留空间。

旅行摘要结构：

```yaml
trip_goal:
confirmed_requirements:
user_preferences:
decisions:
rejected_options:
open_questions:
tool_findings:
stale_findings:
next_step:
```

## 10. 工具与 Skill 策略

所有工具无论通过 Python、HTTP、CLI 还是 Skill 执行，都遵循统一逻辑协议。

Agent-facing `ToolSpec` 保持最小：

```text
ToolSpec
|-- name
|-- description
|-- input_schema
`-- output_schema (optional)
```

Agent 首先看到所有已注册能力的名称和描述；模型选中能力后，Registry 才动态加载该 Tool 或 MCP 的完整 Schema。Skill 选中后再加载完整 Skill 指令。CLI 只暴露参数 Schema，真实命令保留在 Adapter 配置中。

计划实现以下适配器：

- `PythonToolAdapter`
- `HttpApiToolAdapter`
- `CliToolAdapter`
- `SkillAdapter`
- `MockToolAdapter`

首批逻辑工具：

- 目的地搜索。
- 景点搜索。
- 评论总结。
- 天气查询。
- 酒店候选查询。
- 机票候选查询。
- 地图和路线规划。
- 预算计算。

工具必须动态暴露。例如酒店搜索阶段不应同时向模型暴露全部机票、报告、订阅和管理工具。

## 11. 外部证据策略

网页、评论和工具结果属于不可信数据，不能被当作系统指令。

每条证据计划包含：

```text
evidence_id
source_type
source_name
source_url
query
content
structured_data
fetched_at
valid_until
confidence
related_entities
```

规则：

- 网页内容不能修改系统或工具策略。
- 工具参数执行前必须校验。
- 过长工具结果应标准化、截断或只保存在上下文外。
- 动态信息必须显示获取时间。
- 冲突信息应保留多个来源并明确提示，不可静默合并。
- 缺乏足够证据的结果必须标记不确定性。

## 12. 外部操作安全策略

- 只读操作可在配置边界内自动执行。
- 创建订阅、预订、取消、发消息或支付必须获得明确确认。
- 外部写操作应尽可能使用幂等键。
- 重试逻辑不得盲目重复不可逆操作。
- 每个外部操作记录请求、标准化参数、执行结果和外部引用 ID。
- MVP 不执行真实预订和支付。

## 13. 报告策略

模型先生成通过校验的 `TravelPlan`，再由 Jinja2 将其渲染为 Markdown 和 HTML，不直接让模型自由生成整页 HTML。

报告计划包含：

- 用户需求摘要。
- 目的地推荐和理由。
- 每日行程。
- 景点推荐。
- 酒店和交通候选。
- 路线信息。
- 预算明细。
- 风险和提醒。
- 信息来源和抓取时间。
- 未确认、过期、Mock 或无法获取的信息。

报告版本与 `trip_id` 关联。用户修改需求后生成新版本，不覆盖旧版本。

## 14. 可观测性与恢复

使用追加式事件日志记录：

- 用户和 Agent 消息。
- 状态更新。
- Context Manifest。
- 计划及计划修改。
- 工具调用和结果。
- 中断和用户确认。
- 重试决定。
- 上下文压缩摘要。
- 最终输出和停止原因。

必须支持的恢复场景：

- 工具超时或限流。
- 工具参数无效。
- 搜索无结果或结果冲突。
- 用户中途修改目的地、日期、预算或约束。
- 旧证据失效。
- 上下文超限。
- Planner 达到最大步数。
- 进程重启后恢复会话。

## 15. 计划目录结构

```text
docs/
  project_plan.md
  architecture.md
  context_policy.md
  tool_protocol.md
src/
  travel_agent/
    __init__.py
    cli.py
    config.py
    models/
      trip_state.py
      memory.py
      evidence.py
      tools.py
      travel_plan.py
    graph/
      builder.py
      routing.py
      nodes/
      prompts/
    context/
      builder.py
      ranking.py
      compaction.py
      manifest.py
    memory/
      repository.py
      policies.py
    tools/
      registry.py
      adapters/
      travel/
    reports/
      renderer.py
      templates/
    persistence/
      database.py
      events.py
tests/
  unit/
  integration/
  scenarios/
```

## 16. 命名规范

代码统一使用英文标识符；CLI 提示、用户可见错误信息和旅行报告默认使用中文。

### 16.1 文件和目录

- 目录、Python 模块和 Markdown 文件使用小写 `snake_case`。
- Python 包名为 `travel_agent`。
- 测试文件使用 `test_<subject>.py`。
- 路径中不使用空格、混合大小写和含义不清的缩写。

示例：

```text
docs/project_plan.md
src/travel_agent/context/builder.py
tests/unit/test_context_builder.py
```

### 16.2 Python 标识符

- 类、枚举和 Pydantic 模型：`PascalCase`。
- 函数、方法、局部变量和字段：`snake_case`。
- 常量：`UPPER_SNAKE_CASE`。
- 私有实现：以下划线开头，例如 `_rank_candidates`。
- 布尔值名称以 `is_`、`has_`、`can_`、`should_` 或 `requires_` 开头。

示例：

```python
class ContextBuilder: ...
class TripState: ...

def build_context(...): ...
def requires_confirmation(...): ...

DEFAULT_CONTEXT_TOKEN_BUDGET = 24_000
```

### 16.3 LangGraph 节点

- 节点名称使用小写 `snake_case` 动词或动词短语。
- 节点名称描述业务行为，不描述底层技术。

示例：

```text
extract_requirements
check_missing_fields
clarify_with_user
search_destinations
validate_evidence
build_travel_plan
safe_stop
```

### 16.4 CLI 名称

- 可执行程序和子命令使用小写 `kebab-case`。
- CLI 参数使用 `--kebab-case`。

示例：

```text
travel-agent chat
travel-agent export-report --trip-id <id>
travel-agent inspect-context --thread-id <id>
```

### 16.5 环境变量

- 使用 `UPPER_SNAKE_CASE`。
- 项目专用配置以 `TRAVEL_AGENT_` 开头。
- 第三方凭证保留清晰的服务商名称。

示例：

```text
DEEPSEEK_API_KEY
TRAVEL_AGENT_MODEL
TRAVEL_AGENT_DATABASE_PATH
TRAVEL_AGENT_CONTEXT_TOKEN_BUDGET
```

### 16.6 数据库和事件

- 表名和字段名使用小写 `snake_case`。
- 表名使用复数名词。
- 实体 ID 使用 `<entity>_id`。
- 事件类型使用 `<domain>.<event>` 小写点分格式。

示例：

```text
users
trips
memory_items
tool_executions
context_manifests

user.preference_updated
trip.requirements_changed
tool.execution_failed
context.compacted
report.generated
```

### 16.7 工具和 Skill

- 逻辑工具名使用小写 `snake_case`，名称应包含明确动作和领域对象。
- 适配器类以 `Adapter` 结尾。
- 工具输入和输出模型分别以 `Input` 和 `Result` 结尾。

示例：

```text
search_destinations
search_hotels
lookup_weather
plan_route
calculate_budget

CliToolAdapter
HotelSearchInput
HotelSearchResult
```

### 16.8 术语一致性

- 优先使用完整领域术语，避免无法解释的缩写。
- 同一概念在代码、数据库、Prompt、CLI 和文档中使用同一个名称。
- 存在更具体名称时，避免使用 `data`、`info`、`handler`、`manager` 或 `util`。
- `trip` 表示一次具体旅行计划，`travel` 表示旅行领域。
- `evidence` 表示外部检索得到的支持信息。
- `memory` 表示已经持久化的用户或 Agent 知识。
- `context` 只表示某一次模型调用实际接收的输入集合。

## 17. 实施阶段与验收标准

### 阶段 1：架构与数据契约

产出：

- 架构文档。
- 上下文策略文档。
- 工具协议文档。
- State、Memory、Evidence、Tool 和 Report 的 Pydantic 模型。

验收：

- 数据模型可表达无目的地和已知目的地两种场景。
- 记忆与上下文职责明确分离。
- 外部操作具有风险和确认元数据。

### 阶段 2：项目骨架和模型客户端

产出：

- Python 项目结构。
- 依赖配置。
- 环境变量模板。
- DeepSeek 模型适配器。
- 基础 CLI。

验收：

- 用户可以启动 CLI 并获得流式模型回复。
- 密钥通过环境变量加载且不会被提交。

### 阶段 3：持久化、记忆和上下文

产出：

- SQLite Repository。
- LangGraph checkpoint 集成。
- 事件日志。
- Context Builder 和 Context Manifest。
- 上下文压缩实现。

验收：

- 会话可以停止并恢复。
- 上下文选择结果可以检查。
- 长对话可以压缩且不会删除原始历史。

### 阶段 4：Agent 工作流

产出：

- 需求提取。
- 澄清中断。
- 场景路由。
- 两条核心工作流。
- 有边界的自由 Planner。

验收：

- 使用 Mock 工具可以完成两种主要场景。
- 用户修改需求后，受影响状态和旧证据会失效。

### 阶段 5：旅行工具

产出：

- 工具注册中心和适配器。
- 在现有凭证允许范围内接入真实工具。
- 其他能力提供明确标记的 Mock。
- 来源与新鲜度管理。

验收：

- 替换工具实现不需要修改工作流。
- 工具失败后能够有界恢复或安全停止。

### 阶段 6：报告生成

产出：

- `TravelPlan` 模型。
- Markdown 渲染器。
- HTML 渲染器和模板。
- 报告版本管理。

验收：

- 完成对话后生成可读的 Markdown 和 HTML。
- 来源、时间、Mock 和不确定信息清晰可见。

### 阶段 7：可靠性和运行文档

产出：

- 单元测试、集成测试和场景测试。
- 包含安装和运行命令的 README。
- 普通请求、条件改变请求和失败请求的执行轨迹。

验收：

- 按 README 可以从干净环境运行项目。
- 关键失败场景可以稳定复现并解释。

### 阶段 8：MVP 后续扩展

- 接入更多 CLI 和 Skill。
- 真实价格监控调度器。
- Web UI。
- 国际旅行支持。
- 完整评测集和 baseline。

## 18. 参考代码库

### 18.1 主要参考

- [earendil-works/pi](https://github.com/earendil-works/pi)：参考 Agent loop、session tree、compaction、工具结果截断和 HTML 导出；第一版不直接 Fork。MIT License。
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)：用于图编排、checkpoint、interrupt、流式输出和恢复执行；项目核心依赖。MIT License。
- [LangGraph SQLite Checkpoint](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite)：用于本地持久化图状态。
- [langchain-ai/langmem](https://github.com/langchain-ai/langmem)：参考结构化记忆提取、合并和检索；可选集成。MIT License。
- [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk)：参考事件存储和 Context Condenser；不作为核心依赖。MIT License。
- [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)：参考研究规划、来源收集和报告整合；该仓库已归档，只参考设计。MIT License。

### 18.2 归属与许可证策略

- 记录所有复用库及实质性复制的代码。
- 保留许可证要求的版权和许可声明。
- 优先借鉴架构模式，避免大段复制实现。
- 如果复制代码，记录对应仓库、版本或 commit。
- 在最终 README 和课程报告中加入归属说明。

## 19. 尚待确定的外部集成信息

以下问题不影响总体架构，但必须在对应工具进入开发阶段前确定：

- 当前 DeepSeek 账户可以使用哪些搜索能力。
- 是否可以提供高德 API Key、CLI 或 Skill。
- 哪些酒店和机票服务允许查询。
- 美团、小红书、高德等 CLI 或 Skill 的具体接口格式。
- “订阅”是本地价格监控、消息通知，还是服务商侧订阅。

在这些信息确定前，项目使用稳定接口和明确标记的 Mock 实现。
