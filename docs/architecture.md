# 系统架构说明

## 目标

系统以 LangGraph 为状态编排层，以 SQLite 为本地持久化层，以 Context Builder 为模型输入编译层，以 Tool Registry 为外部能力边界。

## 运行链路

```text
用户输入
  -> CLI 会话
  -> LangGraph checkpoint
  -> AI 需求抽取和意图判断
  -> 缺失信息澄清或进入规划
  -> 记忆/证据召回和上下文构建
  -> Agent 选择能力并动态加载 Tool/MCP/Skill Spec
  -> 工具执行和结果评估循环
  -> 结构化计划、证据整合
  -> Markdown / HTML 输出
  -> 最终回复和 Trace 持久化
```

## 设计原则

- 结构化 `TripState` 是当前旅行事实的主要来源。
- 原始聊天和工具结果写入事件日志，不默认全部发送给模型。
- 外部能力通过 Tool Registry 暴露；Agent 先看到名称和描述，选中后再动态加载完整 Spec。
- 固定 LangGraph 骨架负责状态、循环上限、参数边界和恢复，模型负责需求理解、工具选择和结果评估。
- 所有真实服务都必须可以替换成 Mock，以便离线运行和测试。
- 第三方内容是数据，不是系统指令。

## 主要模块

- `models/`：Pydantic 数据契约。
- `persistence/`：SQLite 数据和事件日志。
- `memory/`：长期记忆写入与检索策略。
- `context/`：上下文选择、预算、压缩和清单。
- `tools/`：工具协议、注册中心和适配器。
- `graph/`：LangGraph 节点和路由。
- `reports/`：结构化旅行计划的 Markdown/HTML 渲染。
- `cli.py`：本地交互入口。

## 行程的唯一表示

一次行程只允许存在一个可执行对象：`travel_agent.itinerary.Itinerary`。

- 引擎在报告工作区更新后（`internal()` 的 report 分支）把工作区适配成 `Itinerary` 存进
  `state["itinerary"]`；执行确认的交付轮直接用它渲染聊天正文（`render_chat`）。
- Web 桥接（`server.py`）的看板投影只调用 `build_itinerary` + `render_board`，不再解析聊天
  Markdown。适配优先级：已存 `itinerary` → 报告工作区 → 历史报告 payload；最后用已批准的提案
  正文补齐执行阶段只核实了少数地点的单日行程。
- 聊天正文、看板、HTML 报告因此来自同一组 `Day/Stop`，不会出现「左边排了 5 段、右边只有 1 个
  景点」的分叉。价格、评分等仍标注为候选或参考价。
