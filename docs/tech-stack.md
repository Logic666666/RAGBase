# 技术栈分析

## 项目概述

基于 FastAPI + ChromaDB + Ollama 的知识管理与研究系统,融合 RAG 单轮问答与 ReAct Agent 多步研究两种模式。所有基础设施组件(LLM 客户端、Embedding 客户端、向量存储、文本分块、重试、错误分类)与 Agent 核心(ReAct 循环、工具系统、上下文压缩、收敛检测、可观测性、会话管理)均为手搓实现,不依赖 LangChain 等第三方 agent/LLM 框架。

Agent 架构参考 Claude Code 的公开设计理念:**"简单的循环 + 强大的工具系统,把复杂性放在确定性的基础设施里"**。Claude Code 的 Agent loop 由三个阶段构成——收集上下文(context)、采取行动(action)、验证结果(verification)([官方文档](https://code.claude.com/docs/zh-CN/how-claude-code-works));本项目在同一框架下针对 2B 级本地小模型做了防御性强化。

---

## 分层架构

```
┌─────────────────────────────────────────────────────────┐
│              API 层（app/main.py）                         │
│   FastAPI 路由 · Pydantic Schema · 依赖注入               │
│   · Agent 组装工厂（build_agent）· SSE 事件流             │
├─────────────────────────────────────────────────────────┤
│           业务服务层（app/services/）                       │
│   知识库管理（kb.py）· RAG 问答（rag.py）                  │
│   · search_docs() 共享检索入口                            │
├───────────────────┬─────────────────────────────────────┤
│  Agent 核心        │  基础设施层                            │
│  （agent/）        │  （infrastructure/）                   │
│                    │                                       │
│  · ReAct 循环      │  · llm_client.py（手搓，重试）         │
│  · 规划轮          │  · embeddings.py（手搓）              │
│  · 上下文压缩      │  · vector_store.py（ChromaDB）        │
│  · 收敛检测        │  · text_splitter.py（手搓）           │
│                    │                                       │
│  工具系统(tools/)  │  可靠性（reliability/）                │
│  · BaseTool/ToolSpec│  · errors.py 错误分类                │
│  · ToolRegistry    │  · retry.py 指数退避                  │
│                    │                                       │
│  会话(sessions/)   │  事件（events/）                       │
│  · SessionStore    │  · EventBus 发布-订阅                 │
│  · SessionManager  │                                       │
│                    │  可观测（observability/）              │
│                    │  · tracer.py（Trace）                 │
└───────────────────┴─────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| LLM 框架 | 自实现,不依赖 LangChain | 完全控制调用流程,理解底层协议 |
| 工具调用格式 | 文本 JSON + 容错解析(而非 native tool calling) | 本地小模型的 native tool calling 不可靠;文本 JSON 方案模型无关 |
| Agent 循环 | 单线程 ReAct while-loop | 对齐 Claude Code 的 master loop:简单、可调试、不依赖复杂编排 |
| 任务执行 | 异步会话(提交即返回 + 轮询/SSE) | 研究报告是分钟级长任务,不能同步等待 |
| 上下文管理 | 三层:单轮裁剪 → 历史折叠 → 窗口兜底 | 对齐 Claude Code 的 autoCompact 多层防御 |
| 模型输出防御 | 10 层防御链 | 2B 模型输出不稳定,防御链保证可完成、不编造 |

---

## 基础设施层(app/infrastructure/)

### 1. LLM 客户端(llm_client.py)

直接通过 HTTP 调用 Ollama `/api/chat`,无框架封装。关键能力:

- **重试与退避**:transient 错误(网络闪断/超时/5xx)指数退避重试,permanent 错误直接抛出(重试只会重复失败)——对齐 Claude Code 的 retry with backoff
- **思考内容双通道提取**:`message.thinking` 结构化字段或 `<think>` 标签,统一提取到 `ChatResponse.thinking`
- **截断诊断**:`done_reason`("stop"/"length"/"model_length")记录在响应中——输出被截断不再靠猜,这是评估的基础数据(对齐 Claude Code 的 stop_reason)

### 2. Embedding 客户端(embeddings.py)

直接调用 Ollama `/api/embeddings` 将文本向量化。Embedding 在客户端算好再传入向量库,不依赖向量库自动调用——过程透明、可缓存、可替换提供方。

### 3. 向量存储(vector_store.py)

ChromaDB 原生 SDK,`add_documents` / `similarity_search` / `delete_collection` 三个核心操作。共享 client 池 + 显式 close(Windows 上句柄未释放会导致删除失败)。余弦距离度量,持久化到 `data/vectorstore/<kb>/`。

### 4. 文本分块(text_splitter.py)

递归字符分割器,纯函数实现。分隔符优先级:段落 → 行 → 中文句号 → 英文句号 → 空格 → 字符级。中英文混合优化,支持 chunk_size / chunk_overlap。

---

## Agent 核心(app/agent/)

### 规划轮(Explore → Plan)

任务开始前先执行一次规划(对齐 Claude Code 的 [plan mode](https://code.claude.com/docs/zh-TW/commands) 与 Explore 探索):

1. `list_files` 探索项目结构(工具不可用时降级)
2. LLM 基于真实结构生成研究计划(维度 + 关键词)
3. 探索结果与计划作为 **system 锚点**进入主循环——模型始终能看到"结构已探索、计划是什么",不会重复探索、不会空规划

### ReAct 主循环(orchestrator.py)

```
规划轮锚点 + 用户任务
→ 上下文检查(should_compact → 折叠早期轮次)
→ LLM 推理
→ 解析输出:工具调用 or 最终回答
    ├─ 工具调用 → 收敛检测 → 执行工具 → 结果裁剪写回 → 回到循环
    └─ 最终回答 → 返回 AgentResult
```

安全机制:`max_steps` 防死循环、`max_parse_failures` 连续错误终止、工具执行错误以结构化 `ToolResult` 返回给 LLM 自行决策。

### 防御链(模型输出容错,10 层)

模型输出从"脏文本"到"可执行工具调用"的完整防御,分层拦截不同失败模式:

| 层 | 机制 | 拦截的失败模式 |
| --- | --- | --- |
| 1 | 裸引号/截断修复 | JSON 语法损坏 |
| 2 | 嵌套/缺键名提取 | 独白 + JSON 混合输出 |
| 3 | 响应级复读检测 | 模型复读自身输出(echo mode) |
| 4 | 工具级重复调用检测 | 相同 (工具, 参数) 反复调用(list_files 特化引导直接阅读) |
| 5 | 空工具名容错 | tool=null/空(区分"思考后结束"与"完全退化") |
| 6 | 参数友好校验 | jsonschema 校验缺参/错参 |
| 7 | search_kb 自我拒绝 | 文档检索连续命中代码文件时提示换工具 |
| 8 | 收敛终止 | 连续无进展/错误达到阈值,give_up |
| 9 | 提前收尾质疑 | 0 阅读收尾硬拦截(防编造);部分阅读软质疑(须说明未读原因) |
| 10 | 收尾确认轮 | 模型以 JSON 声明收尾时要求纯文本回答(防"声明当报告") |

**设计参考**:Claude Code 依赖 Claude 模型的原生结构化输出能力;本项目使用 2B 级本地模型,原生能力不可靠,容错层承担了模型原生能力的工作——这是模型约束下的正确取舍。

### 上下文管理(context.py,三层)

```
单轮裁剪(TrimCompressor,8000 字符兜底,截断提示含总量)
→ 历史折叠(SummaryCompressor:字符总量触发 → 摘要继承 → 保留最近 N 轮)
→ num_ctx=16384 窗口兜底
```

**SummaryCompressor(对齐 Claude Code 的 autoCompact)**:

- **字符总量触发**(非轮数):轮次部分字符量 > 阈值(≈ 窗口 60%,与 `LLM_NUM_CTX` 联动)。Claude Code 与 LangChain 均按 token 量估算触发([上下文窗口源码分析](https://github.com/wuwangzhang1216/claude-code-source-all-in-one/blob/main/claude-code-deep-analysis/07-context-window.md)、[ConversationSummaryBufferMemory](https://langchain-doc.readthedocs.io/en/latest/modules/memory/types/summary_buffer.html))
- **增量节流**:压缩后至少新增 N 轮才再次触发,防"压缩后残余仍超阈值 → 立即再压缩"的 thrashing(对齐 Claude Code 的 circuit breaker 思想)
- **保留最近 N 轮原文**:防全量摘要丢失近期状态(Claude Code 已知失败模式)
- **摘要继承**:新摘要 = LLM(旧摘要 + 新轮次),多次压缩不丢早期结论(对齐 LangChain `predict_new_summary`)
- **已读文件清单(代码级维护)**:压缩折叠了"读过哪些文件"的事实,清单由代码提取写入摘要消息,不依赖 LLM 复述;只认成功调用、保序去重。与收敛检测(seen_calls 全程记忆)共享同一套事实,避免"模型不知道已读 → 重读 → 被拦 → 连续错误"
- **失败降级**:摘要生成失败返回原消息,压缩不阻塞主任务

**TrimCompressor(对齐 Claude Code 的 tool output 裁剪)**:来源路径保留(可重新检索),正文截断;截断提示必须揭示总量——模型需要知道"结果有 N 字符,只看到前 M",否则误以为读到完整内容(completeness misjudgment)。

### 工具调用解析(schemas.py)

两段式容错:整段 JSON 解析 → 失败则提取 `{...}` 子串再解析 → 都失败视为最终回答。

---

## 工具系统(app/tools/)

### 工具抽象(base.py)

```python
ToolSpec:      # 工具的"身份证"
    name / description / parameters(完整 JSON Schema)
ToolResult:    # 结构化执行结果(对齐 Claude Code 的 tool_result + is_error)
    ok: bool / content: str
BaseTool:
    spec / async run() / execute()  # 模板方法:校验 + 执行 + 异常兜底
```

**Schema 双重职责**(对齐 Claude Code 的 inputSchema 设计):JSON Schema 同时用于约束 LLM 输出参数与运行时 jsonschema 校验(LLM 输出不可信)。

### 内置工具(builtin/)

| 工具 | 能力 | 场景 |
| --- | --- | --- |
| `search_kb` | 向量检索文档片段 | 语义匹配文档/论文 |
| `grep_code` | 正则搜索代码 | 代码定位(对齐 Claude Code 的 Grep) |
| `read_file` | 读取文本文件(带行号,拒绝二进制/PDF) | 代码阅读(对齐 Read) |
| `list_files` | 列出文件结构 | 结构探索(对齐 Glob) |
| `read_pdf` | PDF 文本提取 | 论文阅读 |
| `note_take` / `read_note` / `list_notes` | 工作区笔记 | 中间产物外置(对齐 Claude Code 的 filesystem as memory) |

**场景驱动分化**:文档用向量检索,代码用精确搜索(grep/read),论文用 read_pdf——工具 description 是模型选择依据,不点名引导。

---

## 可靠性(app/reliability/)

- **错误分类**:transient(网络/超时/5xx)与 permanent 区分,只有 transient 值得重试
- **指数退避重试**:重试延迟指数增长 + 抖动,单轮上限控制

---

## 事件总线(app/events/)

发布-订阅模式(对齐 agent-event-bus:Agent 步骤 → EventBus → SSE 广播):

- Agent 每步发布事件 → EventBus 分发给该会话的订阅队列 → SSE endpoint 推送前端
- **晚订阅补偿**:保存每会话最近事件历史(有界),订阅时重放——SSE 连接晚于事件发布(如规划轮的工具调用在提交后毫秒级完成)也能看到完整过程,前端展示对工具调用一视同仁
- 慢消费者保护:队列满时丢弃新事件,不阻塞 Agent 主流程
- 可替换原则:内存实现单机够用,多进程部署可替换为 Redis 实现,接口不变

---

## 会话管理(app/sessions/)

异步长任务的执行与追溯:

- `SessionManager`:提交任务创建后台 asyncio.Task,持有句柄支持状态查询
- `JsonSessionStore`:结果 + trace + transcript 落盘 JSON 文件
- `SessionRecord`:session_id / task / status / result / trace / messages / created_at
- 历史会话自动清理(超出上限删除最旧)

**双通道设计**:EventBus(实时推送,体验)与 SessionStore(磁盘权威,追溯)——同源事件,不重复埋点。

---

## 可观测性(app/observability/)

### Trace(tracer.py)

记录 Agent 执行的每一步,是调试、前端展示与离线评估的统一数据源:

```json
{
  "run_id": "a1b2c3d4",
  "total_duration_ms": 441613.2,
  "events": [
    {"event": "plan", "detail": "{\"plan\": [...]}", "duration_ms": 16105.0, "step": 1},
    {"event": "tool_call", "detail": "read_pdf {'path': '...'}", "duration_ms": 0.0, "step": 2},
    {"event": "compact", "detail": "折叠早期轮次（保留最近 1 轮）", "duration_ms": 11196.5, "step": 4},
    {"event": "llm", "detail": "{\"thought\": ...}", "duration_ms": 32948.9, "step": 1},
    {"event": "final_answer", "detail": "## 摘要\n...", "duration_ms": 0.7, "step": 12}
  ]
}
```

事件类型:start / tool_call / tool_result / plan / think / llm / compact / final_question / final_answer / max_steps / give_up。

**耗时统计**(每步评估的基础数据):`llm` 事件耗时为 CPU 慢模型的大头;`tool_call → tool_result` 为工具执行耗时;`done_reason` 记录截断原因。对齐 Claude Code 的 trace/transcript 支撑"可复盘"与"改进闭环"。

### 前端展示(static/index.html)

- SSE 流式增量渲染:thought / 工具调用 / 工具结果逐步出现
- 完成后原地升级:用带完整耗时与统计的正式版替换流式预览,不重复渲染
- 工具调用行、可折叠的工具结果、可展开的 Thought 块、统计条(总耗时/步骤/工具调用次数)

---

## 测试体系

169 项自动化测试,分层策略:

| 层级 | 测试对象 | 数量 |
| --- | --- | --- |
| 单元测试 | text_splitter / llm_client / embeddings / schemas / tool_registry | 50+ |
| 单元测试 | agent_orchestrator(循环控制、收敛检测、质疑分级、确认轮) | 25+ |
| 单元测试 | context(TrimCompressor / SummaryCompressor:触发、节流、继承、清单、降级) | 20+ |
| 单元测试 | events(发布订阅、会话隔离、晚订阅重放)、sessions、tracer、workspace | 30+ |
| 集成测试 | vector_store(ChromaDB 操作)、rag(管线编排) | 10+ |
| 冒烟测试 | 项目状态、模块导入 | 5+ |

运行方式:

```bash
pytest tests/ -q       # 全部 169 项
pytest tests/unit/ -q  # 单元测试(毫秒级,无外部依赖)
```

---

## 技术栈总览

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| 框架 | FastAPI | Web 服务 + 依赖注入 |
| 运行时 | Uvicorn | ASGI 服务器 |
| 语言 | Python 3.11+ | 主力开发语言 |
| LLM 服务 | Ollama(qwen3.5:2b + bge-m3) | 本地对话 + 嵌入 |
| 向量数据库 | ChromaDB | 向量索引与相似度搜索 |
| HTTP 客户端 | httpx | 异步 HTTP 请求 |
| 数据校验 | jsonschema | 工具参数运行时校验 |
| PDF 解析 | pypdf | 论文文本提取 |
| Git 操作 | GitPython | 代码仓库克隆 |
| 配置管理 | Pydantic Settings | 环境变量 + 类型校验 |
| 测试 | pytest + pytest-asyncio | 自动化测试框架 |
| 容器化 | Docker + Docker Compose | 生产部署 |

---

## 参考与对照

- Claude Code 如何工作(Agent loop:收集上下文 → 行动 → 验证):<https://code.claude.com/docs/zh-CN/how-claude-code-works>
- Claude Code hooks(compaction 拦截、checkpoints 等):<https://code.claude.com/docs/en/hooks>
- Claude Code 上下文窗口源码分析(autoCompact / snipCompact / contextCollapse 多层压缩):<https://github.com/wuwangzhang1216/claude-code-source-all-in-one/blob/main/claude-code-deep-analysis/07-context-window.md>
- Claude Code 多策略压缩详解(五层防御):<https://github.com/0xtresser/Claude-Code-VS-OpenCode/blob/main/EN/Chapter_11_Claude_Code_Commercial/11.5_Multi_Strategy_Compaction.md>
- LangChain ConversationSummaryBufferMemory(摘要 + 缓冲双轨):<https://langchain-doc.readthedocs.io/en/latest/modules/memory/types/summary_buffer.html>
