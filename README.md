# 产品调研与研发交付 Agent

这是一个基于 LangGraph 的本地 Agent 项目。目标是把产品经理的想法逐步转成**有来源的调研报告、可验收的产品方案，以及后续的小范围代码交付**。项目直接使用原仓库的 `app/` 目录；不再提供旧的技术研究报告和 Python 示例功能。

> **当前实现范围：阶段 3＋只读探索。** 已实现按需定位/阅读项目、查询本地 Chroma RAG 与 Tavily、产品方案、两次人工审核、SQLite 断点恢复、CLI 与 FastAPI。`code_change` 在方案批准后只读检查本次选定的仓库，并在独立隔离工作区生成候选文件与 diff；**尚不运行生成代码、自动测试、提交或推送**。

## 当前工作流

```text
START → Supervisor → Intake（接待 Worker：回答/澄清/请求只读工具）→ Supervisor
                    ├─ 请求查找：Discovery Tool（仓库文件、RAG、网页）→ Supervisor → Intake
                    ├─ 需要用户决定：等待补充（LangGraph interrupt）→ Supervisor → Intake
                    └─ 需求明确：Intake 确定模式，Supervisor 继续派发正式 Worker
                       ↔ Researcher（可补查一次）↔ Supervisor
                          └─ 关键依据仍缺失：等待用户补充 → Supervisor → Researcher
                       ↔ Product Planner（产品方案与验收标准）↔ Supervisor
                       → Human Review（必须由人决定）→ Supervisor
         ├─ 退回：带意见重新生成方案（最多 2 次）
         ├─ 终止：结束，不保存
         └─ 批准：Supervisor 派工保存调研报告和产品方案
                    ├─ proposal_only：完成
                    └─ code_change：Supervisor 继续派工：只读仓库检查、文件级计划、
                       隔离工作区与候选代码 → 展示 diff → 第二次人工审核
                         ├─ 退回：新工作区重做（最多 2 次）
                         ├─ 终止：结束，不提交
                         └─ 批准：阶段 3 完成；等待固定测试阶段
```

Supervisor 是唯一的图内派工入口：从 START 就由它派发 Intake；每个 Worker 完成时把**下一步**明确写入 `State.next_step`，再回到 Supervisor。任务模式在 Intake 确认需求时确定；显式创建的任务也经过 Intake，但不会额外调用澄清模型。Supervisor 不再解读用户需求，也不靠“报告文件是否存在、工作区是否存在”等字段猜进度；它只在 `assess_research` 和 `assess_plan` 两个决策点判断是否补查，其余步骤直接派发。程序仍强制执行工具范围、仓库路径校验、来源编号、人工审核、隔离区与正式任务派工次数上限；模型不能凭一句理由跳过这些门禁。`plan` 只是展示用里程碑，实际路线以 `next_step` 和每次 `supervisor -> ...` 事件为准。接待阶段可以只回答一个问题，不会把每句话强制变成产品任务。报告把已读仓库文件、本地知识库片段、网页摘要和用户补充分开标注：仓库文件只证明读到的片段，Tavily 结果是**搜索摘要**而非已逐篇核实的网页原文。知识库或公网检索失败不会仅因渠道故障而终止任务：有其他依据就继续并标注局限；关键事实仍缺失时先用 `interrupt` 请用户补充。用户可以提供规则，也可以要求先带待确认项出方案。

> 本次状态机重构新增了 checkpoint 字段 `next_step`。旧版本尚未完成的 SQLite 会话不支持直接恢复；请新建会话测试，不要拿旧任务 ID 继续。

## 环境

- Python 3.12；已验证解释器：`C:\Users\Kay\anaconda3\envs\multi-agent-research\python.exe`。
- 安装 `requirements.txt` 中的依赖。
- 项目根目录 `.env` 提供 `DEEPSEEK_API_KEY` 和 `TAVILY_API_KEY`。可选 `KB_RETRIEVAL_URL=http://127.0.0.1:8002/retrieve`，仅允许本机地址。不要把 Key 提交到 Git。
- 可选 `AGENT_PROJECTS_ROOT=C:\Users\Kay\PycharmProjects`：允许 Agent 列出/选择该目录的**直接子仓库**。默认是本项目的父目录；用户在对话里给出的路径也必须处于这个范围，不能借模型输入读取整块磁盘。
- 本地知识库由独立的 `C:\Users\Kay\PycharmProjects\rag-qa-agent` 提供**只读检索接口**；该服务使用自己的 Python 解释器和 Chroma 库。

```powershell
Set-Location -LiteralPath 'C:\Users\Kay\PycharmProjects\multi-agent-research'
$python = 'C:\Users\Kay\anaconda3\envs\multi-agent-research\python.exe'
& $python -m pip install -r 'C:\Users\Kay\PycharmProjects\multi-agent-research\requirements.txt'
```

需要使用本地知识库时，在**另一个 PowerShell 窗口**启动服务；第一次或更新文档后先同步索引：

```powershell
Set-Location -LiteralPath 'C:\Users\Kay\PycharmProjects\rag-qa-agent'
$ragPython = 'C:\Users\Kay\anaconda3\envs\rag-qa-agent\python.exe'
& $ragPython knowledge_base.py
& $ragPython -m uvicorn api:app --host 127.0.0.1 --port 8002
```

`/retrieve` 不需要 DeepSeek Key，只返回最多 3 个知识库片段和文档位置。示例资料在 `rag-qa-agent/data/docs/demo/产品反馈列表背景.md`，**仅用于本地模拟，不是公司知识库**。当前知识库其他文档主要是 Java 笔记；程序会用重排分阈值过滤明显不相关命中，但阈值不是正确性保证。更新索引后需重启知识库服务，使 BM25 内存索引同步。

## 终端对话（适合在 PyCharm 单步调试）

PyCharm 的 Python 运行配置选择 **Module name**：`app.main`，工作目录设为本项目根目录，**Parameters 留空**。也可以在 PowerShell 运行：

```powershell
& $python -m app.main
```

启动后直接说话即可；问候会自然引导，问“演示项目在哪”可以先调用受限仓库发现工具回答，不必马上创建正式需求。具体需求也可先查本地文件、知识库或公开资料；工具选择由模型决定，但实际读取范围由 Python 校验。若只说“我想做一个需求”，Supervisor 会让 Intake 追问具体问题，不会进入正式调研。下一句用同一 thread_id 恢复；观察结果、来源和选定仓库保存在 SQLite checkpoint。只有需求明确后才写入正式 `request` 并生成方案。HTTP 创建任务入口提供完整需求，仍会跳过接待；代码任务需在请求体提供 `repo_root`。

需要继续旧会话时，输入“继续上次对话”或 `/recent` 可列出最近待继续会话的摘要与 ID，再用 `/resume 会话ID` 回到原 checkpoint。恢复旧会话先由命令行选择旧 `thread_id`，**不会新建一个 Intake 任务，也不会自动猜测要恢复哪一项**。调研缺资料时直接回复 Agent 的问题即可；补充内容作为用户来源记录，再由 Researcher 整理。方案与代码 diff 仍分别需要人工明确审核，例如输入“批准”“退回 补充验收标准”或“终止”；“好的”不会被当作批准。`/status` 查看当前状态，`/new` 开始另一项需求，`/retry` 在修复外部服务后按相同需求新建任务，`/exit` 退出对话。已创建会话的 checkpoint 会保留。本地知识库服务不是启动工作流的前提；不可用时会标出局限，关键事实缺失时请用户补充。

入口只支持交互式聊天；新需求直接输入，待继续的会话用 `/recent`、`/resume 会话ID` 恢复。当前工作流的任务状态保存在 `data/agent_sessions.sqlite`，不读取旧版 checkpoint。产品方案批准后写入 `outputs/<task_id>/research.md`、`proposal.md`，代码类任务还会写入 `change_plan.md`；阶段 3 在 `data/workspaces/<task_id>/attempt-N/` 保存独立候选文件，并在 `outputs/<task_id>/diff-attempt-N.patch` 保存待审核 diff。相同任务 ID 可在重启终端后继续审核。

## HTTP 接口

```powershell
& $python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

在 `http://127.0.0.1:8000/docs` 查看接口：`POST /tasks` 创建任务，`GET /tasks/{task_id}` 查询状态，`POST /tasks/{task_id}/clarify` 补充调研资料，`POST /tasks/{task_id}/review` 审核产品方案，`POST /tasks/{task_id}/diff-review` 审核代码 diff。示例创建请求：

```json
{
  "request": "只调研反馈列表的优先级筛选并写产品方案",
  "mode": "proposal_only"
}
```

`POST /tasks` 同步执行到调研补充或审核暂停点。响应中 `waiting_for_research_input=true` 时，向 `/tasks/{task_id}/clarify` 提交 `{"message":"补充说明"}`；之后继续到方案审核。方案获批后同步生成候选 diff 并停在第二个审核点。该服务没有鉴权、任务队列或多实例能力，只能在可信本机演示，不要直接暴露公网。
代码类 HTTP 请求还需传入 `"repo_root": "C:\\Users\\Kay\\PycharmProjects\\product-feedback-demo"`，否则返回 422；终端聊天则可直接把仓库路径告诉 Agent，或让它在授权项目目录里查找。

## 测试与目录

```powershell
& $python -m unittest discover -s tests -v
```

自动化测试使用假模型、假搜索和假知识库响应，不消耗外部 API 配额。关键边界包括 Supervisor 从 START 管理 Intake 与工具、正式 Worker 回环与补查上限、越级派工拒绝、仓库/知识库/网页/用户四类来源、授权仓库范围、资料缺口时的用户补充与恢复、两次人工审核、SQLite 重启恢复、HTTP 流程、越权路径、空/超大 diff、工作区或 diff 被调包及退回次数上限。

读代码建议按这条主线走：`main.py` 启动 → `chat_cli.py`/`api.py` 接收输入 → `state.py` 建状态 → `graph.py` 决定工具回环及审核暂停 → `agents.py` 让模型作结构化决定 → `discovery_tools.py` 运行受限只读工具 → `repo_tools.py`/`workspace_tools.py` 检查并生成隔离候选。`config.py` 只放数据库路径和 `thread_id` 配置。

```text
app/
  config.py   数据库路径与 LangGraph thread_id 配置
  state.py    任务状态
  prompts.py  角色提示词
  agents.py   探索接待 / Supervisor / Researcher / Product Planner / Change Planner / Coder
  graph.py    LangGraph 工具回环、两次审核、仓库分析与隔离改动
  discovery_tools.py  只读仓库文件、知识库与网页工具调度
  repo_tools.py  授权项目目录内的仓库发现、只读检查与路径校验
  workspace_tools.py  隔离工作区、完整文件校验、diff 与哈希检查
  kb_tools.py    本机 RAG 检索接口客户端
  tools.py    Tavily 搜索封装
  main.py     启动入口（打开 SQLite 并构建图）
  chat_cli.py 终端聊天、状态展示与会话恢复
  api.py      FastAPI 入口
tests/        自动化测试
data/         SQLite checkpoint（Git 忽略）
outputs/      报告、方案、变更计划及待审核 diff（Git 忽略）
```

## 仓库与工作区边界

仓库路径不再写死为 `product-feedback-demo`，但目前只允许选择授权项目目录的直接子 Git 仓库。接待阶段可读取其中已跟踪的少量常见文本文件，拒绝隐藏路径、密钥名称、链接、超大文件和越界路径。**代码生成阶段仍是演示工程布局**：要求仓库干净，读取 `README.md`、`requirements.txt`、`app/main.py`、`tests/test_feedback.py`，候选改动仅限文件计划及既有变更白名单；其他项目可能只支持探索，不能直接进入代码生成。程序检查路径、语法、diff 大小和审核前后的哈希。模型只提供候选文件内容，**不执行模型建议的命令，也不写原仓库基线**。

## 下一阶段

阶段 4 才会在 diff 人工批准后运行预先固定的测试命令。当前没有执行生成代码、自动测试、Git 提交或 CI/CD，不能称为完整研发交付闭环；本轮也不会替用户 commit/push。

**数据边界**：项目发现结果、选定仓库的允许文本片段、需求和检索片段会发送给配置的 DeepSeek 模型；公开检索词会发送给 Tavily。请把 `AGENT_PROJECTS_ROOT` 限定在仅含演示或获准外发项目的目录；不要接入旧公司的真实内部资料、凭据或用户数据。路径与文件过滤只是演示期防护，不是完整的企业级数据防泄漏体系。
