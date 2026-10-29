# 多 Agent 研究与报告系统

基于 LangGraph 的本地研究助理：根据任务决定是否需要代码示例，联网收集资料，生成带来源的报告，并在**人工批准后**保存最终文件。项目提供命令行和 HTTP 两种入口，共用同一张状态图与 SQLite checkpoint。

> 这是可运行的本地演示，不是已具备鉴权、任务队列和多实例部署能力的生产服务。

## 工作流程

```text
START → Supervisor → Researcher → Supervisor → [Coder → Supervisor] → Writer
                                                                   ↓
                              END ← Save ← 批准 ← Human Review
                                               ├─ 退回 → Writer 重新撰写
                                               └─ 终止 → END（不保存）
```

- **Supervisor**：调用 DeepSeek 判断是否需要代码，在 `researcher → writer` 和 `researcher → coder → writer` 两条计划中选择；Python 代码校验计划、前置条件和路由，防止模型跳步。
- **Researcher**：生成检索词，调用 Tavily 搜索，整理最多 6 条带编号的来源及研究要点；来源为空时明确标注证据不足。
- **Coder**：仅在需要代码时生成 Python 示例，最多尝试 2 次，用 `ast.parse` 检查语法，**不执行生成的代码**。
- **Writer**：整合研究结论与可选代码，检查正文中的引用编号是否存在；代码和来源列表由程序拼入报告。
- **Human Review**：在 `interrupt()` 处暂停，支持批准、退回并填写意见、终止。最多退回修改 2 次；再次退回会终止任务。
- **Save**：仅在批准后写入 `outputs/<task_id>.md`。重复保存相同内容不会生成第二份文件；若同名文件内容不同则报错。

`task_id` 同时用作 LangGraph 的 `thread_id`。状态存于 `data/checkpoints.sqlite`，因此可以关闭程序后用同一 ID 恢复待审核任务。

## 环境与安装

- Python 3.12（本项目验证环境为 3.12）。
- 可用的 DeepSeek API Key 和 Tavily API Key。真实运行会调用外部服务，可能产生费用或消耗配额。

在项目根目录、使用目标 Python 环境安装依赖：

```powershell
python -m pip install -r requirements.txt
```

在项目根目录创建 `.env`，填写自己的 Key：

```dotenv
DEEPSEEK_API_KEY=填写你的DeepSeek_Key
TAVILY_API_KEY=填写你的Tavily_Key
```

代码从项目根目录的 `.env` 读取配置。不要把真实 Key 写入代码、README 或提交到 Git。模型当前配置为 `deepseek-flash`，关闭 thinking 模式以配合结构化输出。

## 命令行使用

以下命令均在项目根目录执行。

启动一个**不需要代码**的任务：

```powershell
python -m app.main start '比较 BM25 与向量检索，只写简报，不要代码'
```

或启动一个**需要代码示例**的任务：

```powershell
python -m app.main start '调研 LangGraph checkpoint，给出 Python 示例并写报告'
```

命令会先打印 `task_id`，完成研究和写作后在人工审核处暂停并退出。记下该 ID；重新打开终端后仍可恢复：

```powershell
python -m app.main resume 'TASK_ID'
```

将 `TASK_ID` 换成程序打印的实际 ID。按提示输入 `a` 批准、`r` 退回并填写修改意见，或 `q` 终止。退回后 Writer 会生成新草稿并再次暂停；批准后才写入 `outputs/`。

## HTTP 接口

启动本地服务（默认单进程）：

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

浏览 `http://127.0.0.1:8000/docs` 可查看和试用接口。

| 接口 | 作用 |
| --- | --- |
| `POST /tasks` | 创建任务；同步执行到审核暂停点后返回状态和草稿 |
| `GET /tasks/{task_id}` | 读取 checkpoint 中的任务状态、草稿和执行事件 |
| `POST /tasks/{task_id}/review` | 提交 `approve`、`reject` 或 `terminate`；退回时必须提供 `feedback` |

PowerShell 示例（新任务会真实调用 DeepSeek 和 Tavily）：

```powershell
$base = 'http://127.0.0.1:8000'
$body = @{ task = '比较 BM25 与向量检索，只写简报，不要代码' } | ConvertTo-Json -Compress
$created = Invoke-RestMethod -Method Post -Uri "$base/tasks" `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600

$id = $created.task_id
$status = Invoke-RestMethod -Uri "$base/tasks/$id"
$status | Select-Object task_id, review_status, waiting_for_review, revision_count

$review = @{ action = 'approve' } | ConvertTo-Json -Compress
$approved = Invoke-RestMethod -Method Post -Uri "$base/tasks/$id/review" `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($review)) -TimeoutSec 600
$approved | Select-Object task_id, review_status, output_path
```

状态响应中的 `waiting_for_review=true` 表示确实停在审核中断处。未知任务返回 404，当前不在待审核状态却提交审核返回 409；请求字段错误或退回缺少意见返回 422。服务重启后仍可凭 `task_id` 查询或审核同一任务。

## 测试

```powershell
python -m unittest discover -s tests -v
```

当前有 11 项自动化测试，使用假模型与假搜索，不调用外部 API。覆盖两条任务路径、非法路由、步数上限、无搜索结果、虚构来源编号、代码语法错误、无效引用、人工批准/退回/终止、修改次数上限，以及 SQLite 关闭后重新打开的恢复。另已做过本地 HTTP 创建、查询、批准和保存的手工验证。

## 目录与数据

```text
app/
  agents.py       Supervisor / Researcher / Coder / Writer
  graph.py        LangGraph 节点、条件边、审核与保存
  state.py        共享状态与初始值
  prompts.py      各角色提示词
  tools.py        Tavily 搜索封装
  main.py         命令行入口
  api.py          FastAPI 入口
tests/            自动化测试
data/             SQLite checkpoint（运行时生成）
outputs/          批准后的 Markdown 报告（运行时生成）
```

`.env`、`data/`、`outputs/` 已在 `.gitignore` 中排除。SQLite 文件含任务内容和报告草稿，应按敏感数据处理；不要将其或真实报告误传到公开仓库。

## 已知限制

- 检索主要使用 Tavily 返回的搜索摘要，不等于已读取和核实每篇原文。URL 和引用**编号**校验不能证明报告中的每条事实确实得到来源支持；关键结论需要人工核对。
- 生成代码仅通过 Python 语法解析，未执行，也未验证语义正确性。
- Supervisor 目前在两条预设计划中选择，并非允许任意角色和任意顺序的开放式编排。
- HTTP `POST /tasks` 为同步请求，需等待模型和搜索完成；没有后台任务队列、身份认证、跨进程锁或多实例并发支持。只在可信本机以 `127.0.0.1` 访问，不要直接暴露到公网；也避免同时用 CLI 和 HTTP 写同一个 SQLite 文件。
- 未完成系统性的质量、延迟、吞吐或成本评测，不宣称量化收益。网络/API 故障可能使当前请求失败；可通过 checkpoint 查看已保存状态，但并非所有失败都支持自动恢复。
