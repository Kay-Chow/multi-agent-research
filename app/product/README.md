# 产品调研与研发交付 Agent（阶段 1）

当前只实现“需求分类 → 联网研究 → 产品方案 → 人工审核 → 保存/交接”。`code_change` 在方案批准后只标记 `ready_for_repo`，**不会读取或修改任何仓库**；仓库分析和代码交付属于后续阶段。

## 运行

在 `C:\Users\Kay\PycharmProjects\multi-agent-research` 下执行：

```powershell
$python = 'C:\Users\Kay\anaconda3\envs\multi-agent-research\python.exe'

# 真实调用 DeepSeek 和 Tavily；省略 --mode 时由模型判断模式。
& $python -m app.product.main start '只调研反馈列表的优先级筛选，并写产品方案，不要改代码'

# 固定演示需求：明确现有技术背景，避免把其他框架的行为当成项目事实。
& $python -m app.product.main start `
    '现有演示项目使用 Python FastAPI，GET /feedback 当前返回全部反馈，每条反馈已有 priority 字段。产品经理希望列表支持按优先级筛选：priority 只允许 high、medium、low；不传时保持原有列表行为，非法值返回明确错误。请先调研并形成产品方案，后续需要改代码。' `
    --mode code_change

# 把 TASK_ID 换成 start 输出的实际 ID；批准 a、退回 r、终止 q。
& $python -m app.product.main resume 'TASK_ID'
```

`--mode proposal_only` 或 `--mode code_change` 可显式指定模式；不指定时由模型判断，Python 只接受这两种值。

审核前任务保存在 `data/product_checkpoints.sqlite`；批准后分别生成 `outputs/product/<task_id>/research.md` 和 `proposal.md`。这两个目录已在项目 `.gitignore` 中排除。重启终端后仍能通过同一 `task_id` 恢复审核。

## 测试

```powershell
& $python -m unittest discover -s tests -v
```

测试使用假模型和假搜索，不消耗真实 API 配额。搜索结果只提供**摘要**，来源编号校验不等于事实核验。人工审核时应检查来源是否相关、验收标准是否符合需求；错误方案应退回，不应直接批准。
