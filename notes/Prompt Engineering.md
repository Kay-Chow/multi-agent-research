## 原则

1. 提供合适的上下文
2. 警惕 AI 的谄媚效应，prompt 要尽可能的中立
3. 复杂任务 planing 化
4. AI 模型的知识来源分为预训练和 Web Search，都可能会有脏东西，所以在需要准确的知识时，引导 AI 去可靠信息源找
5. 还有一个来源，Deep Research，其实也是 Web Search，就是搜索的网页更多，整合的信息更多，会进行更多思考
6. 在让 AI review 你的东西时，很可能触发它的谄媚效应，所以，可以给它制定一份明确的评分标准，让它对照着来
7. 当然，除了让原 AI review 它做出来的东西，还可以使用其他模型来交叉 review



## 技巧

### Zero-shot prompt

- 直接给指令

### Few-shot promt

- 提供几个输入输出示例

### Chain-of-thought(CoT)

- Let's think step by step

### ReAct(Reasoning + Acting)

- 思考-行动-观察的循环
- Agent 的基础范式