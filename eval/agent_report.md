# Agent / 工具调用评测报告

> backend: **mock** | tasks: 8

## 指标

- task_success_rate: **1.0**
- tool_call_success_rate: **1.0**
- step_efficiency (avg steps): **2.0**

## 逐题结果

| task | success | expected | final | steps |
|---|---|---|---|---|
| t01 | True | 20 | 20 | 2 |
| t02 | True | 36 | 36 | 2 |
| t03 | True | 83 | 83 | 2 |
| t04 | True | 84 | 84 | 2 |
| t05 | True | 40 | 40 | 2 |
| t06 | True | 200 | 200 | 2 |
| t07 | True | 1024 | 1024 | 2 |
| t08 | True | 28 | 28 | 2 |

## 说明

- **task_success_rate**: 最终答案是否正确
- **tool_call_success_rate**: 工具调用格式正确且返回有效
- **step_efficiency**: 完成任务的平均步数 (越少越省 token)
