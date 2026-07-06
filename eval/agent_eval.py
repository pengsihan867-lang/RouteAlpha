"""eval/agent_eval.py — 最小 Agent / 工具调用评测 (ReAct)

================================================================================
教一件事: Agent 评测 = 记录 trajectory + 量化工具调用与多步任务成功率
================================================================================

Agent loop (ReAct):
  plan → act (call tool) → observe → ... → final answer

本文件提供:
  - 1 个 calculator 工具 + 6-8 道算术应用题 (答案可判定)
  - trajectory 表: [task_id, step, thought, action, tool, tool_input, observation]
  - 3 个指标: task_success_rate / tool_call_success_rate / step_efficiency

默认 **离线 mock policy** (可复现); 加 `--real` 用真实 LLM (OpenAI 兼容 API)。

跑法:
  python eval/agent_eval.py
  python eval/agent_eval.py --real   # 需 OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# 任务集: 算术应用题, 期望数值答案
# --------------------------------------------------------------------------- #
DEMO_TASKS = [
    {"id": "t01", "question": "小明有 12 个苹果, 又买了 8 个, 一共有几个?", "answer": 20},
    {"id": "t02", "question": "一本书 45 元, 打 8 折后是多少钱?", "answer": 36},
    {"id": "t03", "question": "仓库里有 150 箱货, 运走 67 箱, 还剩多少箱?", "answer": 83},
    {"id": "t04", "question": "3 个班每班 28 人, 全校一共多少人?", "answer": 84},
    {"id": "t05", "question": "100 除以 4 再加上 15 等于多少?", "answer": 40},
    {"id": "t06", "question": "火车时速 80 公里, 行驶 2.5 小时走多远? (公里)", "answer": 200},
    {"id": "t07", "question": "2 的 10 次方是多少?", "answer": 1024},
    {"id": "t08", "question": "半径为 3 的圆面积是多少? 用 pi≈3.14159, 保留整数。", "answer": 28},
]


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def tool_calculator(expr: str) -> tuple[bool, str]:
    """安全计算器: 只允许数字与 + - * / ( ) . 和 **。"""
    expr = expr.strip().replace("^", "**")
    allowed = re.fullmatch(r"[\d\.\+\-\*/\(\)\s]+", expr.replace("**", ""))
    if not allowed:
        return False, f"非法表达式: {expr}"
    try:
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                return False, "不支持函数调用"
            if isinstance(n, ast.Name):
                return False, "不支持变量"
        val = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}})
        return True, str(round(float(val), 6))
    except Exception as e:
        return False, f"计算失败: {e}"


TOOLS = {"calculator": tool_calculator}


# --------------------------------------------------------------------------- #
# Mock agent: 确定性脚本, 从题目里抽算式
# --------------------------------------------------------------------------- #
EXPR_BY_ID = {
    "t01": "12+8",
    "t02": "45*0.8",
    "t03": "150-67",
    "t04": "28*3",
    "t05": "100/4+15",
    "t06": "80*2.5",
    "t07": "2**10",
    "t08": "3.14159*3*3",
}


def _extract_expr_mock(task_id: str, question: str) -> str:
    if task_id in EXPR_BY_ID:
        return EXPR_BY_ID[task_id]
    if "12" in question and "8" in question:
        return "12+8"
    if "45" in question and "8" in question:
        return "45*0.8"
    if "150" in question and "67" in question:
        return "150-67"
    if "28" in question and "3" in question:
        return "28*3"
    if "100" in question and "4" in question:
        return "100/4+15"
    if "80" in question and "2.5" in question:
        return "80*2.5"
    if "10 次方" in question:
        return "2**10"
    if "半径" in question:
        return "3.14159*3*3"
    return "0"


def run_mock_agent(task: dict, max_steps: int = 3) -> dict:
    """离线 mock: 一步调 calculator, 一步给答案。"""
    traj: list[dict] = []
    tid, q, expected = task["id"], task["question"], task["answer"]

    expr = _extract_expr_mock(tid, q)
    traj.append(
        {
            "task_id": tid,
            "step": 0,
            "thought": "需要计算器",
            "action": "tool_call",
            "tool": "calculator",
            "tool_input": expr,
            "observation": "",
        }
    )
    ok, obs = tool_calculator(expr)
    traj[-1]["observation"] = obs
    tool_ok = ok

    try:
        ans = int(round(float(obs)))
    except Exception:
        ans = None

    traj.append(
        {
            "task_id": tid,
            "step": 1,
            "thought": "根据计算结果作答",
            "action": "final_answer",
            "tool": "",
            "tool_input": str(ans),
            "observation": "",
        }
    )
    success = ans == expected
    return {
        "task_id": tid,
        "question": q,
        "expected": expected,
        "final_answer": ans,
        "success": success,
        "n_steps": len(traj),
        "tool_calls": 1,
        "tool_success": 1 if tool_ok else 0,
        "trajectory": traj,
    }


# --------------------------------------------------------------------------- #
# Real agent: 调 LLM, 解析 JSON action
# --------------------------------------------------------------------------- #
def _llm_client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def run_real_agent(task: dict, max_steps: int = 4) -> dict:
    """真实 LLM ReAct (简化): 每步让模型输出 JSON action。"""
    client = _llm_client()
    model = os.environ.get("AGENT_MODEL", os.environ.get("JUDGE_MODEL", "gpt-4o-mini"))
    tid, q, expected = task["id"], task["question"], task["answer"]
    traj: list[dict] = []
    messages = [
        {
            "role": "system",
            "content": (
                "你是解题 agent。每步只输出一行 JSON, 格式之一:\n"
                '{"action":"tool_call","tool":"calculator","tool_input":"12+8","thought":"..."}\n'
                '{"action":"final_answer","answer":20,"thought":"..."}\n'
                "可用工具: calculator(expr)"
            ),
        },
        {"role": "user", "content": q},
    ]
    final_ans = None
    tool_calls = tool_success = 0

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        try:
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0) if m else text)
        except Exception:
            data = {"action": "final_answer", "answer": None, "thought": text[:80]}

        action = data.get("action", "")
        thought = data.get("thought", "")
        if action == "tool_call":
            tool = data.get("tool", "calculator")
            tin = str(data.get("tool_input", ""))
            tool_calls += 1
            fn = TOOLS.get(tool)
            ok, obs = fn(tin) if fn else (False, "unknown tool")
            if ok:
                tool_success += 1
            traj.append(
                {
                    "task_id": tid,
                    "step": step,
                    "thought": thought,
                    "action": "tool_call",
                    "tool": tool,
                    "tool_input": tin,
                    "observation": obs,
                }
            )
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"工具返回: {obs}"})
        else:
            final_ans = data.get("answer")
            try:
                final_ans = int(round(float(final_ans)))
            except Exception:
                final_ans = None
            traj.append(
                {
                    "task_id": tid,
                    "step": step,
                    "thought": thought,
                    "action": "final_answer",
                    "tool": "",
                    "tool_input": str(final_ans),
                    "observation": "",
                }
            )
            break

    success = final_ans == expected
    return {
        "task_id": tid,
        "question": q,
        "expected": expected,
        "final_answer": final_ans,
        "success": success,
        "n_steps": len(traj),
        "tool_calls": tool_calls,
        "tool_success": tool_success,
        "trajectory": traj,
    }


# --------------------------------------------------------------------------- #
# 评测汇总
# --------------------------------------------------------------------------- #
def run_agent_eval(use_real: bool = False, tasks: list[dict] | None = None) -> dict:
    tasks = tasks or DEMO_TASKS
    runner = run_real_agent if use_real else run_mock_agent
    results = [runner(t) for t in tasks]
    n = len(results)
    n_succ = sum(1 for r in results if r["success"])
    total_tools = sum(r["tool_calls"] for r in results)
    ok_tools = sum(r["tool_success"] for r in results)
    metrics = {
        "backend": "real_llm" if use_real else "mock",
        "n_tasks": n,
        "task_success_rate": round(n_succ / n, 3) if n else 0.0,
        "tool_call_success_rate": round(ok_tools / total_tools, 3) if total_tools else 0.0,
        "step_efficiency": round(sum(r["n_steps"] for r in results) / n, 2) if n else 0.0,
    }
    traj_rows = [row for r in results for row in r["trajectory"]]
    return {"results": results, "metrics": metrics, "trajectory": traj_rows}


def write_report(eval_out: dict, out_path: Path) -> None:
    m = eval_out["metrics"]
    lines = [
        "# Agent / 工具调用评测报告\n",
        f"> backend: **{m['backend']}** | tasks: {m['n_tasks']}\n",
        "## 指标\n",
        f"- task_success_rate: **{m['task_success_rate']}**",
        f"- tool_call_success_rate: **{m['tool_call_success_rate']}**",
        f"- step_efficiency (avg steps): **{m['step_efficiency']}**\n",
        "## 逐题结果\n",
        "| task | success | expected | final | steps |",
        "|---|---|---|---|---|",
    ]
    for r in eval_out["results"]:
        lines.append(
            f"| {r['task_id']} | {r['success']} | {r['expected']} | {r['final_answer']} | {r['n_steps']} |"
        )
    lines.append("\n## 说明\n")
    lines.append("- **task_success_rate**: 最终答案是否正确")
    lines.append("- **tool_call_success_rate**: 工具调用格式正确且返回有效")
    lines.append("- **step_efficiency**: 完成任务的平均步数 (越少越省 token)\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 Agent 工具调用评测")
    parser.add_argument("--real", action="store_true", help="使用真实 LLM (需 OPENAI_API_KEY)")
    args = parser.parse_args()

    if args.real and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("请设置 OPENAI_API_KEY (及可选 OPENAI_BASE_URL)")

    out = run_agent_eval(use_real=args.real)
    m = out["metrics"]
    print("指标:", m)
    print("\n逐题:")
    for r in out["results"]:
        flag = "OK" if r["success"] else "FAIL"
        print(f"  [{flag}] {r['task_id']}: expected={r['expected']} got={r['final_answer']} steps={r['n_steps']}")

    report = ROOT / "eval" / "agent_report.md"
    write_report(out, report)
    print(f"\n报告已写入: {report}")


if __name__ == "__main__":
    main()
