"""eval/failure_analysis.py — MILP 路由失败样本逐条归因 (badcase)

把每条失败 query 归入设计文档 3.6 的五类之一:
  task_hard          oracle 也失败 → 任务本身难, 非路由锅
  prediction_error   预测排序错, 没选到更可能成功的模型
  calibration_error  排序可能对, 但概率刻度误导了分配
  budget_binding     预算紧, 被迫选便宜模型而失败
  label_noise        边界/存疑 (概率接近、多模型结果矛盾)

用法:
  python eval/failure_analysis.py
  或在 test.ipynb 里 import analyze_routing_failures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))

from model import milp as mp  # noqa: E402


def _oracle_choice(mats: mp.RoutingMatrices) -> np.ndarray:
    choice = np.empty(len(mats.queries), dtype=int)
    for i in range(len(mats.queries)):
        succ = np.where(mats.y_true[i] > 0)[0]
        choice[i] = succ[mats.cost[i, succ].argmin()] if len(succ) else int(mats.cost[i].argmin())
    return choice


def _best_pred_success_model(mats: mp.RoutingMatrices, i: int) -> int | None:
    succ = np.where(mats.y_true[i] > 0)[0]
    if len(succ) == 0:
        return None
    return int(succ[mats.p_success[i, succ].argmax()])


def attribute_failure(
    mats: mp.RoutingMatrices,
    choice: np.ndarray,
    i: int,
    oracle: np.ndarray,
    budget_per_query: float,
) -> str:
    """对单条失败 query 打归因标签。"""
    j = int(choice[i])
    if mats.y_true[i, j] >= 1:
        return "ok"

    succ = np.where(mats.y_true[i] > 0)[0]
    if len(succ) == 0:
        return "task_hard"

    oracle_j = int(oracle[i])
    if mats.y_true[i, oracle_j] < 1:
        return "task_hard"

    best_succ = _best_pred_success_model(mats, i)
    cheap_j = int(mats.cost[i].argmin())

    # 概率接近 → 边界样本
    if best_succ is not None and abs(mats.p_success[i, j] - mats.p_success[i, best_succ]) < 0.03:
        return "label_noise"

    # 选了失败模型, 但它的预测概率高于某个能成功的模型 → 校准/过度自信
    for sj in succ:
        if mats.p_success[i, j] > mats.p_success[i, sj] + 0.05 and j != sj:
            return "calibration_error"

    # 选了最便宜且失败, 且存在更贵但能成功的 → 预算绑定
    if j == cheap_j and mats.cost[i, oracle_j] > mats.cost[i, j] + budget_per_query * 0.3:
        return "budget_binding"

    if best_succ is not None and j != best_succ:
        return "prediction_error"

    return "prediction_error"


def analyze_routing_failures(
    pred_df: pd.DataFrame,
    budget_per_query: float = 0.002,
    prob_col: str = "p_success",
    max_rows: int = 15,
    prompt_lookup: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """跑 MILP, 产出 badcase 表 + 归因计数 + 汇总指标。"""
    milp = mp.solve_routing(pred_df, budget_per_query=budget_per_query, prob_col=prob_col)
    mats = mp.to_matrices(pred_df, prob_col=prob_col)
    choice = np.array([mats.models.index(m) for m in milp.assignment["model"]])
    oracle = _oracle_choice(mats)

    rows: list[dict] = []
    for i, q in enumerate(mats.queries):
        j = int(choice[i])
        if mats.y_true[i, j] >= 1:
            continue
        tag = attribute_failure(mats, choice, i, oracle, budget_per_query)
        oracle_j = int(oracle[i])
        best_succ = _best_pred_success_model(mats, i)
        prompt = ""
        if prompt_lookup and q in prompt_lookup:
            prompt = str(prompt_lookup[q])[:80].replace("\n", " ")
        rows.append(
            {
                "sample_id": q,
                "eval_name": pred_df.loc[pred_df["sample_id"] == q, "eval_name"].iloc[0],
                "prompt_preview": prompt or f"[{q}]",
                "chosen_model": mats.models[j],
                "chosen_p": round(float(mats.p_success[i, j]), 3),
                "oracle_model": mats.models[oracle_j],
                "best_pred_model": mats.models[best_succ] if best_succ is not None else "-",
                "failure_tag": tag,
            }
        )

    badcase = pd.DataFrame(rows)
    if len(badcase) == 0:
        summary = pd.DataFrame(columns=["failure_tag", "count"])
        metrics = {"n_failures": 0}
        return badcase, summary, metrics

    summary = (
        badcase["failure_tag"]
        .value_counts()
        .rename_axis("failure_tag")
        .reset_index(name="count")
    )
    metrics = {
        "n_failures": len(badcase),
        "downgrade_failure_rate": round(mp.downgrade_failure_rate(mats, choice), 4),
        "optimality_gap": round(mp.optimality_gap(milp, mp.baselines(pred_df, prob_col=prob_col)["oracle"]), 4),
        "realized_success_rate": round(milp.realized_success_rate, 4),
    }
    return badcase.head(max_rows), summary, metrics


def plot_failure_summary(summary: pd.DataFrame, ax=None):
    """归因计数条形图。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if summary.empty:
        ax.set_title("无失败样本")
        return ax
    order = ["prediction_error", "calibration_error", "budget_binding", "task_hard", "label_noise"]
    colors = {
        "prediction_error": "#e45756",
        "calibration_error": "#f58518",
        "budget_binding": "#4c78a8",
        "task_hard": "#9e9e9e",
        "label_noise": "#b279a2",
    }
    labels = {
        "prediction_error": "预测排序错",
        "calibration_error": "校准/概率误导",
        "budget_binding": "预算太紧",
        "task_hard": "任务本身难",
        "label_noise": "边界/存疑",
    }
    s = summary.set_index("failure_tag").reindex(order).fillna(0)
    x = np.arange(len(order))
    ax.bar(x, s["count"], color=[colors.get(k, "#ccc") for k in order])
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(k, k) for k in order], rotation=15, ha="right")
    ax.set_ylabel("失败条数")
    ax.set_title("MILP 失败样本归因分布")
    for xi, v in zip(x, s["count"]):
        if v > 0:
            ax.text(xi, v + 0.5, str(int(v)), ha="center", fontsize=8)
    return ax


def _df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无数据_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(badcase: pd.DataFrame, summary: pd.DataFrame, metrics: dict, out: Path) -> None:
    lines = [
        "# MILP 路由失败归因报告\n",
        f"> 预算 {metrics.get('budget_per_query', '?')}/query | "
        f"真实成功率 {metrics.get('realized_success_rate', '?')} | "
        f"optimality gap {metrics.get('optimality_gap', '?')} | "
        f"降级失败率 {metrics.get('downgrade_failure_rate', '?')}\n",
        "## 归因计数\n",
        _df_to_md_table(summary) if not summary.empty else "_无失败样本_\n",
        "\n## Badcase 样例（前 15 条）\n",
    ]
    if badcase.empty:
        lines.append("_无失败样本_\n")
    else:
        lines.append(_df_to_md_table(badcase))
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pred = pd.read_parquet(ROOT / "data" / "predictions.parquet")
    budget = 0.002
    prompts = pred.groupby("sample_id")["eval_name"].first().to_dict()
    badcase, summary, metrics = analyze_routing_failures(pred, budget_per_query=budget, prompt_lookup=prompts)
    metrics["budget_per_query"] = budget

    print("指标:", metrics)
    print("\n归因计数:\n", summary.to_string(index=False))
    print("\nBadcase 样例:\n", badcase.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    plot_failure_summary(summary, ax=ax)
    fig.tight_layout()
    out_png = ROOT / "docs" / "failure_attribution.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print("saved", out_png)

    write_report(badcase, summary, metrics, ROOT / "eval" / "failure_report.md")
    print("saved eval/failure_report.md")


if __name__ == "__main__":
    main()
