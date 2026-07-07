"""生成 images/calibration.png — 可靠性图 + 各模型质量对比 (供 README / notebook 复用)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import ml_seperate as ml  # noqa: E402
from scripts.plot_utils import setup_chinese_font  # noqa: E402

OUT = ROOT / "images" / "calibration.png"


def overall_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    ece, n = 0.0, len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        if mask.sum():
            ece += abs(y_true[mask].mean() - p[mask].mean()) * mask.sum() / n
    return ece


def main() -> None:
    setup_chinese_font()
    pred = pd.read_parquet(ROOT / "data" / "predictions.parquet")
    metrics_df = ml.compute_metrics(pred)

    y_true = pred["y_true"].to_numpy(dtype=float)
    p_cal = pred["p_success"].to_numpy(dtype=float)
    has_raw = "p_success_raw" in pred.columns
    p_raw = pred["p_success_raw"].to_numpy(dtype=float) if has_raw else None
    ece_cal = overall_ece(y_true, p_cal)
    ece_raw = overall_ece(y_true, p_raw) if has_raw else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax0 = axes[0]
    axb = ax0.twinx()
    axb.hist(p_cal, bins=20, range=(0, 1), color="#cfcfcf", alpha=0.35, zorder=0)
    axb.set_ylabel("样本数", color="#9e9e9e", fontsize=9)
    ax0.set_zorder(axb.get_zorder() + 1)
    ax0.patch.set_visible(False)
    ax0.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="完美校准", zorder=2)
    if has_raw:
        fr_raw, mp_raw = calibration_curve(y_true, p_raw, n_bins=10, strategy="quantile")
        ax0.plot(mp_raw, fr_raw, "o-", color="#f16913", linewidth=1.6, markersize=5,
                 label=f"校准前 raw (ECE={ece_raw:.3f})", zorder=3)
    fr_cal, mp_cal = calibration_curve(y_true, p_cal, n_bins=10, strategy="quantile")
    ax0.plot(mp_cal, fr_cal, "o-", color="#2171b5", linewidth=1.8, markersize=5,
             label=f"校准后 cal (ECE={ece_cal:.3f})", zorder=4)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("预测成功率")
    ax0.set_ylabel("实际成功率")
    ax0.set_title("可靠性图 · 校准前 → 校准后" if has_raw else "可靠性图（校准后）")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    m = metrics_df[metrics_df["model"] != "__overall__"].copy()
    name_map = {
        "gpt-4-1106-preview": "GPT-4",
        "gpt-3.5-turbo-1106": "GPT-3.5",
        "claude-v2": "Claude-v2",
        "claude-instant-v1": "Claude-Instant",
    }
    labels = [name_map.get(s, s.split("/")[-1][:12]) for s in m["model"]]
    x = np.arange(len(m))
    w = 0.26
    for offset, col, color, lab in [
        (-w, "accuracy", "#4c78a8", "accuracy"),
        (0, "auc", "#f58518", "AUC"),
        (w, "ece", "#54a24b", "1−ECE"),
    ]:
        vals = m[col] if col != "ece" else 1 - m[col]
        bars = ax1.bar(x + offset, vals, w, label=lab, color=color)
        for b in bars:
            ax1.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    ax1.axhline(0.5, color="#999", linestyle="--", linewidth=1, alpha=0.7, label="AUC 随机基线")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("指标值（越高越好）")
    ax1.set_title("各候选模型预测质量（阶段一）")
    ax1.legend(fontsize=7.5, ncol=4, loc="upper center")
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
