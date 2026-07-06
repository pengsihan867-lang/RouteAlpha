"""Generate iteration_progress.png: Pareto baselines + v0→v2→v3 trajectory."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import milp as mp  # noqa: E402
from model import ml_seperate as ml  # noqa: E402

DEMO_BUDGET = 0.001
OUT = ROOT / "docs" / "iteration_progress.png"

VERSIONS = [
    ("v0", "v0_baseline", ROOT / "data" / "predictions_v0.parquet", "#9e9e9e"),
    ("v2", "v2_TE_only", ROOT / "data" / "predictions_v2.parquet", "#6baed6"),
    ("v3", "v3_TE+cross+mglobal", ROOT / "data" / "predictions.parquet", "#08519c"),
]


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def cpq(result, q: int) -> tuple[float, float]:
    return result.total_cost / q, result.realized_success_rate


def plot_iteration_progress(show: bool = False):
    setup_chinese_font()

    ref_pred = pd.read_parquet(VERSIONS[-1][2])
    q = ref_pred["sample_id"].nunique()
    bs = mp.baselines(ref_pred)

    baseline_styles = {
        "oracle": {"c": "#ffbf00", "m": "*", "s": 280, "label": "oracle（作弊上限）", "z": 6},
        "always_cheap": {"c": "#2ca02c", "m": "s", "s": 90, "label": "always-cheap", "z": 5},
        "always_expensive": {"c": "#d62728", "m": "^", "s": 90, "label": "always-expensive", "z": 5},
        "random": {"c": "#9467bd", "m": "D", "s": 70, "label": "random", "z": 4},
    }

    traj_rows = []
    for short, _name, path, color in VERSIONS:
        if not path.exists():
            raise FileNotFoundError(path)
        pred = pd.read_parquet(path)
        milp = mp.solve_routing(pred, budget_per_query=DEMO_BUDGET)
        ov = ml.compute_metrics(pred).query("model == '__overall__'").iloc[0]
        cx, sy = cpq(milp, q)
        traj_rows.append(
            {
                "short": short,
                "color": color,
                "cost_q": cx,
                "success": sy,
                "auc": float(ov["auc"]),
                "gap": bs["oracle"].realized_success_rate - sy,
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    for key, sty in baseline_styles.items():
        cx, sy = cpq(bs[key], q)
        ax.scatter(
            cx,
            sy,
            color=sty["c"],
            marker=sty["m"],
            s=sty["s"],
            label=sty["label"],
            zorder=sty["z"],
            edgecolors="k",
            linewidths=0.5,
        )

    xs = [r["cost_q"] for r in traj_rows]
    ys = [r["success"] for r in traj_rows]
    ax.plot(xs, ys, "-", color="#2171b5", linewidth=2.2, alpha=0.85, zorder=7, label="MILP 迭代轨迹")
    for r in traj_rows:
        ax.scatter(
            r["cost_q"],
            r["success"],
            s=140,
            color=r["color"],
            edgecolors="black",
            linewidths=1.0,
            zorder=8,
        )
        ax.annotate(
            f"{r['short']}\nAUC={r['auc']:.3f}\nSR={r['success']:.3f}",
            (r["cost_q"], r["success"]),
            textcoords="offset points",
            xytext={
                "v0": (-72, -28),
                "v2": (12, -32),
                "v3": (12, 10),
            }[r["short"]],
            fontsize=8,
            ha="left" if r["short"] != "v0" else "right",
        )

    ax.annotate(
        "",
        xy=(traj_rows[-1]["cost_q"], traj_rows[-1]["success"]),
        xytext=(traj_rows[0]["cost_q"], traj_rows[0]["success"]),
        arrowprops=dict(arrowstyle="->", color="#2171b5", lw=1.8, shrinkA=8, shrinkB=8),
        zorder=6,
    )

    v0, v3 = traj_rows[0], traj_rows[-1]
    d_sr = (v3["success"] - v0["success"]) * 100
    d_auc = (v3["auc"] - v0["auc"]) * 100
    d_gap = (v0["gap"] - v3["gap"]) * 100
    ax.text(
        0.02,
        0.03,
        f"@ {DEMO_BUDGET}/query：MILP +{d_sr:.1f}pp | AUC +{d_auc:.1f}pt | gap↓{d_gap:.1f}pp",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    ax.axvline(DEMO_BUDGET, color="#2171b5", linestyle=":", linewidth=1.0, alpha=0.45, zorder=1)

    oracle_c, oracle_s = cpq(bs["oracle"], q)
    ax.text(oracle_c + 0.00004, oracle_s - 0.012, "目标 ↖ oracle", fontsize=8, color="#666666")

    all_x = [oracle_c, *[cpq(bs[k], q)[0] for k in baseline_styles], *xs]
    all_y = [oracle_s, *[cpq(bs[k], q)[1] for k in baseline_styles], *ys]
    ax.set_xlim(0, max(all_x) + 0.00015)
    ax.set_ylim(min(all_y) - 0.025, max(all_y) + 0.015)
    ax.set_box_aspect(0.58)

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.plot(
        [x_lo + 0.00003, x_hi * 0.88],
        [y_lo + 0.008, y_hi - 0.008],
        linestyle="--",
        color="#888888",
        linewidth=1.0,
        alpha=0.45,
        zorder=0,
    )

    ax.set_xlabel("平均成本 / query")
    ax.set_ylabel("真实成功率")
    ax.set_title(f"迭代进步（Pareto 视角）· demo 预算 {DEMO_BUDGET}/query")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5)
    plt.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT)
    print(pd.DataFrame(traj_rows).to_string(index=False))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    plot_iteration_progress(show=False)


if __name__ == "__main__":
    main()
