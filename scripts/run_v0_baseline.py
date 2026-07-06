"""Run v0 baseline (no TE / no cross) and append to iteration_log.csv.

输入: config/config.yaml, data/peek.csv
输出: data/predictions_v0.parquet, data/iteration_log.csv 一行
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import milp as mp  # noqa: E402
from model import ml_seperate as ml  # noqa: E402

ITER_LOG = ROOT / "data" / "iteration_log.csv"


def feature_note(cfg: dict) -> str:
    fc = cfg.get("features", {})
    parts = ["bge384"]
    if fc.get("use_structural", True):
        parts.append("24维结构")
    if fc.get("use_target_encoding", False):
        parts.append(f"TE(alpha={fc.get('target_encoding_alpha', 10)})")
    if fc.get("use_cross_difficulty", False):
        parts.append("cross+mglobal")
    if len(parts) == 2:
        parts.append("无TE")
    return " + ".join(parts)


def record_iteration(version: str, pred_df, cfg, note: str = "", prob_col: str = "p_success") -> pd.DataFrame:
    budget = cfg["milp"]["budget_per_query"]
    q = pred_df["sample_id"].nunique()
    metrics = ml.compute_metrics(pred_df, prob_col=prob_col)
    ov = metrics[metrics["model"] == "__overall__"].iloc[0]
    oracle_s = mp.baselines(pred_df, prob_col=prob_col)["oracle"].realized_success_rate
    milp = mp.solve_routing(pred_df, budget_per_query=budget, prob_col=prob_col)
    row = {
        "版本": version,
        "AUC": round(float(ov["auc"]), 4),
        "ECE": round(float(ov["ece"]), 4),
        "MILP成功率": round(milp.realized_success_rate, 4),
        "成本/q": round(milp.total_cost / q, 5),
        "oracle天花板": round(oracle_s, 4),
        "gap": round(oracle_s - milp.realized_success_rate, 4),
        "预算/q": budget,
        "备注": note or feature_note(cfg),
    }
    if ITER_LOG.exists():
        log = pd.read_csv(ITER_LOG, encoding="utf-8")
        log = log[log["版本"] != version]
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        ITER_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = pd.DataFrame([row])
    log.to_csv(ITER_LOG, index=False, encoding="utf-8")
    return log


def main() -> None:
    cfg = ml.load_config("config/config.yaml")
    cfg_v0 = copy.deepcopy(cfg)
    cfg_v0["features"]["use_target_encoding"] = False
    cfg_v0["features"]["use_cross_difficulty"] = False

    data = ml.load_data(cfg_v0)
    feat = ml.Featurizer(cfg_v0, prompts=data.prompts, eval_names=data.eval_names)
    pred_df = ml.rolling_backtest(cfg_v0, feat, data)

    out = ROOT / "data" / "predictions_v0.parquet"
    pred_df.to_parquet(out, index=False)
    print(f"saved {out}")

    log = record_iteration("v0_baseline", pred_df, cfg_v0)
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()
