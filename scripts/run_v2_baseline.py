"""Run v2 (TE only, alpha=10) and append to iteration_log.csv.

输入: config/config.yaml, data/peek.csv
输出: data/predictions_v2.parquet, data/iteration_log.csv 一行
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_v0_baseline import record_iteration  # noqa: E402
from model import ml_seperate as ml  # noqa: E402


def main() -> None:
    cfg = ml.load_config("config/config.yaml")
    cfg_v2 = copy.deepcopy(cfg)
    cfg_v2["features"]["use_target_encoding"] = True
    cfg_v2["features"]["target_encoding_alpha"] = 10.0
    cfg_v2["features"]["use_cross_difficulty"] = False

    data = ml.load_data(cfg_v2)
    feat = ml.Featurizer(cfg_v2, prompts=data.prompts, eval_names=data.eval_names)
    pred_df = ml.rolling_backtest(cfg_v2, feat, data)

    pred_df.to_parquet(ROOT / "data" / "predictions_v2.parquet", index=False)
    log = record_iteration("v2_TE_only", pred_df, cfg_v2)
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()
