"""eval/judge.py — LLM-as-a-judge 位置偏置量化 + swap-and-aggregate

职责: 比较两答案优劣, 量化位置偏置, 演示 swap-and-aggregate 防污染。
输入: DEMO_PAIRS (prompt, good, bad); 可选真实 API (--real)。
输出: flip_rate / accept_rate / accuracy_on_accepted; eval/judge_report.md

默认离线 mock; Kimi 真实结果见 eval/kimi_judge_results.json。
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # "Route Alpha" 目录


# ============================================================================ #
# 第 1 步: 准备测试数据 —— 几对 (问题, 好答案, 坏答案)
# ----------------------------------------------------------------------------
# 因为我们"人为知道"哪个答案更好, 后面就能拿它当标准答案, 检验 judge 判得对不对。
# 好答案 = 紧扣问题、信息正确;  坏答案 = 跑题 / 空洞 / 错误。
# ============================================================================ #
DEMO_PAIRS = [
    # ---- easy: 好坏差距明显, 用于 mock 演示 baseline ----
    {
        "id": "easy_overfit",
        "category": "easy",
        "prompt": "用一句话解释什么是过拟合 overfitting",
        "good": "过拟合是指模型把训练数据里的噪声也学了进去, 导致在训练集上表现很好, 但在新数据上泛化变差。",
        "bad": "过拟合是一种很常见的现象, 大家都听说过, 它和机器学习有关系。",
    },
    {
        "id": "easy_list_tuple",
        "category": "easy",
        "prompt": "Python 里 list 和 tuple 最主要的区别是什么",
        "good": "list 是可变的 (创建后能增删改元素), tuple 是不可变的 (创建后不能修改), 所以 tuple 更适合做固定数据和字典的 key。",
        "bad": "list 和 tuple 都是 Python 的数据类型, 用起来都差不多, 看个人习惯。",
    },
    {
        "id": "easy_auc",
        "category": "easy",
        "prompt": "AUC 这个指标衡量的是分类器的什么能力",
        "good": "AUC 衡量的是排序能力: 随机取一个正样本和一个负样本, 模型给正样本更高分的概率, 0.5 等于瞎猜, 1.0 是完美排序。",
        "bad": "AUC 就是准确率的另一种说法, 数值越高说明模型预测得越准。",
    },
    {
        "id": "easy_calibration",
        "category": "easy",
        "prompt": "为什么要对预测概率做校准 calibration",
        "good": "因为模型输出的概率可能系统性偏高或偏低, 校准让 '预测 70%' 真的对应约 70% 的发生率, 这样概率才能拿来做决策。",
        "bad": "校准就是让模型更准一点的一个步骤, 做了总比不做好。",
    },
    {
        "id": "easy_milp",
        "category": "easy",
        "prompt": "一句话说明 MILP (混合整数线性规划) 适合解决什么问题",
        "good": "MILP 适合在一堆线性约束 (比如总预算上限) 下, 对 '选或不选' 这类 0/1 决策求全局最优解。",
        "bad": "MILP 是一种数学方法, 可以用来算很多优化的东西, 非常强大。",
    },
    {
        "id": "easy_leakage",
        "category": "easy",
        "prompt": "解释一下什么是数据穿越 data leakage",
        "good": "数据穿越是指训练阶段不小心用到了本不该看到的未来/测试信息, 导致离线指标虚高, 上线后表现暴跌。",
        "bad": "数据穿越就是数据从一个地方传到另一个地方, 在工程里很常见。",
    },
    # ---- hard: 质量接近 / 长度偏置陷阱, 用于压测真实 judge ----
    {
        "id": "hard_dropout_length",
        "category": "hard",
        "trap": "length_bias",
        "prompt": "用一句话解释 dropout 正则化在做什么",
        "good": "训练时随机丢弃部分神经元, 迫使网络不过度依赖单一路径, 从而减轻过拟合。",
        "bad": (
            "Dropout 是深度学习中非常重要、被大量论文和工业界广泛采用的一项核心正则化技术。"
            "它在训练过程中会随机地将神经网络中的部分神经元暂时从计算图中移除, "
            "这样可以防止神经元之间产生复杂的共适应关系, 提高泛化能力, "
            "在 ImageNet、BERT 等 benchmark 上都被证明有效, 是每一个从业者都应该掌握的基础知识。"
        ),
    },
    {
        "id": "hard_routerbench_arena",
        "category": "hard",
        "trap": "subtle_error",
        "prompt": "RouterBench 和 RouterArena 在路由评测里各扮演什么角色",
        "good": "RouterBench 提供 query×model 的训练/回测宽表; RouterArena 是 held-out 打榜集, 禁止在其上训练或调参。",
        "bad": "RouterBench 和 RouterArena 都是路由相关的公开数据集, 都可以用来训练路由器并在上面报告成功率。",
    },
    {
        "id": "hard_ece_brier",
        "category": "hard",
        "trap": "subtle_error",
        "prompt": "ECE 和 Brier score 分别衡量什么, 二者有何不同",
        "good": "ECE 看分箱后预测概率与真实频率的偏差 (校准); Brier 是个体预测概率与 0/1 标签的平方误差, 同时反映校准与分辨率。",
        "bad": "ECE 和 Brier 都是衡量模型好坏的指标, 数值越低越好, 在实际项目里两个都会报告, 没有本质区别。",
    },
    {
        "id": "hard_swap_aggregate",
        "category": "hard",
        "trap": "subtle_error",
        "prompt": "swap-and-aggregate 为什么能减弱 LLM-judge 的位置偏置",
        "good": "同一对答案正反各评一次, 只有两次结论一致才采纳; 若仅因顺序改变 verdict 则 discard, 避免位置偏置污染标签。",
        "bad": "swap-and-aggregate 是一种评测技巧, 把同一题多评几次取平均, 结果会更稳定、更可信。",
    },
    {
        "id": "hard_milp_budget",
        "category": "hard",
        "trap": "subtle_error",
        "prompt": "固定预算 0.002/query 时, MILP 路由相对 always-cheap 的优势是什么",
        "good": "MILP 在全局预算内按每条 query 的 P(success) 做组合优化, 不必每条都选最便宜模型, 从而在相同总成本下提高真实成功率。",
        "bad": "MILP 是数学优化算法, always-cheap 是简单 baseline; 优化算法通常比启发式更先进, 所以 MILP 几乎总是更好。",
    },
    {
        "id": "hard_random_baseline_length",
        "category": "hard",
        "trap": "length_bias",
        "prompt": "LLM 路由实验里 random baseline 应如何定义才公平",
        "good": "每条 query 从候选模型中均匀随机选一个, 通常固定多个 seed 取平均成功率, 作为无信息下界。",
        "bad": (
            "Random baseline 就是随机选模型, 这是最简单、最直观、也是业界普遍采用的 baseline 设定方式。"
            "它的意义在于提供一个参照, 说明任何严肃的路由方法都应该显著优于 random, "
            "否则就没有部署价值; 在实际论文和 benchmark 中 random 的表现通常很差, 这是大家的共识。"
        ),
    },
]


# ============================================================================ #
# 第 2 步: judge 后端 —— 一个能"看两个答案、选一个更好"的函数
# ----------------------------------------------------------------------------
# 我们提供两种:
#   (A) make_mock_judge: 离线模拟, 不用联网。它用一个"质量代理"打分, 再叠加位置偏置。
#   (B) api_judge:       可选, 真用一个大模型 API 当 judge (后面想试再用, 默认不调用)。
# judge 函数统一签名:  judge(prompt, answer_a, answer_b) -> "A" 或 "B"
# ============================================================================ #
def _quality_proxy(prompt: str, answer: str) -> float:
    """一个粗糙的"答案质量"估计, 让 mock judge 有东西可依据。

    思路: 好答案通常更长、信息更密(我们造数据时好答案确实更充实)。
    这只是为了让模拟 judge 行为像那么回事, 不是真的 NLP 评分。
    """
    # 答案字符数(去掉空格)作为信息量的粗略代理; 封顶防止单纯堆字
    char_count = len(answer.replace(" ", ""))
    return min(char_count, 80) * 0.1


def make_mock_judge(position_bias: float = 0.0, noise: float = 0.5, seed: int = 0):
    """造一个模拟 judge。

    参数:
      position_bias: 对 A 位置的偏好强度。0 = 公平; 越大 = 越偏向排在前面的答案。
      noise:         随机扰动, 模拟 judge 不是每次都理性。
      seed:          固定随机种子, 保证每次跑结果可复现。
    返回: 一个 judge(prompt, answer_a, answer_b) 函数。
    """
    rng = random.Random(seed)

    def judge(prompt: str, answer_a: str, answer_b: str) -> str:
        score_a = _quality_proxy(prompt, answer_a) + rng.gauss(0, noise)
        score_b = _quality_proxy(prompt, answer_b) + rng.gauss(0, noise)
        # 关键: 给 A 位置无脑加一个 bias —— 这就是"位置偏置"的来源
        score_a += position_bias
        return "A" if score_a >= score_b else "B"

    return judge


def api_judge(prompt: str, answer_a: str, answer_b: str) -> str:
    """可选: 用真实大模型 API 当 judge (默认不被调用, 想试再说)。

    需要联网 + 在环境变量里设好 OPENAI_API_KEY / OPENAI_BASE_URL。
    这里用 OpenAI 兼容接口写法; 失败会抛错, 不影响 mock 流程。
    """
    import os

    from openai import OpenAI  # 延迟导入: 不用 API 时即使没装也不报错

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),  # 可指向 OpenRouter / GLM 等
    )
    system = "你是一个严格的评审。只回答字母 A 或 B, 表示哪个回答更好, 不要解释。"
    user = f"问题:\n{prompt}\n\n回答 A:\n{answer_a}\n\n回答 B:\n{answer_b}\n\n哪个更好? 只答 A 或 B。"
    resp = client.chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().upper()
    return "A" if text.startswith("A") else "B"


# ============================================================================ #
# 第 3 步: 对一对答案做 swap-and-aggregate (正反各评一次)
# ============================================================================ #
def judge_one_pair(judge, pair: dict) -> dict:
    """对一对 (好答案, 坏答案) 正反各问一次 judge, 返回这对的判定细节。"""
    prompt, good, bad = pair["prompt"], pair["good"], pair["bad"]
    # 顺序 1:  A = 好答案,  B = 坏答案
    verdict_1 = judge(prompt, good, bad)
    winner_1 = "good" if verdict_1 == "A" else "bad"  # 翻译成"内容谁赢了"

    # 顺序 2:  A = 坏答案,  B = 好答案  (把位置换过来)
    verdict_2 = judge(prompt, bad, good)
    winner_2 = "good" if verdict_2 == "B" else "bad"  # 注意: 这次好答案在 B 位置

    consistent = winner_1 == winner_2  # 两次结论是否一致
    return {
        "id": pair.get("id", ""),
        "category": pair.get("category", "unknown"),
        "trap": pair.get("trap", ""),
        "prompt": prompt[:40] + ("..." if len(prompt) > 40 else ""),
        "verdict_1": verdict_1,   # 第一次选的字母
        "verdict_2": verdict_2,   # 第二次选的字母
        "winner_1": winner_1,     # 第一次实际上选了好/坏
        "winner_2": winner_2,     # 第二次实际上选了好/坏
        "consistent": consistent, # 是否一致(可采纳)
        # 只有一致时才采纳判定; 采纳结果就是 winner_1
        "accepted_winner": winner_1 if consistent else None,
    }


def _metrics_from_rows(rows: list[dict]) -> dict:
    n = len(rows)
    n_flipped = sum(1 for r in rows if not r["consistent"])
    accepted = [r for r in rows if r["consistent"]]
    n_correct = sum(1 for r in accepted if r["accepted_winner"] == "good")
    return {
        "n_pairs": n,
        "position_flip_rate": round(n_flipped / n, 3) if n else 0.0,
        "accept_rate": round(len(accepted) / n, 3) if n else 0.0,
        "accuracy_on_accepted": round(n_correct / len(accepted), 3) if accepted else None,
    }


# ============================================================================ #
# 第 4 步: 跑完所有样本, 汇总成三个指标
# ============================================================================ #
def run_judge_eval(judge, pairs: list[dict]) -> dict:
    """对所有 pair 跑 swap-and-aggregate, 返回 {明细, 指标, 分难度指标}。"""
    rows = [judge_one_pair(judge, p) for p in pairs]
    metrics = _metrics_from_rows(rows)

    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        by_category[cat] = _metrics_from_rows(cat_rows)

    return {"rows": rows, "metrics": metrics, "by_category": by_category}


# ============================================================================ #
# 第 5 步: 写一份小报告 (交付物), 顺便对比"公平 judge"和"有偏置 judge"
# ============================================================================ #
def write_report(
    fair: dict,
    biased: dict,
    out_path: Path,
    real: dict | None = None,
    real_meta: dict | None = None,
) -> None:
    lines = []
    lines.append("# LLM-as-a-Judge 偏置诊断报告\n")
    lines.append("> 由 `eval/judge.py` 生成。演示位置偏置如何被 swap-and-aggregate 发现。\n")
    lines.append(f"> 评测集: **{fair['metrics']['n_pairs']} 对** (easy 6 + hard 6, 含长度偏置/细微错误陷阱)\n")
    lines.append("## 指标对比\n")
    lines.append("| judge | 位置翻转率 | 采纳率 | 采纳样本准确率 |")
    lines.append("|---|---|---|---|")
    for name, res in [("公平 judge (mock)", fair), ("有位置偏置 judge (mock)", biased)]:
        m = res["metrics"]
        lines.append(
            f"| {name} | {m['position_flip_rate']} | {m['accept_rate']} | {m['accuracy_on_accepted']} |"
        )
    if real is not None:
        m = real["metrics"]
        model = (real_meta or {}).get("model", "真实 LLM")
        provider = (real_meta or {}).get("provider", "")
        label = f"{provider} {model}".strip() if provider else model
        lines.append(
            f"| {label} (real) | {m['position_flip_rate']} | {m['accept_rate']} | {m['accuracy_on_accepted']} |"
        )
    lines.append("\n## 分难度 (真实 judge)\n")
    if real is not None and real.get("by_category"):
        lines.append("| 子集 | 对数 | 翻转率 | 采纳率 | 采纳准确率 |")
        lines.append("|---|---|---|---|---|")
        for cat, cm in real["by_category"].items():
            lines.append(
                f"| {cat} | {cm['n_pairs']} | {cm['position_flip_rate']} | "
                f"{cm['accept_rate']} | {cm['accuracy_on_accepted']} |"
            )
    else:
        lines.append("_运行 `python eval/judge.py --real` 后自动填充_\n")
    if real is not None:
        flipped = [r for r in real["rows"] if not r["consistent"]]
        if flipped:
            lines.append("\n## 发生位置翻转的样本\n")
            lines.append("| id | trap | verdict_1 | verdict_2 |")
            lines.append("|---|---|---|---|")
            for r in flipped:
                lines.append(f"| {r['id']} | {r.get('trap', '-')} | {r['verdict_1']} | {r['verdict_2']} |")
        wrong = [
            r for r in real["rows"]
            if r["consistent"] and r["accepted_winner"] != "good"
        ]
        if wrong:
            lines.append("\n## 一致但选错 (质量误判)\n")
            lines.append("| id | trap | accepted |")
            lines.append("|---|---|---|")
            for r in wrong:
                lines.append(f"| {r['id']} | {r.get('trap', '-')} | {r['accepted_winner']} |")
    lines.append("\n## 怎么读这张表\n")
    lines.append("- **位置翻转率越低越好**: 它衡量 judge 受答案顺序影响的程度。")
    lines.append("- 有偏置的 judge 翻转率明显更高, 说明很多判定只是因为顺序不同而改变, 不可信。")
    lines.append("- **采纳率** = 两次一致、保留下来的比例; 翻转掉的判定被丢弃, 不进入训练标签 (防污染)。")
    lines.append("- **采纳样本准确率**: 在保留的判定里, judge 是否真的挑出了更好的答案。")
    lines.append("- **hard 子集** 故意加入长度偏置陷阱和细微 factual 错误, 比 easy 更能暴露 judge 弱点。\n")
    lines.append("## 一键切换真实模型\n")
    lines.append("```bash")
    lines.append("export OPENAI_API_KEY=your_key")
    lines.append("# Kimi 示例:")
    lines.append("export OPENAI_BASE_URL=https://api.moonshot.cn/v1")
    lines.append("export JUDGE_MODEL=moonshot-v1-8k")
    lines.append("python eval/judge.py --real")
    lines.append("```\n")
    lines.append("## 结论\n")
    lines.append("swap-and-aggregate 能把受位置影响的判定识别并剔除, 是 LLM-judge 可信度的基础防线。\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _load_dotenv() -> None:
    """若项目根有 .env 则加载 (不提交 git)。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        import os

        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> None:
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="LLM-as-a-judge 偏置诊断")
    parser.add_argument(
        "--real",
        action="store_true",
        help="用真实 LLM API 跑 swap-and-aggregate (需 OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    pairs = DEMO_PAIRS

    # mock 演示: 公平 vs 有偏置
    fair_judge = make_mock_judge(position_bias=0.0, seed=42)
    biased_judge = make_mock_judge(position_bias=5.0, seed=42)

    fair = run_judge_eval(fair_judge, pairs)
    biased = run_judge_eval(biased_judge, pairs)
    real = None
    real_meta = None

    if args.real:
        _load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("请设置 OPENAI_API_KEY (及可选 OPENAI_BASE_URL / JUDGE_MODEL)")
        base = os.environ.get("OPENAI_BASE_URL", "")
        model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        provider = "Kimi" if "moonshot" in base else ("OpenAI" if "openai" in base else "LLM")
        real_meta = {"provider": provider, "model": model, "base_url": base}
        print(f"使用真实 LLM judge: {provider} / {model} ...")
        real = run_judge_eval(api_judge, pairs)
        # 持久化真实结果, 供 README 引用
        out_json = ROOT / "eval" / "kimi_judge_results.json"
        payload = {
            "meta": real_meta,
            "metrics": real["metrics"],
            "by_category": real["by_category"],
            "rows": real["rows"],
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"真实 judge 结果已写入: {out_json}")

    # 终端打印
    print("\n===== 公平 judge (mock) =====")
    print(fair["metrics"])
    print("\n===== 有位置偏置 judge (mock) =====")
    print(biased["metrics"])
    if real is not None:
        print("\n===== 真实 LLM judge =====")
        print(real["metrics"])
        print("分难度:", real["by_category"])

    print("\n----- 有偏置 judge 的逐条明细 -----")
    print(f"{'第1次':<6}{'第2次':<6}{'是否一致':<8}问题")
    for r in biased["rows"]:
        flag = "一致" if r["consistent"] else "翻转!"
        print(f"{r['verdict_1']:<7}{r['verdict_2']:<7}{flag:<9}{r['prompt']}")

    # 写报告
    out = ROOT / "eval" / "judge_report.md"
    write_report(fair, biased, out, real=real, real_meta=real_meta)
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
