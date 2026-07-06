# LLM-as-a-Judge 偏置诊断报告

> 由 `eval/judge.py` 生成。演示位置偏置如何被 swap-and-aggregate 发现。

> 评测集: **12 对** (easy 6 + hard 6, 含长度偏置/细微错误陷阱)

## 指标对比

| judge | 位置翻转率 | 采纳率 | 采纳样本准确率 |
|---|---|---|---|
| 公平 judge (mock) | 0.083 | 0.917 | 0.818 |
| 有位置偏置 judge (mock) | 1.0 | 0.0 | None |
| Kimi moonshot-v1-8k (real) | 0.083 | 0.917 | 1.0 |

## 分难度 (真实 judge)

| 子集 | 对数 | 翻转率 | 采纳率 | 采纳准确率 |
|---|---|---|---|---|
| easy | 6 | 0.0 | 1.0 | 1.0 |
| hard | 6 | 0.167 | 0.833 | 1.0 |

## 发生位置翻转的样本

| id | trap | verdict_1 | verdict_2 |
|---|---|---|---|
| hard_random_baseline_length | length_bias | A | A |

## 怎么读这张表

- **位置翻转率越低越好**: 它衡量 judge 受答案顺序影响的程度。
- 有偏置的 judge 翻转率明显更高, 说明很多判定只是因为顺序不同而改变, 不可信。
- **采纳率** = 两次一致、保留下来的比例; 翻转掉的判定被丢弃, 不进入训练标签 (防污染)。
- **采纳样本准确率**: 在保留的判定里, judge 是否真的挑出了更好的答案。
- **hard 子集** 故意加入长度偏置陷阱和细微 factual 错误, 比 easy 更能暴露 judge 弱点。

## 一键切换真实模型

```bash
export OPENAI_API_KEY=your_key
# Kimi 示例:
export OPENAI_BASE_URL=https://api.moonshot.cn/v1
export JUDGE_MODEL=moonshot-v1-8k
python eval/judge.py --real
```

## 结论

swap-and-aggregate 能把受位置影响的判定识别并剔除, 是 LLM-judge 可信度的基础防线。
