# MILP 路由失败归因报告

> 预算 0.002/query | 真实成功率 0.81 | optimality gap 0.1167 | 降级失败率 0.614

## 归因计数

| failure_tag | count |
| --- | --- |
| calibration_error | 82 |
| task_hard | 66 |
| label_noise | 15 |
| prediction_error | 8 |

## Badcase 样例（前 15 条）

| sample_id | eval_name | prompt_preview | chosen_model | chosen_p | oracle_model | best_pred_model | failure_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chinese_character_riddles.dev.12 | Chinese_character_riddles | Chinese_character_riddles | claude-instant-v1 | 0.368 | claude-instant-v1 | - | task_hard |
| Chinese_character_riddles.dev.20 | Chinese_character_riddles | Chinese_character_riddles | gpt-3.5-turbo-1106 | 0.662 | gpt-3.5-turbo-1106 | - | task_hard |
| Chinese_character_riddles.dev.45 | Chinese_character_riddles | Chinese_character_riddles | gpt-4-1106-preview | 0.8 | claude-instant-v1 | - | task_hard |
| Chinese_character_riddles.dev.64 | Chinese_character_riddles | Chinese_character_riddles | gpt-4-1106-preview | 0.8 | claude-instant-v1 | - | task_hard |
| Chinese_character_riddles.dev.72 | Chinese_character_riddles | Chinese_character_riddles | claude-instant-v1 | 0.719 | claude-instant-v1 | - | task_hard |
| Chinese_character_riddles.dev.77 | Chinese_character_riddles | Chinese_character_riddles | gpt-3.5-turbo-1106 | 0.75 | claude-instant-v1 | - | task_hard |
| Chinese_character_riddles.dev.96 | Chinese_character_riddles | Chinese_character_riddles | gpt-4-1106-preview | 0.75 | claude-instant-v1 | - | task_hard |
| arc-challenge.test.456 | arc-challenge | arc-challenge | gpt-4-1106-preview | 0.899 | claude-instant-v1 | - | task_hard |
| arc-challenge.val.193 | arc-challenge | arc-challenge | gpt-3.5-turbo-1106 | 1.0 | claude-instant-v1 | gpt-4-1106-preview | label_noise |
| bias_detection.dev.223 | bias_detection | bias_detection | gpt-3.5-turbo-1106 | 0.769 | claude-instant-v1 | - | task_hard |
| bias_detection.dev.28 | bias_detection | bias_detection | gpt-3.5-turbo-1106 | 0.778 | claude-instant-v1 | - | task_hard |
| chinese_ancient_poetry.dev.24 | chinese_ancient_poetry | chinese_ancient_poetry | gpt-3.5-turbo-1106 | 0.778 | claude-instant-v1 | - | task_hard |
| chinese_homonym.dev.12 | chinese_homonym | chinese_homonym | claude-v2 | 0.781 | claude-instant-v1 | gpt-3.5-turbo-1106 | calibration_error |
| chinese_modern_poem_identification.test.1 | chinese_modern_poem_identification | chinese_modern_poem_identification | gpt-3.5-turbo-1106 | 0.882 | claude-instant-v1 | - | task_hard |
| chinese_zodiac.dev.109 | chinese_zodiac | chinese_zodiac | gpt-3.5-turbo-1106 | 0.882 | claude-instant-v1 | - | task_hard |