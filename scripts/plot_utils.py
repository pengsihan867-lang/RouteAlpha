"""Matplotlib 中文字体配置 — 供 eval/ 与 scripts/ 出图复用."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager

# macOS / Windows 常见中文字体，按优先级排列
_CJK_FONT_CANDIDATES = [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]


def setup_chinese_font() -> str | None:
    """配置 matplotlib 中文字体，返回实际使用的字体名；找不到则返回 None。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None
