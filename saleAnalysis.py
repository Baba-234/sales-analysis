#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
销售订单统计分析
================

把 ``saleAnalysis.ipynb`` 中的分析流程整理成一个可直接运行的脚本。四条分析：

1. 每天销售金额变化（折线图）
2. 不同城市销售数量对比（柱状图，降序）
3. 不同产品类别销售数量占比（饼图）
4. 不同支付方式订单占比（饼图）

四张图拼成 2×2 的画布，统一输出一张 PNG。

用法示例::

    python saleAnalysis.py
    python saleAnalysis.py --csv sales.csv --out 销售订单统计分析.png
    python saleAnalysis.py --show          # 弹窗显示，不关窗口

依赖：pandas、matplotlib（中文字体使用系统自带的 SimHei / 微软雅黑）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# --------------------------------------------------------------------------- #
# 常量配置
# --------------------------------------------------------------------------- #

#: csv 中必需的列，缺任何一列都无法完成分析
REQUIRED_COLUMNS: tuple[str, ...] = (
    "订单号",
    "产品类别",
    "产品名称",
    "销售数量",
    "单价",
    "客户所在城市",
    "支付方式",
    "订单日期",
)

#: 中文字体候选，按顺序取第一个系统里存在的
FONT_CANDIDATES: tuple[str, ...] = ("SimHei", "Microsoft YaHei", "Arial Unicode MS")

#: 图表标题字号 / 轴标签字号
TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 12
SUPTITLE_FONTSIZE = 23


# --------------------------------------------------------------------------- #
# 数据读取与清洗
# --------------------------------------------------------------------------- #

def configure_chinese_font() -> None:
    """设置 matplotlib 中文字体，避免中文显示成方块。"""
    plt.rcParams["font.sans-serif"] = list(FONT_CANDIDATES)
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def add_sales_amount(df: pd.DataFrame) -> pd.DataFrame:
    """按 ``销售数量 * 单价`` 计算销售金额，返回新的 DataFrame（不改动入参）。"""
    out = df.copy()
    out["销售数量"] = pd.to_numeric(out["销售数量"], errors="coerce")
    out["单价"] = pd.to_numeric(out["单价"], errors="coerce")
    out["销售金额"] = out["销售数量"] * out["单价"]
    return out


def load_sales_data(csv_path: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """读取销售数据并做基础清洗。

    处理内容：
        - 校验必需列是否齐全（缺失直接抛 ValueError）
        - ``产品类别`` 去掉首尾空格，避免 "图书 " 与 "图书" 被当成两个类别
        - ``销售数量`` / ``单价`` 转成数值，``订单日期`` 转成 datetime
        - 新增 ``销售金额`` 列

    :param csv_path: csv 文件路径
    :param encoding: 文件编码，默认 utf-8
    :return: 清洗后的 DataFrame
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_csv(path, encoding=encoding)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必需列: {', '.join(missing)}；实际列为: {', '.join(map(str, df.columns))}")

    df = df[list(REQUIRED_COLUMNS)].copy()

    # 类别名去空格，防止同名不同值导致饼图图例重复
    df["产品类别"] = df["产品类别"].astype("string").str.strip()

    df["订单日期"] = pd.to_datetime(df["订单日期"], errors="coerce")

    return add_sales_amount(df)


# --------------------------------------------------------------------------- #
# 统计：每个图对应一个取数函数，方便单测、也方便复用到别处
# --------------------------------------------------------------------------- #

def daily_sales_by_date(df: pd.DataFrame) -> pd.Series:
    """按订单日期汇总销售金额，index 升序（每天一个点）。"""
    series = df.groupby("订单日期")["销售金额"].sum()
    return series.sort_index()


def sales_by_city(df: pd.DataFrame) -> pd.Series:
    """按客户所在城市汇总销售数量，按数量降序。"""
    series = df.groupby("客户所在城市")["销售数量"].sum()
    return series.sort_values(ascending=False)


def sales_by_category(df: pd.DataFrame) -> pd.Series:
    """按产品类别汇总销售数量。"""
    return df.groupby("产品类别")["销售数量"].sum()


def order_count_by_payment(df: pd.DataFrame) -> pd.Series:
    """按支付方式统计订单条数。"""
    return df.groupby("支付方式")["支付方式"].count()


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """生成一份控制台摘要，便于跑完脚本先看结论。"""
    dates = df["订单日期"].dropna()
    city = sales_by_city(df)
    category = sales_by_category(df)
    return {
        "订单条数": len(df),
        "总销售额": float(df["销售金额"].sum()),
        "日期范围": (
            f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}" if not dates.empty else "无有效日期"
        ),
        "城市数": int(city.shape[0]),
        "销售数量最高城市": f"{city.index[0]} ({int(city.iloc[0])})" if not city.empty else "-",
        "产品类别数": int(category.shape[0]),
        "销售数量最高类别": f"{category.idxmax()} ({int(category.max())})" if not category.empty else "-",
    }


# --------------------------------------------------------------------------- #
# 绘图：通用绘图函数 + 四个业务图
# --------------------------------------------------------------------------- #

def create_layout(
    *,
    nrows: int = 2,
    ncols: int = 2,
    figsize: tuple[float, float] = (20, 12),
    dpi: int = 100,
) -> tuple[Figure, list[Axes]]:
    """创建 2×2 子图画布，返回 (fig, 展平后的 axes 列表)。"""
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    fig.suptitle("销售订单统计分析", fontsize=SUPTITLE_FONTSIZE, x=0.5, y=0.98)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    return fig, list(axes.ravel())


def plot_line(
    ax: Axes,
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "green",
    tick_step: int = 15,
    rot: int = 45,
) -> None:
    """画折线图，x 轴按 ``tick_step`` 隔点取刻度，避免日期标签挤在一起。"""
    ax.clear()
    ax.plot(x, y, color=color)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)

    if len(x):
        ticks = list(x)[::tick_step]
        if ticks[-1] != list(x)[-1]:
            ticks.append(list(x)[-1])
        ax.set_xticks(ticks)

    ax.grid(linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=rot)


def plot_bar(
    ax: Axes,
    series: pd.Series,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "blue",
    width: float = 0.6,
    rot: int = 90,
) -> None:
    """画柱状图，x 轴取 series 的 index。"""
    ax.clear()
    ax.bar(series.index.astype(str), series.values, color=color, width=width)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=rot)


def plot_pie_ratio(
    ax: Axes,
    series: pd.Series,
    *,
    title: str,
    autopct: str = "%1.1f%%",
    radius: float = 1.2,
    legend_ncol: int = 5,
    textprops: dict[str, Any] | None = None,
) -> None:
    """画占比饼图：扇区带类别标签和百分比，图例放底部。

    先 ``ax.clear()``，否则 Jupyter 里重复执行 / 脚本重复调用会叠加多组图例。
    """
    ax.clear()
    if series.empty:
        ax.set_title(f"{title}（无数据）", fontsize=TITLE_FONTSIZE)
        return

    ax.pie(
        series.values,
        labels=series.index.astype(str),
        autopct=autopct,
        startangle=0,
        radius=radius,
        textprops=textprops or {},
    )
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=20)
    ax.legend(loc="lower center", ncol=legend_ncol, bbox_to_anchor=(0.5, -0.2))


def plot_daily_sales(ax: Axes, df: pd.DataFrame, *, date_format: str = "%Y-%m-%d") -> None:
    """图1：每天销售金额变化折线图。"""
    series = daily_sales_by_date(df)
    x = [d.strftime(date_format) for d in series.index]
    plot_line(
        ax,
        x,
        series.values,
        title="每天销售金额变化折线图",
        xlabel="日期",
        ylabel="销售金额",
        tick_step=15,
    )


def plot_city_sales(ax: Axes, df: pd.DataFrame) -> None:
    """图2：不同城市销售数量柱状图。"""
    plot_bar(
        ax,
        sales_by_city(df),
        title="不同城市的销售数量柱状图",
        xlabel="城市",
        ylabel="销售数量",
        color="blue",
    )


def plot_category_ratio(ax: Axes, df: pd.DataFrame) -> None:
    """图3：不同产品类别销售数量占比饼图。"""
    plot_pie_ratio(
        ax,
        sales_by_category(df),
        title="不同产品类型对应的订单比例饼状图",
        textprops={"fontsize": 12},
    )


def plot_payment_ratio(ax: Axes, df: pd.DataFrame) -> None:
    """图4：不同支付方式订单占比饼图。"""
    plot_pie_ratio(
        ax,
        order_count_by_payment(df),
        title="不同支付方式占比饼状图",
        textprops={"fontsize": 15},
    )


def build_report(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (20, 12),
    dpi: int = 100,
) -> Figure:
    """一次性画出 2×2 的完整分析画布并返回 fig。"""
    configure_chinese_font()
    fig, (axes1, axes2, axes3, axes4) = create_layout(figsize=figsize, dpi=dpi)

    plot_daily_sales(axes1, df)
    plot_city_sales(axes2, df)
    plot_category_ratio(axes3, df)
    plot_payment_ratio(axes4, df)

    return fig


def save_figure(fig: Figure, out_path: str | Path, *, dpi: int = 100) -> Path:
    """保存图片，自动创建输出目录。"""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #

def resolve_data_path(csv_arg: str) -> Path:
    """解析 csv 路径：先按当前工作目录找，找不到再去脚本同级目录找。"""
    path = Path(csv_arg).expanduser()
    if path.is_file():
        return path
    if not path.is_absolute():
        beside_script = Path(__file__).resolve().parent / path
        if beside_script.is_file():
            return beside_script
    raise FileNotFoundError(f"找不到数据文件: {csv_arg}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="销售订单统计分析：生成 2×2 统计图表")
    parser.add_argument("--csv", default="sales.csv", help="数据文件路径（默认 sales.csv）")
    parser.add_argument("--out", default="销售订单统计分析.png", help="输出图片路径（默认 销售订单统计分析.png）")
    parser.add_argument("--dpi", type=int, default=100, help="输出图片 DPI（默认 100）")
    parser.add_argument("--encoding", default="utf-8", help="csv 编码（默认 utf-8）")
    parser.add_argument("--show", action="store_true", help="生成后弹窗显示图片（不关闭窗口）")
    parser.add_argument("--no-save", action="store_true", help="只显示不保存图片")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        csv_path = resolve_data_path(args.csv)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    df = load_sales_data(csv_path, encoding=args.encoding)

    print(f"数据文件：{csv_path}")
    print(f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
    for key, value in summarize(df).items():
        print(f"  {key}：{value}")

    fig = build_report(df, dpi=args.dpi)

    if not args.no_save:
        out_path = save_figure(fig, args.out, dpi=args.dpi)
        print(f"图表已保存：{out_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
