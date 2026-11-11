"""
销售订单统计分析
基于 sales.csv 数据，生成包含折线图、柱状图和饼状图的综合分析图表。
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def setup_chinese_font():
    """配置matplotlib中文字体显示。"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def load_data(filepath: str = 'sales.csv') -> pd.DataFrame:
    """读取销售数据CSV文件并进行预处理。

    Args:
        filepath: CSV文件路径

    Returns:
        处理后的DataFrame，包含销售金额列
    """
    columns = ['订单号', '产品类别', '产品名称', '销售数量', '单价', '客户所在城市', '支付方式', '订单日期']
    df = pd.read_csv(filepath, usecols=columns, dtype={'年份': 'Int64'})

    # 计算销售金额
    df['销售金额'] = df['销售数量'] * df['单价']

    # 清理产品类别中的空白字符
    df['产品类别'] = df['产品类别'].str.strip()

    return df


def plot_daily_sales_line(axes: Axes, df: pd.DataFrame):
    """绘制每天销售金额变化折线图。

    Args:
        axes: matplotlib子图对象
        df: 销售数据DataFrame
    """
    daily_sales = df.groupby('订单日期')['销售金额'].sum()
    x = daily_sales.index.to_list()
    y = daily_sales.values

    axes.plot(x, y, color='green')
    axes.set_title('每天销售金额变化折线图', fontsize=15)
    axes.set_xlabel('日期', fontsize=12)
    axes.set_ylabel('销售金额', fontsize=12)
    axes.set_xticks(x[::15])
    axes.grid(linestyle='--', alpha=0.5)
    axes.tick_params(axis='x', rotation=45)


def plot_city_sales_bar(axes: Axes, df: pd.DataFrame):
    """绘制不同城市销售数量柱状图。

    Args:
        axes: matplotlib子图对象
        df: 销售数据DataFrame
    """
    city_sales = df.groupby('客户所在城市')['销售数量'].sum().sort_values(ascending=False)
    x = city_sales.index.to_list()
    y = city_sales.values

    axes.bar(x, y, color='blue', width=0.6)
    axes.set_title('不同城市的销售数量柱状图', fontsize=15)
    axes.set_xlabel('城市', fontsize=12)
    axes.set_ylabel('销售数量', fontsize=12)
    axes.grid(linestyle='--', alpha=0.5)
    axes.tick_params(axis='x', rotation=90)


def plot_category_pie(axes: Axes, df: pd.DataFrame):
    """绘制不同产品类型销售数量占比饼状图。

    Args:
        axes: matplotlib子图对象
        df: 销售数据DataFrame
    """
    cate_amount = df.groupby('产品类别')['销售数量'].sum()

    axes.pie(
        cate_amount.values,
        labels=cate_amount.index,
        autopct='%1.1f%%',
        startangle=0,
        radius=1.2,
        textprops={'fontsize': 20}
    )
    axes.set_title('不同产品类型对应的订单比例饼状图', fontsize=15, pad=20)
    axes.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.2))


def plot_payment_pie(axes: Axes, df: pd.DataFrame):
    """绘制不同支付方式占比饼状图。

    Args:
        axes: matplotlib子图对象
        df: 销售数据DataFrame
    """
    pay_method = df.groupby('支付方式')['支付方式'].count()

    axes.pie(
        pay_method.values,
        labels=pay_method.index,
        autopct='%1.1f%%',
        startangle=0,
        radius=1.2,
        textprops={'fontsize': 20}
    )
    axes.set_title('不同支付方式占比饼状图', fontsize=15, pad=20)
    axes.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.2))


def generate_analysis_report(df: pd.DataFrame, save_path: str = '销售订单统计分析.png'):
    """生成完整的销售分析图表并保存。

    Args:
        df: 销售数据DataFrame
        save_path: 图表保存路径
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), dpi=100)
    fig.suptitle('销售订单统计分析', fontsize=23, x=0.5, y=0.98)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    axes1: Axes = axes[0][0]
    axes2: Axes = axes[0][1]
    axes3: Axes = axes[1][0]
    axes4: Axes = axes[1][1]

    # 绘制各子图
    plot_daily_sales_line(axes1, df)
    plot_city_sales_bar(axes2, df)
    plot_category_pie(axes3, df)
    plot_payment_pie(axes4, df)

    # 保存图表
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"图表已保存至: {save_path}")


def main():
    """主函数：执行完整的销售数据分析流程。"""
    # 1. 配置中文字体
    setup_chinese_font()

    # 2. 加载数据
    df = load_data('sales.csv')
    print(f"数据加载完成，共 {len(df)} 条记录")

    # 3. 生成分析图表
    generate_analysis_report(df)


if __name__ == '__main__':
    main()
