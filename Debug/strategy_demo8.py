import sys
import os

# 获取当前脚本(demo.py)所在的目录（即Debug目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录（即project目录，因为Debug的父目录是project）
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到Python的模块搜索路径中
sys.path.append(parent_dir)
print(parent_dir)

import json
import pandas as pd
from typing import Dict, TypedDict

from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid, Scatter
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
import xgboost as xgb
from strategy_demo5 import stragety_feature

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def build_candlestick_df(chan):
    """
    将Chan对象中的K线数据转换为DataFrame，用于pyecharts绘图
    """
    records = []
    for klc in chan[0]:
        for klu in klc:
            records.append({
                "time": klu.time.to_str(),
                "open": klu.open,
                "high": klu.high,
                "low": klu.low,
                "close": klu.close,
                "volume": getattr(klu, "volume", 0),
            })
    df = pd.DataFrame(records)
    # 使用format='mixed'让pandas自动推断每个元素的日期格式
    df["time"] = pd.to_datetime(df["time"], format='mixed')
    return df


def predict_bsp(model: xgb.Booster, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = -9999999
    feature_arr = [missing] * len(meta)
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
    feature_arr = [feature_arr]
    dtest = xgb.DMatrix(feature_arr, missing=missing)
    return model.predict(dtest)


def build_kline(df):
    """
    使用pyecharts构造K线数据
    """
    x = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    y = df[["open", "close", "low", "high"]].values.tolist()
    return x, y


def build_bsp_scatter(df, bsp_records, threshold, display_options):
    """
    构造买卖点散点数据
    """
    buy_points = []
    sell_points = []
    for bsp in bsp_records:
        # 如果选择了只显示有效信号，则过滤
        if "valid_only" in display_options and not bsp["is_valid"]:
            continue
        # 根据新的阈值重新判断有效性
        is_valid = bsp["score"] > threshold
        if "valid_only" in display_options and not is_valid:
            continue

        bsp_time = bsp["time"]
        try:
            bsp_time = pd.to_datetime(bsp_time)
            if bsp_time in df["time"].values:
                idx = df[df["time"] == bsp_time].index[0]
                y_pos = df.iloc[idx]["high"] if bsp["is_buy"] else df.iloc[idx]["low"]
                marker_text = f"{'√' if is_valid else '×'} {bsp['score']:.3f}"
                item = {
                    "name": bsp_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "value": [bsp_time.strftime("%Y-%m-%d %H:%M:%S"), y_pos],
                    "symbol": "pin",
                    "symbol_size": 40,
                    "itemstyle": {
                        "color": "green" if bsp["is_buy"] else "red"
                    },
                    "label": {
                        "show": True,
                        "formatter": marker_text,
                        "color": "black",
                        "font_size": 12,
                        "position": "top" if bsp["is_buy"] else "bottom"
                    }
                }
                if bsp["is_buy"]:
                    buy_points.append(item)
                else:
                    sell_points.append(item)
        except Exception as e:
            print(f"Error processing BSP: {e}")
    return buy_points, sell_points


def build_stat_table(bsp_records, threshold, display_options):
    """
    构造买卖点统计表格
    """
    valid_bsp = []
    for bsp in bsp_records:
        if "valid_only" in display_options and not bsp["is_valid"]:
            continue
        is_valid = bsp["score"] > threshold
        if "valid_only" in display_options and not is_valid:
            continue
        valid_bsp.append({**bsp, "is_valid": is_valid})

    buy_count = sum(1 for bsp in valid_bsp if bsp["is_buy"])
    sell_count = sum(1 for bsp in valid_bsp if not bsp["is_buy"])
    valid_count = sum(1 for bsp in valid_bsp if bsp["is_valid"])

    table = (
        opts.Table(
            title_opts=opts.TitleOpts(title="买卖点统计", pos_left="center"),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        .add(
            headers=["指标", "数量"],
            rows=[
                ["买入信号", f"{buy_count}个"],
                ["卖出信号", f"{sell_count}个"],
                ["有效信号", f"{valid_count}个 (阈值: {threshold})"]
            ]
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="买卖点统计", pos_left="center"),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    return table


if __name__ == "__main__":
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    并使用pyecharts提供Web页面进行交互式展示
    """
    code = "MES"
    begin_time = "20200320220000000"
    end_time = "20200400181400000"
    data_src = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_1M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    model = xgb.Booster()
    model.load_model("model.json")
    meta = json.load(open("feature.meta", "r"))

    treated_bsp_idx = set()
    bsp_records = []  # 存储买卖点记录，用于交互式展示

    for chan_snapshot in chan.step_load():
        # 策略逻辑要对齐demo5
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[0]

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx in treated_bsp_idx or cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        last_bsp.features.add_feat(stragety_feature(last_klu))  # 开仓K线特征

        # 买卖点打分
        score = predict_bsp(model, last_bsp, meta)[0]

        # 根据分数判断有效性（使用0.5作为阈值）
        is_valid = score > 0.3
        marker_text = f"{'√' if is_valid else '×'} {score:.3f}"
        direction = "down" if last_bsp.is_buy else "up"

        # 记录买卖点信息
        bsp_records.append({
            "time": last_bsp.klu.time.to_str(),
            "score": score,
            "is_buy": last_bsp.is_buy,
            "is_valid": is_valid,
            "marker_text": marker_text,
            "direction": direction,
        })

        print(f"{last_bsp.klu.time} - Score: {score:.4f} - {'BUY' if last_bsp.is_buy else 'SELL'} - {'VALID' if is_valid else 'INVALID'}")
        treated_bsp_idx.add(last_bsp.klu.idx)

    # 准备K线数据
    df = build_candlestick_df(chan)

    # 使用pyecharts生成图表
    x_data, y_data = build_kline(df)

    kline = (
        Kline(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
        .add_xaxis(x_data)
        .add_yaxis(
            "K线",
            y_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",
                color0="#14b143",
                border_color="#ef232a",
                border_color0="#14b143",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Chan策略买卖点分析", pos_left="center"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    xaxis_index=[0],
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    type_="slider",
                    xaxis_index=[0],
                    range_start=0,
                    range_end=100,
                    pos_top="95%",
                ),
            ],
        )
    )

    # 构造买卖点散点
    buy_points, sell_points = build_bsp_scatter(df, bsp_records, 0.3, ["kline", "bsp"])

    if buy_points:
        buy_scatter = (
            Scatter()
            .add_xaxis([p["value"][0] for p in buy_points])
            .add_yaxis(
                "买入信号", 
                [p["value"][1] for p in buy_points],
                symbol="pin",
                symbol_size=40,
                itemstyle_opts=opts.ItemStyleOpts(color="green"),
                label_opts=opts.LabelOpts(is_show=True, position="top")
            )
        )
        kline.overlap(buy_scatter)

    if sell_points:
        sell_scatter = (
            Scatter()
            .add_xaxis([p["value"][0] for p in sell_points])
            .add_yaxis(
                "卖出信号", 
                [p["value"][1] for p in sell_points],
                symbol="pin",
                symbol_size=40,
                itemstyle_opts=opts.ItemStyleOpts(color="red"),
                label_opts=opts.LabelOpts(is_show=True, position="bottom")
            )
        )
        kline.overlap(sell_scatter)

    # 生成HTML文件
    kline.render("chan_strategy_demo8.html")
    print("\n已生成 chan_strategy_demo8.html，请用浏览器打开查看交互式图表")
