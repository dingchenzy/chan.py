# demo.py
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
from typing import Dict, TypedDict

import xgboost as xgb

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


def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 400,
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("label.png")


def stragety_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
    }


if __name__ == "__main__":
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    code = "MES"
    begin_time = "20190505220000000"
    end_time = "20250928235900000"
    data_src = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_5M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
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

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[0]

        cur_lv_chan = chan_snapshot[0]  # 获取当前级别的K线列表（第0级别，即5分钟级别）
        
        # 判断买卖点是否满足交易条件：
        # 1. last_bsp.klu.idx not in bsp_dict: 该买卖点尚未被记录过（避免重复交易）
        # 2. cur_lv_chan[-2].idx == last_bsp.klu.klc.idx: 买卖点所在的合并K线是倒数第二根K线
        #    这意味着买卖点分形已经确认（第三元素已出现），可以进行交易
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            # 将买卖点信息存储到字典中，用于后续特征分析和模型训练
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,  # 买卖点的特征数据
                "is_buy": last_bsp.is_buy,     # 是否为买点（True）或卖点（False）
                "open_time": last_klu.time,    # 开仓时间（当前最新K线的时间）
            }
            # 打印买卖点字典内容
            print("bsp_dict 内容:", bsp_dict)
            # 为该买卖点添加策略特征（基于开仓K线的特征）
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            # 打印买卖点信息：时间和买卖方向
            print(last_bsp.klu.time, last_bsp.is_buy)

    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_latest_bsp(number=0)]
    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fid = open("feature.libsvm", "w")
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
        features = []  # List[(idx, value)]
        for feature_name, value in feature_info['feature'].items():
            if feature_name not in feature_meta:
                feature_meta[feature_name] = cur_feature_idx
                cur_feature_idx += 1
            features.append((feature_meta[feature_name], value))
        features.sort(key=lambda x: x[0])
        feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        fid.write(f"{label} {feature_str}\n")
        plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")
    fid.close()

    # 将特征与对应索引保存下来，下次可直接对其，防止特征与索引不对应
    with open("feature.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    plot(chan, plot_marker)

    # load sample file & train model
    dtrain = xgb.DMatrix("feature.libsvm?format=libsvm")  # load sample
    param = {'max_depth': 2, 'eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    evals_result = {}
    bst = xgb.train(
        param,
        dtrain=dtrain,
        num_boost_round=10,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
        verbose_eval=True,
    )
    bst.save_model("model.json")
    # 打印训练过程中的AUC
    print("训练集AUC:", evals_result["train"]["auc"])

    # load model
    model = xgb.Booster()
    model.load_model("model.json")
    # predict
    print(model.predict(dtrain))
