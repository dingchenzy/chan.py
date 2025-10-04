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
from DataAPI.csvAPI import CSV_API


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
    
def align_klu_by_time(base_klu, kl_dict: Dict, lv_list: list):

    """
    根据60分钟K线的时间，对齐其他级别K线
    返回对应时间窗口内的K线列表
    使用上根K线的结束时间和当前K线的结束时间之间的K线组成
    确保父子级别K线在同一天内，避免时间不一致错误
    """
    aligned = {KL_TYPE.K_60M: [base_klu]}

    # 获取时间范围：从上根K线的结束时间到当前K线的结束时间
    if base_klu.pre is not None:
        base_begin = base_klu.pre.time
    else:
        # 如果没有前一根K线，使用当前K线时间作为开始时间
        base_begin = base_klu.time
    base_end = base_klu.time
    
    # 获取基准K线的日期，用于过滤同一天的K线
    base_date = base_klu.time
    
    for lv in lv_list:
        if lv == KL_TYPE.K_60M:
            continue
        aligned[lv] = []
        for klu in kl_dict[lv]:
            # 检查时间范围和日期一致性
            if (klu.time > base_begin and klu.time <= base_end and
                klu.time.year == base_date.year and 
                klu.time.month == base_date.month and 
                klu.time.day == base_date.day):
                aligned[lv].append(klu)
    return aligned
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
    # 多级别：1m,5m,15m,30m,60m,240m
    lv_list = [KL_TYPE.K_60M, KL_TYPE.K_30M,  KL_TYPE.K_15M, KL_TYPE.K_5M, KL_TYPE.K_1M]

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
        "print_err_time": True,
        "kl_data_check": True,
        "max_kl_inconsistent_cnt": 1000,
        "max_kl_misalgin_cnt": 1000,
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
    
    CSV_API.do_init()
    data_src_1h = CSV_API(code, k_type=KL_TYPE.K_60M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    data_src_30m = CSV_API(code, k_type=KL_TYPE.K_30M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    kl_30m_all = list(data_src_30m.get_kl_data())
    
    data_src_15m = CSV_API(code, k_type=KL_TYPE.K_15M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    kl_15m_all = list(data_src_15m.get_kl_data())
    
    data_src_5m = CSV_API(code, k_type=KL_TYPE.K_5M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    kl_5m_all = list(data_src_5m.get_kl_data())
    
    data_src_1m = CSV_API(code, k_type=KL_TYPE.K_1M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    kl_1m_all = list(data_src_1m.get_kl_data())
    
    # 建立K线之间的前后关系
    def setup_kline_links(kl_list):
        """为K线列表建立前后链接关系"""
        for i in range(1, len(kl_list)):
            kl_list[i].set_pre_klu(kl_list[i-1])
    
    # 为所有级别的K线建立链接关系
    kl_60m_all = list(data_src_1h.get_kl_data())
    setup_kline_links(kl_60m_all)
    setup_kline_links(kl_30m_all)
    setup_kline_links(kl_15m_all)
    setup_kline_links(kl_5m_all)
    setup_kline_links(kl_1m_all)

    # 组织成字典方便对齐
    kl_dict = {
        KL_TYPE.K_60M: kl_60m_all,
        KL_TYPE.K_30M: kl_30m_all,
        KL_TYPE.K_15M: kl_15m_all,
        KL_TYPE.K_5M: kl_5m_all,
        KL_TYPE.K_1M: kl_1m_all,
    }
    
    # 按60分钟级别逐根触发，并对齐其他级别K线
    for klu_60m in kl_dict[KL_TYPE.K_60M]:
        aligned_klu = align_klu_by_time(klu_60m, kl_dict, lv_list)
        chan.trigger_load(aligned_klu)
    
    CSV_API.do_close()
    
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[0]

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            print(last_bsp.klu.time, last_bsp.is_buy)

    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_latest_bsp(idx=0)]
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
    # plot(chan, plot_marker)

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
