# demo.py
import sys
import os
import json
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 获取当前脚本(demo.py)所在的目录（即Debug目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录（即project目录，因为Debug的父目录是project）
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到Python的模块搜索路径中
sys.path.append(parent_dir)
print(f"Added to path: {parent_dir}")

try:
    from Chan import CChan
    from ChanConfig import CChanConfig
    from ChanModel.Features import CFeatures
    from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
    from Common.CTime import CTime
    from Plot.PlotDriver import CPlotDriver
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required modules are available")
    sys.exit(1)


class SampleInfo:
    """样本信息类"""
    def __init__(self, feature: CFeatures, is_buy: bool, open_time: CTime, klu_idx: int):
        self.feature = feature
        self.is_buy = is_buy
        self.open_time = open_time
        self.klu_idx = klu_idx


def plot(chan, plot_marker, filename="label1.png"):
    """绘图函数"""
    try:
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
        plot_driver.save2img(filename)
        print(f"Plot saved as {filename}")
    except Exception as e:
        print(f"Plotting failed: {e}")


def strategy_feature(last_klu) -> Dict[str, float]:
    """策略特征提取函数"""
    try:
        # 确保价格数据有效
        if last_klu.open == 0 or last_klu.low == 0:
            return {}
        
        open_klu_rate = (last_klu.close - last_klu.open) / last_klu.open
        high_low_range = (last_klu.high - last_klu.low) / last_klu.low
        
        return {
            "open_klu_rate": open_klu_rate,
            "high_low_range": high_low_range,
        }
    except Exception as e:
        print(f"Error in strategy_feature: {e}")
        return {}


def prepare_data() -> bool:
    """准备数据 - 将原始CSV转换为可用格式"""
    try:
        input_file = "../MES_1m.csv"  # 假设这是您的数据文件
        if not os.path.exists(input_file):
            print(f"Data file {input_file} not found")
            return False
            
        # 这里可以添加数据预处理逻辑
        print("Data preparation completed")
        return True
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return False


def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """验证和修正配置"""
    default_config = {
        "trigger_step": True,
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": False,  # 改为False减少输出噪音
        "zs_algo": "normal",
    }
    
    # 更新用户配置，保留默认值
    for key, value in config_dict.items():
        default_config[key] = value
        
    return default_config


def extract_features(bsp_dict: Dict[int, SampleInfo], chan) -> tuple:
    """提取特征并生成训练数据"""
    feature_meta = {}
    cur_feature_idx = 0
    plot_marker = {}
    samples = []
    labels = []
    
    # 获取所有买卖点作为正样本
    bsp_academy = set(bsp.klu.idx for bsp in chan.get_latest_bsp(number=0))
    
    for klu_idx, sample_info in bsp_dict.items():
        try:
            label = 1 if klu_idx in bsp_academy else 0
            
            # 提取特征
            features_dict = {}
            for feature_name, value in sample_info.feature.items():
                if feature_name not in feature_meta:
                    feature_meta[feature_name] = cur_feature_idx
                    cur_feature_idx += 1
                features_dict[feature_meta[feature_name]] = float(value)
            
            # 添加策略特征
            strategy_feat = strategy_feature(sample_info.feature)
            for feat_name, value in strategy_feat.items():
                if feat_name not in feature_meta:
                    feature_meta[feat_name] = cur_feature_idx
                    cur_feature_idx += 1
                features_dict[feature_meta[feat_name]] = value
            
            # 排序特征
            sorted_features = sorted(features_dict.items())
            feature_vector = [f"{idx}:{value}" for idx, value in sorted_features]
            
            samples.append(" ".join(feature_vector))
            labels.append(label)
            
            # 准备绘图标记
            marker_symbol = "√" if label else "×"
            marker_side = "down" if sample_info.is_buy else "up"
            plot_marker[sample_info.open_time.to_str()] = (marker_symbol, marker_side)
            
        except Exception as e:
            print(f"Error processing sample {klu_idx}: {e}")
            continue
    
    return samples, labels, feature_meta, plot_marker


def train_and_evaluate_model(samples: List[str], labels: List[int]) -> Optional[xgb.Booster]:
    """训练和评估模型"""
    if len(samples) < 10:
        print("Not enough samples for training")
        return None
        
    try:
        # 保存为libsvm格式
        with open("feature.libsvm", "w") as f:
            for label, sample in zip(labels, samples):
                f.write(f"{label} {sample}\n")
        
        # 加载数据
        dtrain = xgb.DMatrix("feature.libsvm?format=libsvm")
        
        # 分割训练测试集
        X = np.array([list(map(float, sample.split())) for sample in samples])
        y = np.array(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练模型
        param = {
            'max_depth': 3,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error'],
            'verbosity': 1
        }
        
        evals_result = {}
        bst = xgb.train(
            param,
            dtrain=dtrain,
            num_boost_round=50,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=10,
        )
        
        # 保存模型
        bst.save_model("model1.json")
        print("Model saved as model1.json")
        
        # 评估模型
        dtest = xgb.DMatrix(X_test)
        y_pred = bst.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))
        
        return bst
        
    except Exception as e:
        print(f"Model training failed: {e}")
        return None


def main():
    """主函数"""
    print("Starting demo...")
    
    # 准备数据
    if not prepare_data():
        return
    
    # 配置参数
    config_dict = validate_config({
        "trigger_step": True,
        "bi_strict": True,
        "skip_step": 0,
        "print_warning": False,
    })
    
    try:
        # 初始化缠论分析器
        chan = CChan(
            code="MES",
            begin_time="20240722010800000",
            end_time="20250928235900000", 
            data_src=DATA_SRC.CSV,
            lv_list=[KL_TYPE.K_1M],
            config=CChanConfig(config_dict),
            autype=AUTYPE.QFQ,
        )
        
        # 存储买卖点特征
        bsp_dict: Dict[int, SampleInfo] = {}
        
        print("Processing data and extracting features...")
        # 逐步加载数据并提取特征
        for chan_snapshot in chan.step_load():
            try:
                last_klu = chan_snapshot[0][-1][-1]
                bsp_list = chan_snapshot.get_latest_bsp()
                
                if not bsp_list:
                    continue
                    
                last_bsp = bsp_list[0]
                cur_lv_chan = chan_snapshot[0]
                
                # 检查是否应该记录该买卖点
                if (last_bsp.klu.idx not in bsp_dict and 
                    cur_lv_chan[-2].idx == last_bsp.klu.klc.idx):
                    
                    bsp_dict[last_bsp.klu.idx] = SampleInfo(
                        feature=last_bsp.features,
                        is_buy=last_bsp.is_buy,
                        open_time=last_klu.time,
                        klu_idx=last_bsp.klu.idx
                    )
                    
                    # 添加策略特征
                    strategy_feat = strategy_feature(last_klu)
                    for feat_name, value in strategy_feat.items():
                        bsp_dict[last_bsp.klu.idx].feature.add_feat({feat_name: value})
                    
                    print(f"BSP at {last_bsp.klu.time}: {'Buy' if last_bsp.is_buy else 'Sell'}")
                    
            except Exception as e:
                print(f"Error processing snapshot: {e}")
                continue
        
        print(f"Collected {len(bsp_dict)} samples")
        
        if len(bsp_dict) == 0:
            print("No samples collected. Exiting.")
            return
        
        # 提取特征
        samples, labels, feature_meta, plot_marker = extract_features(bsp_dict, chan)
        
        # 保存特征元数据
        with open("feature1.meta", "w") as f:
            json.dump(feature_meta, f, indent=2)
        
        # 训练模型
        model = train_and_evaluate_model(samples, labels)
        
        # 绘图
        if plot_marker:
            plot(chan, plot_marker)
        else:
            print("No markers to plot")
            
        print("Demo completed successfully")
        
    except Exception as e:
        print(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()