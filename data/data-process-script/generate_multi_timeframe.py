#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线数据时间周期转换脚本
将1分钟OHLCV数据转换为5分钟、15分钟、30分钟、1小时、4小时数据
"""

import pandas as pd
from datetime import datetime
import os

def parse_time_key(time_key):
    """
    解析时间戳格式 YYYYMMDDHHMMSS000
    """
    time_str = str(time_key)[:14]  # 取前14位，去掉末尾的000
    return datetime.strptime(time_str, '%Y%m%d%H%M%S')

def format_time_key(dt):
    """
    将datetime对象转换回时间戳格式
    """
    return int(dt.strftime('%Y%m%d%H%M%S') + '000')

def aggregate_kline_data(df, timeframe_minutes):
    """
    聚合K线数据到指定时间周期
    
    Args:
        df: 包含1分钟数据的DataFrame
        timeframe_minutes: 目标时间周期（分钟）
    
    Returns:
        聚合后的DataFrame
    """
    # 创建副本避免修改原数据
    data = df.copy()
    
    # 解析时间戳
    data['datetime'] = data['time_key'].apply(parse_time_key)
    
    # 设置datetime为索引
    data.set_index('datetime', inplace=True)
    
    # 按指定时间周期重采样
    freq = f'{timeframe_minutes}min'  # min表示分钟
    
    # 聚合规则：开盘价取第一个，收盘价取最后一个，最高价取最大值，最低价取最小值，成交量求和
    aggregated = data.resample(freq).agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # 删除包含NaN的行（可能由于数据不完整导致）
    aggregated = aggregated.dropna()
    
    # 重新生成time_key
    aggregated['time_key'] = aggregated.index.map(format_time_key)
    
    # 重新排列列的顺序
    aggregated = aggregated[['time_key', 'open', 'high', 'low', 'close', 'volume']]
    
    # 重置索引
    aggregated.reset_index(drop=True, inplace=True)
    
    return aggregated

def main():
    """
    主函数：读取1分钟数据并生成多时间周期数据
    """
    # 输入文件路径
    input_file = "MES_1m.csv"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    print("正在读取1分钟数据...")
    try:
        # 读取CSV文件
        df_1m = pd.read_csv(input_file)
        print(f"成功读取 {len(df_1m)} 条1分钟数据")
        
        # 显示数据基本信息
        print(f"数据时间范围：{df_1m['time_key'].min()} 到 {df_1m['time_key'].max()}")
        
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return
    
    # 生成不同时间周期的数据
    timeframes = [5, 15, 30, 60, 240]  # 增加1小时(60分钟)和4小时(240分钟)
    
    for timeframe in timeframes:
        print(f"\n正在生成 {timeframe} 分钟数据...")
        
        try:
            # 聚合数据
            df_aggregated = aggregate_kline_data(df_1m, timeframe)
            
            # 输出文件名
            output_file = f"MES_{timeframe}m.csv"
            
            # 保存到CSV文件
            df_aggregated.to_csv(output_file, index=False)
            
            print(f"成功生成 {len(df_aggregated)} 条 {timeframe} 分钟数据，保存到 {output_file}")
            
            # 显示前几行数据作为验证
            print(f"{timeframe}分钟数据示例：")
            print(df_aggregated.head(3))
            
        except Exception as e:
            print(f"生成 {timeframe} 分钟数据时出错：{e}")
    
    print("\n所有时间周期数据生成完成！")

if __name__ == "__main__":
    main()