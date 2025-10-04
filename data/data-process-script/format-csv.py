import pandas as pd
from datetime import datetime

# 直接指定文件名
input_file = "MES_data_1m.csv"  # 替换为您的输入文件名
output_file = "MES_1m.csv"  # 替换为您想要的输出文件名

# 读取数据
df = pd.read_csv(input_file)

# 处理重复的时间戳 - 保留volume较大的行
print("处理重复的时间戳...")
# 按时间戳分组，对于每个时间戳，保留volume最大的行
df = df.sort_values('ts_event')
df = df.groupby('ts_event').apply(lambda x: x.loc[x['volume'].idxmax()]).reset_index(drop=True)

# 转换时间格式并选择所需列
converted_data = []

for _, row in df.iterrows():
    # 转换时间格式
    original_time = row['ts_event']
    if original_time.endswith('Z'):
        original_time = original_time[:-1]
    
    dt = datetime.fromisoformat(original_time)
    time_key = dt.strftime("%Y%m%d%H%M%S") + str(int(dt.microsecond / 1000)).zfill(3)
    
    # 添加转换后的行
    converted_row = {
        'time_key': time_key,
        'open': float(row['open']),
        'high': float(row['high']),
        'low': float(row['low']),
        'close': float(row['close']),
        'volume': int(row['volume'])
    }
    
    converted_data.append(converted_row)

# 创建新的DataFrame并保存
converted_df = pd.DataFrame(converted_data)
converted_df.to_csv(output_file, index=False)

print(f"格式转换完成！")
print(f"转换后的数据保存至: {output_file}")
print(f"共转换 {len(converted_df)} 行数据")