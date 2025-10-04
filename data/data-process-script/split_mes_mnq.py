import pandas as pd

# 直接指定文件名
input_file = "glbx-mdp3-20250928-20251003.ohlcv-1m.csv"  # 替换为您的文件名

# 读取数据
df = pd.read_csv(input_file)

# 筛选MES和MNQ数据，并修改symbol列
mes_data = df[df['symbol'].str.startswith('MES')].copy()
mes_data['symbol'] = 'MES'

# mnq_data = df[df['symbol'].str.startswith('MNQ')].copy()
# mnq_data['symbol'] = 'MNQ'

# 保存到新文件
mes_data.to_csv("MES_data_1m.csv", index=False)
# mnq_data.to_csv("MNQ_data.csv", index=False)

print(f"MES数据已保存到 MES_data.csv，包含 {len(mes_data)} 行")
# print(f"MNQ数据已保存到 MNQ_data.csv，包含 {len(mnq_data)} 行")