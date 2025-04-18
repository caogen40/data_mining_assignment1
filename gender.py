import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（以SimHei为例）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

start_time = time.time()

paths_10G = [f"../10G_data/part-{str(index).zfill(5)}.parquet" for index in range(8)]
paths_30G = [f"../30G_data/part-{str(index).zfill(5)}.parquet" for index in range(16)]

# 合并所有gender数据
all_genders = []
for path in paths_30G:
    print("开始读取数据:", path)
    temp_df = pd.read_parquet(path)
    temp_df = temp_df.dropna(subset=['gender'])  # 只删除gender列的NA值
    all_genders.extend(temp_df['gender'].tolist())

# 转换为Series用于分析
gender_series = pd.Series(all_genders)

# 绘制性别分布
gender_counts = gender_series.value_counts()
plt.figure(figsize=(10, 6))
gender_counts.plot(kind='pie', autopct='%1.1f%%', title='性别分布')
plt.ylabel('')
plt.tight_layout()
plt.savefig('gender_distribution.png', dpi=600)
plt.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.4f} 秒")