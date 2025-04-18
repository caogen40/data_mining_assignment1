import time
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（以SimHei为例）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

start_time = time.time()

paths_10G = [f"../10G_data/part-{str(index).zfill(5)}.parquet" for index in range(8)]
paths_30G = [f"../30G_data/part-{str(index).zfill(5)}.parquet" for index in range(16)]


# 合并所有gender数据
all_incomes = []
for path in paths_30G:
    print("开始读取数据:", path)
    temp_df = pd.read_parquet(path)
    temp_df = temp_df.dropna(subset=['income'])  # 只删除gender列的NA值
    all_incomes.extend(temp_df['income'].tolist())

income_series = pd.Series(all_incomes)

# 绘制收入分布的KDE曲线
plt.figure(figsize=(10, 6))
sns.kdeplot(income_series, fill=True)
plt.title('收入分布的核密度估计曲线')
plt.xlabel('收入')
plt.ylabel('密度')
plt.tight_layout()  # 紧凑型布局
plt.savefig('income_kde_plot.png', dpi=600)
plt.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.4f} 秒")