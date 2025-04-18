import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（以SimHei为例）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

start_time = time.time()

paths_10G = [f"../10G_data/part-{str(index).zfill(5)}.parquet" for index in range(8)]
paths_30G = [f"../30G_data/part-{str(index).zfill(5)}.parquet" for index in range(16)]

all_categories = []

for path in paths_30G:
    print("开始读取数据:", path)
    temp_df = pd.read_parquet(path)

    # 仅处理purchase_history列的缺失值
    temp_df = temp_df.dropna(subset=['purchase_history'])


    # 安全解析JSON数据
    def parse_history(x):
        try:
            return json.loads(x) if isinstance(x, str) else x
        except:
            return None


    temp_df['purchase_history'] = temp_df['purchase_history'].apply(parse_history)

    # 提取有效商品类别
    valid_categories = []
    for hist in temp_df['purchase_history']:
        if isinstance(hist, dict) and 'categories' in hist:
            valid_categories.append(hist['categories'])

    all_categories.extend(valid_categories)

# 转换为Series用于分析
category_series = pd.Series(all_categories)

# 绘制购买类别分布
plt.figure(figsize=(14, 8))
category_counts = category_series.value_counts().head(10)  # 仅显示前10个类别
category_counts.plot(kind='bar', title='Top10 商品类别分布')
plt.xlabel('商品类别')
plt.ylabel('购买次数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('purchase_category_distribution.png', dpi=600)
plt.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.4f} 秒")